from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import time
import os
import sys
import re

# Attempt to import GLiNER
try:
    from gliner2 import GLiNER2
except ImportError:
    # Fallback to standard gliner if gliner2 isn't available
    from gliner import GLiNER as GLiNER2

# ==========================================
# 1. INITIALIZE MODEL & REGEX PATTERNS
# ==========================================
MODEL_DIR = "/app/model"
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(
        f"WARNING: Model config missing at {MODEL_DIR}/config.json. "
        "Startup will continue, but inference endpoints will return errors until model is available."
    )

print(f"--- Initializing Inference Engine from {MODEL_DIR} ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_model = None

# Default labels mapping
DEFAULT_LABELS = [
    "Aadhaar Number", 
    "VPA", 
    "IFSC Code", 
    "Bank Name", 
    "Transaction ID", 
    "Driving Licence",
    "PAN Number",
    "Account Number",
    "Beneficiary Name"
]

# Strict Regex Patterns for Indian KYC and Banking Data
REGEX_PATTERNS = {
    "Aadhaar Number": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "PAN Number": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "IFSC Code": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "VPA": r"\b[\w.\-_]+@[a-zA-Z]+\b",
    "Driving Licence": r"\b[A-Z]{2}\d{2}[-\s]?\d{4}[-\s]?\d{7}\b",
    "Account Number": r"\b\d{9,18}\b"
}

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    labels: Optional[List[str]] = None
    threshold: float = Field(default=0.5, ge=0.0)


def _load_model() -> None:
    global eval_model
    try:
        eval_model = GLiNER2.from_pretrained(
            MODEL_DIR, 
            local_files_only=True, 
            proxies=None,          
            resume_download=True   
        ).to(device)
        print("SUCCESS: Model loaded.")
    except Exception as e:
        print(
            "SEVERE: Failed to load GLiNER2 model during startup. "
            f"MODEL_DIR={MODEL_DIR}. Error: {e}"
        )
        eval_model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/extract")
def extract_entities(payload: ExtractRequest) -> dict[str, Any]:
    if eval_model is None:
        raise HTTPException(status_code=500, detail="Model failed to initialize on server startup.")

    text = payload.text
    labels = payload.labels or DEFAULT_LABELS
    threshold = payload.threshold

    print(f"\nAnalyzing text (length {len(text)})...")
    start_time = time.time()

    try:
        # 1. GLiNER Inference Extraction
        raw_result = eval_model.extract_entities(text, labels, threshold=threshold)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")

        formatted_entities: list[dict[str, Any]] = []

        if isinstance(raw_result, dict):
            entities_dict = raw_result.get("entities", raw_result)

            if isinstance(entities_dict, dict):
                for label, matches in entities_dict.items():
                    for match in matches:
                        if isinstance(match, str):
                            formatted_entities.append(
                                {"label": label, "text": match, "confidence": 100.0, "source": "model"}
                            )
                        elif isinstance(match, dict):
                            text_val = match.get("text", str(match))
                            score = match.get("confidence", match.get("score", 0.0))
                            if isinstance(score, float) and score <= 1.0:
                                score *= 100

                            formatted_entities.append(
                                {
                                    "label": label,
                                    "text": text_val,
                                    "confidence": round(float(score), 2),
                                    "source": "model"
                                }
                            )
        elif isinstance(raw_result, list):
            for ent in raw_result:
                if isinstance(ent, dict):
                    formatted_entities.append(
                        {
                            "label": ent.get("label", "Unknown"),
                            "text": ent.get("text", ""),
                            "confidence": round(float(ent.get("score", 1.0)) * 100, 2),
                            "source": "model"
                        }
                    )

        # 2. Regex Hard-Match Extraction
        for target_label in labels:
            pattern = REGEX_PATTERNS.get(target_label)
            if pattern:
                # Use IGNORECASE for VPA to catch varying casing in email formats
                flags = re.IGNORECASE if target_label == "VPA" else 0
                for match in re.finditer(pattern, text, flags=flags):
                    matched_text = match.group()
                    
                    # Deduplication check against model outputs
                    already_exists = any(
                        e["label"].lower() == target_label.lower() and e["text"] == matched_text 
                        for e in formatted_entities
                    )
                    
                    if not already_exists:
                        formatted_entities.append({
                            "label": target_label,
                            "text": matched_text,
                            "confidence": 100.0,
                            "source": "regex"
                        })

        # 3. Aadhaar Specific Parsing
        aadhaar_candidates: list[str] = []
        for entity in formatted_entities:
            label = str(entity.get("label", "")).strip().lower()
            if label != "aadhaar number":
                continue

            entity_text = str(entity.get("text", "")).strip()
            digits_only = re.sub(r"\D", "", entity_text)
            
            if len(digits_only) == 12:
                aadhaar_candidates.append(digits_only)
            elif entity_text:
                aadhaar_candidates.append(entity_text)

        # Keep order and remove duplicates.
        unique_aadhaar_numbers = list(dict.fromkeys(aadhaar_candidates))

        return {
            "success": True,
            "inference_time_ms": round(inference_time * 1000, 2),
            "entities": formatted_entities,
            "aadhaar_number": unique_aadhaar_numbers[0] if unique_aadhaar_numbers else None,
            "aadhaar_numbers": unique_aadhaar_numbers,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": eval_model is not None,
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)