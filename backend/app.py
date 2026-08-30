"""
backend/app.py
==============
FastAPI service wrapping a fine-tuned GLiNER2 extractor for Indian financial PII.

Request flow:
  1. EXTRACT   — GLiNER2 proposes labelled spans.
  2. RECALL    — regex sweeps the text for well-formed identifiers the model missed.
  3. VALIDATE  — each candidate is checked against the structure its label requires.
  4. ARBITRATE — overlapping survivors are resolved by structural specificity.

Steps 3 and 4 live in `validation.py` and are covered by `test_validation.py`,
which runs without the model.
"""

from contextlib import asynccontextmanager
from typing import Any, List, Optional
import os
import re
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from validation import REGEX_PATTERNS, validate_and_arbitrate

# GLiNER2 is a hard requirement. The previous version fell back to GLiNER v1 on
# ImportError, which is a different package with a different API: v1 exposes
# predict_entities, not extract_entities, so the fallback turned a missing
# dependency into an AttributeError on every request instead of a startup error.
try:
    from gliner2 import GLiNER2
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "gliner2 is required but not installed. The fine-tuned weights are a "
        "GLiNER2 'extractor' model and cannot be loaded by the gliner (v1) "
        "package. Install it with: pip install gliner2"
    ) from exc

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")

# Browsers reject credentialed requests against a wildcard origin, so pairing
# allow_origins=["*"] with allow_credentials=True never worked as intended.
# Default to the dev frontend and let deployments name their own origins.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]

DEFAULT_LABELS = [
    "Aadhaar Number",
    "VPA",
    "IFSC Code",
    "Bank Name",
    "Transaction ID",
    "Driving Licence",
    "PAN Number",
    "Account Number",
    "Beneficiary Name",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_model: Optional[Any] = None


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    labels: Optional[List[str]] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def _load_model() -> None:
    global eval_model
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(
            f"SEVERE: no config.json under MODEL_DIR={MODEL_DIR}. The weights are "
            "not baked into the image; they are mounted at runtime. Fetch them "
            "with: huggingface-cli download VK1402/AADHAAR_Extractor "
            "--local-dir ./model"
        )
        eval_model = None
        return
    try:
        eval_model = GLiNER2.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
        print(f"SUCCESS: model loaded from {MODEL_DIR} on {device}.")
    except Exception as exc:
        print(f"SEVERE: failed to load model from {MODEL_DIR}: {exc}")
        eval_model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Indian Financial PII Extractor",
    description="Fine-tuned NER with a structural validation layer.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _locate(haystack: str, needle: str, claimed: set[tuple[int, int]]) -> tuple[int, int]:
    """Find the first occurrence of `needle` not already claimed by another span."""
    if not needle:
        return -1, -1
    start = haystack.find(needle)
    while start != -1:
        span = (start, start + len(needle))
        if span not in claimed:
            claimed.add(span)
            return span
        start = haystack.find(needle, start + 1)
    return -1, -1


def _normalise_score(raw: Any) -> float:
    """Model scores arrive as 0-1 probabilities or 0-100 percentages."""
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # The old guard was isinstance(score, float), which silently skipped ints,
    # so an integer score of 1 was reported as 1% instead of 100%.
    return round(score * 100, 2) if score <= 1.0 else round(score, 2)


def _model_candidates(raw: Any, text: str, claimed: set) -> list[dict]:
    out: list[dict] = []

    def add(label: str, value: str, score: Any) -> None:
        value = str(value).strip()
        if not value:
            return
        start, end = _locate(text, value, claimed)
        out.append({
            "label": label,
            "text": value,
            "start": start,
            "end": end if end >= 0 else -1,
            "confidence": _normalise_score(score),
            "source": "model",
        })

    if isinstance(raw, dict):
        entities = raw.get("entities", raw)
        if isinstance(entities, dict):
            for label, matches in entities.items():
                for match in matches or []:
                    if isinstance(match, str):
                        add(label, match, 1.0)
                    elif isinstance(match, dict):
                        add(
                            label,
                            match.get("text", ""),
                            match.get("confidence", match.get("score", 0.0)),
                        )
    elif isinstance(raw, list):
        for ent in raw:
            if isinstance(ent, dict):
                add(ent.get("label", "Unknown"), ent.get("text", ""), ent.get("score", 1.0))
    return out


def _regex_candidates(text: str, labels: list[str], claimed: set) -> list[dict]:
    """Sweep for well-formed identifiers the model may have missed.

    These are recall, not truth: every one still goes through validation and
    arbitration alongside the model's own candidates.
    """
    out: list[dict] = []
    for label in labels:
        pattern = REGEX_PATTERNS.get(label)
        if not pattern:
            continue
        flags = re.IGNORECASE if label == "VPA" else 0
        for match in re.finditer(pattern, text, flags=flags):
            span = match.span()
            if span in claimed:
                continue
            claimed.add(span)
            out.append({
                "label": label,
                "text": match.group(),
                "start": span[0],
                "end": span[1],
                # A pattern match is not a probability. Reporting it as 100%
                # confidence overstated it; provenance is carried by `source`.
                "confidence": None,
                "source": "regex",
            })
    return out


@app.post("/api/extract")
def extract_entities(payload: ExtractRequest) -> dict[str, Any]:
    if eval_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. See server logs; weights mount at MODEL_DIR.",
        )

    text = payload.text
    labels = payload.labels or DEFAULT_LABELS

    print(f"Analyzing text (length {len(text)})...")
    start = time.perf_counter()

    try:
        raw = eval_model.extract_entities(text, labels, threshold=payload.threshold)
    except Exception as exc:
        # Never echo the exception to the client: this endpoint handles PII and
        # the message can embed fragments of the submitted text.
        print(f"ERROR during inference: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    inference_ms = (time.perf_counter() - start) * 1000

    claimed: set[tuple[int, int]] = set()
    candidates = _model_candidates(raw, text, claimed)
    candidates += _regex_candidates(text, labels, claimed)

    kept, rejected = validate_and_arbitrate(candidates)
    total_ms = (time.perf_counter() - start) * 1000

    aadhaar = [
        re.sub(r"\D", "", e["text"])
        for e in kept
        if e["label"].strip().lower() == "aadhaar number"
    ]
    aadhaar = list(dict.fromkeys(aadhaar))

    print(
        f"  {len(candidates)} candidates -> {len(kept)} kept, "
        f"{len(rejected)} rejected ({inference_ms:.0f}ms inference)"
    )

    return {
        "success": True,
        "inference_time_ms": round(inference_ms, 2),
        "total_time_ms": round(total_ms, 2),
        "entities": kept,
        "rejected": rejected,
        "counts": {
            "candidates": len(candidates),
            "kept": len(kept),
            "rejected": len(rejected),
        },
        "aadhaar_number": aadhaar[0] if aadhaar else None,
        "aadhaar_numbers": aadhaar,
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy" if eval_model is not None else "degraded",
        "model_loaded": eval_model is not None,
        "model_dir": MODEL_DIR,
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
