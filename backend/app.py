from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import time
import os
import sys

# Attempt to import GLiNER
try:
    from gliner2 import GLiNER2
except ImportError:
    # Fallback to standard gliner if gliner2 isn't available
    from gliner import GLiNER as GLiNER2

app = Flask(__name__)
# Enable CORS for all routes (since we use a separate Vite frontend)
CORS(app)

# ==========================================
# 1. INITIALIZE MODEL
# ==========================================
# We expect the model to be extracted in the 'model' directory in the project root
# Or in D:/AADHAAR 2 Final/model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Check if model exists, if not, fallback to D:/Adhar/Fine_tuned
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    fallback_dir = "D:\\Adhar\\Fine_tuned"
    if os.path.exists(os.path.join(fallback_dir, "config.json")):
        MODEL_DIR = fallback_dir
    else:
        print(f"WARNING: No model found at {MODEL_DIR} or {fallback_dir}!")

print(f"--- Initializing Inference Engine from {MODEL_DIR} ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    eval_model = GLiNER2.from_pretrained(MODEL_DIR).to(device)
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"ERROR loading model: {e}")
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

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    if eval_model is None:
        return jsonify({"error": "Model failed to initialize on server startup."}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No 'text' field provided in JSON body."}), 400

    text = data['text']
    # Use custom labels if provided, else defaults
    labels = data.get('labels', DEFAULT_LABELS)
    threshold = data.get('threshold', 0.5)

    print(f"\nAnalyzing text (length {len(text)})...")
    start_time = time.time()
    
    try:
        # Run GLiNER inference
        raw_result = eval_model.extract_entities(text, labels, threshold=threshold)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Clean processing specific to gliner2's dict structure
        formatted_entities = []
        
        if isinstance(raw_result, dict):
            # Sometimes wrapped in 'entities'
            entities_dict = raw_result.get('entities', raw_result)
            
            for label, matches in entities_dict.items():
                for match in matches:
                    if isinstance(match, str):
                        formatted_entities.append({
                            "label": label,
                            "text": match,
                            "confidence": 100.0
                        })
                    elif isinstance(match, dict):
                        text_val = match.get('text', str(match))
                        score = match.get('confidence', match.get('score', 0.0))
                        # Normalize score (sometimes it's 0-1, sometimes 0-100)
                        if score <= 1.0 and isinstance(score, float):
                            score *= 100
                        
                        formatted_entities.append({
                            "label": label,
                            "text": text_val,
                            "confidence": round(score, 2)
                        })
        elif isinstance(raw_result, list):
            # Standard GLiNER format fallback
            for ent in raw_result:
                formatted_entities.append({
                    "label": ent.get('label', 'Unknown'),
                    "text": ent.get('text', ''),
                    "confidence": round(ent.get('score', 1.0) * 100, 2)
                })

        return jsonify({
            "success": True,
            "inference_time_ms": round(inference_time * 1000, 2),
            "entities": formatted_entities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": eval_model is not None,
        "device": str(device)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
