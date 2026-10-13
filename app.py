"""
Flask backend + web page for the Gastric Tissue Screening tool.

Run:   python app.py
Open:  http://127.0.0.1:5000

The decision threshold defaults to 0.50 (the setting with a measured
test-set sensitivity of about 86%). To try another value without editing
the file, in PowerShell:   $env:THRESHOLD="0.65"; python app.py
"""

import io
import os

import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, UnidentifiedImageError

from model import build_model
from dataset import eval_transform, LABEL_TO_IDX

# ---- CONFIG ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_PATH = os.path.join(BASE_DIR, "gastric_cancer_model_finetuned.pth")
ORIGINAL_PATH = os.path.join(BASE_DIR, "gastric_cancer_model.pth")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.50"))
MAX_UPLOAD_MB = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prefer the fine-tuned model; fall back to the original one if it is missing
if os.path.exists(FINETUNED_PATH):
    MODEL_PATH = FINETUNED_PATH
elif os.path.exists(ORIGINAL_PATH):
    MODEL_PATH = ORIGINAL_PATH
    print("WARNING: fine-tuned model not found, using the original model.")
else:
    raise FileNotFoundError(
        "No model file found. Put gastric_cancer_model_finetuned.pth "
        "in the same folder as app.py."
    )

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---- LOAD MODEL ONCE AT STARTUP ----
print(f"Loading model from {os.path.basename(MODEL_PATH)} ...")
model = build_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded. Decision threshold: {THRESHOLD}")


@app.after_request
def add_cors_headers(resp):
    # Lets index.html work even if it is opened straight from disk
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"The image is larger than {MAX_UPLOAD_MB} MB."}), 413


FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def frontend_files(filename):
    # Serves style.css, script.js and anything else in the frontend folder
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_file": os.path.basename(MODEL_PATH),
        "threshold_percent": round(THRESHOLD * 100, 1),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image was sent."}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The file could not be read as an image."}), 400

    # Same preprocessing as training/evaluation
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)

    cancer_prob = probs[0][LABEL_TO_IDX["cancer"]].item()
    is_cancer = cancer_prob >= THRESHOLD
    confidence = cancer_prob if is_cancer else 1 - cancer_prob

    if is_cancer:
        message = (
            "This image shows features that the AI model associates with "
            "tumor tissue. This is not a diagnosis. Please have the image "
            "reviewed by a qualified pathologist or doctor."
        )
    else:
        message = (
            "This image does not show strong features that the AI model "
            "associates with tumor tissue. This does not rule out cancer. "
            "Please consult a doctor about any concerns."
        )

    return jsonify({
        "prediction": "cancer" if is_cancer else "non_cancer",
        "confidence_percent": round(confidence * 100, 2),
        "cancer_probability_percent": round(cancer_prob * 100, 2),
        "threshold_percent": round(THRESHOLD * 100, 1),
        "message": message,
        "disclaimer": (
            "This tool gives a preliminary AI-based screening signal and is "
            "not a medical diagnosis. Always ask a qualified healthcare "
            "professional to interpret the result."
        ),
    })


if __name__ == "__main__":
    # 127.0.0.1 keeps the tool reachable only from this computer
    app.run(host="127.0.0.1", port=5000, debug=False)