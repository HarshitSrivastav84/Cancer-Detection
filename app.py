"""
Flask backend for the Gastric Cancer Detection tool.

Serves a single /predict endpoint that accepts an uploaded image,
runs it through the trained CNN, and returns a prediction with
plain-language explanation - written to be safely understood by
both doctors and patients, since this app has no login/role split.
"""

import torch
from flask import Flask, request, jsonify
from PIL import Image
import io

from model import build_model
from dataset import eval_transform, LABEL_TO_IDX

# ---- CONFIG ----
MODEL_PATH = "gastric_cancer_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}  # reverse the mapping

app = Flask(__name__)

# ---- LOAD MODEL ONCE AT STARTUP (not per-request) ----
print("Loading model...")
model = build_model(num_classes=2, freeze_backbone=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read the uploaded file as an image'}), 400

    # Preprocess using the SAME pipeline used during training/evaluation
    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_label = IDX_TO_LABEL[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    cancer_probability = round(probabilities[0][1].item() * 100, 2)  # index 1 = cancer

    # Plain-language message, safe for both doctors and patients
    if predicted_label == "cancer":
        message = (
            "This image shows features that the AI model associates with "
            "tumor tissue. This is NOT a diagnosis - please have this image "
            "reviewed by a qualified pathologist or doctor."
        )
    else:
        message = (
            "This image does not show strong features associated with "
            "tumor tissue in this AI model. This does NOT rule out cancer - "
            "please continue routine checkups and consult a doctor with any "
            "concerns."
        )

    return jsonify({
        'prediction': predicted_label,
        'confidence_percent': confidence_pct,
        'cancer_probability_percent': cancer_probability,
        'message': message,
        'disclaimer': (
            "This tool provides a preliminary AI-based screening signal "
            "and is NOT a medical diagnosis. Always consult a qualified "
            "healthcare professional to interpret this result."
        )
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Gastric Cancer Detection API is running',
        'usage': 'POST an image file to /predict'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)