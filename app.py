"""
Flask Application for Sentiment Analysis (PyTorch)
====================================================
Serves a premium web UI and exposes a REST API for predictions.

Usage:
    python app.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

from flask import Flask, render_template, request, jsonify
from src.preprocess import preprocess_single, decode_prediction, load_object
from src.model import build_model


# Paths
SAVE_DIR       = "saved_model"
MODEL_PATH     = os.path.join(SAVE_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
CONFIG_PATH    = os.path.join(SAVE_DIR, "config.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flask App
app = Flask(__name__)

# Load model on startup
print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model = build_model(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    lstm_units=checkpoint["lstm_units"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = load_object(TOKENIZER_PATH)
config    = load_object(CONFIG_PATH)
max_len   = config["max_len"]
print("[OK] Model loaded successfully!")


@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API endpoint for sentiment prediction.

    Request JSON:  { "text": "I love this product" }
    Response JSON: { "label": "Positive", "confidence": 94.2, "probabilities": {...} }
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Predict
    padded = preprocess_single(text, tokenizer, max_len)
    tensor = torch.from_numpy(padded).long().to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    result = decode_prediction(probs)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
