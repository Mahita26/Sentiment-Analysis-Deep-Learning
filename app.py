"""
Flask Application for Sentiment Analysis (LSTM)
===============================================
Serves a premium web UI and exposes a REST API for predictions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from flask import Flask, render_template, request, jsonify
from src.preprocess import preprocess_single, decode_prediction, load_object, scaled_softmax
from src.model import build_model

SAVE_DIR       = "saved_model"
MODEL_PATH     = os.path.join(SAVE_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
CONFIG_PATH    = os.path.join(SAVE_DIR, "config.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

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
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    padded = preprocess_single(text, tokenizer, max_len)
    tensor = torch.from_numpy(padded).long().to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor)
        logits_np = logits.cpu().numpy()[0]
        probs = scaled_softmax(logits_np, temperature=0.5)
        
    result = decode_prediction(probs)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
