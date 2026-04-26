"""
Prediction Script for Sentiment Analysis (LSTM)
===============================================
Loads the saved LSTM model and provides interactive CLI for predictions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from src.preprocess import preprocess_single, decode_prediction, load_object, scaled_softmax
from src.model import build_model

SAVE_DIR       = "saved_model"
MODEL_PATH     = os.path.join(SAVE_DIR, "model.pt")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
CONFIG_PATH    = os.path.join(SAVE_DIR, "config.pkl")

EMOJI = {"Positive": ":)", "Neutral": ":|", "Negative": ":("}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = build_model(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        lstm_units=checkpoint["lstm_units"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def predict_sentiment(text: str, model, tokenizer, max_len: int) -> dict:
    padded = preprocess_single(text, tokenizer, max_len)
    tensor = torch.from_numpy(padded).long().to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        logits_np = logits.cpu().numpy()[0]
        probs = scaled_softmax(logits_np, temperature=0.5)
    return decode_prediction(probs)

def main():
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model not found. Please run python train.py first.")
        sys.exit(1)

    print("Loading model...")
    model     = load_trained_model()
    tokenizer = load_object(TOKENIZER_PATH)
    config    = load_object(CONFIG_PATH)
    max_len   = config["max_len"]
    print("[OK] Model loaded successfully!\n")

    print("=" * 60)
    print("  SENTIMENT ANALYSIS - LIVE PREDICTION (LSTM)")
    print("  Type a sentence and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            text = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        result = predict_sentiment(text, model, tokenizer, max_len)
        emoji  = EMOJI.get(result["label"], "")

        print(f"\n   Sentiment  : {result['label']} {emoji}")
        print(f"   Confidence : {result['confidence']:.1f}%")
        print(f"   -------------------------------")
        for cls, prob in result["probabilities"].items():
            bar = "#" * int(prob / 5) + "." * (20 - int(prob / 5))
            print(f"   {cls:>8s}  [{bar}]  {prob:.1f}%")

if __name__ == "__main__":
    main()
