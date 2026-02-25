"""
Training Script for Sentiment Analysis (PyTorch)
==================================================
Loads data -> preprocesses -> builds LSTM model -> trains -> saves artifacts.

Usage:
    python train.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.preprocess import load_and_preprocess, save_object
from src.model import build_model


# ── Configuration ──────────────────────────────────────────────
DATASET_PATH  = os.path.join("data", "dataset.csv")
SAVE_DIR      = "saved_model"
MAX_WORDS     = 5000
MAX_LEN       = 50
EMBEDDING_DIM = 128
LSTM_UNITS    = 64
EPOCHS        = 50
BATCH_SIZE    = 16
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 60)
    print("  SENTIMENT ANALYSIS - LSTM TRAINING (PyTorch)")
    print("=" * 60)

    # ── 1. Preprocess ──────────────────────────────────────────
    print("\n[STEP 1] Loading & preprocessing dataset...")
    X_train, X_test, y_train, y_test, tokenizer, config = load_and_preprocess(
        csv_path=DATASET_PATH,
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
    )

    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).long()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t  = torch.from_numpy(X_test).long()
    y_test_t  = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ── 2. Build Model ─────────────────────────────────────────
    print("\n[STEP 2] Building LSTM model...")
    vocab_size = min(tokenizer.vocab_size, MAX_WORDS)
    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
    ).to(DEVICE)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── 3. Train ───────────────────────────────────────────────
    print("\n[STEP 3] Training model...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        print(f"   Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.1f}%")

    # ── 4. Final Evaluation ────────────────────────────────────
    print(f"\n[STEP 4] Final Test Accuracy: {val_acc * 100:.2f}%")

    # ── 5. Save ────────────────────────────────────────────────
    print("\n[STEP 5] Saving model artifacts...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path     = os.path.join(SAVE_DIR, "model.pt")
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer.pkl")
    config_path    = os.path.join(SAVE_DIR, "config.pkl")

    # Save model state dict + architecture config
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
    }, model_path)
    print(f"   Model     -> {model_path}")

    save_object(tokenizer, tokenizer_path)
    save_object(config, config_path)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Final Accuracy: {val_acc * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
