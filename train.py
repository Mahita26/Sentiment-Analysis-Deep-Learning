"""
Training Script for Sentiment Analysis (PyTorch LSTM)
=====================================================
Loads data -> preprocesses -> builds LSTM model -> trains -> saves artifacts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.preprocess import load_and_preprocess, save_object
from src.model import build_model

# ── Configuration ──────────────────────────────────────────────
DATASET_PATH  = os.path.join("data", "dataset.csv")
SAVE_DIR      = "saved_model"
MAX_WORDS     = 5000
MAX_LEN       = 80
EMBEDDING_DIM = 128
LSTM_UNITS    = 64
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 60)
    print("  SENTIMENT ANALYSIS - LSTM TRAINING (PyTorch)")
    print("=" * 60)

    # 1. Preprocess
    print("\n[STEP 1] Loading & preprocessing dataset...")
    X_train, X_test, y_train, y_test, tokenizer, config = load_and_preprocess(
        csv_path=DATASET_PATH,
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
    )

    X_train_t = torch.from_numpy(X_train).long()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t  = torch.from_numpy(X_test).long()
    y_test_t  = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. Build Model
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
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 3. Train
    print("\n[STEP 3] Training model...\n")
    best_val_acc = 0.0
    
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

        print(f"   Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "embedding_dim": EMBEDDING_DIM,
                "lstm_units": LSTM_UNITS,
            }, os.path.join(SAVE_DIR, "model.pt"))
            save_object(tokenizer, os.path.join(SAVE_DIR, "tokenizer.pkl"))
            save_object(config, os.path.join(SAVE_DIR, "config.pkl"))

    # 4. Final
    print(f"\n[STEP 4] Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
