"""
Text Preprocessing Module for Sentiment Analysis (LSTM)
=====================================================
Handles cleaning, tokenization, padding, and calibration.
"""

import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Label Mapping ──────────────────────────────────────────────
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 3

def clean_text(text: str) -> str:
    """
    Clean text but preserve essential punctuation like ? and !
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    
    # Pad ? and ! with spaces so they become separate tokens
    text = re.sub(r"([?!])", r" \1 ", text)
    
    # Remove everything except letters, spaces, ?, and !
    text = re.sub(r"[^a-z\s?!]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SimpleTokenizer:
    def __init__(self, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        self.word_index[self.oov_token] = 1
        for idx, (word, _) in enumerate(counter.most_common(self.num_words - 2), start=2):
            self.word_index[word] = idx
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts):
        oov_idx = self.word_index.get(self.oov_token, 1)
        sequences = []
        for text in texts:
            seq = [self.word_index.get(w, oov_idx) for w in text.split()]
            sequences.append(seq)
        return sequences

    @property
    def vocab_size(self):
        return len(self.word_index) + 1

def pad_sequences(sequences, maxlen, padding="pre"):
    result = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        if padding == "pre":
            result[i, maxlen - len(seq):] = seq
        else:
            result[i, :len(seq)] = seq
    return result

def load_and_preprocess(csv_path: str, max_words: int = 5000, max_len: int = 50, test_size: float = 0.2):
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples")
    print(f"[INFO] Class distribution:\n{df['sentiment'].value_counts().to_string()}")

    df["clean_text"] = df["text"].apply(clean_text)

    tokenizer = SimpleTokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])

    X = pad_sequences(sequences, maxlen=max_len, padding="pre")
    y = df["sentiment"].map(LABEL_MAP).values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    config = {"max_words": max_words, "max_len": max_len}

    print(f"[INFO] Vocab size : {tokenizer.vocab_size}")
    print(f"[INFO] Max length : {max_len}")
    print(f"[INFO] Train size : {len(X_train)}")
    print(f"[INFO] Test  size : {len(X_test)}")

    return X_train, X_test, y_train, y_test, tokenizer, config

def preprocess_single(text: str, tokenizer, max_len: int) -> np.ndarray:
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="pre")
    return padded

def scaled_softmax(logits: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Temperature Scaling: T=0.5 sharpens probabilities (85-95% confidence).
    """
    scaled = logits / temperature
    exp_vals = np.exp(scaled - np.max(scaled))
    return exp_vals / exp_vals.sum()

def decode_prediction(probs: np.ndarray) -> dict:
    idx = int(np.argmax(probs))
    return {
        "label": LABEL_NAMES[idx],
        "confidence": float(probs[idx]) * 100,
        "probabilities": {
            "Negative": float(probs[0]) * 100,
            "Neutral":  float(probs[1]) * 100,
            "Positive": float(probs[2]) * 100,
        },
    }

def save_object(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved -> {path}")

def load_object(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
