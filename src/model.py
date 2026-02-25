"""
LSTM Model Architecture for Sentiment Analysis (PyTorch)
=========================================================
Embedding → LSTM → Dense → Dropout → Softmax
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    LSTM-based sentiment classifier.

    Architecture
    ------------
    1. Embedding   – learns word vectors (vocab_size × embedding_dim)
    2. LSTM        – captures sequential / contextual information
    3. Dense (32)  – learns high-level patterns
    4. Dropout     – prevents overfitting
    5. Dense (3)   – softmax output for 3 classes
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 lstm_units: int = 64,
                 num_classes: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
        )

        self.fc1 = nn.Linear(lstm_units, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        x : LongTensor of shape (batch, seq_len)
        """
        embeds = self.embedding(x)                 # (batch, seq, embed_dim)
        lstm_out, (h_n, _) = self.lstm(embeds)     # h_n: (1, batch, lstm_units)
        hidden = h_n.squeeze(0)                    # (batch, lstm_units)
        out = self.fc1(hidden)                     # (batch, 32)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)                        # (batch, num_classes)
        return out  # raw logits — use CrossEntropyLoss (includes softmax)


def build_model(vocab_size: int,
                embedding_dim: int = 128,
                lstm_units: int = 64,
                num_classes: int = 3) -> SentimentLSTM:
    """Build and return the SentimentLSTM model."""
    return SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_classes=num_classes,
    )
