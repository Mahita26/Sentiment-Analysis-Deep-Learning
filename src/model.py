"""
LSTM Model Architecture for Sentiment Analysis (PyTorch)
=========================================================
Embedding -> Bidirectional LSTM -> Global Max Pooling -> Dropout -> Dense -> Softmax
"""

import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM-based sentiment classifier.
    Features Global Max Pooling to handle padding naturally and extract strong features.
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
            bidirectional=True,
        )

        # Bidirectional doubles the output size
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x : LongTensor of shape (batch, seq_len)
        """
        embeds = self.embedding(x)                 # (batch, seq, embed_dim)
        lstm_out, _ = self.lstm(embeds)            # lstm_out: (batch, seq, lstm_units*2)

        # Global Max Pooling across sequence dimension
        hidden, _ = torch.max(lstm_out, dim=1)     # (batch, lstm_units*2)

        out = self.fc1(hidden)                     # (batch, 64)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)                        # (batch, num_classes)
        return out  # raw logits

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
