# -*- coding: utf-8 -*-
"""
Temporal baselines without graph structure.

Both models share weights across nodes and treat each countline-direction as an
independent sequence. This keeps the parameter count independent of the number
of nodes and matches the purpose of these baselines: testing how far prediction
can go without introducing road-network structure. One forward pass outputs all
four forecasting horizons.
"""
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_horizon=4, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_horizon)

    def forward(self, batch):
        x = batch["recent"]                      # (B, T, N)
        b, t, n = x.shape
        s = x.permute(0, 2, 1).reshape(b * n, t, 1)
        out, _ = self.lstm(s)
        y = self.head(out[:, -1])                # (B*N, H)
        return y.view(b, n, -1).permute(0, 2, 1)  # (B, H, N)


class CNNLSTM(nn.Module):
    """Use 1D convolution to extract local temporal patterns, then LSTM for longer-range memory."""

    def __init__(self, n_horizon=4, channels=32, hidden=64, layers=1,
                 kernel=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_horizon)

    def forward(self, batch):
        x = batch["recent"]
        b, t, n = x.shape
        s = x.permute(0, 2, 1).reshape(b * n, 1, t)
        c = self.conv(s).permute(0, 2, 1)        # (B*N, T, C)
        out, _ = self.lstm(c)
        y = self.head(self.drop(out[:, -1]))
        return y.view(b, n, -1).permute(0, 2, 1)
