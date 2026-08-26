# -*- coding: utf-8 -*-
"""Project interfaces for STAEformer and iTransformer."""
import math

import torch
import torch.nn as nn


class AxisSelfAttention(nn.Module):
    """Apply Transformer encoding along a selected axis of a four-dimensional tensor."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, axis):
        # x: (B, T, N, D). Move the attention axis to the penultimate dimension
        # and fold the remaining dimensions into the batch.
        if axis == 1:
            y = x.permute(0, 2, 1, 3)
            b, n, t, d = y.shape
            y = y.reshape(b * n, t, d)
            restore = lambda z: z.reshape(b, n, t, d).permute(0, 2, 1, 3)
        elif axis == 2:
            b, t, n, d = x.shape
            y = x.reshape(b * t, n, d)
            restore = lambda z: z.reshape(b, t, n, d)
        else:
            raise ValueError(f"Unsupported attention axis {axis}")
        a, _ = self.attn(y, y, y, need_weights=False)
        y = self.norm1(y + self.drop1(a))
        y = self.norm2(y + self.drop2(self.ff(y)))
        return restore(y)


class STAEformer(nn.Module):
    """Adaptive spatio-temporal embeddings with temporal and spatial self-attention."""

    def __init__(
        self,
        n_node,
        n_horizon=4,
        in_steps=12,
        steps_per_day=288,
        input_emb=16,
        tod_emb=16,
        dow_emb=8,
        adaptive_emb=32,
        n_heads=4,
        n_layers=2,
        d_ff=144,
        dropout=0.1,
    ):
        super().__init__()
        self.n_node = n_node
        self.in_steps = in_steps
        self.n_horizon = n_horizon
        self.input_proj = nn.Linear(1, input_emb)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_emb)
        self.dow_embedding = nn.Embedding(7, dow_emb)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(in_steps, n_node, adaptive_emb)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        d_model = input_emb + tod_emb + dow_emb + adaptive_emb
        if d_model % n_heads:
            raise ValueError(f"STAEformer d_model={d_model} is not divisible by heads={n_heads}")
        self.temporal_layers = nn.ModuleList(
            AxisSelfAttention(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.spatial_layers = nn.ModuleList(
            AxisSelfAttention(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(in_steps * d_model, n_horizon)

    def forward(self, batch):
        speed = batch["recent"]
        b, t, n = speed.shape
        x = self.input_proj(speed[..., None])
        tod = self.tod_embedding(batch["recent_tod"])
        dow = self.dow_embedding(batch["recent_dow"])
        adaptive = self.adaptive_embedding[None].expand(b, -1, -1, -1)
        x = torch.cat((x, tod[..., None, :].expand(-1, -1, n, -1),
                       dow[..., None, :].expand(-1, -1, n, -1), adaptive), dim=-1)
        for layer in self.temporal_layers:
            x = layer(x, axis=1)
        for layer in self.spatial_layers:
            x = layer(x, axis=2)
        x = x.transpose(1, 2).reshape(b, n, -1)
        return self.output_proj(x).transpose(1, 2)


class ITransformer(nn.Module):
    """Embed the full input sequence of each node as one token."""

    def __init__(self, in_steps=12, n_horizon=4, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(in_steps, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, n_horizon)

    def forward(self, batch):
        x = batch["recent"]  # (B, T, N)
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)
        x = (x - mean) / std
        tokens = self.value_embedding(x.transpose(1, 2))
        encoded = self.encoder(tokens)
        out = self.projector(encoded).transpose(1, 2)
        return out * std + mean
