# -*- coding: utf-8 -*-
"""
STGCN, corresponding to Spatio-Temporal Graph Convolutional Networks by Yu,
Yin, and Zhu (arXiv:1709.04875). The structure follows the PyTorch
implementation from hazdzz/STGCN with three changes: the data interface uses
this project's speed matrix and OSM adjacency matrix, the output layer predicts
the four forecasting horizons in one pass, and normalisation and metrics are
handled by the unified project pipeline.

The graph shift operator supports both Chebyshev expansion and first-order
approximation. The adjacency matrix contains isolated nodes, so zero degrees
are replaced with 1 to avoid division by zero during normalisation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_laplacian(adj):
    """Symmetrically normalised Laplacian, with isolated-node degrees set to 1."""
    a = np.asarray(adj, dtype=np.float64)
    deg = a.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    dinv = 1.0 / np.sqrt(deg_safe)
    a_norm = a * dinv[:, None] * dinv[None, :]
    return np.eye(a.shape[0]) - a_norm


def build_gso(adj, kind="cheb"):
    """
    cheb returns the scaled Laplacian for Chebyshev expansion.
    gcn returns the symmetrically normalised adjacency with self-loops for
    first-order approximation.
    """
    if kind == "cheb":
        lap = normalized_laplacian(adj)
        # The spectral radius of the normalised Laplacian is at most 2; scaling
        # directly by 2 avoids repeated eigenvalue computations.
        return (lap - np.eye(adj.shape[0])).astype(np.float32)
    a = np.asarray(adj, dtype=np.float64) + np.eye(adj.shape[0])
    deg = a.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    return (a * dinv[:, None] * dinv[None, :]).astype(np.float32)


class Align(nn.Module):
    """Align channels for residual addition."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.conv = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        return x


class TemporalConv(nn.Module):
    """Causal temporal convolution with a gated linear unit; each layer shortens the time axis by Kt-1 steps."""

    def __init__(self, c_in, c_out, kt):
        super().__init__()
        self.kt = kt
        self.align = Align(c_in, c_out)
        self.conv = nn.Conv2d(c_in, 2 * c_out, (kt, 1))
        self.c_out = c_out

    def forward(self, x):
        res = self.align(x)[:, :, self.kt - 1:, :]
        h = self.conv(x)
        p, q = h[:, :self.c_out], h[:, self.c_out:]
        return (p + res) * torch.sigmoid(q)


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, ks):
        super().__init__()
        self.ks = ks
        self.weight = nn.Parameter(torch.empty(ks, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, gso):
        # x: (B, C, T, N)
        h = x.permute(0, 2, 3, 1)                      # (B, T, N, C)
        terms = [h]
        if self.ks > 1:
            terms.append(torch.einsum("ij,btjc->btic", gso, h))
        for k in range(2, self.ks):
            terms.append(2 * torch.einsum("ij,btjc->btic", gso, terms[-1]) - terms[-2])
        out = sum(torch.einsum("btic,co->btio", t, self.weight[k])
                  for k, t in enumerate(terms))
        return (out + self.bias).permute(0, 3, 1, 2)


class GraphConv(nn.Module):
    """First-order graph convolution approximation."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out)

    def forward(self, x, gso):
        h = x.permute(0, 2, 3, 1)
        h = torch.einsum("ij,btjc->btic", gso, h)
        return self.lin(h).permute(0, 3, 1, 2)


class STConvBlock(nn.Module):
    """Temporal convolution, graph convolution, temporal convolution, and layer normalisation."""

    def __init__(self, c_in, c_mid, c_out, kt, ks, n_node, conv="cheb", dropout=0.1):
        super().__init__()
        self.t1 = TemporalConv(c_in, c_mid, kt)
        self.g = (ChebGraphConv(c_mid, c_mid, ks) if conv == "cheb"
                  else GraphConv(c_mid, c_mid))
        self.t2 = TemporalConv(c_mid, c_out, kt)
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gso):
        x = self.t1(x)
        x = F.relu(self.g(x, gso))
        x = self.t2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.drop(x)


class OutputBlock(nn.Module):
    """Compress the remaining time dimension to one step, then map to the four forecasting horizons."""

    def __init__(self, c_in, c_hidden, n_horizon, t_left):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_hidden, (t_left, 1))
        self.fc1 = nn.Linear(c_hidden, c_hidden)
        self.fc2 = nn.Linear(c_hidden, n_horizon)

    def forward(self, x):
        h = torch.relu(self.conv(x))                   # (B, C, 1, N)
        h = h.squeeze(2).permute(0, 2, 1)              # (B, N, C)
        h = torch.relu(self.fc1(h))
        return self.fc2(h).permute(0, 2, 1)            # (B, H, N)


class STGCN(nn.Module):
    def __init__(self, adj, n_horizon=4, t_in=12, kt=3, ks=3,
                 channels=(64, 16, 64), c_out=128, conv="cheb", dropout=0.1):
        super().__init__()
        gso = build_gso(adj, "cheb" if conv == "cheb" else "gcn")
        self.register_buffer("gso", torch.as_tensor(gso))
        c1, cm, c2 = channels
        self.b1 = STConvBlock(1, cm, c1, kt, ks, adj.shape[0], conv, dropout)
        self.b2 = STConvBlock(c1, cm, c2, kt, ks, adj.shape[0], conv, dropout)
        t_left = t_in - 4 * (kt - 1)
        if t_left < 1:
            raise ValueError(f"Time dimension exhausted by convolutions, t_in={t_in} kt={kt}")
        self.out = OutputBlock(c2, c_out, n_horizon, t_left)

    def forward(self, batch):
        x = batch["recent"].unsqueeze(1)               # (B, 1, T, N)
        h = self.b1(x, self.gso)
        h = self.b2(h, self.gso)
        return self.out(h)
