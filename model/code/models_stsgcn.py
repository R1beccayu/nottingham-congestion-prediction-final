# -*- coding: utf-8 -*-
"""
STSGCN, corresponding to Spatial-Temporal Synchronous Graph Convolutional
Networks by Song et al. (AAAI 2020). The original authors released an MXNet
implementation; this version rewrites the model in PyTorch following the paper
so it can share the same data loader and metrics with STGCN and the proposed
model.

The core idea is a local spatio-temporal graph: nodes from three adjacent time
steps are flattened into 3N nodes, road-network edges connect nodes within the
same time step, and each node is connected to itself across adjacent time steps.
A single graph convolution can therefore perform both spatial aggregation and
short-range temporal aggregation. Each temporal position has independent module
parameters, which is why this model has a relatively large parameter count.
"""
import numpy as np
import torch
import torch.nn as nn


def build_st_adj(adj, n_step=3):
    """Build the (n_step*N, n_step*N) local spatio-temporal adjacency with self-loops and symmetric normalisation."""
    n = adj.shape[0]
    big = np.zeros((n_step * n, n_step * n), dtype=np.float64)
    a = np.asarray(adj, dtype=np.float64)
    for s in range(n_step):
        big[s * n:(s + 1) * n, s * n:(s + 1) * n] = a
    eye = np.eye(n)
    for s in range(n_step - 1):
        i, j = s * n, (s + 1) * n
        big[i:i + n, j:j + n] = eye
        big[j:j + n, i:i + n] = eye
    big += np.eye(n_step * n)
    deg = big.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    return (big * dinv[:, None] * dinv[None, :]).astype(np.float32)


class GLUGraphConv(nn.Module):
    """One graph convolution on the local spatio-temporal graph, followed by a gated linear unit."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.lin = nn.Linear(c_in, 2 * c_out)
        self.c_out = c_out

    def forward(self, x, gso):
        # x: (B, S, C)，S = n_step * N
        h = torch.einsum("ij,bjc->bic", gso, x)
        h = self.lin(h)
        p, q = h[..., :self.c_out], h[..., self.c_out:]
        return p * torch.sigmoid(q)


class STSGCM(nn.Module):
    """
    Several graph-convolution layers are stacked. The outputs of all layers are
    aggregated with an element-wise maximum, and then the side time steps are
    removed so only the centre step remains. The centre step has already
    absorbed information from neighbouring steps through temporal edges.
    """

    def __init__(self, c_in, c_out, n_layer=3, n_node=144, n_step=3):
        super().__init__()
        self.n_node, self.n_step = n_node, n_step
        chans = [c_in] + [c_out] * n_layer
        self.convs = nn.ModuleList(
            GLUGraphConv(chans[i], chans[i + 1]) for i in range(n_layer))

    def forward(self, x, gso):
        outs = []
        h = x
        for conv in self.convs:
            h = conv(h, gso)
            outs.append(h)
        h = torch.stack(outs, dim=0).max(dim=0).values
        mid = self.n_step // 2
        return h[:, mid * self.n_node:(mid + 1) * self.n_node, :]


class STSGCL(nn.Module):
    """
    Slide a window of n_step time steps along the time axis, with an independent
    STSGCM at each position. Learnable spatio-temporal positional embeddings are
    added to the input.
    """

    def __init__(self, t_in, n_node, c_in, c_out, n_layer=3, n_step=3, emb=True):
        super().__init__()
        self.t_in, self.n_node, self.n_step = t_in, n_node, n_step
        self.t_out = t_in - n_step + 1
        self.modules_ = nn.ModuleList(
            STSGCM(c_in, c_out, n_layer, n_node, n_step) for _ in range(self.t_out))
        self.emb = emb
        if emb:
            self.t_emb = nn.Parameter(torch.zeros(1, t_in, 1, c_in))
            self.n_emb = nn.Parameter(torch.zeros(1, 1, n_node, c_in))
            nn.init.xavier_uniform_(self.t_emb)
            nn.init.xavier_uniform_(self.n_emb)

    def forward(self, x, gso):
        # x: (B, T, N, C)
        if self.emb:
            x = x + self.t_emb + self.n_emb
        outs = []
        for t in range(self.t_out):
            seg = x[:, t:t + self.n_step]                    # (B, S, N, C)
            b = seg.shape[0]
            flat = seg.reshape(b, self.n_step * self.n_node, -1)
            outs.append(self.modules_[t](flat, gso))
        return torch.stack(outs, dim=1)                       # (B, T', N, C')


class STSGCN(nn.Module):
    def __init__(self, adj, n_horizon=4, t_in=12, channels=64, n_block=3,
                 n_conv=3, n_step=3, c_hidden=128, dropout=0.1):
        super().__init__()
        n_node = adj.shape[0]
        self.register_buffer("gso", torch.as_tensor(build_st_adj(adj, n_step)))
        self.inp = nn.Linear(1, channels)
        layers, t = [], t_in
        for _ in range(n_block):
            layers.append(STSGCL(t, n_node, channels, channels, n_conv, n_step))
            t = t - n_step + 1
            if t < 1:
                raise ValueError("Time dimension exhausted; reduce n_block or n_step")
        self.blocks = nn.ModuleList(layers)
        self.t_left = t
        self.drop = nn.Dropout(dropout)
        # One independent output head per forecasting horizon, consistent with the paper.
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(t * channels, c_hidden), nn.ReLU(),
                          nn.Linear(c_hidden, 1))
            for _ in range(n_horizon))

    def forward(self, batch):
        x = batch["recent"].unsqueeze(-1)          # (B, T, N, 1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, self.gso)
        b, t, n, c = h.shape
        h = self.drop(h.permute(0, 2, 1, 3).reshape(b, n, t * c))
        out = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(out, dim=1)             # (B, H, N)
