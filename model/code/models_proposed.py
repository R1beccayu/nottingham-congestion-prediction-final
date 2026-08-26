# -*- coding: utf-8 -*-
"""
Improved model. The backbone remains STGCN, with two ablatable additions.

1. Spatio-temporal enhancement. In addition to the recent branch, daily and
weekly periodic branches are added. The three branches share one STGCN encoder.
The encoded outputs first pass through a local spatio-temporal graph
convolution so neighbouring nodes can exchange information across adjacent time
steps, and then temporal attention assigns branch weights for each sample. When
a periodic branch is unavailable, it is masked in attention so it receives no
weight and does not contaminate the fused representation.

An optional spatial-attention module is also provided. It reweights neighbours
on the fixed OSM graph only at existing edge locations, without introducing new
connections. It is disabled by default and is used to examine its contribution
on this dataset separately.

2. Weather gating. Weather features at the target time are independently mapped
to channel-wise gates. In the form of 1 plus tanh, they adjust the fused
spatio-temporal representation separately for each forecasting horizon, with
node-specific response strengths. Weather is not concatenated directly with
spatio-temporal features, avoiding the same hourly value being copied to all
nodes and overwhelming structural information. In the delivered configuration,
the gate uses only weather columns, while time-of-day information is handled by
the periodic branches in the spatio-temporal enhancement. This is controlled by
wg_part.

Both parts can be disabled for the A0 to A3 ablation study. The output
projections of the three added modules are zero-initialised, so the model is
equivalent to the original STGCN backbone at the start of training.
"""
import torch
import torch.nn as nn

from models_stgcn import STConvBlock, build_gso
from models_stsgcn import build_st_adj


class STEncoder(nn.Module):
    """Two spatio-temporal convolution blocks from the STGCN backbone, shared by the three branches."""

    def __init__(self, c_mid, c_out, kt, ks, n_node, conv, dropout):
        super().__init__()
        self.b1 = STConvBlock(1, c_mid, c_out, kt, ks, n_node, conv, dropout)
        self.b2 = STConvBlock(c_out, c_mid, c_out, kt, ks, n_node, conv, dropout)

    def forward(self, x, gso):
        return self.b2(self.b1(x, gso), gso)


class LocalSTConv(nn.Module):
    """
    Local spatio-temporal graph convolution. Adjacent n_step time steps are
    flattened into one graph, so a single convolution performs both spatial
    aggregation and short-range temporal aggregation. The idea follows STSGCN,
    but only one layer is stacked and the backbone remains STGCN.

    Two design choices are intentional. Replicate padding is applied at both
    ends of the time axis so the output length matches the input length and no
    time steps are lost by adding this module. The output projection is
    zero-initialised, so the module starts as an identity mapping and learns how
    much to add to the backbone during training, preventing the enhancement from
    disturbing an already stable backbone at the beginning.
    """

    def __init__(self, c_in, c_out, n_node, n_step=3):
        super().__init__()
        assert n_step % 2 == 1, "The local spatio-temporal graph span must be odd to align the centre step"
        self.n_node, self.n_step = n_node, n_step
        self.lin = nn.Linear(c_in, 2 * c_out)
        self.proj = nn.Linear(c_out, c_in)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.c_out = c_out

    def forward(self, x, st_gso):
        # x: (B, C, T, N)
        b, c, t, n = x.shape
        pad = self.n_step // 2
        xp = nn.functional.pad(x, (0, 0, pad, pad), mode="replicate")
        h = xp.permute(0, 2, 3, 1)                     # (B, T+2p, N, C)
        outs = []
        for s in range(t):
            seg = h[:, s:s + self.n_step].reshape(b, self.n_step * n, c)
            g = self.lin(torch.einsum("ij,bjc->bic", st_gso, seg))
            p, q = g[..., :self.c_out], g[..., self.c_out:]
            gated = p * torch.sigmoid(q)
            outs.append(gated[:, pad * n:(pad + 1) * n, :])
        out = torch.stack(outs, dim=1)                 # (B, T, N, C_out)
        return x + self.proj(out).permute(0, 3, 1, 2)


class SpatialAttention(nn.Module):
    """
    Reweight neighbours on the fixed OSM graph. Attention is computed only on
    existing graph edges, without adding new connections. The road-distance
    prior is therefore preserved, and isolated nodes are handled by self-loops.

    The output projection is zero-initialised, so the module starts as an
    identity mapping.
    """

    def __init__(self, c_in, adj, c_att=32):
        super().__init__()
        n = adj.shape[0]
        edge = torch.as_tensor(adj) > 0
        edge = edge | torch.eye(n, dtype=torch.bool)
        self.register_buffer("edge", edge)
        self.q = nn.Linear(c_in, c_att)
        self.k = nn.Linear(c_in, c_att)
        self.v = nn.Linear(c_in, c_att)
        self.proj = nn.Linear(c_att, c_in)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.scale = c_att ** -0.5

    def forward(self, x):
        # x: (B, C, T, N)
        h = x.permute(0, 2, 3, 1)                      # (B, T, N, C)
        q, k, v = self.q(h), self.k(h), self.v(h)
        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        score = score.masked_fill(~self.edge, float("-inf"))
        att = torch.softmax(score, dim=-1)
        out = torch.matmul(att, v)                     # (B, T, N, C_att)
        self.last_att = att.detach()
        return x + self.proj(out).permute(0, 3, 1, 2)


class BranchFusion(nn.Module):
    """
    Temporal-attention fusion. The recent branch always serves as the backbone,
    while the daily and weekly branches are added as residual terms. Weights are
    scored from each periodic representation together with the recent
    representation and are assigned at node level.

    The output layer of the scoring network is zero-initialised, so periodic
    branch weights are zero at the start of training and the model is equivalent
    to using only the recent branch. This avoids early contamination from
    periodic representations that have not yet been learned. When a branch is
    unavailable, its weight is multiplied by the availability flag and set to
    zero.
    """

    def __init__(self, c, hidden=32, n_period=2):
        super().__init__()
        self.nets = nn.ModuleList()
        for _ in range(n_period):
            net = nn.Sequential(nn.Linear(2 * c, hidden), nn.Tanh(),
                                nn.Linear(hidden, 1))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

    def forward(self, recent, periods, avail):
        # recent: (B, C, N)；periods: list of (B, C, N)；avail: list of (B,)
        out = recent
        ws = []
        r = recent.permute(0, 2, 1)                    # (B, N, C)
        for net, p, ok in zip(self.nets, periods, avail):
            q = p.permute(0, 2, 1)
            w = net(torch.cat([r, q], dim=-1)).squeeze(-1)    # (B, N)
            w = w * ok[:, None]
            ws.append(w)
            out = out + w[:, None, :] * p
        self.last_weight = torch.stack([w.detach() for w in ws], dim=1)
        return out


class WeatherGate(nn.Module):
    """
    External feature adjustment. Weather and time-calendar features are handled
    separately because they affect traffic speed in different ways.

    Weather conditions such as precipitation, visibility, and wind speed tend to
    change travel speed proportionally, so they are modelled with a
    multiplicative gate. Time-calendar information such as hour, weekday, and
    term status defines the baseline speed level for the target time, so it is
    modelled as an additive bias. Mixing both feature types into one
    multiplicative gate would force the model to learn a compromise and was
    empirically worse than leaving the module out.

    The output layers of both branches are zero-initialised, so the gate starts
    at 1 and the bias starts at 0. The model is therefore equivalent to not using
    this module at the start of training.
    """

    def __init__(self, n_weather, n_time, c, hidden=32, n_horizon=4,
                 n_node=144, node_specific=True):
        super().__init__()
        self.n_horizon = n_horizon
        self.n_weather, self.n_time = n_weather, n_time
        self.node_specific = node_specific
        self.gate_net = self._zero_head(n_weather, hidden, c) if n_weather else None
        self.bias_net = self._zero_head(n_time, hidden, c) if n_time else None
        if node_specific:
            # Weather is available as a single city-level series shared by all
            # nodes. A purely global adjustment would be highly redundant with
            # the speed sequence itself, so each node receives its own response
            # strength. The same rainfall event can then have different effects
            # on faster roads and city-centre roads.
            self.gate_scale = nn.Parameter(torch.ones(n_node, c))
            self.bias_scale = nn.Parameter(torch.ones(n_node, c))

    @staticmethod
    def _zero_head(n_in, hidden, c):
        net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, c))
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net

    def forward(self, h, weather, time_feat):
        # h: (B, C, N); weather / time_feat: (B, H, D), taken at each target horizon.
        out = h[:, None]                               # (B, 1, C, N)
        if self.gate_net is not None:
            raw = self.gate_net(weather)                       # (B, H, C)
            if self.node_specific:
                raw = raw[..., None] * self.gate_scale.t()[None, None]
            else:
                raw = raw[..., None]
            g = 1.0 + torch.tanh(raw)                          # (B, H, C, N)
            self.last_gate = g.detach()
            out = out * g
        if self.bias_net is not None:
            raw = self.bias_net(time_feat)
            if self.node_specific:
                raw = raw[..., None] * self.bias_scale.t()[None, None]
            else:
                raw = raw[..., None]
            self.last_bias = raw.detach()
            out = out + raw
        return out                                     # (B, H, C, N)


WEATHER_PREFIX = ("temp", "humidity", "precip", "windspeed", "visibility",
                  "is_rain", "icon_")
TIME_COLS = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
             "is_school_holiday", "is_christmas_period")


def split_dyn_columns(columns):
    """Split external feature columns into weather and time-calendar groups."""
    weather, timef = [], []
    for i, c in enumerate(columns):
        if c in TIME_COLS:
            timef.append(i)
        elif c.startswith(WEATHER_PREFIX):
            weather.append(i)
        else:
            timef.append(i)
    return weather, timef


class ProposedModel(nn.Module):
    def __init__(self, adj, n_dyn, n_horizon=4, t_in=12, kt=3, ks=3,
                 c_mid=16, c_out=64, c_hidden=128, conv="cheb", dropout=0.1,
                 use_st_enhance=True, use_weather_gate=True, n_step=3,
                 dyn_columns=None, wg_part="both", use_daily=True,
                 use_weekly=True, use_spatial_att=False,
                 output_residual="none"):
        super().__init__()
        n_node = adj.shape[0]
        self.n_node = n_node
        if dyn_columns is None:
            self.w_idx, self.t_idx = list(range(n_dyn)), []
        else:
            self.w_idx, self.t_idx = split_dyn_columns(dyn_columns)
        # Diagnostic split: examine the weather multiplicative gate and the time
        # additive bias separately, so gains from time encoding are not assigned
        # to the weather branch.
        if wg_part == "weather":
            self.t_idx = []
        elif wg_part == "time":
            self.w_idx = []
        elif wg_part != "both":
            raise ValueError(f"Unknown wg_part {wg_part}")
        self.wg_part = wg_part
        self.use_st = use_st_enhance
        self.use_wg = use_weather_gate
        self.use_sa = use_spatial_att
        self.n_horizon = n_horizon
        self.output_residual = output_residual

        self.register_buffer("gso", torch.as_tensor(
            build_gso(adj, "cheb" if conv == "cheb" else "gcn")))
        self.encoder = STEncoder(c_mid, c_out, kt, ks, n_node, conv, dropout)

        # The daily periodic branch and the time additive bias both describe the
        # normal speed level for a node at a given time, so their information can
        # overlap. The two periodic branches can therefore be disabled
        # separately to examine their contributions.
        self.periods = ([k for k, on in (("daily", use_daily), ("weekly", use_weekly))
                         if on] if use_st_enhance else [])
        if self.use_sa:
            self.sa = SpatialAttention(c_out, adj)
        t_enc = t_in - 4 * (kt - 1)
        if self.use_st:
            self.register_buffer("st_gso", torch.as_tensor(build_st_adj(adj, n_step)))
            self.local = LocalSTConv(c_out, c_out, n_node, n_step)
            self.fuse = (BranchFusion(c_out, n_period=len(self.periods))
                         if self.periods else None)
        # Local spatio-temporal convolution preserves the time dimension, so the
        # temporal-pooling kernel length is consistent across both settings.
        self.t_pool = t_enc
        self.pool = nn.Conv2d(c_out, c_out, (t_enc, 1))

        if self.use_wg:
            self.gate = WeatherGate(len(self.w_idx), len(self.t_idx),
                                    c_out, 32, n_horizon, n_node)
            self.head = nn.Sequential(nn.Linear(c_out, c_hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(c_hidden, 1))
        else:
            self.head = nn.Sequential(nn.Linear(c_out, c_hidden), nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(c_hidden, n_horizon))
        if self.output_residual != "none":
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def _encode(self, x):
        h = self.encoder(x.unsqueeze(1), self.gso)
        if self.use_st:
            h = self.local(h, self.st_gso)
        if self.use_sa:
            h = self.sa(h)
        return h

    def forward(self, batch):
        if self.use_st:
            h = self._encode(batch["recent"])
            fused = torch.relu(self.pool(h)).squeeze(2)
            if self.fuse is not None:
                periods, avail = [], []
                for key in self.periods:
                    hp = self._encode(batch[key])
                    periods.append(torch.relu(self.pool(hp)).squeeze(2))
                    avail.append(batch[f"{key}_ok"])
                fused = self.fuse(fused, periods, avail)
        else:
            h = self._encode(batch["recent"])
            fused = torch.relu(self.pool(h)).squeeze(2)

        if self.use_wg:
            dyn = batch["dyn"]
            weather = dyn[..., self.w_idx] if self.w_idx else None
            timef = dyn[..., self.t_idx] if self.t_idx else None
            g = self.gate(fused, weather, timef)           # (B, H, C, N)
            out = self.head(g.permute(0, 1, 3, 2)).squeeze(-1)
            if self.output_residual == "ha":
                out = batch["ha_norm"] + out
            elif self.output_residual == "last":
                out = batch["recent"][:, -1, None, :] + out
            return out                                      # (B, H, N)
        out = self.head(fused.permute(0, 2, 1))            # (B, N, H)
        out = out.permute(0, 2, 1)
        if self.output_residual == "ha":
            out = batch["ha_norm"] + out
        elif self.output_residual == "last":
            out = batch["recent"][:, -1, None, :] + out
        return out
