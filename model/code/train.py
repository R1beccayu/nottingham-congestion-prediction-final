# -*- coding: utf-8 -*-
"""
Unified training entry point. All models use the same data interface,
metric definitions, and early-stopping rule. Configurations are selected
only on the validation set, and the test set is used only for final
evaluation after the configuration has been fixed.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from metrics import evaluate, show
from models_seq import CNNLSTM, LSTMBaseline
from models_proposed import ProposedModel
from models_stgcn import STGCN
from models_stsgcn import STSGCN
from models_transformer import ITransformer, STAEformer
from torch_data import Batcher, masked_huber, masked_mae, masked_mse
from windows import WindowIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY = {
    "lstm": lambda w, a: LSTMBaseline(len(w.horizons), a.hidden, a.layers, a.dropout),
    "cnn_lstm": lambda w, a: CNNLSTM(len(w.horizons), a.channels, a.hidden,
                                     max(1, a.layers - 1), a.kernel, a.dropout),
    "stgcn": lambda w, a: STGCN(w.adj, len(w.horizons), w.recent_len, a.kt, a.ks,
                                (a.st_channels, a.st_bottleneck, a.st_channels),
                                a.hidden * 2, a.conv, a.dropout),
    "stsgcn": lambda w, a: STSGCN(w.adj, len(w.horizons), w.recent_len,
                                  a.sts_channels, a.sts_blocks, a.sts_convs,
                                  a.sts_step, a.hidden * 2, a.dropout),
    "staeformer": lambda w, a: STAEformer(
        w.n_node, len(w.horizons), w.recent_len, w.steps_per_day,
        a.stae_input_emb, a.stae_tod_emb, a.stae_dow_emb, a.stae_adaptive_emb,
        a.tf_heads, a.tf_layers, a.tf_ff, a.dropout,
    ),
    "itransformer": lambda w, a: ITransformer(
        w.recent_len, len(w.horizons), a.tf_d_model, a.tf_heads,
        a.tf_layers, a.tf_ff, a.dropout,
    ),
    # A0 to A3 share the same constructor and are switched by two flags.
    "proposed": lambda w, a: ProposedModel(
        w.adj, w.dyn.shape[1], len(w.horizons), w.recent_len, a.kt, a.ks,
        a.st_bottleneck, a.st_channels, a.hidden * 2, a.conv, a.dropout,
        use_st_enhance=not a.no_st_enhance, use_weather_gate=not a.no_weather_gate,
        n_step=a.sts_step, dyn_columns=w.meta["dyn_columns"],
        wg_part=a.wg_part, use_daily=not a.no_daily,
        use_weekly=not a.no_weekly, use_spatial_att=a.spatial_att,
        output_residual=getattr(a, "output_residual", "none")),
}


def collect(model, batcher, device):
    model.eval()
    ys, ps, ms, ts = [], [], [], []
    with torch.no_grad():
        for b in batcher:
            p = model(b) * batcher.std + batcher.mean
            ys.append(b["y"].cpu().numpy())
            ps.append(p.float().cpu().numpy())
            ms.append(b["mask"].cpu().numpy())
            ts.append(b["tod"].cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(ms), np.concatenate(ts))


def run(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    path = os.path.join(ROOT, "data", "processed", f"dataset_{args.tag}.npz")
    w = WindowIndex(path)
    tr = Batcher(w, "train", device, args.batch, shuffle=True, seed=args.seed)
    va = Batcher(w, "val", device, args.batch)
    te = Batcher(w, "test", device, args.batch)
    # About 40% of the validation period falls within school holidays or the
    # Christmas period, when traffic is smoother and easier to predict. The
    # test period contains only regular workdays. Using the full validation
    # set for early stopping may therefore favour configurations tuned to
    # smoother days, so this flag restricts model selection to regular
    # validation workdays.
    va_sel = va
    if args.val_workday_only:
        cols = w.meta["dyn_columns"]
        si, ci = cols.index("is_school_holiday"), cols.index("is_christmas_period")
        keep = [d for d in range(len(w.days))
                if not (w.dyn[w.day_of_row == d, si].max() > 0
                        or w.dyn[w.day_of_row == d, ci].max() > 0)]
        va_sel = Batcher(w, "val", device, args.batch, keep_days=keep)
        print(f"Early stopping uses only regular validation workdays: "
              f"{va_sel.n_sample():,} samples "
              f"({va_sel.n_sample()/va.n_sample()*100:.0f}% of validation)")

    name = args.run_name or args.model
    model = REGISTRY[args.model](w, args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=5)
    print(f"{name} params {n_par:,}  train samples {tr.n_sample():,}  "
          f"batches {len(tr)}  device {device}")

    ck_dir = os.path.join(ROOT, "checkpoints", args.group) if args.group else os.path.join(ROOT, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    ck = os.path.join(ck_dir, f"{name}_{args.tag}_seed{args.seed}.pt")

    best, best_ep, bad, hist = float("inf"), -1, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in tr:
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            if args.loss == "mae":
                loss = masked_mae(pred, b["y_norm"], b["mask"])
            elif args.loss == "huber":
                loss = masked_huber(pred, b["y_norm"], b["mask"], args.huber_delta)
            else:
                loss = masked_mse(pred, b["y_norm"], b["mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for b in va_sel:
                p = model(b) * va_sel.std + va_sel.mean
                vl += masked_mae(p, b["y"], b["mask"]).item() * b["mask"].sum().item()
                vn += b["mask"].sum().item()
        vmae = vl / vn
        sched.step(vmae)
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_mae=vmae,
                         lr=opt.param_groups[0]["lr"]))
        flag = ""
        if vmae < best - 1e-5:
            best, best_ep, bad = vmae, ep, 0
            torch.save(model.state_dict(), ck)
            flag = " *"
        else:
            bad += 1
        if ep % args.log_every == 0 or flag:
            print(f"  ep {ep:>3}  train {tot/nb:.5f}  val MAE {vmae:.4f}{flag}")
        if bad >= args.patience:
            print(f"  Validation did not improve for {args.patience} epochs; "
                  f"stopped at epoch {ep}")
            break
    secs = time.time() - t0
    print(f"Best epoch {best_ep}, validation MAE {best:.4f} mph, "
          f"elapsed {secs/60:.1f} minutes")

    model.load_state_dict(torch.load(ck))
    frames = []
    for split, bt in (("val", va), ("test", te)):
        y, p, m, tod = collect(model, bt, device)
        frames.append(evaluate(y, p, m, tod, w.horizons, name, split))
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = args.seed
    df["dataset_tag"] = args.tag

    res = os.path.join(ROOT, "results", args.group) if args.group else os.path.join(ROOT, "results")
    os.makedirs(res, exist_ok=True)
    df.to_csv(os.path.join(res, f"metrics_{name}_{args.tag}_seed{args.seed}.csv"),
              index=False, encoding="utf-8-sig")
    log = os.path.join(ROOT, "logs", args.group) if args.group else os.path.join(ROOT, "logs")
    os.makedirs(log, exist_ok=True)
    with open(os.path.join(log, f"{name}_{args.tag}_seed{args.seed}.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), n_params=n_par, best_epoch=best_ep,
                       best_val_mae=best, minutes=secs / 60, history=hist),
                  f, ensure_ascii=False, indent=2)
    print("\nAll day")
    print(show(df, "all"))
    print("\nAM peak")
    print(show(df[df.split == "test"], "am_peak"))
    print("\nPM peak")
    print(show(df[df.split == "test"], "pm_peak"))
    return df


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--tag", default="eps0.3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--loss", default="mse", choices=["mse", "mae", "huber"])
    ap.add_argument("--huber_delta", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tf_d_model", type=int, default=64)
    ap.add_argument("--tf_heads", type=int, default=4)
    ap.add_argument("--tf_layers", type=int, default=2)
    ap.add_argument("--tf_ff", type=int, default=128)
    ap.add_argument("--stae_input_emb", type=int, default=16)
    ap.add_argument("--stae_tod_emb", type=int, default=16)
    ap.add_argument("--stae_dow_emb", type=int, default=8)
    ap.add_argument("--stae_adaptive_emb", type=int, default=32)
    ap.add_argument("--kt", type=int, default=3, help="Temporal convolution kernel size")
    ap.add_argument("--ks", type=int, default=3, help="Chebyshev polynomial order")
    ap.add_argument("--conv", default="cheb", choices=["cheb", "gcn"])
    ap.add_argument("--st_channels", type=int, default=64)
    ap.add_argument("--st_bottleneck", type=int, default=16)
    ap.add_argument("--sts_channels", type=int, default=64)
    ap.add_argument("--sts_blocks", type=int, default=3, help="Number of STSGCL layers")
    ap.add_argument("--sts_convs", type=int, default=3, help="Number of graph-convolution layers in each STSGCM")
    ap.add_argument("--sts_step", type=int, default=3, help="Number of time steps spanned by the local spatio-temporal graph")
    ap.add_argument("--no_st_enhance", action="store_true", help="Disable spatio-temporal enhancement")
    ap.add_argument("--no_weather_gate", action="store_true", help="Disable weather gating")
    ap.add_argument("--wg_part", default="both", choices=["both", "weather", "time"],
                    help="Keep only the weather multiplicative gate or only the time additive bias for diagnostic separation")
    ap.add_argument("--spatial_att", action="store_true",
                    help="Enable spatial attention on the fixed graph to reweight neighbours")
    ap.add_argument("--output_residual", default="none",
                    choices=["none", "last", "ha"],
                    help="Predict residuals relative to the latest value or the training-period node time-slot mean")
    ap.add_argument("--no_daily", action="store_true", help="Disable the daily periodic branch")
    ap.add_argument("--no_weekly", action="store_true", help="Disable the weekly periodic branch")
    ap.add_argument("--run_name", default=None, help="Name used for result files; defaults to the model name")
    ap.add_argument("--group", default=None,
                    help="Subdirectory for results, checkpoints, and logs to isolate different datasets")
    ap.add_argument("--val_workday_only", action="store_true",
                    help="Use only regular validation workdays for early stopping to match the test-period distribution")
    ap.add_argument("--log_every", type=int, default=5)
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
