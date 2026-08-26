# -*- coding: utf-8 -*-
"""Run the added B4-B7 configurations with node-level time-slot baselines and summarise B0-B7."""
import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
PY = sys.executable
CONFIGS = {
    "B4": ["--no_st_enhance", "--wg_part", "time"],
    "B5": ["--wg_part", "time"],
    "B6": ["--no_st_enhance", "--wg_part", "both"],
    "B7": ["--wg_part", "both"],
}


def run(cmd, log_path):
    started = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8") as stream:
        proc = subprocess.run(cmd, cwd=CODE, stdout=stream,
                              stderr=subprocess.STDOUT, env=env)
    print(f"{os.path.basename(log_path)} return code {proc.returncode}, "
          f"elapsed {(time.time() - started) / 60:.1f} minutes")
    if proc.returncode:
        raise SystemExit(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val_workday_only", action="store_true")
    args = ap.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.group)
    os.makedirs(log_dir, exist_ok=True)
    common = [
        "--tag", args.tag, "--group", args.group, "--device", args.device,
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch", str(args.batch),
    ]
    if args.val_workday_only:
        common.append("--val_workday_only")

    for seed in args.seeds:
        for name, flags in CONFIGS.items():
            cmd = [PY, "-u", os.path.join(CODE, "train.py"),
                   "--model", "proposed", "--run_name", name,
                   "--seed", str(seed)] + common + flags
            run(cmd, os.path.join(log_dir, f"step_{name}_seed{seed}.txt"))

    collect = [PY, "-u", os.path.join(CODE, "collect_time_ablation.py"),
               "--tag", args.tag, "--group", args.group]
    run(collect, os.path.join(log_dir, "step_collect_time_ablation.txt"))


if __name__ == "__main__":
    main()
