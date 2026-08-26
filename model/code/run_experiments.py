# -*- coding: utf-8 -*-
"""Run the novfeb baselines and the complete A/B ablation series."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
PYTHON = sys.executable

A_SERIES = {
    "A0": ["--no_st_enhance", "--no_weather_gate"],
    "A1": ["--no_weather_gate"],
    "A2": ["--no_st_enhance", "--wg_part", "weather"],
    "A3": ["--wg_part", "weather"],
    "A4": ["--wg_part", "weather", "--spatial_att"],
}

B_SERIES = {
    "B4": ["--no_st_enhance", "--wg_part", "time"],
    "B5": ["--wg_part", "time"],
    "B6": ["--no_st_enhance", "--wg_part", "both"],
    "B7": ["--wg_part", "both"],
}


def execute(command: list[str], log_path: str) -> None:
    started = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=CODE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(f"{os.path.basename(log_path)} return code {completed.returncode}, "
          f"elapsed {(time.time() - started) / 60:.1f} minutes")
    if completed.returncode:
        raise SystemExit(completed.returncode)


def train_command(model: str, name: str, seed: int, args, flags=None):
    command = [
        PYTHON,
        "-u",
        os.path.join(CODE, "train.py"),
        "--model",
        model,
        "--run_name",
        name,
        "--tag",
        args.tag,
        "--group",
        args.group,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch",
        str(args.batch),
        "--val_workday_only",
    ]
    return command + (flags or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="novfeb_eps0.3")
    parser.add_argument("--group", default="novfeb_155nodes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_a", action="store_true")
    parser.add_argument("--skip_b", action="store_true")
    args = parser.parse_args()

    dataset = os.path.join(ROOT, "data", "processed", f"dataset_{args.tag}.npz")
    if not os.path.isfile(dataset):
        raise SystemExit(f"Missing modelling input {dataset}")

    log_dir = os.path.join(ROOT, "logs", args.group)
    os.makedirs(log_dir, exist_ok=True)

    if not args.skip_baselines:
        execute(
            [PYTHON, "-u", os.path.join(CODE, "baseline_ha.py"),
             "--tag", args.tag, "--group", args.group],
            os.path.join(log_dir, "run_historical_average.txt"),
        )
        execute(
            [PYTHON, "-u", os.path.join(CODE, "baseline_recent_average.py"),
             "--tag", args.tag, "--group", args.group],
            os.path.join(log_dir, "run_recent_average.txt"),
        )
        single_seed = [
            ("lstm", "lstm", []),
            ("cnn_lstm", "cnn_lstm", []),
            ("stsgcn", "stsgcn", []),
            ("itransformer", "itransformer", ["--batch", "128"]),
        ]
        for model, name, flags in single_seed:
            execute(
                train_command(model, name, 0, args, flags),
                os.path.join(log_dir, f"run_{name}_seed0.txt"),
            )
        for seed in args.seeds:
            execute(
                train_command("stgcn", "stgcn", seed, args),
                os.path.join(log_dir, f"run_stgcn_seed{seed}.txt"),
            )
            execute(
                train_command(
                    "staeformer", "staeformer", seed, args,
                    ["--batch", "64", "--epochs", "60", "--patience", "10"],
                ),
                os.path.join(log_dir, f"run_staeformer_seed{seed}.txt"),
            )

    if not args.skip_a:
        for seed in args.seeds:
            for name, flags in A_SERIES.items():
                execute(
                    train_command("proposed", name, seed, args, flags),
                    os.path.join(log_dir, f"run_{name}_seed{seed}.txt"),
                )

    if not args.skip_b:
        for seed in args.seeds:
            for name, flags in B_SERIES.items():
                execute(
                    train_command("proposed", name, seed, args, flags),
                    os.path.join(log_dir, f"run_{name}_seed{seed}.txt"),
                )

    execute(
        [PYTHON, "-u", os.path.join(CODE, "collect_results.py"),
         "--tag", args.tag, "--group", args.group],
        os.path.join(log_dir, "run_collect_results.txt"),
    )
    execute(
        [PYTHON, "-u", os.path.join(CODE, "collect_time_ablation.py"),
         "--tag", args.tag, "--group", args.group],
        os.path.join(log_dir, "run_collect_time_ablation.txt"),
    )


if __name__ == "__main__":
    main()
