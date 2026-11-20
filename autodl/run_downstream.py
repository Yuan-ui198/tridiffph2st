"""Portable Linux wrapper for the Lightning downstream entrypoint."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional unique suffix for the timestamped result directory.",
    )
    parser.add_argument(
        "--ablation",
        choices=["full", "no_diffusion", "no_omiclip", "no_h3"],
        default="full",
        help="Controlled large-module ablation variant.",
    )
    args = parser.parse_args()

    command = [sys.executable, str(ROOT / "src" / "main.py"), "--config_name", args.config, "--mode", "cv"]
    if args.all_folds:
        command.append("--all_folds")
    else:
        command.extend(["--fold", str(args.fold)])
    if args.resume:
        command.extend(["--resume_ckpt_path", args.resume])
    if args.max_epochs is not None:
        command.extend(["--max_epochs", str(args.max_epochs)])
    if args.run_tag:
        command.extend(["--run_tag", args.run_tag])
    if args.ablation != "full":
        command.extend(["--ablation", args.ablation])
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
