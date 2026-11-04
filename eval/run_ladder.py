# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Evaluate trained ladder checkpoints in the mode(s) the paper uses for each.

Aegis, Wild, and Mix are evaluated in fast mode, Reason and Adapt in slow mode,
and the Hybrid family in both. For each (variant, seed) this prints, or runs
with --run, one evaluate_universal.py command per mode.

Checkpoints are expected at <model_root>/<BASE>_<VARIANT>_lr-<LR>_seed-<SEED>,
the layout written by training/recipe.py.

Examples:
    python eval/run_ladder.py --base Qwen3-8B --model_root output
    python eval/run_ladder.py --base Qwen3-32B --model_root output --variants Hybrid-PN --seeds 42 --run
"""
import argparse
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from recipe import MODELS, VARIANTS, SEEDS, profile_for  # noqa: E402
from variants import modes_for  # noqa: E402


def main():
    """Print (or run) one evaluate_universal.py command per checkpoint and mode."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", choices=list(MODELS), required=True, help="Base model")
    p.add_argument("--model_root", required=True, help="Directory with the trained checkpoints")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--suite", default="ALL")
    p.add_argument("--input_dir", default="datasets", help="Directory with test_ready_*.jsonl")
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--run", action="store_true", help="Execute instead of printing")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    lr = profile_for(args.base)["lr"]
    for variant in args.variants:
        for seed in args.seeds:
            ckpt = os.path.join(args.model_root, f"{args.base}_{variant}_lr-{lr}_seed-{seed}")
            for mode in modes_for(variant):
                cmd = [sys.executable, os.path.join(here, "evaluate_universal.py"),
                       "--model", ckpt, "--mode", mode, "--suite", args.suite,
                       "--input_dir", args.input_dir, "--output_dir", args.output_dir]
                print(shlex.join(cmd))
                if args.run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
