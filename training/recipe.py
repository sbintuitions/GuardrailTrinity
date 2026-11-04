# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Paper training recipe: per-model hyperparameters and the launch command.

Variants use the paper names (Aegis, Wild, Mix, Reason, Adapt, Hybrid, each with
optional -P and -PN augmentations).

Prints (or runs, with --run) the `accelerate launch` command for one base
model, ablation variant, and seed. All paper runs used one node with 8 GPUs
(H100 80GB), LoRA, 20 epochs, early stopping with patience 5, and a
gradient-accumulation factor chosen per dataset so every run takes roughly
3,000 optimizer steps.

Examples:
    python training/recipe.py --model Qwen3-8B --variant Hybrid-PN --seed 42
    python training/recipe.py --model Qwen3-32B --variant Hybrid --seed 0 --run
    python training/recipe.py --table --model Qwen3-8B     # grad-accum per variant
"""
import argparse
import math
import os
import re
import shlex
import subprocess
import sys

MODELS = {
    "Llama-3.1-8B-Instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "template": "<|start_header_id|>assistant<|end_header_id|>\\n\\n",
    },
    "Qwen3-8B": {"hf_id": "Qwen/Qwen3-8B", "template": "<|im_start|>assistant\\n"},
    "Qwen3-14B": {"hf_id": "Qwen/Qwen3-14B", "template": "<|im_start|>assistant\\n"},
    "Qwen3-32B": {"hf_id": "Qwen/Qwen3-32B", "template": "<|im_start|>assistant\\n"},
    "Qwen3-30B-A3B-Instruct-2507": {
        "hf_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "template": "<|im_start|>assistant\\n",
        "is_moe": True,
    },
}

VARIANTS = ["Aegis", "Wild", "Mix", "Reason", "Reason-P", "Reason-PN",
            "Adapt", "Adapt-P", "Adapt-PN", "Hybrid", "Hybrid-P", "Hybrid-PN"]
SEEDS = [0, 1, 42]

NUM_EPOCHS = 20
TARGET_TOTAL_STEPS = 3000
GPUS = 8  # per node; the global batch below assumes this

_DENSE_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Keyed by (size class, is_moe). The learning rate is lower above 8B and for MoE.
PROFILES = {
    ("small", False): {"lr": "1e-4", "batch": 4, "modules": _DENSE_MODULES,
                       "accelerate": "deepspeed", "ds_config": "deepspeed_config_lora.json"},
    ("medium", False): {"lr": "5e-5", "batch": 4, "modules": _DENSE_MODULES,
                        "accelerate": "deepspeed", "ds_config": "deepspeed_config_lora.json"},
    # Attention-only LoRA: adapting the expert projections breaks FSDP's
    # per-parameter all-gather ordering across ranks.
    ("medium", True): {"lr": "5e-5", "batch": 2, "modules": "q_proj,k_proj,v_proj,o_proj",
                       "accelerate": "fsdp", "ds_config": None},
    # ZeRO-3: the 64 GB of bf16 base weights do not fit replicated under ZeRO-2.
    ("large", False): {"lr": "5e-5", "batch": 4, "modules": _DENSE_MODULES,
                       "accelerate": "deepspeed", "ds_config": "deepspeed_config_lora_zero3.json"},
}

COMMON = {"context_length": 4096, "lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
          "max_grad_accum": 32}


def profile_for(model):
    """Return the hyperparameter profile for a base model.

    Classifies the model name into a size class (small ≤10B, medium ≤30B,
    large >30B) from the "NB" suffix and looks up PROFILES keyed by
    (size class, is_moe).
    """
    info = MODELS[model]
    size_b = float(re.search(r"(\d+\.?\d*)B", model, re.IGNORECASE).group(1))
    size = "small" if size_b <= 10 else "medium" if size_b <= 30 else "large"
    return PROFILES[(size, info.get("is_moe", False))]


def compute_grad_accum(n_samples, batch, max_grad_accum=COMMON["max_grad_accum"]):
    """Power-of-two accumulation that brings total steps closest to the target.

    global_batch = batch x grad_accum x GPUS; steps = (n // global_batch) x epochs.
    """
    steps_per_epoch = TARGET_TOTAL_STEPS / NUM_EPOCHS
    raw = (n_samples / steps_per_epoch) / (batch * GPUS)
    ga = 2 ** round(math.log2(max(1.0, raw)))
    return max(2, min(max_grad_accum, ga))


def count_lines(path):
    """Return the number of lines in a file (rows in a JSONL dataset)."""
    with open(path) as f:
        return sum(1 for _ in f)


def build_command(model, variant, seed, dataset_dir, output_dir, n_train,
                  use_mlflow=False, repo_root=None):
    """Assemble the `accelerate launch` command line for one (model, variant, seed).

    Picks train_moe.py vs train.py, the FSDP or DeepSpeed backend config, and the
    LoRA target modules from the model's profile; derives the run name
    {model}_{variant}_lr-{lr}_seed-{seed} and the gradient-accumulation factor
    from the training-set size n_train.

    Returns: (run_name, cmd_as_list, grad_accum, global_batch, approx_total_steps).
    """
    info, prof = MODELS[model], profile_for(model)
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ga = compute_grad_accum(n_train, prof["batch"])
    run_name = f"{model}_{variant}_lr-{prof['lr']}_seed-{seed}"
    train_script = "train_moe.py" if info.get("is_moe") else "train.py"

    if prof["accelerate"] == "fsdp":
        backend = ["--config_file", os.path.join(root, "training", "fsdp_config.yaml")]
    else:
        backend = ["--use_deepspeed", "--deepspeed_config_file",
                   os.path.join(root, "training", prof["ds_config"])]

    cmd = (["accelerate", "launch"] + backend +
           ["--num_machines", "1", "--num_processes", str(GPUS),
            os.path.join(root, "training", train_script),
            "--model_name_or_path", info["hf_id"],
            "--train_file", os.path.join(dataset_dir, f"train_{variant}.jsonl"),
            "--val_file", os.path.join(dataset_dir, f"val_{variant}.jsonl"),
            "--output_dir", os.path.join(output_dir, run_name),
            "--run_name", run_name,
            "--response_template", info["template"],
            "--context_length", str(COMMON["context_length"]),
            "--learning_rate", prof["lr"],
            "--seed", str(seed),
            "--num_train_epochs", str(NUM_EPOCHS),
            "--per_device_train_batch_size", str(prof["batch"]),
            "--per_device_eval_batch_size", str(prof["batch"]),
            "--gradient_accumulation_steps", str(ga),
            "--save_strategy", "epoch", "--eval_strategy", "epoch", "--do_eval",
            "--load_best_model_at_end", "True",
            "--metric_for_best_model", "eval_loss",
            "--logging_steps", "5",
            "--report_to", "mlflow" if use_mlflow else "none",
            "--bf16", "True",
            "--use_lora", "true",
            "--lora_r", str(COMMON["lora_r"]),
            "--lora_alpha", str(COMMON["lora_alpha"]),
            "--lora_dropout", str(COMMON["lora_dropout"]),
            "--lora_target_modules", prof["modules"]])
    global_batch = prof["batch"] * ga * GPUS
    steps = (n_train // global_batch) * NUM_EPOCHS
    return run_name, cmd, ga, global_batch, steps


def main():
    """Parse arguments; print the grad-accum table or the launch command, optionally running it."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--seed", type=int, default=42, help=f"paper seeds: {SEEDS}")
    p.add_argument("--dataset_dir", default="datasets",
                   help="Directory holding train_<variant>.jsonl and val_<variant>.jsonl")
    p.add_argument("--output_dir", default="output", help="Checkpoints are written here")
    p.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    p.add_argument("--run", action="store_true", help="Execute instead of printing")
    p.add_argument("--table", action="store_true",
                   help="Print grad-accum, global batch, and steps for every variant")
    args = p.parse_args()

    if args.table:
        batch = profile_for(args.model)["batch"]
        print(f"{'variant':<8} {'N':>7} {'ga':>3} {'global_batch':>12} {'~steps':>7}")
        for v in VARIANTS:
            path = os.path.join(args.dataset_dir, f"train_{v}.jsonl")
            if not os.path.isfile(path):
                print(f"{v:<8} missing: {path}")
                continue
            n = count_lines(path)
            ga = compute_grad_accum(n, batch)
            gb = batch * ga * GPUS
            print(f"{v:<8} {n:>7} {ga:>3} {gb:>12} {(n // gb) * NUM_EPOCHS:>7}")
        return
    if not args.variant:
        p.error("--variant is required unless --table is given")

    train_path = os.path.join(args.dataset_dir, f"train_{args.variant}.jsonl")
    if not os.path.isfile(train_path):
        sys.exit(f"Missing {train_path}. Generate it first: "
                 f"python data/generate_train_datasets.py --models {args.variant}")
    run_name, cmd, ga, gb, steps = build_command(
        args.model, args.variant, args.seed, args.dataset_dir, args.output_dir,
        count_lines(train_path), use_mlflow=args.mlflow)
    print(f"# {run_name}: grad_accum={ga}, global_batch={gb}, ~{steps} steps "
          f"over {NUM_EPOCHS} epochs (early stopping, patience 5)")
    print(shlex.join(cmd))
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
