# Training

All paper models are trained with supervised fine-tuning and LoRA on one node
with 8 H100 80GB GPUs. `training/recipe.py` holds the per-model settings and
prints the exact launch command.

```bash
python training/recipe.py --model Qwen3-8B --variant Hybrid-PN --seed 42
python training/recipe.py --model Qwen3-8B --variant Hybrid-PN --seed 42 --run
python training/recipe.py --model Qwen3-8B --table      # grad-accum per variant
```

The run name, and the output directory under `--output_dir`, is
`{BASE}_{VARIANT}_lr-{LR}_seed-{SEED}`, for example
`Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42`. The paper uses seeds 0, 1, and 42.
Pass `--mlflow` to log to MLflow (off by default).

For a manual launch the trainers can be invoked directly; the defaults in
`train.py` and `train_moe.py` already match the paper (LoRA r=64, alpha=128,
context 4096).

## Base models

| Base model | Type | Trainer | Parallelism | Config |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | Dense | `train.py` | DeepSpeed ZeRO-2 | `deepspeed_config_lora.json` |
| Qwen3-8B, Qwen3-14B | Dense | `train.py` | DeepSpeed ZeRO-2 | `deepspeed_config_lora.json` |
| Qwen3-32B | Dense | `train.py` | DeepSpeed ZeRO-3 | `deepspeed_config_lora_zero3.json` |
| Qwen3-30B-A3B-Instruct-2507 | MoE | `train_moe.py` | FSDP | `fsdp_config.yaml` |

The MoE model is fully supported but its results are not in the paper.
`deepspeed_config.json` (ZeRO-3 with CPU offload) is for full fine-tuning only,
which is not the paper setting.

**Why ZeRO-3 for 32B.** ZeRO-2 replicates the roughly 64 GB of bf16 base weights
on every GPU, leaving too little of the 80 GB for activations and the eval logits.
ZeRO-3 shards them to about 8 GB per GPU on one node, and keeps the all-gathers on
NVLink.

**Why FSDP for MoE.** ZeRO-3 deadlocks when some ranks skip experts that received
no tokens. `train_moe.py` also sets `output_router_logits = False`, because
retaining router logits for every layer defeats gradient checkpointing and runs
out of memory; the router is pretrained and needs no load-balancing loss during
LoRA fine-tuning.

## LoRA and optimization

| | 8B dense | 14B dense | 32B dense | 30B MoE |
|---|---|---|---|---|
| Rank r / alpha / dropout | 64 / 128 / 0.05 | 64 / 128 / 0.05 | 64 / 128 / 0.05 | 64 / 128 / 0.05 |
| Target modules | q, k, v, o, gate, up, down | same | same | q, k, v, o |
| Learning rate | 1e-4 | 5e-5 | 5e-5 | 5e-5 |
| Per-GPU train / eval batch | 4 / 4 | 4 / 4 | 4 / 4 | 2 / 2 |
| Max `grad_accum` | 32 | 32 | 32 | 32 |

Shared settings: bf16, AdamW with linear learning-rate decay (the `SFTConfig`
default), gradient checkpointing with `use_reentrant=True`, 20 epochs, early
stopping on validation loss with patience 5 epochs, context length 4096, and
checkpoint selection by the best evaluation loss. The separate 8B learning-rate
rescue runs are described in the paper's appendix.

**MoE LoRA targets are attention-only.** Adapting the expert projections creates
about 10,752 per-parameter FSDP tensors (64 experts, 3 projections, 2 LoRA
matrices, 28 layers). Their all-gathers interleave differently across ranks and
cause a shape-mismatch protocol violation. Attention-only LoRA yields 224 tensors
with a rank-consistent order.

**Context length and eval batch.** Qwen3 has a 151,936-token vocabulary, so the
eval logits tensor is `eval_batch x seq_len x 151,936 x 2 bytes`. A batch of 16 at
length 8192 peaked near 25 GB and ran out of memory. A batch of 4 at length 4096
is the setting that fits; do not raise it without checking memory.

## Gradient accumulation and step budget

`grad_accum` is chosen per dataset so each run takes roughly 3,000 optimizer
steps over 20 epochs:

```python
steps_per_epoch = 3000 / 20                       # 150
raw_ga = (n_samples / steps_per_epoch) / (train_batch * 8 GPUs)
ga = 2 ** round(log2(max(1.0, raw_ga)))           # nearest power of two
ga = max(2, min(32, ga))
```

The dense profiles reach the same global batch: 256 for the single-source fast
variants and 512 for the mixed-source variants. The MoE profile halves the
per-GPU batch and doubles `ga` to match.

| Variant | Train rows | `ga` (dense) | Global batch | Max steps |
|---|---|---|---|---|
| Aegis | 30,003 | 8 | 256 | 2,350 |
| Wild | 30,000 | 8 | 256 | 2,350 |
| Mix | 60,003 | 16 | 512 | 2,350 |
| Reason, Adapt | 60,001 | 16 | 512 | 2,350 |
| Reason-P, Reason-PN, Adapt-P, Adapt-PN | about 63,400 | 16 | 512 | 2,500 |
| Hybrid | 80,001 | 16 | 512 | 3,150 |
| Hybrid-P, Hybrid-PN | about 81,100 | 16 | 512 | 3,200 |

Row counts are what `data/generate_train_datasets.py` produces with the default
seed (42) and the dataset versions used for the paper; `recipe.py --table`
recomputes everything from your own files. Max steps are the computed step counts
rounded up to the nearest 50, and early stopping usually ends a run sooner.

**Comparing variants.** Under a decaying schedule, total steps shapes the
learning-rate curve, so it matters more than global batch for a fair comparison.
Within a family (for example `Adapt`, `Adapt-P`, `Adapt-PN`) the global batch is
identical and the step budgets differ by 1 to 5%, because the contrastive pairs
created by the exemption add records to the `-P` and `-PN` variants. That is a
minor confound that, if anything, favors the augmented variants. Cross-family
comparisons test intentionally different data compositions.

## Checkpointing and merging

Training keeps every epoch checkpoint so `load_best_model_at_end` can always
restore the best one. After training, the LoRA adapter is merged into the base
weights so the model loads directly in vLLM, and the intermediate checkpoints are
removed.

- **ZeRO (dense):** the adapter is merged and saved from rank 0.
- **FSDP (MoE):** the adapter is merged and saved with the FSDP-aware sharded save.

If a checkpoint exists in the output directory, a rerun resumes from the latest
one. With `--report_to mlflow`, the MLflow run ID is stored in
`mlflow_run_id.txt` in the output directory so a resumed job appends to the same
run; do not delete it mid-training.

<!-- Copyright (c) 2026 SB Intuitions Corp. SPDX-License-Identifier: MIT -->
