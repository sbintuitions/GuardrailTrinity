# Reproducing the paper

This walkthrough takes you from the public datasets to the paper's tables and
figures. The full grid is large: training all 12 variants across the four base
models took about 5,800 H100 GPU-hours (reported in the paper). Each step below also
works for a single model, variant, and seed, which is the sensible way to start.

## 0. Environment

The paper runs used an NVIDIA PyTorch 26.01 container with the packages pinned in
the requirements files.

```bash
pip install -r requirements.txt          # data, metrics, tables, plots (CPU)
pip install -r requirements-gpu.txt      # training and evaluation (CUDA)
```

`requirements-gpu.txt` includes vLLM and DeepSpeed. flash-attn is installed as a
separate step because it is compiled against your CUDA toolkit and PyTorch: after
the two files above, run `pip install flash-attn==2.8.3 --no-build-isolation`, using
a CUDA toolkit whose major and minor version match the one PyTorch was built with
(the build stops with a version-mismatch error otherwise). If you cannot build it,
append `--use_flash_attn False` to the command that `training/recipe.py` prints to
use PyTorch attention instead. The pinned set was checked end to end (train, merge,
vLLM evaluation) with that flag, not with flash-attn 2.8.3.
Models and datasets download from the Hugging Face Hub, and Llama models require
accepting the license there.
For vLLM on H100 the paper runs used:

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1
```

## 1. Build the datasets

```bash
python data/generate_train_datasets.py        # 12 variants -> datasets/train_*.jsonl, val_*.jsonl
python data/generate_test_datasets.py         # 8 benchmarks -> datasets/test_ready_*.jsonl
python data/sample_datasets_batch.py datasets/train_Hybrid-PN.jsonl -n 3   # spot-check; writes datasets/dataset_samples.txt
```

Check `datasets/train_statistics.csv` against the data pipeline description in
[data_pipeline.md](data_pipeline.md). Generation needs network access to the Hub,
the DynaHate GitHub CSV, and the `unitary/toxic-bert` proxy; the proxy scores run
on CPU or GPU.

## 2. Train

```bash
python training/recipe.py --model Qwen3-8B --table                    # step budget per variant
python training/recipe.py --model Qwen3-8B --variant Hybrid-PN --seed 42 --run
```

The recipe prints the exact `accelerate launch` command (drop `--run` to inspect
it). It assumes one node with 8 GPUs. Checkpoints land in
`output/{BASE}_{VARIANT}_lr-{LR}_seed-{SEED}/`, merged and ready for vLLM. Repeat
for the 12 variants and seeds 0, 1, 42; see [training.md](training.md).

## 3. Evaluate

```bash
python eval/run_ladder.py --base Qwen3-8B --model_root output --run   # every variant, native modes
python eval/evaluate_baselines.py --model all --suite ALL             # 15 baseline variants
```

Results go to `eval_results/`. See [evaluation.md](evaluation.md) for modes,
suites, and flags.

## 4. Tables and figures

```bash
python eval/generate_pivot_tables.py --dir eval_results --pattern Qwen3-8B
python eval/generate_pivot_tables.py --dir eval_results --pattern baselines
python eval/generate_plots.py --results_dir eval_results --out_dir figures
```

Rows are labeled with the paper's names: `Aegis`, `Wild`, `Mix`, `Reason`,
`Adapt`, and the `Hybrid` family with `-1` or `-2` for the inference mode.
The tables report the mean and standard deviation over the three seeds.

## 5. Analyses

```bash
python eval/audit_reasoning_chains.py --input_dir eval_results
python eval/audit_chain_faithfulness.py --dataset-dir datasets
python eval/analyze_hn_proxy_overlap.py
python eval/permission_flip_examples.py --model output/Qwen3-32B_Hybrid-PN_lr-5e-5_seed-42
```

## Expectations

- **Seed variance is real.** Several procedural variants at 8B swing widely
  across seeds on DynaHate recall; the paper reports mean and standard deviation
  for that reason. Compare means with the spread in view.
- **Source data can drift.** Hub datasets are not versioned by the scripts. A
  re-download can change row counts slightly, so `recipe.py --table` recomputes the
  step budget from the files you generated.
- **Baselines depend on model-card interfaces.** Each baseline uses the prompt its
  authors document, so scores can move if a model card or tokenizer template
  changes.

<!-- Copyright (c) 2026 SB Intuitions Corp. SPDX-License-Identifier: MIT -->
