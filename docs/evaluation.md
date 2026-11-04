# Evaluation

`eval/evaluate_universal.py` evaluates a trained model with vLLM on the benchmark
suites. `eval/evaluate_baselines.py` evaluates the 15 baseline guard variants with
each model's documented interface. `eval/generate_pivot_tables.py` and
`eval/generate_plots.py` turn the result files into the paper's tables and
figures.

## Modes

`--mode fast` uses the `text_fast` prompt column and at most 192 output tokens.
`--mode slow` uses `text_slow` and at most 1,024 tokens. Each test row carries both
prompts, so one test set serves both modes. Decoding is greedy.

The paper evaluates each variant in its native mode(s): `Aegis`, `Wild`, and `Mix`
in fast mode; `Reason` and `Adapt` in slow mode; and the `Hybrid` family in both.
Result rows carry a suffix only for the `Hybrid` family: `-1` for System-1 (fast)
and `-2` for System-2 (slow), for example `Hybrid-PN-2`.

## Benchmark suites

| Suite | Benchmarks | Primary metric |
|---|---|---|
| `A` | AEGIS, WildGuard, ToxicChat | Macro F1 |
| `B` | DynaHate | Recall, FPR, Adv-BA |
| `C` | DynaGuard, BeaverTails-PolicyShift | Guideline adherence, Macro F1 |
| `D` | XSTest, BeaverTails | FRR and Cal-BA; OOD accuracy |
| `ALL` | all of the above | |

Metrics are defined in `eval/metrics.py`. Every metric scores only the binary
safe/unsafe verdict; category outputs are never scored. Companion metrics
(balanced accuracy, FPR, unsafe recall) are reported next to each headline metric
so a degenerate "always unsafe" or "always safe" strategy is visible.

## Output parsing

`parse_model_output` strips `<think>` blocks and the `<analysis>` chain, then reads
the JSON verdict. If the JSON is malformed it falls back to detecting `unsafe`
without `safe`. The **application-layer override** then applies: if the model
outputs `unsafe` with explicit `Safety Categories` and none of them exist in the
`<unsafe_categories>` block of the prompt, the verdict is corrected to `safe`. It
fires only when the model provides a categories field. This matters for the `-P`
and `-PN` models, which learn to answer `safe` when every flagged category is
permitted. Result files carry the `_PRIME` suffix to mark this parser.

## Running

```bash
# One checkpoint (tensor parallelism defaults to all visible GPUs)
python eval/evaluate_universal.py --model output/Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42 \
    --mode slow --suite ALL

# The whole ladder for one base model, in each variant's native mode(s)
python eval/run_ladder.py --base Qwen3-8B --model_root output            # print commands
python eval/run_ladder.py --base Qwen3-8B --model_root output --run      # run them
```

Useful flags: `--api_url http://localhost:8000` queries a running vLLM
OpenAI-compatible server instead of loading the model, `-tp` sets the tensor
parallel size, `--max_num_seqs` (default 256) and `--gpu_memory_utilization`
(default 0.90) tune the vLLM scheduler, and `--recompute_from_jsonl` recomputes
metrics from an existing generations file after a parser or metric change.

Outputs go to `--output_dir` (default `eval_results/`):

| File | Content |
|---|---|
| `eval__{model_slug}_{suite}_{mode}_PRIME.json` | Metrics per benchmark |
| `generated__{model_slug}_{suite}_{mode}.jsonl` | Raw generations and parsed verdicts |

The model slug is the checkpoint directory name, so it already encodes the base
model, variant, learning rate, and seed.

## Baselines

`evaluate_baselines.py` loads each baseline from the Hugging Face Hub (or from
`--model_root`, a local mirror laid out as `<org>/<name>`) and feeds it the prompt
format it documents.

| Key | Model | Backend | Prompt |
|---|---|---|---|
| `llama-guard-3` | `meta-llama/Llama-Guard-3-8B` | vLLM | Tokenizer template (MLCommons S1 to S14) |
| `llama-guard-4` | `meta-llama/Llama-Guard-4-12B` | vLLM | Tokenizer template via `AutoProcessor` |
| `qwen3guard` | `Qwen/Qwen3Guard-Gen-8B` | vLLM | Tokenizer template; Safe/Unsafe/Controversial |
| `nemotron` | `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | vLLM | Model-card template (S1 to S23) |
| `gpt-oss-{20b,120b}-{low,medium,high}` | `openai/gpt-oss-safeguard-{20b,120b}` | vLLM | Generic policy as system message; three reasoning efforts |
| `shieldgemma` | `google/shieldgemma-9b` | HF logits | Four built-in policies, P(Yes) over a threshold |
| `wildguard` | `allenai/wildguard` | vLLM | Model-card `[INST]` template |
| `granite` | `ibm-granite/granite-guardian-3.1-8b` | HF logits | `guardian_config` risk name `harm` |
| `dynaguard-8b-{fast,cot}` | `tomg-group-umd/DynaGuard-8B` | vLLM | Its documented policy format |

The 120B gpt-oss variants need two 80 GB GPUs. ShieldGemma and Granite Guardian
score logits and do not generate. Baseline files are named
`eval__{model_key}_{suite}_PRIME.json`, with no mode token.

```bash
python eval/evaluate_baselines.py --model llama-guard-3 --suite ALL
python eval/evaluate_baselines.py --model shieldgemma --suite A --shieldgemma_threshold 0.4
python eval/evaluate_baselines.py --model all --suite ALL
```

## Tables and figures

```bash
# One table per base model (mean and standard deviation over seeds)
python eval/generate_pivot_tables.py --dir eval_results --pattern Qwen3-8B
# Baselines
python eval/generate_pivot_tables.py --dir eval_results --pattern baselines
# Figures from the aggregated CSVs
python eval/generate_plots.py --results_dir eval_results --out_dir figures
```

The pivot script writes `aggregated_<base>_results.csv` and
`aggregated_baselines_results.csv` next to the results; the plot script reads them.
`generate_plots.py` expects the three Qwen3 models, Llama-3.1-8B, and the baselines
to be present.

## Analyses from the paper

| Script | What it does | Needs a GPU |
|---|---|---|
| `eval/audit_reasoning_chains.py` | Checks that slow-mode chains are consistent with the verdict and cite codes that exist in the prompt taxonomy; exports a sample for manual review | No |
| `eval/audit_chain_faithfulness.py` | Input grounding and case-description accuracy of the synthetic chains over the training sets | No |
| `eval/analyze_hn_proxy_overlap.py` | Overlap between hard-negative sets chosen by two toxicity proxies | Optional |
| `eval/permission_flip_examples.py` | Runs curated prompts with and without an exemption note and writes a Markdown report | Yes |

```bash
python eval/audit_reasoning_chains.py --input_dir eval_results --pattern Qwen3-32B
python eval/audit_chain_faithfulness.py --dataset-dir datasets
python eval/analyze_hn_proxy_overlap.py --limit 2000
python eval/permission_flip_examples.py --model output/Qwen3-32B_Hybrid-PN_lr-5e-5_seed-42
```

The LLM-judge agreement audit reported in the paper is not part of this release: its
inputs contain excerpts of third-party benchmark data, which this repository does not
redistribute.

<!-- Copyright (c) 2026 SB Intuitions Corp. SPDX-License-Identifier: MIT -->
