# The Guardrail Trinity

**Procedural Safety Schemas, Reasoning Chain Decomposition, and Dual-Mode Inference for LLM Guardrails**

Huy H. Nguyen<sup>1,2</sup>, Pride Kavumba<sup>1</sup>, Jiaxuan Li<sup>1</sup>, Duc Minh Vo<sup>1</sup>, Koki Wataoka<sup>1</sup>

<sup>1</sup>SB Intuitions Corp., Japan &nbsp;&nbsp; <sup>2</sup>National Institute of Informatics, Japan

Accepted to [AACL-IJCNLP 2026](https://2026.aaclnet.org/), Hengqin, China: the 5th
Conference of the Asia-Pacific Chapter of the Association for Computational
Linguistics and the 15th International Joint Conference on Natural Language
Processing.

> **Status: code released.** This repository holds the data construction, training,
> and evaluation code for every method in the paper. See
> [What is verified](#what-is-verified) for what has been run end to end.

> **Warning:** the datasets used in this repository contain harmful text.

## Abstract

Effective LLM safety guardrails must balance adversarial robustness, custom-policy
adaptability, and precision on benign prompts with sensitive vocabulary. We introduce
the **Guardrail Trinity**, a data-curation recipe combining *procedural safety
schemas*, *reasoning chain decomposition* with hard-negative supervision, and a
*hybrid data blend* for fast and slow inference. Experiments across Llama-3.1 and
Qwen3 models yield three practical insights. **First**, strong standard moderation
scores can conceal a custom-policy compliance gap; procedural schema variation
improves adaptation on our held-out policy-shift probe. **Second**, augmentation must
be matched to model scale and training recipe: smaller models can exhibit training
instability, while hard negatives at larger scales can reduce false alarms at a cost
to harmful-input recall. **Third**, hybrid training supports both direct
classification and reasoning, with the fast mode retaining strong custom-policy
compliance without generating a chain. Together, these findings motivate a deployment
playbook that selects data composition, inference mode, and scale jointly, and
validates both policy adherence and the trade-off between missed harms and false
alarms.

## The recipe

A guardrail classifies content under the policy it is given, so the policy it was
trained on should not be the only one it can follow. The Trinity curates the training
data along three axes.

1. **Procedural safety schemas.** Policy rendering (list, JSON object, or paragraph)
   and category identifiers are randomized during training, so the same harm concept
   appears under different codes and formats. The target cites the code shown in the
   prompt, which keeps the policy block and the verdict linked after randomization.
2. **Reasoning chain decomposition.** The System-2 target describes user intent, agent
   behavior, and policy match in that order, before the verdict. This separates a
   harmful request from an agent's safe refusal, and benign sensitive vocabulary from
   a prohibited use. Hard-negative supervision sharpens the second distinction.
3. **Hybrid data blend.** Fast (System-1) and slow (System-2) targets are mixed at a
   nominal 3:1 ratio, so one set of weights serves a direct verdict or an explicit
   chain, selected through the inference instruction.

## Findings

Evaluated on Llama-3.1-8B and Qwen3 at 8B, 14B, and 32B, against 15 open-weight
baseline configurations. All figures are means over seeds {0, 1, 42}.

- **Standard moderation scores hide a compliance gap.** The strongest open baseline
  averages 85.63 Macro-F1 on standard moderation but scores 64.84 on a custom policy.
  Our Qwen3-32B hybrid model in fast mode reaches 71.87 ± 0.86 Macro-F1 on
  BeaverTails-PolicyShift, about 7 points above that baseline.
- **Fast inference keeps the policy benefit.** Across all nine paired hybrid
  checkpoints, System-1 has higher mean PolicyShift Macro-F1 than System-2, so the
  chain is not required to follow a custom policy.
- **Augmentation trades off against scale.** At 32B in fast mode, adding hard
  negatives to the permission-augmented hybrid recipe lowers the DynaHate
  false-positive rate by 19.7 percentage points, at a cost of 16.1 points of
  harmful-input recall. Validate the trade-off before selecting an operating point.

The paper also reports an audit of the synthesized reasoning chains and documents
seed instability at 8B. See the Limitations section for the scope of these claims.

## Release

The release covers every method introduced in the paper.

- **Data construction** (`data/`). The curation pipeline end to end: procedural schema
  sampling (taxonomy codes, category ordering, and list, JSON, or paragraph rendering,
  with the semantic collision filter on distractors), reasoning chain decomposition,
  hard-negative selection, permission augmentation, and the fast and slow blend that
  assembles a dual-mode training set, plus the generator for the eight test benchmarks.
- **Training** (`training/`). The supervised fine-tuning loops for dense and
  mixture-of-experts base models, the LoRA, DeepSpeed, and FSDP configurations, and
  `recipe.py`, which holds the per-model hyperparameters and prints the exact launch
  command for a base model, variant, and seed.
- **Evaluation** (`eval/`). The evaluation harness for the benchmark suites in the
  paper, including our BeaverTails-PolicyShift probe, the baseline runners, the scripts
  that aggregate results into the reported tables and figures, and the paper's analyses.
- **Documentation** (`docs/`) and **tests** (`tests/`).

### Names

Variants use the paper's names: `Aegis`, `Wild`, and `Mix` (fast, System-1 only),
`Reason` and `Adapt` (slow, System-2 only), and `Hybrid` (both modes), each with
optional `-P` (category permission) and `-PN` (permission plus hard negatives)
augmentations. A `-1` or `-2` suffix on a Hybrid result row is the inference mode:
System-1 (fast) or System-2 (slow). Runs are named
`{BASE}_{VARIANT}_lr-{LR}_seed-{SEED}`, for example
`Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42`.

### Layout

```
data/          dataset construction (train and test) and sampling
training/      train.py, train_moe.py, recipe.py, DeepSpeed and FSDP configs
eval/          evaluate_universal.py, run_ladder.py, baselines, tables, figures, analyses
docs/          reproduce.md, data_pipeline.md, training.md, evaluation.md
tests/         CPU unit tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # data, metrics, tables, plots (CPU)
pip install -r requirements-gpu.txt    # training and evaluation (CUDA GPUs)
```

The requirements files pin exact versions (for example Transformers 4.57.6, as
reported in the paper, and vLLM 0.19.1). flash-attn is installed separately because
it is compiled against your CUDA toolkit; see [docs/reproduce.md](docs/reproduce.md)
for that step and for the `--use_flash_attn False` fallback.

We tested with Python 3.12 and CUDA GPUs (NVIDIA H100 80GB). The paper runs used one
node with 8 GPUs. A short Qwen3-8B training run (batch size 4, context length 4096)
and its vLLM evaluation also ran on a single 80GB GPU. Models and datasets download
from the Hugging Face Hub, so set `HF_HOME` to a directory with enough space and log
in with a token (`HF_TOKEN` or `huggingface-cli login`) for the gated repositories
listed below.

## Datasets and models

The data scripts download the public source datasets and build every training and
test file locally. Each source keeps its own license; please check it before use.

| Dataset | Used for | Hub or source | License | Access |
|---|---|---|---|---|
| AEGIS 2.0 | Training, `aegis` test | [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | CC BY 4.0 | Open |
| WildGuardMix | Training, `wildguard` test | [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) | ODC-BY | Gated (accept the terms) |
| ToxicChat | `toxicchat` test | [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | CC BY-NC 4.0 | Open |
| DynaHate v0.2.3 | `dynahate` test | [GitHub CSV](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) | CC BY 4.0 | Open |
| DynaBench | `dynaguard` test | [montehoover/DynaBench](https://huggingface.co/datasets/montehoover/DynaBench) | MIT | Open |
| BeaverTails | `beavertails`, `beavertails_dynamic` (PolicyShift) tests | [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | CC BY-NC 4.0 | Open |
| XSTest | `xstest` test | [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) | CC BY 4.0 | Open |

ToxicChat and BeaverTails are for non-commercial use only. The hard-negative selector
uses [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert) (Apache 2.0),
which the training-data script downloads on first use. See
[docs/data_pipeline.md](docs/data_pipeline.md) for how each source is processed.

We train and evaluate the following base models. Pass the name in the first column to
`training/recipe.py --model` and `eval/run_ladder.py --base`.

| Name | Checkpoint | License | Access |
|---|---|---|---|
| `Llama-3.1-8B-Instruct` | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | Llama 3.1 Community License | Gated (request access) |
| `Qwen3-8B` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache 2.0 | Open |
| `Qwen3-14B` | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | Apache 2.0 | Open |
| `Qwen3-32B` | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Apache 2.0 | Open |
| `Qwen3-30B-A3B-Instruct-2507` | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Apache 2.0 | Open |

The baseline guard models are listed in [docs/evaluation.md](docs/evaluation.md).

## Quickstart

```bash
python data/generate_train_datasets.py                       # 12 variants -> datasets/
python data/generate_test_datasets.py                        # 8 benchmarks -> datasets/
python training/recipe.py --model Qwen3-8B --variant Aegis --seed 0 --run
python eval/run_ladder.py --base Qwen3-8B --model_root output --variants Aegis --seeds 0 --suite A --run
```

`recipe.py` assumes one node with 8 GPUs; drop `--run` to print the command and adapt
it. [docs/reproduce.md](docs/reproduce.md) walks through the full reproduction: data,
training for every variant and seed, evaluation, tables and figures, and analyses.
Run the unit tests with `pip install pytest && pytest tests`.

## What is verified

- Dataset generation: a fresh run reproduces the training and validation files used
  for the paper byte for byte for all 12 variants, and the statistics table matches.
- Aggregation: the table scripts reproduce our aggregated result tables exactly from
  the per-run result files, for all four base models and the baselines.
- A short training run, LoRA merge, and vLLM evaluation of a Qwen3-8B model ran end
  to end on one GPU with the pinned packages, with `--use_flash_attn False`.
- Not rerun for this release: flash-attn 2.8.3, the multi-GPU launch and the
  mixture-of-experts path, and the test-benchmark generator.

## Citation

```bibtex
@inproceedings{nguyen2026guardrail,
  title     = {The Guardrail Trinity: Procedural Safety Schemas, Reasoning Chain Decomposition, and Dual-Mode Inference for {LLM} Guardrails},
  author    = {Nguyen, Huy H. and Kavumba, Pride and Li, Jiaxuan and Vo, Duc Minh and Wataoka, Koki},
  booktitle = {Proceedings of the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 15th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2026)},
  year      = {2026}
}
```

The ACL Anthology identifier and page numbers will be added once the proceedings are
published.

## Contact

Questions about the paper or the release: `hong.huy.nguyen@sbintuitions.co.jp`, or
open an issue.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 SB Intuitions Corp.
