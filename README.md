# The Guardrail Trinity

**Procedural Safety Schemas, Reasoning Chain Decomposition, and Dual-Mode Inference for LLM Guardrails**

Huy H. Nguyen<sup>1,2</sup>, Pride Kavumba<sup>1</sup>, Jiaxuan Li<sup>1</sup>, Duc Minh Vo<sup>1</sup>, Koki Wataoka<sup>1</sup>

<sup>1</sup>SB Intuitions Corp., Japan &nbsp;&nbsp; <sup>2</sup>National Institute of Informatics, Japan

Accepted to [AACL-IJCNLP 2026](https://2026.aaclnet.org/), Hengqin, China: the 5th
Conference of the Asia-Pacific Chapter of the Association for Computational
Linguistics and the 15th International Joint Conference on Natural Language
Processing.

> **Status: the code release is in preparation.** This repository currently holds the
> license and this README. The data construction, training, and evaluation code for
> every method in the paper will be published here soon. Watch the repository to be
> notified.

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

## Planned release

The release covers every method introduced in the paper.

- **Data construction.** The curation pipeline end to end: procedural schema sampling
  (taxonomy codes, category ordering, and list, JSON, or paragraph rendering, with the
  semantic collision filter on distractors), reasoning chain decomposition,
  hard-negative selection, permission augmentation, and the fast and slow blend that
  assembles a dual-mode training set.
- **Training.** The supervised fine-tuning loops and the LoRA and accelerator
  configurations behind the reported runs, for dense and mixture-of-experts base
  models.
- **Evaluation.** The evaluation harness for the benchmark suites in the paper,
  including our BeaverTails-PolicyShift probe, together with the scripts that
  aggregate results into the reported tables.

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
