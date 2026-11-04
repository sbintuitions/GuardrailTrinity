# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Hard-negative proxy overlap analysis (no training required).

Measures proxy dependence at the data level: how similar is the hard-negative
set selected by the paper's proxy (unitary/toxic-bert) to the set selected by an
independent alternative (s-nlp/roberta_toxicity_classifier, a RoBERTa model
from a different team)?

The candidate population mirrors the training pipeline exactly: AEGIS train
rows with safe prompt labels that pass is_hard_negative's early exits (length
filter + demographic-keyword exclusion). Both proxies are then applied with the
pipeline's decision rule (score > 0.70 on a non-negative label). High overlap
means the hard-negative mechanism is not an artifact of one specific
classifier.

Usage:
    python eval/analyze_hn_proxy_overlap.py                    # full pool
    python eval/analyze_hn_proxy_overlap.py --limit 2000       # quick pass
    python eval/analyze_hn_proxy_overlap.py --alt_proxy <hf_model_id>

Outputs (in --output_dir, default audit_results/):
    hn_proxy_overlap.json          — aggregate statistics
    hn_proxy_disagreements.csv     — rows where the two proxies disagree
"""

import argparse
import csv
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import pipeline

# Single source of truth for the demographic filter and label conventions.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from generate_train_datasets import _DEMOGRAPHIC_PAT, _NON_TOXIC_LABELS  # noqa: E402

PAPER_PROXY = "unitary/toxic-bert"
THRESHOLD = 0.70


def eligible(text: str) -> bool:
    """Mirror is_hard_negative's early exits (length + demographic exclusion)."""
    if not text or len(str(text).strip()) < 5:
        return False
    if _DEMOGRAPHIC_PAT.search(str(text)):
        return False
    return True


def fires(results) -> bool:
    """Mirror is_hard_negative's decision rule on one pipeline output."""
    if isinstance(results, dict):
        results = [results]
    if results and isinstance(results[0], list):
        results = results[0]
    return any(
        r["score"] > THRESHOLD and r["label"].lower() not in _NON_TOXIC_LABELS
        for r in results
    )


def score_all(model_id: str, texts, batch_size: int):
    """Run one proxy classifier over all texts; return a fire/no-fire decision per text."""
    device = 0 if torch.cuda.is_available() else -1
    print(f">>> Loading proxy {model_id} (device={'cuda' if device == 0 else 'cpu'})...")
    clf = pipeline("text-classification", model=model_id, device=device)
    decisions = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:512] for t in texts[i:i + batch_size]]
        outs = clf(batch, truncation=True, max_length=512)
        decisions.extend(fires(o) for o in outs)
        if (i // batch_size) % 20 == 0:
            print(f"    {model_id}: {i + len(batch)}/{len(texts)}")
    return decisions


def main():
    """Build the candidate pool, score it with both proxies, and write overlap stats + disagreements."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alt_proxy", default="s-nlp/roberta_toxicity_classifier")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap candidate pool size for a quick pass (0 = full pool)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="audit_results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed; must match the data pipeline (42)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Candidate pool: AEGIS train, shuffled with the pipeline seed, safe prompts
    # only (hard negatives are benign-by-label), passing the early-exit filters.
    print(">>> Loading AEGIS train split...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    ds = ds.shuffle(seed=args.seed)

    texts = []
    for row in ds:
        if str(row.get("prompt_label", "")).lower() != "safe":
            continue
        text = row.get("prompt") or ""
        if eligible(text):
            texts.append(text)
        if args.limit and len(texts) >= args.limit:
            break
    print(f">>> Candidate pool: {len(texts)} eligible safe AEGIS prompts")

    paper = score_all(PAPER_PROXY, texts, args.batch_size)
    alt = score_all(args.alt_proxy, texts, args.batch_size)

    n = len(texts)
    p_set = {i for i, d in enumerate(paper) if d}
    a_set = {i for i, d in enumerate(alt) if d}
    inter = p_set & a_set
    union = p_set | a_set

    stats = {
        "paper_proxy": PAPER_PROXY,
        "alt_proxy": args.alt_proxy,
        "threshold": THRESHOLD,
        "candidates": n,
        "paper_fires": len(p_set),
        "alt_fires": len(a_set),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": len(inter) / len(union) if union else None,
        "overlap_over_min": len(inter) / min(len(p_set), len(a_set)) if p_set and a_set else None,
        "decision_agreement": sum(p == a for p, a in zip(paper, alt)) / n if n else None,
        "paper_fire_rate": len(p_set) / n if n else None,
        "alt_fire_rate": len(a_set) / n if n else None,
    }

    out_json = os.path.join(args.output_dir, "hn_proxy_overlap.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    out_csv = os.path.join(args.output_dir, "hn_proxy_disagreements.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "paper_fires", "alt_fires", "prompt"])
        for i in sorted(p_set ^ a_set):
            w.writerow([i, i in p_set, i in a_set, texts[i][:300]])

    print("\n" + "=" * 60)
    for k, v in stats.items():
        print(f"  {k:20}: {v}")
    print("=" * 60)
    print(f"Saved: {out_json}\n       {out_csv}")


if __name__ == "__main__":
    main()
