# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""
Generate evaluation/test datasets for GuardrailTrinity safety moderation models.

This script produces test-ready JSONL files from multiple safety evaluation benchmarks.
Each output row contains pre-formatted prompts for both System-1 (fast) and System-2
(explain) evaluation modes, along with ground-truth labels.

Test benchmarks:
  - AEGIS (nvidia): Prompt + optional response safety evaluation
  - WildGuard (allenai): Prompt + optional response harm evaluation
  - ToxicChat (lmsys): Real-world user toxicity detection
  - DynaHate (aps): Dynamic hate speech benchmark
  - XSTest (Paul): Safe prompts designed to trigger false positives
  - BeaverTails (PKU-Alignment): Prompt + response with multi-category labels
  - BeaverTails-PolicyShift: BeaverTails with a held-out 6-category taxonomy and
    category removal (legacy dataset identifier: beavertails_dynamic)
  - DynaGuard/DynaBench (montehoover): Multi-turn guideline adherence evaluation

Each output row contains:
  - instruction, response, label: raw content and ground-truth
  - text_fast: pre-formatted System-1 prompt
  - text_slow: pre-formatted System-2 prompt
  - source: benchmark identifier

Usage:
  python generate_test_datasets.py                       # all benchmarks
  python generate_test_datasets.py --datasets aegis xstest  # specific benchmarks
"""

import argparse
import json
import os
import random
import pandas as pd
from datasets import load_dataset

from prompts import (
    UNIFIED_TEMPLATE, INST_FAST, INST_EXPLAIN,
    JSON_FMT_FAST, JSON_FMT_EXPLAIN,
    DYNABENCH_INST_FAST, DYNABENCH_INST_EXPLAIN,
    direct_format, get_taxonomy_str,
)
from taxonomies import (
    AEGIS_TAXONOMY, WILD_TAXONOMY,
    NOVEL_TAXONOMY, BEAVER_TO_NOVEL,
)

# ==============================================================================
# 1. TAXONOMIES & MAPPINGS
# ==============================================================================
# All taxonomy definitions live in taxonomies.py — the single source of truth
# shared with generate_train_datasets.py.

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def build_prompts(instruction, response, taxonomy_dict):
    """Build both fast and explain prompts for a given conversation and taxonomy."""
    tax_block = get_taxonomy_str(taxonomy_dict)

    # Test prompts never use exemption notes (empty string)
    fast_text = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_block, "", INST_FAST, JSON_FMT_FAST)
    slow_text = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_block, "", INST_EXPLAIN, JSON_FMT_EXPLAIN)
    return fast_text, slow_text

# ==============================================================================
# 3. DATA PROCESSORS
# ==============================================================================
# Each processor loads a benchmark dataset, applies safety moderation logic
# (prompt-only → prompt label; prompt+response → response label), and formats
# each row into a dict with pre-built fast and slow prompts.
#
# Label convention: 1 = unsafe/harmful, 0 = safe/unharmful.

def process_aegis():
    """Process AEGIS test split — prompt + optional response safety evaluation."""
    try: ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
    except Exception as e: print(f"[Warning] Failed to load aegis: {e}"); return []
    out = []
    for row in ds:
        prompt_label = 1 if row['prompt_label'] == 'unsafe' else 0
        resp_label_raw = row.get('response_label')

        # Safety moderation: if response exists and is labeled, use response label
        if resp_label_raw and str(resp_label_raw).lower() != "nan":
            final_response_text = row.get('response') or "N/A"
            final_label = 1 if str(resp_label_raw).lower() == 'unsafe' else 0
        else:
            final_response_text = "N/A"
            final_label = prompt_label

        p_fast, p_slow = build_prompts(row['prompt'], final_response_text, AEGIS_TAXONOMY)

        out.append({
            "instruction": row['prompt'], "response": final_response_text, "label": final_label,
            "source": "aegis", "text_fast": p_fast, "text_slow": p_slow
        })
    return out

def process_wildguard():
    """Process WildGuard test split — prompt + optional response harm evaluation."""
    try: ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    except Exception as e: print(f"[Warning] Failed to load wildguard: {e}"); return []
    out = []
    for row in ds:
        prompt_label = 1 if row['prompt_harm_label'] == 'harmful' else 0
        resp_label_raw = row.get('response_harm_label')

        if resp_label_raw and str(resp_label_raw).lower() != "nan":
            final_response_text = row.get('response') or "N/A"
            final_label = 1 if resp_label_raw == 'harmful' else 0
        else:
            final_response_text = "N/A"
            final_label = prompt_label

        p_fast, p_slow = build_prompts(row['prompt'], final_response_text, WILD_TAXONOMY)

        out.append({
            "instruction": row['prompt'], "response": final_response_text, "label": final_label,
            "source": "wildguard", "text_fast": p_fast, "text_slow": p_slow
        })
    return out

def process_toxicchat():
    """Process ToxicChat test split — prompt-only toxicity detection (no response)."""
    try: ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    except Exception as e: print(f"[Warning] Failed to load toxicchat: {e}"); return []
    out = []
    for row in ds:
        p_fast, p_slow = build_prompts(row['user_input'], "N/A", AEGIS_TAXONOMY)
        out.append({
            "instruction": row['user_input'], "response": "N/A", "label": 1 if row['toxicity'] == 1 else 0,
            "source": "toxicchat", "text_fast": p_fast, "text_slow": p_slow
        })
    return out

def process_dynahate():
    """Process DynaHate test split — dynamic hate speech classification (prompt-only).

    aps/dynahate on HuggingFace only ships a loading script (no Parquet files),
    and datasets 3.x dropped script support entirely. Load the source CSV directly
    from the original GitHub repo instead.
    """
    _URL = ("https://raw.githubusercontent.com/bvidgen/"
            "Dynamically-Generated-Hate-Speech-Dataset/main/"
            "Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv")
    try:
        df = pd.read_csv(_URL)
    except Exception as e:
        print(f"[Warning] Failed to load dynahate: {e}"); return []
    out = []
    for _, row in df[df['split'] == 'test'].iterrows():
        label = 1 if str(row['label']).lower() == 'hate' else 0
        p_fast, p_slow = build_prompts(row['text'], "N/A", WILD_TAXONOMY)
        out.append({
            "instruction": row['text'], "response": "N/A", "label": label,
            "source": "dynahate", "text_fast": p_fast, "text_slow": p_slow
        })
    return out

def process_xstest():
    """Process XSTest — safe prompts designed to trigger over-refusal (false positives).

    Note: This dataset only has a 'train' split; we use it as a test benchmark.
    """
    try:
        ds = load_dataset("Paul/XSTest", split="train")
    except Exception as e:
        print(f"Error loading XSTest: {e}")
        return []

    out = []

    for row in ds:
        # Directly use the explicit 'label' column provided in the dataset
        label_str = str(row.get('label', '')).strip().lower()
        is_unsafe = 1 if label_str == 'unsafe' else 0

        p_fast, p_slow = build_prompts(row['prompt'], "N/A", AEGIS_TAXONOMY)

        out.append({
            "instruction": row['prompt'],
            "response": "N/A",
            "label": is_unsafe,
            "source": "xstest",
            "text_fast": p_fast,
            "text_slow": p_slow
        })
    return out

def process_beavertails():
    """Process BeaverTails 30k test split — prompt + response with static WildGuard taxonomy."""
    try: ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    except Exception as e: print(f"[Warning] Failed to load beavertails: {e}"); return []
    out = []
    for row in ds:
        p_fast, p_slow = build_prompts(row['prompt'], row.get('response') or "N/A", WILD_TAXONOMY)
        out.append({
            "instruction": row['prompt'], "response": row.get('response') or "N/A",
            "label": 1 if not row['is_safe'] else 0,
            "source": "beavertails", "text_fast": p_fast, "text_slow": p_slow
        })
    return out

def process_beavertails_dynamic():
    """Construct BeaverTails-PolicyShift (legacy key: beavertails_dynamic).

    Uses fixed N1–N6 definitions and category removal, independently of the
    procedural training-schema generator; test prompts have no exemption notes.

    This benchmark tests schema-agnostic generalization by:
      1. Mapping BeaverTails categories to a novel 6-category taxonomy (N1–N6).
      2. With p=0.5, removing the violated category from the taxonomy — the model
         must then label it as "safe" since the violation no longer matches any
         presented category. This tests whether the model evaluates against the
         given schema rather than applying internalized safety knowledge.
      3. With p=0.3 for safe samples, randomly removing one category to ensure the
         model doesn't rely on taxonomy completeness as a signal.
    """
    try: ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    except Exception as e: print(f"[Warning] Failed to load beavertails_dynamic: {e}"); return []
    out = []
    data_list = list(ds)
    random.shuffle(data_list)

    for row in data_list:
        is_originally_safe = row['is_safe']
        final_resp = row.get('response') or "N/A"

        # Map BeaverTails fine-grained categories to novel 6-category codes
        violation_codes = set()
        if not is_originally_safe:
            for cat_key, is_present in row.get('category', {}).items():
                if is_present and cat_key in BEAVER_TO_NOVEL:
                    violation_codes.add(BEAVER_TO_NOVEL[cat_key])

        taxonomy_to_use = NOVEL_TAXONOMY.copy()
        final_label = 0 if is_originally_safe else 1

        # Adaptive schema: remove violated categories with p=0.5 → label flips to safe
        if not is_originally_safe and violation_codes:
            if random.random() < 0.5:
                for code in violation_codes:
                    taxonomy_to_use.pop(code, None)
                final_label = 0
        elif is_originally_safe:
            # Random category removal for safe samples (noise injection)
            if random.random() < 0.3:
                keys = list(taxonomy_to_use.keys())
                if keys: del taxonomy_to_use[random.choice(keys)]

        # Only report violation codes that still exist in the active taxonomy
        active_violation_codes = sorted(c for c in violation_codes if c in taxonomy_to_use)

        p_fast, p_slow = build_prompts(row['prompt'], final_resp, taxonomy_to_use)

        out.append({
            "instruction": row['prompt'], "response": final_resp, "label": final_label,
            "source": "beavertails_dynamic", "text_fast": p_fast, "text_slow": p_slow,
            "violation_codes": active_violation_codes
        })
    return out

def process_dynaguard():
    """Process DynaBench/DynaGuard — multi-turn guideline adherence evaluation.

    Unlike other benchmarks, DynaBench evaluates whether the agent's response
    complies with custom-defined guidelines (not a fixed safety taxonomy). This
    tests the model's ability to follow arbitrary policies.

    The transcript is split into conversation history + final agent response,
    and guidelines are rendered as numbered rules in the taxonomy block.
    """
    try:
        ds = load_dataset("montehoover/DynaBench", split="test")
    except Exception as e:
        print(f"Error loading DynaBench: {e}")
        return []

    out = []

    # DYNABENCH_INST_FAST and DYNABENCH_INST_EXPLAIN are imported from prompts.py

    for row in ds:
        transcript = row['transcript']
        last_agent_idx = transcript.rfind("Agent:")

        if last_agent_idx != -1:
            # Split the transcript into everything BEFORE the last agent response, and the final response itself
            history_and_latest_user = transcript[:last_agent_idx].strip()
            last_resp = transcript[last_agent_idx + len("Agent:"):].strip()

            # Sub-divide the history to clearly isolate the final User prompt
            last_user_idx = history_and_latest_user.rfind("User:")
            if last_user_idx != -1 and last_user_idx > 0:
                history = history_and_latest_user[:last_user_idx].strip()
                latest_user = history_and_latest_user[last_user_idx + len("User:"):].strip()

                # Format clearly for the LLM
                instruction = f"[Conversation History]\n{history}\n\n[Latest User Input]\n{latest_user}"
            elif last_user_idx == 0:
                instruction = history_and_latest_user[len("User:"):].strip()
            else:
                instruction = history_and_latest_user
        else:
            instruction = transcript
            last_resp = "N/A"

        # Handle the custom policy/guidelines (DynaBench often stores these as a list)
        guidelines = row.get('guidelines', row.get('policy', row.get('instruction', '')))
        if isinstance(guidelines, list):
            custom_pol_str = "\n".join([f"RULE-{i+1}: {g}" for i, g in enumerate(guidelines)])
        else:
            custom_pol_str = str(guidelines)

        # Map labels (FAIL = violated guidelines = Unsafe = 1)
        label_val = str(row.get('label', '')).strip().upper()
        final_label = 1 if label_val == 'FAIL' else 0

        # Build the prompts using the custom DynaBench instructions
        fast_text = direct_format(UNIFIED_TEMPLATE, instruction, last_resp, custom_pol_str, "", DYNABENCH_INST_FAST, JSON_FMT_FAST)
        slow_text = direct_format(UNIFIED_TEMPLATE, instruction, last_resp, custom_pol_str, "", DYNABENCH_INST_EXPLAIN, JSON_FMT_EXPLAIN)

        out.append({
            "instruction": instruction,
            "response": last_resp,
            "label": final_label,
            "source": "dynaguard",
            "text_fast": fast_text,
            "text_slow": slow_text
        })
    return out

def generate_datasets(args):
    """Run every requested benchmark processor and write test JSONL + statistics.

    Dispatches the processors keyed by benchmark name, writes
    ``test_ready_<name>.jsonl`` per benchmark into ``args.output_dir``, and
    prints/persists a safe-vs-unsafe label-count table (CSV and JSON).
    """
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = {
        "aegis": process_aegis, "wildguard": process_wildguard, "toxicchat": process_toxicchat,
        "dynahate": process_dynahate, "dynaguard": process_dynaguard,
        "beavertails_dynamic": process_beavertails_dynamic, "xstest": process_xstest, "beavertails": process_beavertails
    }

    test_stats = []
    for name, func in tasks.items():
        if "all" not in args.datasets and name not in args.datasets: continue
        print(f"Processing {name}...")
        data = func()
        if not data: continue
        out_path = os.path.join(args.output_dir, f"test_ready_{name}.jsonl")
        with open(out_path, 'w') as f:
            for item in data: f.write(json.dumps(item) + "\n")
        print(f"  -> Saved {len(data)} ready samples to {out_path}")

        safe   = sum(1 for r in data if r["label"] == 0)
        unsafe = len(data) - safe
        test_stats.append({
            "Dataset": name.replace("_", " ").title(),
            "Total": len(data),
            "Safe (Label 0)": safe,
            "Unsafe (Label 1)": unsafe,
            "Safe%": f"{safe/len(data)*100:.1f}%",
            "Unsafe%": f"{unsafe/len(data)*100:.1f}%",
        })

    if test_stats:
        df = pd.DataFrame(test_stats).sort_values("Dataset").reset_index(drop=True)
        w = 105
        print("\n" + "=" * w)
        print("TEST DATASET STATISTICS".center(w))
        print("=" * w)
        print(df.to_markdown(index=False))

        out_csv  = os.path.join(args.output_dir, "test_statistics.csv")
        out_json = os.path.join(args.output_dir, "test_statistics.json")
        df.to_csv(out_csv, index=False)
        df.to_json(out_json, orient="records", indent=2)
        print(f"\nStatistics saved to {out_csv} and {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_datasets(args)
