# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Aggregate evaluation result JSONs into the paper's pivot tables.

Scans a directory of eval__*.json files produced by evaluate_universal.py (our
variants, per model family, with per-seed "mean ± std") or by
evaluate_baselines.py (--pattern baselines), and renders one table per model
family in Markdown and ASCII plus a CSV summary. Columns interleave each
benchmark's headline metric with its gameability-check companion (and the
Adv-BA / Cal-BA summaries for DynaHate and XSTest).

Usage:
    python eval/generate_pivot_tables.py --dir eval_results --pattern Qwen3-8B
    python eval/generate_pivot_tables.py --dir eval_results --pattern baselines
"""
import os
import json
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from variants import ROW_ORDER, row_label, split_run

# ==========================================
# 1. CONFIGURATION & MAPPINGS
# ==========================================

# Desired row order for proposed models (each row = one model variant)
COLUMN_ORDER = ROW_ORDER

# Row order for baseline models (each row = one baseline variant)
BASELINE_COLUMN_ORDER = [
    "llama-guard-3",
    "llama-guard-4",
    "qwen3guard",
    "nemotron",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b-high",
    "gpt-oss-120b-low",
    "gpt-oss-120b-medium",
    "gpt-oss-120b-high",
    "shieldgemma",
    "wildguard",
    "granite",
    "dynaguard-8b-fast",
    "dynaguard-8b-cot",
]

# Headline column names (↑/↓ = optimization direction)
ROW_MAPPING = {
    "aegis":               "aegis (F1)↑",
    "wildguard":           "wildgrd (F1)↑",
    "toxicchat":           "toxic (F1)↑",
    "dynahate":            "dynahate(Recall)↑",
    "dynaguard":           "dynagrd (GA)↑",
    "beavertails_dynamic": "bt_dyn (GA)↑",
    "xstest":              "xstest (FRR)↓",
    "beavertails":         "bt_ood (Acc)↑",
}

# First companion: transparency metric that exposes the specific gameability hole.
COMPANION_HIGHLIGHT = {
    "aegis":               "balanced_acc",
    "wildguard":           "balanced_acc",
    "toxicchat":           "balanced_acc",
    "dynahate":            "fpr",           # exposes "predict all unsafe" strategy
    "dynaguard":           "balanced_acc",
    "beavertails_dynamic": "macro_f1",
    "xstest":              "unsafe_recall",  # exposes "predict all safe" strategy
    "beavertails":         "balanced_acc",
}

COMPANION_ROW_MAPPING = {
    "aegis":               "aegis (BalAcc)↑",
    "wildguard":           "wildgrd (BalAcc)↑",
    "toxicchat":           "toxic (BalAcc)↑",
    "dynahate":            "dynahate(FPR)↓",
    "dynaguard":           "dynagrd (BalAcc)↑",
    "beavertails_dynamic": "bt_dyn (MacF1)↑",
    "xstest":              "xstest (URR)↑",
    "beavertails":         "bt_ood (BalAcc)↑",
}

# Second companion (derived summary): Adv-BA and Cal-BA — only for DynaHate and XSTest.
# Both equal balanced_acc for their respective benchmarks, so we read balanced_acc
# from the result JSON (always present) rather than the adv_ba/cal_ba alias keys
# (only present in results produced after the metrics.py update).
DERIVED_COMPANION_HIGHLIGHT = {
    "dynahate": "balanced_acc",   # = adv_ba = (Recall + (100-FPR)) / 2
    "xstest":   "balanced_acc",   # = cal_ba = (URR + (100-FRR)) / 2
}

DERIVED_COMPANION_ROW_MAPPING = {
    "dynahate": "dynahate(Adv-BA)↑",
    "xstest":   "xstest(Cal-BA)↑",
}

# Merged column order: headline → first companion → derived summary (where applicable).
MERGED_ROW_ORDER = [
    "aegis (F1)↑",           "aegis (BalAcc)↑",
    "wildgrd (F1)↑",         "wildgrd (BalAcc)↑",
    "toxic (F1)↑",           "toxic (BalAcc)↑",
    "dynahate(Recall)↑",     "dynahate(FPR)↓",      "dynahate(Adv-BA)↑",
    "dynagrd (GA)↑",         "dynagrd (BalAcc)↑",
    "bt_dyn (GA)↑",          "bt_dyn (MacF1)↑",
    "xstest (FRR)↓",         "xstest (URR)↑",       "xstest(Cal-BA)↑",
    "bt_ood (Acc)↑",         "bt_ood (BalAcc)↑",
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def generate_markdown_table(df):
    """Generates a Markdown table strictly preserving the .00 format."""
    def format_cell(item):
        """Render numeric cells with two decimals; pass non-numeric strings through."""
        if isinstance(item, (int, float)):
            return f"{item:.2f}"
        if isinstance(item, str):
            try:
                return f"{float(item):.2f}"
            except ValueError:
                return item  # Safely ignores "Model", "-", or "86.50 ± 0.42"
        return str(item)

    cols = list(df.columns)
    formatted_data = [[format_cell(x) for x in row] for _, row in df.iterrows()]

    md_str = []
    md_str.append("| " + " | ".join(cols) + " |")
    md_str.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in formatted_data:
        md_str.append("| " + " | ".join(row) + " |")

    return "\n".join(md_str)

def generate_ascii_table(df):
    """Generates a strict ASCII grid table matching the standard layout."""
    cols = list(df.columns)

    col_widths = [max(len(str(item)) for item in df[col].tolist() + [col]) for col in cols]
    col_widths[0] = max(col_widths[0], 22)  # Min width for model name column

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    ascii_str = []
    ascii_str.append(separator)

    header = "|" + "|".join(f" {col.center(w)} " for col, w in zip(cols, col_widths)) + "|"
    ascii_str.append(header)
    ascii_str.append(separator)

    for _, row in df.iterrows():
        row_str = "|" + "|".join(f" {str(item).ljust(w) if i==0 else str(item).center(w)} " for i, (item, w) in enumerate(zip(row, col_widths))) + "|"
        ascii_str.append(row_str)

    ascii_str.append(separator)
    return "\n".join(ascii_str)

# ==========================================
# 3. BASELINE PIVOT TABLE
# ==========================================

def generate_baseline_table(args):
    """Generate a single pivot table for all SOTA baseline models.

    Layout: rows = baseline model variants, columns = benchmark datasets.

    Matches files named  eval__{model_key}_{suite}_PRIME.json  (no mode suffix)
    produced by eval/evaluate_baselines.py.  Pass --pattern baselines to trigger
    this path.
    """
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        return

    # Baseline result files have no mode token; match the known model keys exactly.
    baseline_keys_re = "|".join(re.escape(k) for k in BASELINE_COLUMN_ORDER)
    file_pattern = re.compile(
        rf"^eval__(?P<model_key>{baseline_keys_re})_(?P<suite>[A-Za-z0-9]+)_PRIME\.json$"
    )

    # col_name -> model_key -> score  (covers both headline and companion columns)
    merged_data: dict[str, dict[str, float]] = defaultdict(dict)
    files_processed = 0

    print(f"Scanning '{args.dir}' for baseline evaluation results...\n")

    for filename in sorted(os.listdir(args.dir)):
        m = file_pattern.match(filename)
        if not m:
            continue
        model_key = m.group("model_key")
        filepath = os.path.join(args.dir, filename)
        try:
            with open(filepath) as fh:
                results = json.load(fh)
            for res in results:
                ds_key = res.get("dataset", "")
                comp = res.get("companion", {})
                if ds_key in ROW_MAPPING:
                    merged_data[ROW_MAPPING[ds_key]][model_key] = res.get("score", 0.0)
                if ds_key in COMPANION_ROW_MAPPING:
                    highlight_key = COMPANION_HIGHLIGHT.get(ds_key, "balanced_acc")
                    comp_score = comp.get(highlight_key)
                    if comp_score is not None:
                        merged_data[COMPANION_ROW_MAPPING[ds_key]][model_key] = comp_score
                if ds_key in DERIVED_COMPANION_ROW_MAPPING:
                    derived_key = DERIVED_COMPANION_HIGHLIGHT[ds_key]
                    derived_score = comp.get(derived_key)
                    if derived_score is not None:
                        merged_data[DERIVED_COMPANION_ROW_MAPPING[ds_key]][model_key] = derived_score
            files_processed += 1
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    if not merged_data:
        print("No baseline evaluation files found.")
        print("Expected filenames like: eval__llama-guard-3_ALL_PRIME.json")
        return

    print(f"Successfully processed {files_processed} baseline evaluation files.\n")

    rows = []
    for model_key in BASELINE_COLUMN_ORDER:
        row = {"Model": model_key}
        for col in MERGED_ROW_ORDER:
            v = merged_data.get(col, {}).get(model_key)
            row[col] = f"{v:.2f}" if v is not None else "-"
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"{'='*90}")
    print(" SOTA BASELINE MODELS")
    print(f"{'='*90}\n")
    print("### Markdown Table\n")
    print(generate_markdown_table(df))
    print("\n\n### ASCII Pivot Table\n")
    print(generate_ascii_table(df))
    print("\n")

    out_csv = os.path.join(args.dir, "aggregated_baselines_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Baseline table complete. CSV saved to '{out_csv}'.")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main(args):
    """Dispatch to the baseline table, or aggregate our variants per model family."""
    if args.pattern and args.pattern.lower() == "baselines":
        generate_baseline_table(args)
        return

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        return

    # col_name -> family -> variant -> list of scores (headline + companion in one dict)
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Regex to extract model slug, suite, and mode from filename
    file_pattern = re.compile(r"^eval__(?P<model_slug>.*)_(?P<suite>[A-Za-z0-9]+)_(?P<mode>fast|slow)_PRIME\.json$")

    print(f"Scanning '{args.dir}' for evaluation results...")
    if args.pattern:
        print(f"Filtering with pattern: '{args.pattern}'\n")
    else:
        print("\n")

    files_processed = 0

    for filename in os.listdir(args.dir):
        match = file_pattern.match(filename)
        if not match:
            continue

        model_slug = match.group("model_slug")
        mode = match.group("mode")

        if args.pattern and not re.search(args.pattern, model_slug):
            continue

        parsed = split_run(model_slug)
        if not parsed:
            continue

        family, base_variant = parsed
        variant = row_label(base_variant, mode)
        if variant is None:
            continue

        filepath = os.path.join(args.dir, filename)
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)

            for res in results:
                ds_key = res.get("dataset", "")
                comp = res.get("companion", {})
                if ds_key in ROW_MAPPING:
                    all_data[family][ROW_MAPPING[ds_key]][variant].append(res.get("score", 0.0))
                if ds_key in COMPANION_ROW_MAPPING:
                    highlight_key = COMPANION_HIGHLIGHT.get(ds_key, "balanced_acc")
                    comp_score = comp.get(highlight_key)
                    if comp_score is not None:
                        all_data[family][COMPANION_ROW_MAPPING[ds_key]][variant].append(comp_score)
                if ds_key in DERIVED_COMPANION_ROW_MAPPING:
                    derived_key = DERIVED_COMPANION_HIGHLIGHT[ds_key]
                    derived_score = comp.get(derived_key)
                    if derived_score is not None:
                        all_data[family][DERIVED_COMPANION_ROW_MAPPING[ds_key]][variant].append(derived_score)

            files_processed += 1
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    if not all_data:
        print("No valid evaluation data found matching your criteria.")
        return

    print(f"Successfully processed {files_processed} evaluation files.\n")

    def _fmt_scores(scores):
        """Format a seed list as one score, or "mean ± std" when several seeds ran."""
        if not scores:
            return "-"
        if len(scores) == 1:
            return f"{scores[0]:.2f}"
        avg = np.mean(scores)
        std = np.std(scores, ddof=1)
        return f"{avg:.2f} ± {std:.2f}"

    # One merged table per model family: headline immediately followed by its
    # companion for each dataset (interleaved per MERGED_ROW_ORDER).
    for family, fam_data in all_data.items():
        safe_family_name = "".join(c if c.isalnum() else "_" for c in family)

        print(f"{'='*90}")
        print(f" MODEL FAMILY: {family}")
        print(f"{'='*90}\n")

        table_rows = []
        for variant in COLUMN_ORDER:
            row_data = {"Model": variant}
            for col in MERGED_ROW_ORDER:
                row_data[col] = _fmt_scores(fam_data.get(col, {}).get(variant, []))
            table_rows.append(row_data)

        df = pd.DataFrame(table_rows)
        print("### Markdown Table\n")
        print(generate_markdown_table(df))
        print("\n\n### ASCII Pivot Table\n")
        print(generate_ascii_table(df))
        print("\n")

        out_csv = os.path.join(args.dir, f"aggregated_{safe_family_name}_results.csv")
        df.to_csv(out_csv, index=False)

    print(f"✅ Aggregation complete. CSV summaries saved in '{args.dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pivot tables from evaluation results.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the eval__*.json result files.")
    parser.add_argument("--pattern", type=str, default=None, help="Regex pattern to filter model slugs (e.g., 'Qwen3-8B' or 'Llama').")

    args = parser.parse_args()
    main(args)
