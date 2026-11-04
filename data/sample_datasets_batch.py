# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Batch-sample rows from generated datasets for manual or LLM inspection.

Reads JSONL or CSV dataset files (default: datasets/*.jsonl), draws a fixed
number of random rows per file with a seeded generator, and writes a single
human-readable text report with pretty-printed samples. Used to eyeball the
output of generate_train_datasets.py and generate_test_datasets.py before
training or evaluation.

Usage:
    python data/sample_datasets_batch.py                       # datasets/*.jsonl, 5 rows each
    python data/sample_datasets_batch.py datasets/*.jsonl -n 3 -o samples.txt
"""
import argparse
import json
import random
import os
import pandas as pd
from glob import glob

def load_and_sample(filepath, n, seed):
    """
    Reads a file (JSONL or CSV), samples n rows, and returns a formatted string.
    """
    if seed is not None:
        random.seed(seed)

    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    output_block = []
    output_block.append(f"{'='*60}")
    output_block.append(f"DATASET: {filepath}")

    try:
        # --- CASE 1: JSONL Files ---
        if ext == '.jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            total = len(lines)
            output_block.append(f"TYPE: JSONL | TOTAL ROWS: {total}")
            output_block.append(f"{'-'*60}\n")

            if total > 0:
                # Sample
                sample_size = min(n, total)
                sampled_lines = random.sample(lines, sample_size)

                for i, line in enumerate(sampled_lines, 1):
                    try:
                        # Pretty print JSON for LLM readability
                        obj = json.loads(line)
                        pretty_json = json.dumps(obj, indent=2)
                        output_block.append(f"--- SAMPLE {i} ---\n{pretty_json}\n")
                    except json.JSONDecodeError:
                        output_block.append(f"--- SAMPLE {i} (Invalid JSON) ---\n{line}\n")
            else:
                output_block.append("(File is empty)")

        # --- CASE 2: CSV Files ---
        elif ext == '.csv':
            try:
                df = pd.read_csv(filepath)
                total = len(df)
                output_block.append(f"TYPE: CSV | TOTAL ROWS: {total}")
                output_block.append(f"{'-'*60}\n")

                if total > 0:
                    sample_size = min(n, total)
                    # Use pandas sample if possible, else head
                    sampled_df = df.sample(n=sample_size, random_state=seed) if total > sample_size else df

                    # Convert to Markdown table for readability
                    output_block.append(sampled_df.to_markdown(index=False))
                    output_block.append("\n")
                else:
                    output_block.append("(File is empty)")
            except Exception as e:
                output_block.append(f"Error reading CSV: {e}")

        else:
            output_block.append(f"Skipping unsupported file type: {ext}")

    except Exception as e:
        output_block.append(f"ERROR: Could not process file. {str(e)}")

    output_block.append("\n\n")
    return "\n".join(output_block)

def main():
    """Parse arguments, sample every matching dataset, and write the report."""
    parser = argparse.ArgumentParser(description="Batch sample rows from datasets for LLM verification.")

    # Input can be a list of files or glob patterns; defaults to datasets/*.jsonl
    parser.add_argument("files", nargs='*', default=["datasets/*.jsonl"],
                        help="Files or glob patterns to sample (default: 'datasets/*.jsonl')")
    parser.add_argument("--output", "-o", default="datasets/dataset_samples.txt", help="Output text file path")
    parser.add_argument("--num", "-n", type=int, default=5, help="Number of samples per file (default: 5)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Expand glob patterns (e.g. if user passed "datasets/*.jsonl" in quotes or shell didn't expand)
    all_files = []
    for f in args.files:
        expanded = glob(f)
        if expanded:
            all_files.extend(expanded)
        else:
            # Maybe it's a specific file that doesn't exist, just add it to report error later
            all_files.append(f)

    # Sort for consistent output order
    all_files = sorted(list(set(all_files)))

    print(f"Processing {len(all_files)} files...")
    print(f"Sampling {args.num} rows per file...")

    with open(args.output, 'w', encoding='utf-8') as outfile:
        # Write Header
        outfile.write(f"DATASET SAMPLING REPORT\n")
        outfile.write(f"Generated from {len(all_files)} files\n")
        outfile.write(f"Samples per file: {args.num}\n")
        outfile.write("\n")

        for filepath in all_files:
            if os.path.exists(filepath):
                print(f"  -> {filepath}")
                content = load_and_sample(filepath, args.num, args.seed)
                outfile.write(content)
            else:
                print(f"  [Warning] File not found: {filepath}")

    print(f"\nDone! Samples saved to: {args.output}")

if __name__ == "__main__":
    main()
