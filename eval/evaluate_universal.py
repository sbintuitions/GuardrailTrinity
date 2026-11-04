# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Evaluate one trained Guardrail Trinity checkpoint on the test benchmarks.

Runs greedy vLLM inference (local weights, or an already-running
OpenAI-compatible server via --api_url) over the test_ready_*.jsonl files
produced by data/generate_test_datasets.py, parses each output into a binary
safe/unsafe verdict, computes the headline and companion metrics from
metrics.py, and writes the per-example generations (generated__*.jsonl) plus a
results JSON (eval__*_PRIME.json).

--mode fast uses the text_fast prompt (label-only output), --mode slow uses
text_slow (<analysis> chain + JSON verdict). A model's "unsafe" verdict counts
only when at least one cited category code appears in the prompt's
<unsafe_categories> block; citing a code that does not exist downgrades the
verdict to safe.

Usage:
    python eval/evaluate_universal.py --model <ckpt> --suite A --mode fast
"""
import argparse
import asyncio
import json
import os
import re
import gc
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from openai import AsyncOpenAI
    import httpx as _httpx
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from metrics import (
    SUITES, METRIC_CONFIG, COMPANION_HIGHLIGHT, COMPANION_GOAL,
    normalize_label, calculate_metrics, load_metrics_from_jsonl,
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
# normalize_label and calculate_metrics are imported from metrics.py

async def _async_completions(prompts, api_url, model_id, max_tokens, stop, max_concurrent):
    """Fan prompts out to the server as chat completions, concurrency-capped.

    The semaphore caps in-flight requests at max_concurrent (the server's
    max_num_seqs); firing all prompts at once would saturate the TCP backlog
    and fail spuriously on connect. Returns the message contents in prompt order.
    """
    # Semaphore caps in-flight requests to max_concurrent (= server's max_num_seqs).
    # Without it, firing all N prompts at once saturates the OS TCP backlog and
    # causes ConnectTimeout before the server can accept connections.
    #
    # Use AsyncOpenAI as a context manager so its httpx connection pool is fully
    # drained before this coroutine returns. Without it, teardown tasks fire after
    # asyncio.run() closes the loop, producing "Event loop is closed" noise.
    sem = asyncio.Semaphore(max_concurrent)
    async with AsyncOpenAI(
        base_url=f"{api_url.rstrip('/')}/v1",
        api_key="EMPTY",
        # connect: fail fast if server is unreachable.
        # read: generous budget for slow-mode long outputs (up to 120 s per request).
        timeout=_httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    ) as client:
        async def _one(prompt):
            """Send one prompt through the semaphore and return its message content."""
            async with sem:
                r = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.0,
                    stop=stop,
                )
                return r.choices[0].message.content or ""

        return await asyncio.gather(*[_one(p) for p in prompts])

def generate_via_api(prompts, api_url, model_id, max_tokens, stop, max_concurrent=256):
    """Send prompts to a vLLM OpenAI-compatible server, bounded by max_concurrent."""
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package required for --api_url: pip install openai")
    return asyncio.run(_async_completions(prompts, api_url, model_id, max_tokens, stop, max_concurrent))

def parse_model_output(output_text: str, prompt_text: str) -> int:
    """Parse one raw generation into a binary verdict: 1 = unsafe, 0 = safe.

    Strips <think>/<analysis> reasoning blocks, extracts the "Safety label"
    field of the JSON verdict, then applies the application-layer check: an
    "unsafe" verdict only stands if at least one of its cited Safety Categories
    codes exists in the prompt's <unsafe_categories> taxonomy block (a
    number-only citation such as "1" is matched against codes like "RULE-1").
    Otherwise the verdict is downgraded to safe.
    """
    text = output_text.lower().strip()

    # Strip model-native thinking tokens (Qwen3 <think>...</think>, always empty after SFT)
    # and our <analysis>...</analysis> reasoning chain before parsing the JSON verdict.
    # This ensures the JSON extraction regex operates only on the compact verdict block,
    # making parsing robust regardless of what appears inside the reasoning chain.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL).strip()

    base_label = 0
    label_match = re.search(r'[\"\']safety label[\"\']\s*:\s*[\"\'](safe|unsafe)[\"\']', text)

    if label_match:
        base_label = 1 if label_match.group(1) == 'unsafe' else 0
    elif ('"unsafe"' in text or "'unsafe'" in text) and '"safe"' not in text and "'safe'" not in text:
        base_label = 1

    # Application Layer Override (The Prime Logic)
    if base_label == 1:
        match = re.search(r'[\"\']safety categories[\"\']\s*:\s*[\"\'](.*?)[\"\']', text)
        if not match:
            match = re.search(r'[\"\']safety categories[\"\']\s*:\s*\[(.*?)\]', text)
        if match:
            flagged_cats = [c.strip().upper() for c in match.group(1).replace(";", ",").replace('"', '').replace("'", "").split(",") if c.strip()]
            tax_block_match = re.search(r'<unsafe_categories>(.*?)</unsafe_categories>', prompt_text, re.DOTALL | re.IGNORECASE)

            if tax_block_match:
                tax_block = tax_block_match.group(1).upper()
                is_valid_unsafe = False

                for cat in flagged_cats:
                    # 1. Try exact match (e.g. "RULE-1")
                    if re.search(rf'\b{re.escape(cat)}\b', tax_block):
                        is_valid_unsafe = True
                        break

                    # 2. Try number-only match to fix DynaGuard (e.g., model outputs "1" for "RULE-1").
                    # Require the digit(s) to follow a letter/separator so plain numbers in
                    # description text (e.g. "Tier 1 content") don't produce false matches.
                    num_only = re.sub(r'\D', '', cat)
                    if num_only and re.search(rf'[A-Z_\-]{re.escape(num_only)}\b', tax_block):
                        is_valid_unsafe = True
                        break

                if not is_valid_unsafe:
                    return 0

    return base_label


# ==========================================
# 3. MAIN EVALUATION LOGIC
# ==========================================

def _print_results(results: list, model_label: str, out_file: str) -> None:
    """Print headline + companion tables and save results JSON."""
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n" + "="*69)
    print(f"Model: {model_label}")
    print("="*69)

    print_df = df.copy()
    print_df["score"] = print_df["score"].apply(lambda x: f"{x:.2f}%")
    print(print_df[["dataset", "metric", "score", "goal"]].to_markdown(index=False))

    companion_rows = []
    for rec in results:
        ds = rec["dataset"]
        comp = rec.get("companion", {})
        highlight_key = COMPANION_HIGHLIGHT.get(ds, "balanced_acc")
        comp_score = comp.get(highlight_key)
        if comp_score is not None:
            companion_rows.append({
                "dataset":          ds,
                "companion_metric": highlight_key,
                "score":            f"{comp_score:.2f}%",
                "goal":             COMPANION_GOAL.get(highlight_key, "maximize"),
            })
    if companion_rows:
        print("\n--- Companion Metrics (gameability check) ---")
        print(pd.DataFrame(companion_rows).to_markdown(index=False))

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    df.to_json(out_file, orient="records", indent=2)

def run_evaluation(args):
    """Drive the full evaluation for one checkpoint over one suite and mode.

    Loads the test files for the suite, generates outputs (vLLM locally or via
    --api_url), parses verdicts with parse_model_output, computes metrics with
    calculate_metrics, and writes generated__*.jsonl + eval__*_PRIME.json into
    args.output_dir. With --recompute_from_jsonl, metrics are recomputed from
    the stored generations without re-running inference.
    """
    target_datasets = SUITES.get(args.suite.upper(), [args.suite.lower()])

    data_files = []
    for ds in target_datasets:
        path = os.path.join(args.input_dir, f"test_ready_{ds}.jsonl")
        if os.path.exists(path):
            data_files.append((ds, path))
        else:
            print(f"[Warning] Dataset file not found: {path}")

    if not data_files: return

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = os.path.basename(args.model.rstrip('/'))
    gen_file   = os.path.join(args.output_dir, f"generated__{model_slug}_{args.suite}_{args.mode}.jsonl")
    out_file   = os.path.join(args.output_dir, f"eval__{model_slug}_{args.suite}_{args.mode}_PRIME.json")

    # ── Early-return: recompute metrics from existing JSONL (opt-in) ─────────
    if args.recompute_from_jsonl:
        cached = load_metrics_from_jsonl(gen_file, [ds for ds, _ in data_files])
        if cached is not None:
            _print_results(cached, os.path.basename(args.model), out_file)
            return
        print(f"[Warning] --recompute_from_jsonl set but '{gen_file}' not found or incomplete — running full inference.")

    api_mode = bool(args.api_url)
    max_toks = 1024 if args.mode == "slow" else 192
    stop_seqs = ["</agent>", "```"]

    # Tokenizer for chat-template formatting — local mode only; API server handles templating.
    tokenizer = None
    if not api_mode and not args.no_chat_template:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if not tokenizer.chat_template:
                tokenizer = None
        except Exception as e:
            print(f"[Warning] Failed to load tokenizer: {e}")

    llm = None
    if api_mode:
        if not _OPENAI_AVAILABLE:
            print("[Error] --api_url requires the openai package: pip install openai"); return
        print(f">>> API mode: {args.api_url}  model={args.model}")
    else:
        tp_size = args.tensor_parallel_size
        if tp_size <= 0:
            tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f">>> Loading Model: {args.model}  (tp={tp_size}, max_num_seqs={args.max_num_seqs}, gpu_mem={args.gpu_memory_utilization})")
        try:
            llm = LLM(
                model=args.model,
                tensor_parallel_size=tp_size,
                trust_remote_code=True,
                dtype="auto",
                max_model_len=8192,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        except Exception as e:
            print(f"[Error] Failed to load model: {e}"); return

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_toks, stop=stop_seqs)
    results = []
    prompt_col = "text_slow" if args.mode == "slow" else "text_fast"

    for ds_key, path in data_files:
        with open(path, 'r') as f:
            raw_data = [json.loads(line) for line in f]
        if not raw_data: continue

        final_inputs, y_true = [], []
        for row in raw_data:
            prompt_text = row.get(prompt_col, "")
            if not api_mode and tokenizer:
                msgs = [{"role": "user", "content": prompt_text}]
                prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            final_inputs.append(prompt_text)
            y_true.append(normalize_label(row['label']))

        if api_mode:
            raw_texts = generate_via_api(final_inputs, args.api_url, args.model, max_toks, stop_seqs,
                                         max_concurrent=args.max_num_seqs)
        else:
            raw_texts = [o.outputs[0].text for o in llm.generate(final_inputs, sampling_params)]

        y_pred = []
        generated_records = []
        for i, txt in enumerate(raw_texts):
            original_prompt = raw_data[i].get(prompt_col, "")
            pred = parse_model_output(txt, original_prompt)
            y_pred.append(pred)
            generated_records.append({
                "dataset": ds_key,
                "index": i,
                "prompt": original_prompt,
                "raw_output": txt,
                "y_pred": pred,
                "y_true": y_true[i],
            })

        metric_info = calculate_metrics(ds_key, y_true, y_pred)
        results.append({
            "dataset":   ds_key,
            "metric":    metric_info["metric_name"],
            "score":     metric_info["score"],
            "goal":      metric_info["goal"],
            "companion": metric_info.get("companion", {}),
        })

        write_mode = "w" if ds_key == target_datasets[0] else "a"
        with open(gen_file, write_mode) as gf:
            for rec in generated_records:
                gf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if results:
        _print_results(results, os.path.basename(args.model), out_file)

    if llm is not None: del llm
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--suite", type=str, default="ALL")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "slow"])
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--api_url", type=str, default=None,
                        help="vLLM OpenAI-compatible server URL (e.g. http://localhost:8000). "
                             "When set, skips local LLM loading and queries the running server instead.")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=-1,
                        help="vLLM tensor parallel size. -1 = all visible CUDA devices.")
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="Max concurrent sequences in the vLLM scheduler. "
                             "Lower this on memory-constrained nodes.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory reserved for vLLM KV cache (default 0.90).")
    parser.add_argument("--recompute_from_jsonl", action="store_true",
                        help="Skip inference and recompute metrics from an existing "
                             "generated__{slug}_{suite}_{mode}.jsonl in --output_dir. "
                             "Falls back to full inference if the file is missing or incomplete.")
    args = parser.parse_args()
    run_evaluation(args)
