# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Example-level faithfulness audit of the synthetic System-2 reasoning chains.

Reproduces the numbers reported in the paper's Appx. E.10 ("Example-Level Semantic
Assessment of the Synthetic Chains"). Two automated checks run over the full slow
partition of the generated training sets, no sampling:

  input grounding      Does the chain share any content word with the judged turn
                       once the cited category codes/names and a stoplist of
                       template vocabulary are masked out? A lexical lower bound on
                       input specificity: a chain can be apposite while sharing no
                       content word, so this bounds specificity, not correctness.

  case-description     On safe-labeled rows whose agent turn opens with an
  accuracy             unambiguous refusal, does the chain say the agent declined?
                       A chain asserting the agent "responded helpfully to the
                       user's benign inquiry" states the wrong reason for a correct
                       verdict. High-precision by construction: only canonical
                       refusal openers are counted, so regex false negatives are
                       excluded from the denominator rather than scored as errors.

Also reports chain/label stance consistency and template-shell diversity per
canonical case (Appx. A.2).

Usage:
    python eval/audit_chain_faithfulness.py                       # all 9 slow variants
    python eval/audit_chain_faithfulness.py --files train_Hybrid-PN   # one variant
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# The nine slow-bearing variants of the ablation ladder (Aegis, Wild, and Mix are
# fast-only).
SLOW_VARIANTS = [
    "train_Reason", "train_Reason-P", "train_Reason-PN",
    "train_Adapt", "train_Adapt-P", "train_Adapt-PN",
    "train_Hybrid", "train_Hybrid-P", "train_Hybrid-PN",
]

STOP = set("""a an the and or but if then than that this these those there here of in on at to for
with without from by as is are was were be been being am do does did doing have has had having it its
i you he she they we me him her them us my your his their our not no nor so such can could may might
must shall should will would about into over under again further once all any both each few more most
other some only own same too very don now how what which who whom when where why while because during
before after above below up down out off between against""".split())

# Vocabulary the templates themselves inject; cannot count as input grounding.
TEMPLATE_WORDS = set("""user users request requests intent content category categories safety policy
policies prompt prompts message messages agent agents response responses harmful harm harms unsafe safe
benign malicious violates violate violation violating restricted flagged flag flags seeks seeking
directly explicitly appears apparent risk risks posing poses falls fall aligning aligns align related
relates targeting targets provides provide providing fulfilling fulfil fulfills declining decline
declines declined refusal refuses refused compliance complies complied assistance assists assisting
concerning terminology terms term used use uses using sense manner context contextual nature non
metaphorical technical educational informational legitimate appropriate appropriately interaction
makes make made rather instead specific specifically none applicable prohibited guidelines guideline
standard standards evaluation evaluate assessed assessment""".split())

REFUSAL_OPENER = re.compile(
    r"^\W{0,5}(i'm sorry|i am sorry|sorry, but|i cannot|i can't|i can not|i won't|i will not|"
    r"i'm unable|i am unable|i'm not able|i am not able)\b", re.I)
CHAIN_DECLINED = re.compile(
    r"\b(declin|refus|did not comply|withheld|rejected the request|not provide the requested)", re.I)
CHAIN_HELPED = re.compile(
    r"\b(responded helpfully|helpful and appropriate|provided a helpful|"
    r"benign (?:inquiry|request|conversation)|without any safety concern|"
    r"appropriate response to the user's safe)", re.I)


def content_tokens(text):
    """Content words of a text: lowercase [a-z]{3,} tokens minus stopwords and template vocabulary."""
    return {t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in STOP} - TEMPLATE_WORDS


def parse_row(row):
    """Return the fields needed by both checks, or None for fast records."""
    msgs = row["messages"]
    target = msgs[-1]["content"]
    if "<analysis>" not in target:
        return None
    source = msgs[0]["content"]
    conv = re.search(r"<conversation>(.*?)</conversation>", source, re.S)
    conv = conv.group(1) if conv else source
    user = re.search(r"<user>(.*?)</user>", conv, re.S)
    agent = re.search(r"<agent>(.*?)</agent>", conv, re.S)
    user_txt = user.group(1).strip() if user else ""
    agent_txt = agent.group(1).strip() if agent else ""
    has_agent = bool(agent_txt) and agent_txt.upper() not in {"N/A", "NA", "NONE"}
    taxonomy = re.search(r"<unsafe_categories>(.*?)</unsafe_categories>", source, re.S)
    label = re.search(r'"Safety label"\s*:\s*"(\w+)"', target)
    cats = re.search(r'"Safety Categories"\s*:\s*"([^"]*)"', target)
    codes = [c.strip() for c in (cats.group(1) if cats else "").split(",")
             if c.strip() and c.strip().lower() != "none"]
    if label is None:
        return None
    return dict(
        analysis=re.search(r"<analysis>(.*?)</analysis>", target, re.S).group(1).strip(),
        label=label.group(1),
        codes=codes,
        taxonomy=taxonomy.group(1) if taxonomy else "",
        user_txt=user_txt,
        agent_txt=agent_txt,
        has_agent=has_agent,
    )


def category_names(codes, taxonomy):
    """Recover each cited code's prose name as rendered in this row's taxonomy."""
    names = []
    for code in codes:
        m = re.search(re.escape(code) + r"[^A-Za-z0-9]{0,4}(?:which involves|:|\)|-)?\s*([^;.\n]{3,80})",
                      taxonomy)
        if m:
            names.append(m.group(1).strip())
    return names


def canonical_case(r):
    """Appx. A.2 cases. 1-3 are exact; 4 vs. 5 uses a refusal cue and is approximate."""
    if not r["has_agent"]:
        return "1_prompt_safe" if r["label"] == "safe" else "2_prompt_unsafe"
    if r["label"] == "unsafe":
        return "3_agent_compliance"
    cue = re.search(r"\b(cannot|can't|won't|unable|sorry|refus|declin|as an ai)\b", r["agent_txt"].lower())
    return "4_agent_refusal" if cue else "5_benign_assistance"


def audit(paths):
    """Run both faithfulness checks over every slow record in the given JSONL files.

    Returns (per_case, shells, cat_ground, refusal): per-canonical-case counts of
    chains/input grounding/stance consistency, the template-shell distribution
    per case, category-grounding counts, and the refusal case-description tally.
    """
    per_case = defaultdict(lambda: dict(n=0, grounded=0, consistent=0))
    shells = defaultdict(Counter)
    cat_ground = Counter()
    refusal = Counter()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                r = parse_row(json.loads(line))
                if r is None:
                    continue
                case = canonical_case(r)
                bucket = per_case[case]
                bucket["n"] += 1

                names = category_names(r["codes"], r["taxonomy"])
                low = r["analysis"].lower()
                if r["codes"]:
                    cat_ground["n"] += 1
                    if (any(c.lower() in low for c in r["codes"])
                            or any(n.lower()[:20] in low for n in names if len(n) > 4)):
                        cat_ground["hit"] += 1

                masked = low
                for token in r["codes"] + names:
                    masked = masked.replace(token.lower(), " ")
                judged = r["agent_txt"] if r["has_agent"] else r["user_txt"]
                if content_tokens(masked) & content_tokens(judged):
                    bucket["grounded"] += 1

                unsafe_stance = re.search(r"\b(violat|unsafe|harmful|restricted|prohibited)", low)
                safe_stance = re.search(
                    r"\b(benign|no apparent|safe|non-harmful|does not violate|no violation)", low)
                if (r["label"] == "unsafe" and unsafe_stance) or (r["label"] == "safe" and safe_stance):
                    bucket["consistent"] += 1

                shells[case][re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", masked)).strip()] += 1

                # case-description accuracy on unambiguous refusals
                if r["label"] == "safe" and r["has_agent"] and REFUSAL_OPENER.match(r["agent_txt"][:80]):
                    refusal["n"] += 1
                    declined = CHAIN_DECLINED.search(r["analysis"])
                    helped = CHAIN_HELPED.search(r["analysis"])
                    if declined and not helped:
                        refusal["correct"] += 1
                    elif helped and not declined:
                        refusal["misattributed"] += 1
                    else:
                        refusal["unclassified"] += 1
    return per_case, shells, cat_ground, refusal


def main():
    """Parse args, resolve the variant files, run the audit, and print the summary tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="datasets")
    ap.add_argument("--files", nargs="*", default=SLOW_VARIANTS,
                    help="variant stems, e.g. train_Hybrid-PN (default: all 9 slow variants)")
    args = ap.parse_args()

    paths = [Path(args.dataset_dir) / f"{f}.jsonl" for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(str(p) for p in missing)}\n"
                         "Run `python data/generate_train_datasets.py` first.")

    per_case, shells, cat_ground, refusal = audit(paths)
    total = sum(b["n"] for b in per_case.values())
    grounded = sum(b["grounded"] for b in per_case.values())
    consistent = sum(b["consistent"] for b in per_case.values())

    print(f"variants: {', '.join(args.files)}")
    print(f"slow records audited: {total:,}\n")
    hdr = f"{'canonical case':24s} {'chains':>9s} {'input-ground':>13s} {'shells':>8s} {'top shell':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for case in sorted(per_case):
        b, sh = per_case[case], shells[case]
        top = 100.0 * sh.most_common(1)[0][1] / b["n"]
        print(f"{case:24s} {b['n']:9,d} {100*b['grounded']/b['n']:12.1f}% {len(sh):8,d} {top:9.1f}%")
    print("-" * len(hdr))
    all_shells = Counter()
    for sh in shells.values():
        all_shells.update(sh)
    print(f"{'ALL':24s} {total:9,d} {100*grounded/total:12.1f}% {len(all_shells):8,d}\n")

    print(f"category grounding (rows citing a category, n={cat_ground['n']:,}): "
          f"{100*cat_ground['hit']/cat_ground['n']:.2f}%")
    print(f"chain/label stance consistency: {100*consistent/total:.2f}%")
    n = refusal["n"]
    print(f"\ncase-description accuracy on unambiguous refusals (n={n:,}):")
    for key in ("correct", "misattributed", "unclassified"):
        print(f"  {key:14s} {refusal[key]:7,d}  {100*refusal[key]/n:6.2f}%")
    print(f"  misattributed as a share of the slow partition: "
          f"{100*refusal['misattributed']/total:.2f}%")


if __name__ == "__main__":
    main()
