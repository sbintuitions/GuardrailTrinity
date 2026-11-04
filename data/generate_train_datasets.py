# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""
Generate training datasets for GuardrailTrinity safety moderation models (the twelve ablation variants).

This script produces chat-formatted JSONL files for fine-tuning a family of twelve
model variants across three axes of the Guardrail Trinity methodology:

  1. **Adaptive Safety Schemas** — Procedurally generated taxonomies with randomized
     codes, formats, and optional category exemptions teach the model to generalize
     beyond any fixed policy.

  2. **Semantic Decoupling** — System-2 (explain) models write a free-form reasoning
     chain inside <analysis>...</analysis> tags (user intent → agent behavior →
     policy alignment), then emit a compact JSON verdict. This model-agnostic format
     works identically for Qwen3, Llama, and other model families without relying on
     native reasoning tokens such as <think>.

  3. **Hybrid Data Blend** — Hybrid variants mix System-1 (fast, ~60k samples) with
     System-2 (explain, ~20k samples) in a 3:1 ratio, combining broad coverage
     with deep reasoning capability.

Model naming convention:
  - Aegis:  AEGIS taxonomy, fast (System-1), static schema
  - Wild:   WildGuard taxonomy, fast, static schema
  - Mix:    Mixed AEGIS+WildGuard, fast, static schema
  - Reason: Static schema + rationale (System-2)
  - Adapt:  Procedural schema + rationale (System-2)
  - Hybrid: Hybrid blend (fast + explain, procedural)
  - -P:    + category permission (exemption mechanism, p=0.10, slow records only)
  - -PN:   + category permission + hard-negative injection (toxic-bert proxy, threshold=0.70, AEGIS-only)

Data sources:
  - AEGIS-AI-Content-Safety-Dataset-2.0 (nvidia) — prompt+response with categories
  - WildGuardMix (allenai) — prompt+response with harm labels & subcategories

System-2 training uses synthetic reasoning generated directly from ground-truth
labels in AEGIS and WildGuard, avoiding reliance on noisy external reasoning chains.

Usage:
  python generate_train_datasets.py                    # all models
  python generate_train_datasets.py --models Adapt Adapt-P   # specific models
"""

import argparse
import json
import random
import os
import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import torch
from transformers import pipeline

from prompts import (
    UNIFIED_TEMPLATE, INST_FAST, INST_EXPLAIN,
    JSON_FMT_FAST, JSON_FMT_EXPLAIN, direct_format,
)
from taxonomies import (
    AEGIS_MAP, AEGIS_TAXONOMY,
    WILD_MAP, WILD_TAXONOMY,
    SEMANTIC_CLUSTERS, MASTER_DESC,
)

# ==============================================================================
# 1. BASE TAXONOMIES & MAPPINGS
# ==============================================================================
# Two source taxonomies are used as the foundation:
#   - AEGIS (23 categories, S1–S23): from NVIDIA's Aegis-AI-Content-Safety-Dataset-2.0
#   - WildGuard (14 categories, W1–W14): from AllenAI's WildGuardMix
#
# These are used directly for static-schema models (Aegis, Wild, Mix, Reason) and as the semantic
# pool from which procedural taxonomies are sampled (Adapt, Hybrid).
#
# SEMANTIC_CLUSTERS groups categories with overlapping meanings across both
# taxonomies (e.g., "Violence" from AEGIS and "Violence And Physical Harm" from
# WildGuard). During procedural taxonomy generation, distractors are prevented
# from being drawn from the same cluster as the violated category, avoiding
# near-duplicate entries that would make classification trivially easy.
#
# All taxonomy definitions live in taxonomies.py — the single source of truth
# shared with generate_test_datasets.py.


# NOTE: Prompt templates (UNIFIED_TEMPLATE, INST_FAST, INST_EXPLAIN, JSON_FMT_FAST,
# JSON_FMT_EXPLAIN, direct_format) are imported from prompts.py — the single source
# of truth shared with generate_test_datasets.py.

# Sequences whose total message-content length exceeds this threshold are dropped.
# Empirically < 0.02% of any dataset exceeds it; keeping them risks OOM during
# eval (Qwen3's vocab_size=151936 makes the logits tensor enormous for long seqs).
# Conservative proxy: 4096 tokens × 4 chars/token = 16384 chars.
MAX_CONTENT_CHARS = 16_384


def _content_chars(row: dict) -> int:
    """Total character count across all messages in a chat-format row."""
    return sum(len(m.get("content", "")) for m in row.get("messages", []))

# ==============================================================================
# 2. FORMAT-AGNOSTIC TAXONOMY GENERATOR (Adaptive Safety Schemas)
# ==============================================================================
# The core of the Guardrail Trinity's schema-agnostic design. Instead of always
# presenting the same fixed taxonomy, this generator builds a unique taxonomy
# for each training sample by:
#
#   1. Selecting the violated category description(s) from the source dataset.
#   2. Sampling distractor categories from a merged pool (AEGIS ∪ WildGuard),
#      filtered through SEMANTIC_CLUSTERS to avoid near-duplicates.
#   3. Assigning randomized code prefixes (RULE-, CAT_, POL-, C, S, W, OPT_)
#      and shuffling the order, so the model can't memorize positional shortcuts.
#   4. Rendering in one of three random formats: list, JSON, or paragraph — so
#      the model generalizes across presentation styles.
#
# For modes with 'prime', a category exemption mechanism is activated with p=0.10:
#   - One violated category is marked as "explicitly permitted."
#   - The model must learn to INVERT its evaluation for that category.
#   - This teaches context-dependent policy interpretation.
#   - Exemption applies ONLY to slow (explain) records; fast records always use the
#     ground-truth label regardless. Applying exemption to fast records causes dynahate
#     collapse: the model learns "this category can be permitted" and generalises to
#     output "safe" for all hate/harmful content even without a permission note.
#   - Rate set to 0.10: doubled from original p=0.05 for more contrastive pair volume,
#     but below p=0.20 which caused 14-17pp xstest FRR spikes in Hybrid-P variants by
#     over-priming category-pattern sensitivity in the slow reasoning chain.

def generate_procedural_taxonomy(violated_desc_list: List[str], mode: str, base_type: str,
                                 force_exempt: Optional[bool] = None) -> Tuple[str, List[str], str, str]:
    """Generate a taxonomy string with mapped codes and optional exemption.

    Args:
        violated_desc_list: Category descriptions that the sample actually violates.
        mode: Controls behavior — 'static'|'proc' × ''|'_prime' × ''|'_hard_neg'.
        base_type: 'aegis' or 'wild' — determines which static taxonomy to use.
        force_exempt: Override the random exemption decision. True = always exempt,
            False = never exempt, None = use the default p=0.10 random draw.

    Returns:
        (taxonomy_string, mapped_violation_codes, exempted_code, exempted_desc)
    """
    is_static = 'static' in mode
    eligible = ('prime' in mode) and len(violated_desc_list) > 0
    if force_exempt is None:
        should_exempt = eligible and (random.random() < 0.10)
    else:
        should_exempt = eligible and force_exempt

    if is_static:
        tax_dict = AEGIS_TAXONOMY if base_type == 'aegis' else WILD_TAXONOMY
        lines = [f"{k}: {v}" for k, v in sorted(tax_dict.items(), key=lambda x: (x[0][0], int(re.sub(r'\D', '', x[0])) if re.search(r'\d', x[0]) else 0))]
        tax_str = "\n".join(lines)
        mapped_codes = [k for k, v in tax_dict.items() if v in violated_desc_list]
        exempted_code = mapped_codes[0] if (should_exempt and mapped_codes) else None
        return tax_str, mapped_codes, exempted_code, tax_dict.get(exempted_code) if exempted_code else None

    all_descs = list(MASTER_DESC.values())
    target_descs = list(violated_desc_list)

    # Semantic collision filter
    excluded_distractors = set(target_descs)
    target_descs_lower = {t.lower() for t in target_descs}

    for cluster in SEMANTIC_CLUSTERS:
        cluster_lower = {c.lower() for c in cluster}
        if target_descs_lower.intersection(cluster_lower):
            excluded_distractors.update(cluster)

    distractors = [d for d in all_descs if d not in excluded_distractors]

    if random.random() < 0.20: num_distractors = random.randint(15, 30)
    else: num_distractors = random.randint(4, 9)

    selected_distractors = random.sample(distractors, min(num_distractors, len(distractors)))
    final_descs = target_descs + selected_distractors
    random.shuffle(final_descs)

    prefix = random.choice(["RULE-", "CAT_", "POL-", "C", "S", "W", "OPT_"])
    tax_dict = {f"{prefix}{i+1}": desc for i, desc in enumerate(final_descs)}

    mapped_codes = [k for k, v in tax_dict.items() if v in violated_desc_list]
    exempted_code = mapped_codes[0] if (should_exempt and mapped_codes) else None

    fmt_choice = random.choice(["list", "json", "paragraph"])
    if fmt_choice == "list":
        tax_str = "\n".join([f"{k}: {v}" for k, v in tax_dict.items()])
    elif fmt_choice == "json":
        tax_str = json.dumps(tax_dict, indent=2)
    else:
        parts = [f"flag {k}, which involves {v}" for k, v in tax_dict.items()]
        if len(parts) > 1: parts[-1] = "and finally, " + parts[-1]
        tax_str = "The assistant must adhere to the following safety categories. It must " + "; ".join(parts) + "."

    return tax_str, mapped_codes, exempted_code, tax_dict.get(exempted_code) if exempted_code else None

# ==============================================================================
# 3. HELPER FUNCTIONS & HARD-NEGATIVE INJECTION
# ==============================================================================

# --- Hard-negative injection (for -PN variants) ---
# Uses a small proxy classifier (unitary/toxic-bert) to identify benign samples
# that use sensitive vocabulary. These "hard negatives" are concentrated in the
# System-2 partition to force the reasoning chain to explicitly distinguish
# between harmful intent and benign use of sensitive terms.
#
# DEMOGRAPHIC EXCLUSION: prompts that mention identity groups (race, gender, religion,
# sexuality, nationality) are excluded from hard-neg treatment even when toxic-bert fires.
# DynaHate contains adversarially-subtle hate speech — derogatory statements about
# identity groups that are ideologically harmful but not lexically explicit. Training
# these as "looks harmful but is safe" teaches the model to dismiss all subtle identity-
# group-adjacent content, collapsing dynahate ADR by ~10pp.
_DEMOGRAPHIC_PAT = re.compile(
    r'\b(?:black|white|asian|hispanic|latino|latina|arab|jewish|muslim|christian|'
    r'hindu|buddhist|gay|lesbian|bisexual|transgender|queer|lgbt|'
    r'women|woman|men|man|female|male|girl|boy|feminist|'
    r'immigrant|refugee|mexican|chinese|indian|african|native|indigenous|'
    r'race|racial|ethnic|gender|sexuality|religion|nationality)\b',
    re.IGNORECASE,
)

hn_detector = None

# Default proxy is unitary/toxic-bert (the paper configuration). Overridable via
# --hn_proxy for the proxy-swap ablation, e.g.
# s-nlp/roberta_toxicity_classifier. Alternative proxies use different negative-class
# label names ('neutral' vs. toxic-bert's implicit multi-label scheme), so the label
# check consults a set. For toxic-bert this set is behavior-identical to the original
# `!= 'non-toxic'` check: toxic-bert's labels (toxic, severe_toxic, obscene, threat,
# insult, identity_hate) are never in it.
HN_PROXY_MODEL = "unitary/toxic-bert"
_NON_TOXIC_LABELS = {"non-toxic", "non_toxic", "not_toxic", "neutral", "normal"}

def is_hard_negative(text):
    """Classify text as a potential hard negative using a toxicity proxy classifier.

    Returns False immediately for prompts mentioning identity groups (race, gender,
    religion, etc.) — DynaHate adversarial hate speech uses ideologically harmful but
    lexically mild phrasing about these groups, and training such examples as "safe
    hard negatives" collapses hate-speech detection.
    """
    global hn_detector
    if not text or len(str(text).strip()) < 5: return False
    if _DEMOGRAPHIC_PAT.search(str(text)): return False

    if hn_detector is None:
        print(f"\n>>> Loading small proxy model ({HN_PROXY_MODEL}) for Hard Negative detection...")
        device = 0 if torch.cuda.is_available() else -1
        hn_detector = pipeline("text-classification", model=HN_PROXY_MODEL, device=device)

    try:
        result = hn_detector(str(text)[:512], truncation=True, max_length=512)
        if isinstance(result, list) and isinstance(result[0], list): result = result[0]
        elif isinstance(result, list) and isinstance(result[0], dict): pass
        else: result = [result]

        for res in result:
            if res['score'] > 0.70 and res['label'].lower() not in _NON_TOXIC_LABELS: return True
    except Exception: pass
    return False

def parse_aegis_descs(cat_str: str) -> List[str]:
    """Parse AEGIS's `violated_categories` field into category descriptions.

    Accepts the raw string form ("S1: Violence, S9: Other"), strips the code
    prefixes, and maps each known category name through AEGIS_MAP to its code,
    returning the corresponding AEGIS_TAXONOMY descriptions. Unknown or empty
    entries are dropped.
    """
    if not cat_str or str(cat_str) == "nan": return []
    clean = re.sub(r'S\d+:\s*', '', str(cat_str))
    items = [c.strip() for c in clean.split(",")]
    return [AEGIS_TAXONOMY[AEGIS_MAP[item]] for item in items if item in AEGIS_MAP]

# Matches definitional or educational queries like "What is X?", "Define Y", "Explain what Z is".
# These are the primary source of false refusals in slow mode: the model applies harmful
# intent templates to questions that are informational in nature. When detected on safe samples,
# we use an educational user_intent framing so the model learns to distinguish curiosity from harm.
DEFINITIONAL_PAT = re.compile(
    r'^\s*(?:what\s+(?:is|are|does|constitutes)\b|define\b|explain\s+what\b|tell\s+me\s+(?:what|about)\b)',
    re.IGNORECASE,
)

# --- Synthetic reasoning template lists for generate_decoupled_rationale ---

_INTENT_HARMFUL = [
    "The user is explicitly requesting assistance with {cat_desc}, which constitutes a harmful request.",
    "The user seeks to obtain content related to {cat_desc}, posing a direct safety risk.",
    "The user's request is for {cat_desc} — content that falls under restricted safety categories.",
    "The user wants help with {cat_desc}, which represents a clearly harmful use case.",
    "The intent behind this message is to acquire {cat_desc}, a restricted and harmful category of content.",
]
_INTENT_DEFINITIONAL = [
    "The user is seeking a factual or educational definition, not requesting harmful assistance.",
    "The user wants to understand a concept for educational or informational purposes.",
    "This is a definitional query — the user is curious about a topic, not seeking to cause harm.",
    "The user is asking for a factual explanation of a concept, which is an educational rather than harmful request.",
]
_INTENT_HARD_NEG = [
    "The user is asking a safe question that uses potentially sensitive vocabulary in a harmless context.",
    "Despite using terms that may appear concerning, the user's request is benign and non-harmful.",
    "The user's phrasing includes sensitive language, but the underlying intent is clearly safe and informational.",
    "This appears to be a safe request framed with terminology that could trigger false positives — the intent is benign.",
]
_INTENT_BENIGN = [
    "The user is engaging in a benign conversation, asking for information, creative writing, or assistance.",
    "The user's request is benign — they are seeking factual or helpful information.",
    "The user appears to have a legitimate, non-harmful purpose for this request.",
    "This is a safe inquiry; the user's intent is informational rather than harmful.",
    "The user is seeking information or assistance with no apparent harmful intent.",
]
_AGENT_COMPLIANCE = [
    "The agent's response provides content related to {cat_desc}, directly fulfilling the harmful request rather than declining it.",
    "Rather than refusing, the agent supplied {cat_desc}, thereby complying with the harmful request.",
    "The agent chose to fulfill the request by providing {cat_desc}, constituting compliance with a harmful instruction.",
    "The agent's reply contains {cat_desc}, meaning it complied with the harmful request instead of refusing.",
]
_AGENT_REFUSAL = [
    "The agent refused to fulfill the harmful request, declining to provide the requested content.",
    "The agent appropriately declined the user's harmful request without providing the harmful information.",
    "The agent explicitly refused to assist with the harmful request, maintaining safety compliance.",
    "The agent recognized the harmful nature of the request and declined to engage with it.",
    "Rather than complying, the agent refused the request, avoiding any unsafe output.",
]
_AGENT_BENIGN = [
    "The agent provided a helpful and appropriate response to the user's safe request.",
    "The agent responded helpfully to the user's benign inquiry without any safety concerns.",
    "The agent's response addressed the user's question in a safe and appropriate manner.",
    "The agent engaged constructively with the user's request, which posed no safety concern.",
    "The agent provided relevant assistance to a safe and legitimate request.",
]
_MATCH_COMPLIANCE_RESP = [
    "The agent's response directly facilitates harmful content related to {cat_desc} (Category {code}), violating the safety policy. The agent's compliance with this request makes the interaction unsafe.",
    "This exchange violates {code} ({cat_desc}): the agent complied with the harmful request rather than refusing it.",
    "The interaction is unsafe because the agent provided {cat_desc} ({code}), directly fulfilling a harmful request.",
    "Policy {code} is violated: the user requested {cat_desc} and the agent provided it, constituting compliance with a harmful instruction.",
]
_MATCH_UNSAFE_PROMPT = [
    "The user's request directly violates the safety policy, specifically aligning with {cat_desc} (Category {code}).",
    "This prompt falls under {code} ({cat_desc}) — a restricted category that the user is explicitly targeting.",
    "The request maps to {code} ({cat_desc}), making it a direct violation of the applicable safety policy.",
    "Category {code} ({cat_desc}) is violated by the user's request, which seeks restricted content.",
]
_MATCH_REFUSAL = [
    "While the user's prompt raises safety concerns related to {cat_desc} ({code}), the agent's refusal prevents a policy violation. The interaction is evaluated as safe.",
    "Despite the request involving {code} ({cat_desc}), the agent's refusal renders the outcome safe.",
    "The agent's decision to refuse the {cat_desc} request ({code}) keeps the overall interaction safe.",
    "Although the prompt targets {code} ({cat_desc}), the agent declined to comply, resulting in a safe interaction.",
]
_MATCH_REFUSAL_NO_CODE = [
    "While the user's prompt raises safety concerns, the agent's response appropriately refuses to engage with the harmful request. The overall interaction is evaluated as safe.",
    "The agent's refusal prevents a policy violation — the interaction is safe despite the harmful request.",
    "The user's request was harmful, but the agent's refusal keeps the overall interaction safe.",
]
_MATCH_DEFINITIONAL = [
    "The request is educational or definitional in nature — seeking to understand a concept, not to cause harm. It does not align with any restricted safety categories.",
    "This is a definitional inquiry; the mere use of sensitive vocabulary in an educational context does not constitute a policy violation.",
    "The conversation involves a factual question about a sensitive topic, which does not violate any of the provided categories.",
    "The user's intent is to understand a concept — not to cause harm — so no category violation applies.",
]
_MATCH_HARD_NEG = [
    "The terminology is used in a non-malicious, metaphorical, or technical sense. It does not align with any restricted safety categories.",
    "Although sensitive vocabulary is present, the underlying intent is benign and no policy is violated.",
    "The surface-level language may appear concerning, but the context reveals a safe and legitimate use — no categories are violated.",
    "This is a case where sensitive terms are used in a harmless context; the intent is non-harmful and no safety category is triggered.",
]
_MATCH_BENIGN = [
    "The context is entirely harmless and does not match any restricted categories in the provided policy.",
    "No policy violation is present; the conversation is benign and does not trigger any unsafe category.",
    "The interaction falls within acceptable bounds — none of the provided categories are violated.",
    "The user's request is legitimate and does not constitute a safety concern under any of the listed categories.",
    "The content is safe and appropriate — there is no match to any of the provided restricted categories.",
]

# Figurative / idiomatic language patterns — safe prompts whose surface vocabulary
# triggers keyword-based false positives (e.g., "murder of crows", "strangle options").
# Priority over generic hard-neg in generate_decoupled_rationale for safe samples.
_FIGURATIVE_PAT = re.compile(
    r'\bmurder\s+of\s+\w+'
    r'|\bsteal(?:ing)?\s+the\s+show\b'
    r'|\bkill(?:ing)?\s+(?:it\b|time\b|two\s+birds)'
    r'|\bset\b.{0,40}on\s+fire\b'
    r'|\bbomb(?:ed|ing)?\s+(?:the\s+stage|it\b|at\s+comedy|a\s+test|the\s+enemy|an?\s+enemy|the\s+base|enemy\s+base)'
    r'|\bstrangl\w*\s+\w*(?:option|position|hold|strategy)\b'
    r'|\bshoot(?:ing)?\s+(?:a\s+|the\s+)?(?:scene|video|film|movie|photo|picture)\b'
    r'|\becstasy\b(?!\s*(?:tablet|pill|drug|MDMA))'
    r'|\bdead\s+(?:ringer|serious|heat)\b'
    r'|\bcrush(?:ing)?\s+(?:it\b|the\s+competition)\b'
    r'|\blit\b(?!\s*(?:cigarette|fire|match|candle|lamp|torch|fuse))'
    r'|\bgoing\s+(?:nuclear|ballistic|viral)\b'
    r'|\bblown?\s+(?:away|out\s+of\s+proportion)\b'
    # Gaming contexts — violent verbs scoped to a named video game
    r'|\b(?:kill|shoot|murder|assassinate|poison|bomb|execute|eliminate)\b.{0,60}'
    r'(?:in\s+(?:call\s+of\s+duty|battlefield|witcher|gta|grand\s+theft|fortnite|minecraft|'
    r'halo|doom|counter.strike|cs\s*:\s*go|valorant|apex|cod|the\s+game|a\s+game|this\s+game))'
    # "crack open" as beverage idiom, "drown(?:ing) in" as overwhelmed metaphor
    r'|\bcrack(?:ing)?\s+(?:open|a)\b'
    r'|\bdrown(?:ing)?\s+in\b'
    # Impossible-subject questions: non-human subject + discriminatory predicate
    # e.g. "Why should ants earn less than men?", "Why can sea urchins not vote?"
    r'|\bwhy\s+(?:should|can|do|would|must|did)\s+\w+(?:\s+\w+){0,4}\s+'
    r'(?:not\s+)?(?:earn\s+less|be\s+kicked|be\s+banned|be\s+fired|be\s+excluded|'
    r'deserve\s+less|get\s+paid|vote|work|live|breathe|shout|levitat|timetravel|fly)\b',
    re.IGNORECASE,
)
_INTENT_FIGURATIVE = [
    "The user's message employs a common idiom or figurative expression; the literal wording appears sensitive but the phrase is a standard figure of speech with no harmful meaning.",
    "This request uses a metaphorical or domain-specific phrase — the concerning-looking word is part of an established idiom or technical term and carries no harmful intent.",
    "The user is using figurative language; the word that superficially resembles harmful content is employed in its idiomatic or contextual sense, not its literal one.",
    "The phrasing is idiomatic — the apparent sensitivity is a surface-level false positive, and the actual intent behind the request is entirely benign.",
    "The user's question or request is set in a fictional, gaming, or hypothetical context where the action described cannot cause real-world harm — the framing is clearly non-literal.",
    "The question uses an impossible, non-human, or absurd subject as a rhetorical device; the premise is logically impossible and the request carries no harmful intent.",
]
_MATCH_FIGURATIVE = [
    "The phrasing is idiomatic or domain-specific and does not constitute a genuine safety violation under any of the listed categories — the concerning word is a figure of speech.",
    "The apparent sensitivity is a surface-level false positive from figurative language; the underlying intent is harmless and no policy category is violated.",
    "No policy violation is triggered; the language is metaphorical or idiomatic, and context confirms the request is benign.",
    "This is a case of figurative or technical vocabulary that superficially resembles harmful content but is not harmful in its actual usage context.",
    "The fictional or gaming context makes the described action non-harmful — no real-world harm can result from actions taken in a game or hypothetical scenario.",
    "The question's premise is logically impossible or involves a non-human subject, making it an absurd rhetorical exercise rather than a harmful request.",
]

# Implicit / coded hate speech — hostile or dehumanizing language toward groups
# expressed through leading questions, stereotypes, or dismissive framing rather
# than explicit slurs. Added to the template pool for hate/harassment categories
# to teach the model to recognise indirect as well as direct hate speech.
_HATE_CAT_KEYWORDS = frozenset({
    'hate', 'harassment', 'discriminat', 'demean', 'denigrat',
    'bigot', 'bias', 'prejudice', 'hostility', 'slur',
})

def _is_hate_category(cat_desc: str) -> bool:
    """Return True if the category description is hate/harassment-adjacent.

    Selects the implicit-hate template pools (_INTENT_IMPLICIT_HATE /
    _MATCH_IMPLICIT_HATE) for the reasoning chain of these categories.
    """
    cat_lower = cat_desc.lower()
    return any(kw in cat_lower for kw in _HATE_CAT_KEYWORDS)

_INTENT_IMPLICIT_HATE = [
    "The user's message employs coded language, a dehumanizing leading question, or stereotyping framing that targets a group — constituting implicit hate speech related to {cat_desc}.",
    "Despite the absence of explicit slurs, the user's statement expresses hostility or dehumanization toward a group through {cat_desc}-related framing.",
    "The user's phrasing uses indirect or loaded language to convey bias or contempt toward a group, which falls under {cat_desc}.",
    "The message encodes a prejudiced or dismissive stance toward a specific group through implicit framing, representing {cat_desc} even without explicit hate speech markers.",
]
_MATCH_IMPLICIT_HATE = [
    "The statement encodes hostility or dehumanization through coded language or implicit framing, violating {code} ({cat_desc}) even without explicit slurs.",
    "The dehumanizing or stereotyping intent is expressed through indirect phrasing, making this a {code} ({cat_desc}) violation.",
    "Implicit hate speech — through leading questions, dismissive framing, or stereotyping — constitutes a violation of {code} ({cat_desc}) regardless of the indirect phrasing.",
    "The bias or hostility expressed toward a group through coded language triggers {code} ({cat_desc}), even though explicit slurs are absent.",
]


def generate_decoupled_rationale(base_status, mapped_cat_code, cat_desc, exempted_code,
                                  is_hn_flag, has_response, is_prompt_harmful, user_text=""):
    """Build the <analysis> reasoning chain for slow-mode training targets.

    Generates a free-form prose chain — user intent, optional agent behavior, policy
    alignment — that becomes the content of the <analysis>...</analysis> block in the
    slow-mode assistant output. All reasoning is assembled as plain text sentences;
    the caller wraps it in <analysis> tags and appends a compact JSON verdict.

    This format is model-agnostic: the same output works for Qwen3, Llama, and other
    model families without relying on any model-native reasoning tokens.

    Args:
        base_status:        "safe"|"unsafe" from source dataset label.
        mapped_cat_code:    The procedural code assigned to the primary violation.
        cat_desc:           Human-readable description of the violated category.
        exempted_code:      Code of the exempted category, or None.
        is_hn_flag:         True if this is a hard-negative (benign but sensitive vocab).
        has_response:       True if an agent response exists (not "N/A").
        is_prompt_harmful:  True if the user's prompt is harmful (independent of response).
        user_text:          Raw user prompt text, used for definitional/figurative detection.

    Returns:
        (analysis_str, final_label) where analysis_str is the prose content for
        <analysis> and final_label is "safe" or "unsafe".
    """
    is_definitional = bool(DEFINITIONAL_PAT.search(user_text)) if user_text else False
    is_figurative = bool(_FIGURATIVE_PAT.search(user_text)) if user_text else False

    # --- user_intent ---
    if base_status == "unsafe" or is_prompt_harmful:
        if _is_hate_category(cat_desc):
            intent = random.choice(_INTENT_HARMFUL + _INTENT_IMPLICIT_HATE).format(cat_desc=cat_desc)
        else:
            intent = random.choice(_INTENT_HARMFUL).format(cat_desc=cat_desc)
    elif is_definitional:
        intent = random.choice(_INTENT_DEFINITIONAL)
    elif is_figurative:
        intent = random.choice(_INTENT_FIGURATIVE)
    elif is_hn_flag:
        intent = random.choice(_INTENT_HARD_NEG)
    else:
        intent = random.choice(_INTENT_BENIGN)

    # --- agent_intent (only when response exists) ---
    agent_intent = None
    if has_response:
        is_refusal = is_prompt_harmful and base_status == "safe"
        is_compliance = is_prompt_harmful and base_status == "unsafe"
        if is_refusal:
            agent_intent = random.choice(_AGENT_REFUSAL)
        elif is_compliance:
            agent_intent = random.choice(_AGENT_COMPLIANCE).format(cat_desc=cat_desc)
        else:
            agent_intent = random.choice(_AGENT_BENIGN)

    # --- policy_match & final_label ---
    code = mapped_cat_code or ""
    if base_status == "unsafe":
        if exempted_code and mapped_cat_code == exempted_code:
            match = f"Although the intent involves sensitive content related to {cat_desc}, Category {exempted_code} is explicitly permitted for this interaction based on the system instructions. Therefore, it is evaluated as safe."
            final_label = "safe"
            if has_response and agent_intent and is_prompt_harmful:
                agent_intent = f"The agent's response involves content related to {cat_desc}, but this category is explicitly permitted."
        else:
            if has_response:
                if _is_hate_category(cat_desc):
                    match = random.choice(_MATCH_COMPLIANCE_RESP + _MATCH_IMPLICIT_HATE).format(cat_desc=cat_desc, code=code)
                else:
                    match = random.choice(_MATCH_COMPLIANCE_RESP).format(cat_desc=cat_desc, code=code)
            else:
                if _is_hate_category(cat_desc):
                    match = random.choice(_MATCH_UNSAFE_PROMPT + _MATCH_IMPLICIT_HATE).format(cat_desc=cat_desc, code=code)
                else:
                    match = random.choice(_MATCH_UNSAFE_PROMPT).format(cat_desc=cat_desc, code=code)
            final_label = "unsafe"
    else:
        final_label = "safe"
        if has_response and is_prompt_harmful:
            if code:
                match = random.choice(_MATCH_REFUSAL).format(cat_desc=cat_desc, code=code)
            else:
                match = random.choice(_MATCH_REFUSAL_NO_CODE)
        elif is_definitional:
            match = random.choice(_MATCH_DEFINITIONAL)
        elif is_figurative:
            match = random.choice(_MATCH_FIGURATIVE)
        elif is_hn_flag:
            match = random.choice(_MATCH_HARD_NEG)
        else:
            match = random.choice(_MATCH_BENIGN)

    lines = [intent]
    if agent_intent is not None:
        lines.append(agent_intent)
    lines.append(match)
    analysis_str = "\n".join(lines)
    return analysis_str, final_label

# ==============================================================================
# 4. ROW PROCESSORS
# ==============================================================================
# Three processors handle the output modes:
#
# process_fast_row:  System-1 (Aegis, Wild, Mix, Hybrid fast partition)
#   Input:  Raw source row from AEGIS or WildGuard
#   Output: {"Safety label": ..., "Safety Categories": ...}
#   Flow:   Extract labels → generate taxonomy → apply exemption → format prompt
#
# process_slow_row:  System-2 (Reason, Adapt, Hybrid slow partition, raw data, all cases)
#   Input:  Raw source row from AEGIS or WildGuard
#   Output: <analysis>[reasoning chain]</analysis>
#           {"Safety label": ..., "Safety Categories": ...}
#   Flow:   Extract labels → generate taxonomy → build synthetic <analysis> chain
#
# process_paired_row: System-1+2 paired (Hybrid slow pool rows only)
#   Input:  Raw source row from AEGIS or WildGuard
#   Output: [fast_record, slow_record] sharing one taxonomy — explicit mode-switching signal
#   Flow:   Extract labels → generate taxonomy ONCE → build both fast and slow outputs

def process_fast_row(row, source_type, mode):
    """Process a source dataset row into a System-1 (fast) training sample.

    Handles the safety moderation two-case logic:
      - If response exists and has a label → use response_label (refusal = safe)
      - If response is missing or unlabeled → use prompt_label, set response to "N/A"
    """
    instruction = row.get('prompt') or ""
    response = row.get('response') or "N/A"

    # --- Extract labels and violation categories from source dataset ---
    if source_type == 'aegis':
        raw_user_safe = "unsafe" if str(row.get('prompt_label', '')).lower() == 'unsafe' else "safe"
        r_label_raw = row.get('response_label')
        violated_descs = parse_aegis_descs(row.get('violated_categories', ''))
    else:
        raw_user_safe = "unsafe" if row.get('prompt_harm_label') == 'harmful' else "safe"
        r_label_raw = row.get('response_harm_label')
        subcat = row.get('subcategory')
        violated_descs = [WILD_TAXONOMY[WILD_MAP[subcat]]] if (subcat and subcat in WILD_MAP) else [WILD_TAXONOMY['W14']]

    # --- Apply safety moderation two-case logic ---
    is_resp_unsafe = str(r_label_raw).lower() in ['unsafe', 'harmful']
    if r_label_raw and str(r_label_raw).lower() != "nan" and response != "N/A":
        base_status = "unsafe" if is_resp_unsafe else "safe"
    else:
        base_status = raw_user_safe
        response = "N/A"

    # Clear categories for safe samples; default for unsafe without categories
    if base_status == "safe": violated_descs = []
    elif not violated_descs: violated_descs = ["Other"] if source_type == 'aegis' else ["Others"]

    def _build_fast_record(force_exempt: Optional[bool]) -> dict:
        """Assemble one fast-mode chat record for the current row.

        Callers always pass force_exempt=False: fast records keep the
        ground-truth label and never carry permission notes (see the comment
        below — exemption is a slow-mode-only capability).
        """
        tax_str, mapped_codes, exempted_code, _ = generate_procedural_taxonomy(
            violated_descs, mode, source_type, force_exempt=force_exempt)
        codes = mapped_codes[:]
        if base_status == "unsafe" and exempted_code:
            codes = [c for c in codes if c != exempted_code]
            label = "safe" if not codes else "unsafe"
        else:
            label = base_status
        note = f"\n\nNote: Category {exempted_code} is explicitly permitted for this interaction." if exempted_code else ""
        prompt = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_str, note, INST_FAST, JSON_FMT_FAST)
        out = {"Safety label": label}
        if label == "unsafe" and codes: out["Safety Categories"] = ", ".join(codes)
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(out, indent=2)}]}

    # --- Generate taxonomy, build output ---
    # Fast records NEVER receive exemption notes or modified labels. Policy flexibility
    # is a slow-mode capability: the reasoning chain in slow mode provides the
    # disambiguating context that teaches the model WHY the label flips under a
    # permission note. Without that context, fast records with exempted labels
    # shift the model's calibration for harmful categories (e.g., hate speech),
    # causing dynahate collapse at inference time even when no permission note is present.
    return _build_fast_record(force_exempt=False)


def process_slow_row(row, source_type, mode):
    """Generate a System-2 (explain) training sample from a raw AEGIS or WildGuard row.

    Replaces process_explain_row + process_compliance_slow_row. Instead of relying on
    GuardReasoner reasoning chains (which are refusal-dominated ~186:1 and require
    noisy cross-reference lookups), this function generates synthetic reasoning
    directly from the ground-truth labels in the source dataset.

    Handles five cases:
      1. Prompt-only, prompt safe:   benign user_intent, no agent_intent, label=safe
      2. Prompt-only, prompt unsafe: harmful user_intent, no agent_intent, label=unsafe
      3. Has response, compliance:   harmful user_intent, compliance agent_intent, label=unsafe
      4. Has response, refusal:      harmful user_intent, refusal agent_intent, label=safe
      5. Has response, benign:       benign user_intent, benign agent_intent, label=safe
    """
    instruction = row.get('prompt') or ""
    response = str(row.get('response') or "N/A").strip()
    if response.lower() in ['none', 'nan', '']:
        response = "N/A"
    has_response = response != "N/A"

    # Extract ground-truth labels and categories
    if source_type == 'aegis':
        prompt_label = str(row.get('prompt_label', 'safe')).lower()
        r_label_raw = row.get('response_label')
        response_label = str(r_label_raw).lower() if (has_response and r_label_raw and str(r_label_raw).lower() not in ['nan', 'none']) else None
        violated_descs = parse_aegis_descs(row.get('violated_categories', ''))
    else:
        prompt_label = str(row.get('prompt_harm_label', 'unharmful')).lower()
        r_label_raw = row.get('response_harm_label')
        response_label = str(r_label_raw).lower() if (has_response and r_label_raw and str(r_label_raw).lower() not in ['nan', 'none']) else None
        subcat = row.get('subcategory', '')
        violated_descs = [WILD_TAXONOMY[WILD_MAP[subcat]]] if (subcat and subcat in WILD_MAP and subcat != 'benign') else []

    is_prompt_harmful = prompt_label in ('unsafe', 'harmful')

    # Determine overall safety label using two-case moderation logic
    if has_response and response_label is not None:
        base_status = 'unsafe' if response_label in ('unsafe', 'harmful') else 'safe'
    else:
        base_status = 'unsafe' if is_prompt_harmful else 'safe'
        response = "N/A"
        has_response = False

    # cat_desc_str: human-readable description for use in reasoning templates.
    # For refusal cases (prompt harmful, response safe), preserve violated_descs so
    # the taxonomy includes the violated code — this lets policy_match reference it.
    # For benign safe cases (prompt safe), violated_descs is empty.
    cat_desc_str = violated_descs[0] if violated_descs else ("Other" if source_type == 'aegis' else "Others")
    if base_status == "unsafe" and not violated_descs:
        violated_descs = ["Other"] if source_type == 'aegis' else ["Others"]
        cat_desc_str = violated_descs[0]

    # Hard-negative detection: benign safe samples with elevated toxicity proxy score.
    # Restricted to slow partition AND AEGIS-sourced rows only. WildGuard's 'unharmful'
    # samples include implicit hate framing (e.g. leading prejudice questions) that toxic-bert
    # correctly flags but that should NOT be trained as "benign sensitive vocabulary" —
    # doing so collapses the reasoning chain to boilerplate "benign intent" for all content.
    is_hn = False
    if 'hard_neg' in mode and base_status == 'safe' and not is_prompt_harmful and source_type == 'aegis':
        is_hn = is_hard_negative(instruction)

    def _build_slow_record(force_exempt: Optional[bool] = None) -> dict:
        """Assemble one slow-mode chat record: taxonomy + <analysis> chain + JSON verdict."""
        # For refusal cases, pass violated_descs to taxonomy so the violated code
        # appears in the taxonomy and can be referenced in the analysis. For benign
        # safe cases (not is_prompt_harmful), pass empty list — no violation codes.
        tax_violated = violated_descs if (base_status == "unsafe" or is_prompt_harmful) else []
        tax_str, mapped_codes, exempted_code, _ = generate_procedural_taxonomy(
            tax_violated, mode, source_type, force_exempt=force_exempt)
        mapped_cat = mapped_codes[0] if mapped_codes else None
        analysis_str, final_label = generate_decoupled_rationale(
            base_status, mapped_cat, cat_desc_str, exempted_code,
            is_hn, has_response, is_prompt_harmful, user_text=instruction)
        note = f"\n\nNote: Category {exempted_code} is explicitly permitted for this interaction." if exempted_code else ""
        prompt = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_str, note, INST_EXPLAIN, JSON_FMT_EXPLAIN)
        verdict = {"Safety label": final_label}
        if final_label == "unsafe" and mapped_cat:
            verdict["Safety Categories"] = mapped_cat
        assistant_content = f"<analysis>\n{analysis_str}\n</analysis>\n\n{json.dumps(verdict, indent=2)}"
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_content}]}

    # Contrastive pair logic (slow partition only): when the p=0.10 exemption fires,
    # emit both the exempted record (safe) and a non-exempted counterpart (unsafe).
    primary = _build_slow_record(force_exempt=None)
    exempt_fired = 'is explicitly permitted' in primary['messages'][0]['content']
    if exempt_fired:
        return [primary, _build_slow_record(force_exempt=False)]
    return primary


def process_paired_row(row, source_type, mode):
    """Generate a paired fast + slow record from a single raw row, sharing one taxonomy.

    Used in Hybrid slow-pool rows so the model explicitly sees both output formats for
    identical input content: INST_FAST → label-only, INST_EXPLAIN → full chain.
    The shared taxonomy ensures the fast and slow records are genuinely paired
    (same category codes, same exemption note) rather than independently randomised.

    Returns a list of 2 records normally, or 3 when the p=0.10 exemption fires
    (fast-exempted, slow-exempted, slow-non-exempted counterpart).
    Fast contrastive counterparts are intentionally omitted — adding extra unsafe
    fast records raises FRR without a compensating reasoning signal.
    """
    instruction = row.get('prompt') or ""
    response = str(row.get('response') or "N/A").strip()
    if response.lower() in ['none', 'nan', '']:
        response = "N/A"
    has_response = response != "N/A"

    if source_type == 'aegis':
        prompt_label = str(row.get('prompt_label', 'safe')).lower()
        r_label_raw = row.get('response_label')
        response_label = str(r_label_raw).lower() if (has_response and r_label_raw and str(r_label_raw).lower() not in ['nan', 'none']) else None
        violated_descs = parse_aegis_descs(row.get('violated_categories', ''))
    else:
        prompt_label = str(row.get('prompt_harm_label', 'unharmful')).lower()
        r_label_raw = row.get('response_harm_label')
        response_label = str(r_label_raw).lower() if (has_response and r_label_raw and str(r_label_raw).lower() not in ['nan', 'none']) else None
        subcat = row.get('subcategory', '')
        violated_descs = [WILD_TAXONOMY[WILD_MAP[subcat]]] if (subcat and subcat in WILD_MAP and subcat != 'benign') else []

    is_prompt_harmful = prompt_label in ('unsafe', 'harmful')

    if has_response and response_label is not None:
        base_status = 'unsafe' if response_label in ('unsafe', 'harmful') else 'safe'
    else:
        base_status = 'unsafe' if is_prompt_harmful else 'safe'
        response = "N/A"
        has_response = False

    cat_desc_str = violated_descs[0] if violated_descs else ("Other" if source_type == 'aegis' else "Others")
    if base_status == "unsafe" and not violated_descs:
        violated_descs = ["Other"] if source_type == 'aegis' else ["Others"]
        cat_desc_str = violated_descs[0]

    is_hn = False
    if 'hard_neg' in mode and base_status == 'safe' and not is_prompt_harmful and source_type == 'aegis':
        is_hn = is_hard_negative(instruction)

    tax_violated = violated_descs if (base_status == "unsafe" or is_prompt_harmful) else []

    def _build_pair(force_exempt: Optional[bool] = None):
        """Build the fast+slow record pair sharing one procedural taxonomy.

        Returns (fast_rec, slow_rec, exempted_code). The fast record always uses
        the ground-truth label without a permission note; the slow record carries
        the note when the exemption fires, forming the contrastive mode pair.
        """
        tax_str, mapped_codes, exempted_code, _ = generate_procedural_taxonomy(
            tax_violated, mode, source_type, force_exempt=force_exempt)
        mapped_cat = mapped_codes[0] if mapped_codes else None

        # Fast record — ALWAYS uses ground-truth label, no exemption note.
        # Applying exemption to fast records shifts fast-mode calibration for harmful
        # categories (e.g., hate speech): the model learns "this category can be
        # permitted" and generalises to output "safe" at inference time when no
        # permission note is present. Fast mode is speed-priority and not the
        # policy-flexibility mode — exemption training belongs in slow mode only.
        fast_codes = mapped_codes[:]
        fast_prompt = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_str, "", INST_FAST, JSON_FMT_FAST)
        fast_out = {"Safety label": base_status}
        if base_status == "unsafe" and fast_codes:
            fast_out["Safety Categories"] = ", ".join(fast_codes)
        fast_rec = {"messages": [{"role": "user", "content": fast_prompt},
                                  {"role": "assistant", "content": json.dumps(fast_out, indent=2)}]}

        # Slow record — <analysis> chain + compact verdict, same taxonomy, with exemption note
        # if applicable. The resulting fast=unsafe / slow=safe pairing (when exemption fires)
        # is an explicit contrastive signal: model learns the permission note — not the content
        # — is what flips the label.
        note = f"\n\nNote: Category {exempted_code} is explicitly permitted for this interaction." if exempted_code else ""
        analysis_str, slow_label = generate_decoupled_rationale(
            base_status, mapped_cat, cat_desc_str, exempted_code,
            is_hn, has_response, is_prompt_harmful, user_text=instruction)
        slow_prompt = direct_format(UNIFIED_TEMPLATE, instruction, response, tax_str, note, INST_EXPLAIN, JSON_FMT_EXPLAIN)
        slow_verdict = {"Safety label": slow_label}
        if slow_label == "unsafe" and mapped_cat:
            slow_verdict["Safety Categories"] = mapped_cat
        slow_content = f"<analysis>\n{analysis_str}\n</analysis>\n\n{json.dumps(slow_verdict, indent=2)}"
        slow_rec = {"messages": [{"role": "user", "content": slow_prompt},
                                  {"role": "assistant", "content": slow_content}]}

        return fast_rec, slow_rec, exempted_code

    fast_primary, slow_primary, exempted_code = _build_pair(force_exempt=None)
    records = [fast_primary, slow_primary]

    # Contrastive pair: only emit the slow unsafe counterpart (not a fast counterpart)
    # to preserve the "no fast contrastive pairs" invariant.
    if exempted_code:
        _, slow_counterpart, _ = _build_pair(force_exempt=False)
        records.append(slow_counterpart)

    return records


# ==============================================================================
# 5. MAIN GENERATION LOGIC
# ==============================================================================

def generate_datasets(args):
    """Orchestrate the full training dataset generation pipeline.

    Pipeline stages:
      1. Load AEGIS and WildGuard source datasets from HuggingFace.
      2. Split each source into train/val pools, define model recipes.
      3. For each target model, apply the appropriate row processor to its
         data pool, then write the results as chat-formatted JSONL.
    """
    global HN_PROXY_MODEL
    suffix = getattr(args, "variant_suffix", "") or ""
    hn_proxy = getattr(args, "hn_proxy", HN_PROXY_MODEL)
    if hn_proxy != HN_PROXY_MODEL:
        # Overwrite guard: a non-default proxy MUST write to suffixed files
        # (e.g. train_Hybrid-PNX.jsonl); otherwise it would silently replace the
        # paper datasets and break reproducibility.
        if not suffix:
            raise SystemExit(
                "--hn_proxy is set to a non-default model but --variant_suffix is empty. "
                "Refusing to overwrite the paper datasets; pass e.g. --variant_suffix X."
            )
        HN_PROXY_MODEL = hn_proxy
        print(f">>> Hard-negative proxy override: {HN_PROXY_MODEL} (variant suffix: '{suffix}')")

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    stats = []

    target_models = args.models if "all" not in args.models else [
        "Aegis", "Wild", "Mix",
        "Reason", "Reason-P", "Reason-PN",
        "Adapt", "Adapt-P", "Adapt-PN",
        "Hybrid", "Hybrid-P", "Hybrid-PN"
    ]
    print(f"--- Generating Datasets for {target_models} ---")

    # --- Stage 1: Load source datasets ---
    try: ds_aegis_train = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    except Exception as e: print(f"WARNING: Failed to load AEGIS train: {e}"); ds_aegis_train=[]
    try: ds_aegis_val = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="validation")
    except Exception as e: print(f"WARNING: Failed to load AEGIS val: {e}"); ds_aegis_val=[]
    try: ds_wild_full = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    except Exception as e: print(f"WARNING: Failed to load WildGuard: {e}"); ds_wild_full=[]

    # --- Stage 2: Split into train/val pools ---
    VAL_SIZE = 1000
    def get_shuffled(ds):
        """Shuffle a source dataset with the run seed; fail fast on empty input."""
        if not ds: raise RuntimeError("Cannot proceed: required dataset is empty or failed to load.")
        return ds.shuffle(seed=args.seed)

    pool_aegis_std_train = get_shuffled(ds_aegis_train)
    pool_aegis_std_val = ds_aegis_val if len(ds_aegis_val) > 0 else []
    if not pool_aegis_std_val and len(pool_aegis_std_train) > VAL_SIZE:
         pool_aegis_std_val = pool_aegis_std_train.select(range(len(pool_aegis_std_train)-VAL_SIZE, len(pool_aegis_std_train)))
         pool_aegis_std_train = pool_aegis_std_train.select(range(len(pool_aegis_std_train)-VAL_SIZE))

    ds_wild_shuffled = get_shuffled(ds_wild_full)
    if len(ds_wild_shuffled) <= VAL_SIZE:
        raise RuntimeError(f"WildGuard dataset too small ({len(ds_wild_shuffled)}) to split with VAL_SIZE={VAL_SIZE}")
    pool_wild_std_train = ds_wild_shuffled.select(range(min(30000, len(ds_wild_shuffled)-VAL_SIZE)))
    pool_wild_std_val = ds_wild_shuffled.select(range(len(pool_wild_std_train), len(pool_wild_std_train)+VAL_SIZE))

    # Paired pool: first SLOW_CAP_PER_SOURCE rows from each source.
    # Each row generates both a fast AND slow record sharing one taxonomy —
    # explicitly teaching the model to switch output modes for identical content.
    # Fast-only pool: remaining rows — fast records only, no slow counterpart.
    # The two pools are disjoint to avoid duplicate fast records.
    SLOW_CAP_PER_SOURCE = 10_000
    aegis_paired_train = pool_aegis_std_train.select(range(min(SLOW_CAP_PER_SOURCE, len(pool_aegis_std_train))))
    aegis_fast_only_train = pool_aegis_std_train.select(range(min(SLOW_CAP_PER_SOURCE, len(pool_aegis_std_train)), len(pool_aegis_std_train)))
    wild_paired_train = pool_wild_std_train.select(range(min(SLOW_CAP_PER_SOURCE, len(pool_wild_std_train))))
    wild_fast_only_train = pool_wild_std_train.select(range(min(SLOW_CAP_PER_SOURCE, len(pool_wild_std_train)), len(pool_wild_std_train)))

    aegis_paired_val = pool_aegis_std_val.select(range(min(500, len(pool_aegis_std_val))))
    aegis_fast_only_val = pool_aegis_std_val.select(range(min(500, len(pool_aegis_std_val)), len(pool_aegis_std_val))) if len(pool_aegis_std_val) > 500 else pool_aegis_std_val.select([])
    wild_paired_val = pool_wild_std_val.select(range(min(500, len(pool_wild_std_val))))
    wild_fast_only_val = pool_wild_std_val.select(range(min(500, len(pool_wild_std_val)), len(pool_wild_std_val))) if len(pool_wild_std_val) > 500 else pool_wild_std_val.select([])

    # --- Stage 3: Define model recipes ---
    # Each recipe specifies: train pool, val pool, and processor function.
    #
    # Model architecture:
    #   Aegis, Wild, Mix: System-1 (fast) — process_fast_row on raw source data
    #   Reason:      System-2 (slow) + static taxonomy, raw AEGIS/WildGuard rows
    #   Adapt:       System-2 (slow) + procedural taxonomy, raw AEGIS/WildGuard rows
    #   Hybrid:      Hybrid 3:1 blend (fast full pools + slow capped pools, procedural)
    #   -P:      + category permission (exemption mechanism)
    #   -PN:     + category permission + hard-negative injection (slow partition only)
    recipes = {
        # --- System-1 (fast) models: label-only output ---
        "Aegis": { "train": pool_aegis_std_train, "val": pool_aegis_std_val, "proc": lambda r: process_fast_row(r, 'aegis', 'static') },
        "Wild": { "train": pool_wild_std_train, "val": pool_wild_std_val, "proc": lambda r: process_fast_row(r, 'wild', 'static') },
        "Mix": { "train": list(pool_aegis_std_train) + list(pool_wild_std_train), "val": list(pool_aegis_std_val) + list(pool_wild_std_val),
                "proc": lambda r: process_fast_row(r, 'aegis', 'static') if 'prompt_label' in r else process_fast_row(r, 'wild', 'static') },

        # --- System-2 (slow) models: semantic decoupling chain, synthetic reasoning ---
        # Reason: Static taxonomy
        "Reason": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'static') },
        "Reason-P": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'static_prime') },
        "Reason-PN": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'static_prime_hard_neg') },

        # Adapt: Procedural taxonomy
        "Adapt": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'proc') },
        "Adapt-P": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'proc_prime') },
        "Adapt-PN": {
            "train": [(r,'aegis') for r in pool_aegis_std_train] + [(r,'wild') for r in pool_wild_std_train],
            "val":   [(r,'aegis') for r in pool_aegis_std_val]   + [(r,'wild') for r in pool_wild_std_val],
            "proc": lambda item: process_slow_row(item[0], item[1], 'proc_prime_hard_neg') },

        # Hybrid: Hybrid blend — fast-only pool (remaining rows) + paired pool (first 10k rows).
        # Paired rows emit both fast and slow records sharing one taxonomy, giving the model
        # an explicit mode-switching signal. Fast-only and paired pools are disjoint to avoid
        # duplicate fast records. Hard-neg in slow partition only (paired rows).
        "Hybrid": {
            "train": [(r, 'aegis_fast_only') for r in aegis_fast_only_train] + [(r, 'wild_fast_only') for r in wild_fast_only_train] +
                     [(r, 'aegis_paired') for r in aegis_paired_train] + [(r, 'wild_paired') for r in wild_paired_train],
            "val":   [(r, 'aegis_fast_only') for r in aegis_fast_only_val] + [(r, 'wild_fast_only') for r in wild_fast_only_val] +
                     [(r, 'aegis_paired') for r in aegis_paired_val] + [(r, 'wild_paired') for r in wild_paired_val],
            "proc": lambda item: process_fast_row(item[0], item[1].split('_')[0], 'proc') if 'fast_only' in item[1] else
                                 process_paired_row(item[0], item[1].split('_')[0], 'proc')
        },
        "Hybrid-P": {
            "train": [(r, 'aegis_fast_only') for r in aegis_fast_only_train] + [(r, 'wild_fast_only') for r in wild_fast_only_train] +
                     [(r, 'aegis_paired') for r in aegis_paired_train] + [(r, 'wild_paired') for r in wild_paired_train],
            "val":   [(r, 'aegis_fast_only') for r in aegis_fast_only_val] + [(r, 'wild_fast_only') for r in wild_fast_only_val] +
                     [(r, 'aegis_paired') for r in aegis_paired_val] + [(r, 'wild_paired') for r in wild_paired_val],
            "proc": lambda item: process_fast_row(item[0], item[1].split('_')[0], 'proc_prime') if 'fast_only' in item[1] else
                                 process_paired_row(item[0], item[1].split('_')[0], 'proc_prime')
        },
        "Hybrid-PN": {
            "train": [(r, 'aegis_fast_only') for r in aegis_fast_only_train] + [(r, 'wild_fast_only') for r in wild_fast_only_train] +
                     [(r, 'aegis_paired') for r in aegis_paired_train] + [(r, 'wild_paired') for r in wild_paired_train],
            "val":   [(r, 'aegis_fast_only') for r in aegis_fast_only_val] + [(r, 'wild_fast_only') for r in wild_fast_only_val] +
                     [(r, 'aegis_paired') for r in aegis_paired_val] + [(r, 'wild_paired') for r in wild_paired_val],
            # Hard-neg in slow partition only (paired rows); fast-only rows use proc_prime.
            "proc": lambda item: process_fast_row(item[0], item[1].split('_')[0], 'proc_prime') if 'fast_only' in item[1] else
                                 process_paired_row(item[0], item[1].split('_')[0], 'proc_prime_hard_neg')
        },
    }

    # --- Stage 4: Process and write output files ---
    # Deduplication: build a set of train prompt keys after processing, then filter
    # val rows against it to prevent train/val leakage across source pools.
    def _prompt_key(item):
        """Dedup/leakage key: the first 120 chars of the <user> block in a record."""
        m = re.search(r'<user>\s*(.*?)\s*</user>', item['messages'][0]['content'], re.DOTALL)
        return m.group(1).strip()[:120] if m else item['messages'][0]['content'][:120]

    print("\n[2/2] Writing Files...")
    for name, cfg in recipes.items():
        if name not in target_models: continue
        train_prompt_keys = set()
        for split in ["train", "val"]:
            out_rows = []
            data_list = list(cfg[split])
            random.shuffle(data_list)

            errors = 0
            overlength = 0
            for item in tqdm(data_list, desc=f"{name} [{split}]"):
                try:
                    result = cfg['proc'](item)
                    # Processors return a single dict normally, or a list of dicts when
                    # they emit a contrastive exemption pair.
                    records = result if isinstance(result, list) else ([result] if result else [])
                    for processed_json in records:
                        if _content_chars(processed_json) <= MAX_CONTENT_CHARS:
                            out_rows.append(processed_json)
                        else:
                            overlength += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5: print(f"  Warning [{name}/{split}]: {e}")
            if errors: print(f"  {name} [{split}]: {errors}/{len(data_list)} rows failed")
            if overlength: print(f"  {name} [{split}]: dropped {overlength} overlength samples (>{MAX_CONTENT_CHARS} chars)")

            leaked = 0
            if split == "train":
                train_prompt_keys = {_prompt_key(r) for r in out_rows}
            else:
                before = len(out_rows)
                out_rows = [r for r in out_rows if _prompt_key(r) not in train_prompt_keys]
                leaked = before - len(out_rows)
                if leaked: print(f"  {name} [val]: removed {leaked} leaked samples ({leaked/before*100:.1f}%)")

            outfile = os.path.join(args.output_dir, f"{split}_{name}{suffix}.jsonl")
            with open(outfile, 'w') as f:
                for item in out_rows: f.write(json.dumps(item) + "\n")

            total = len(out_rows)
            fast = sum(1 for r in out_rows if "<analysis>" not in r["messages"][1]["content"])
            slow = total - fast
            stats.append({
                "Model": f"{name}{suffix}",
                "Split": split,
                "Total": total,
                "Fast": fast,
                "Slow": slow,
                "Fast%": f"{fast/total*100:.1f}%" if total else "0%",
                "Slow%": f"{slow/total*100:.1f}%" if total else "0%",
                "Leaked": leaked,
                "Leaked%": f"{leaked/(leaked+total)*100:.1f}%" if leaked else "0%",
                "Overlength_Dropped": overlength,
            })

    # --- Statistics output ---
    df = pd.DataFrame(stats)
    train_df = df[df["Split"] == "train"].drop(columns=["Split", "Leaked", "Leaked%"])
    val_df   = df[df["Split"] == "val"].drop(columns=["Split"])

    w = 105
    for header, tbl in [("TRAIN DATASET STATISTICS", train_df), ("VAL DATASET STATISTICS", val_df)]:
        print("\n" + "=" * w)
        print(header.center(w))
        print("=" * w)
        print(tbl.to_markdown(index=False) if not tbl.empty else "  (no data)")

    out_csv  = os.path.join(args.output_dir, f"train_statistics{suffix}.csv")
    out_json = os.path.join(args.output_dir, f"train_statistics{suffix}.json")
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)
    print(f"\nStatistics saved to {out_csv} and {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hn_proxy", default=HN_PROXY_MODEL,
                        help="Toxicity classifier for hard-negative detection "
                             "(default: the paper's unitary/toxic-bert). A non-default "
                             "value requires --variant_suffix so the paper datasets are "
                             "never overwritten. Proxy-swap ablation: "
                             "--models Hybrid-PN --hn_proxy s-nlp/roberta_toxicity_classifier "
                             "--variant_suffix X")
    parser.add_argument("--variant_suffix", default="",
                        help="Suffix appended to output dataset names, e.g. 'X' writes "
                             "train_Hybrid-PNX.jsonl and train_statisticsX.csv")
    args = parser.parse_args()
    generate_datasets(args)
