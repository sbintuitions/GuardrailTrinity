# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Paper names for the ablation ladder, and how they appear in run directories.

A trained model directory is named {BASE_MODEL}_{VARIANT}_lr-{LR}_seed-{SEED},
for example Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42. A result row is labeled with the
variant plus, for the Hybrid family only, the inference mode: -1 for System-1
(fast) and -2 for System-2 (slow), for example Hybrid-PN-2.
"""
import re

BASES = ["Aegis", "Wild", "Mix", "Reason", "Adapt", "Hybrid"]
FAST_ONLY = {"Aegis", "Wild", "Mix"}
VARIANT_RE = r"(?:Aegis|Wild|Mix|(?:Reason|Adapt|Hybrid)(?:-PN|-P)?)"

# Row order of the result tables.
ROW_ORDER = [
    "Aegis", "Wild", "Mix",
    "Reason", "Reason-P", "Reason-PN",
    "Adapt", "Adapt-P", "Adapt-PN",
    "Hybrid-1", "Hybrid-2", "Hybrid-P-1", "Hybrid-P-2", "Hybrid-PN-1", "Hybrid-PN-2",
]

_RUN = re.compile(rf"^(?P<family>.*?)_(?P<variant>{VARIANT_RE})(?:_lr-|_|$)")


def split_run(slug):
    """'Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42' -> ('Qwen3-8B', 'Hybrid-PN'), else None."""
    m = _RUN.match(slug)
    return (m.group("family"), m.group("variant")) if m else None


def modes_for(variant):
    """Inference modes the paper evaluates for a variant."""
    base = variant.split("-")[0]
    if base in FAST_ONLY:
        return ["fast"]
    return ["fast", "slow"] if base == "Hybrid" else ["slow"]


def row_label(variant, mode):
    """Row label for a variant evaluated in 'fast' or 'slow' mode.

    Returns None for a mode the paper does not evaluate (for example Reason in
    fast mode), so callers can skip those result files.
    """
    if mode not in modes_for(variant):
        return None
    if variant.startswith("Hybrid"):
        return f"{variant}-{1 if mode == 'fast' else 2}"
    return variant
