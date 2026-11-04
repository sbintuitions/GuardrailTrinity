# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Tests for eval/variants.py: run-name parsing, mode selection, and table row labels."""
import pytest

import variants


@pytest.mark.parametrize("slug,expected", [
    ("Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42", ("Qwen3-8B", "Hybrid-PN")),
    ("Qwen3-8B_Hybrid-P_lr-1e-4_seed-0", ("Qwen3-8B", "Hybrid-P")),
    ("Qwen3-8B_Hybrid_lr-1e-4_seed-1", ("Qwen3-8B", "Hybrid")),
    ("Llama-3.1-8B-Instruct_Aegis_lr-1e-4_seed-42", ("Llama-3.1-8B-Instruct", "Aegis")),
    ("Qwen3-30B-A3B-Instruct-2507_Adapt-PN_lr-5e-5_seed-0",
     ("Qwen3-30B-A3B-Instruct-2507", "Adapt-PN")),
    ("Qwen3-8B_Reason", ("Qwen3-8B", "Reason")),
])
def test_split_run(slug, expected):
    """Well-formed run directory names split into (base model, variant)."""
    assert variants.split_run(slug) == expected


@pytest.mark.parametrize("slug", ["Qwen3-8B_Foo_lr-1e-4_seed-42", "Qwen3-8B", "random"])
def test_split_run_rejects_other_names(slug):
    """Unknown variants and non-run slugs return None."""
    assert variants.split_run(slug) is None


def test_modes():
    """Fast-only variants evaluate once, Hybrid variants in both modes."""
    assert variants.modes_for("Aegis") == ["fast"]
    assert variants.modes_for("Wild") == ["fast"]
    assert variants.modes_for("Mix") == ["fast"]
    assert variants.modes_for("Reason-PN") == ["slow"]
    assert variants.modes_for("Adapt") == ["slow"]
    assert variants.modes_for("Hybrid-P") == ["fast", "slow"]


def test_row_label():
    """Hybrid rows get a -1/-2 mode suffix; fast-only/slow-only rows keep the variant name."""
    assert variants.row_label("Hybrid-PN", "fast") == "Hybrid-PN-1"
    assert variants.row_label("Hybrid-PN", "slow") == "Hybrid-PN-2"
    assert variants.row_label("Hybrid", "fast") == "Hybrid-1"
    assert variants.row_label("Adapt-P", "slow") == "Adapt-P"
    assert variants.row_label("Aegis", "fast") == "Aegis"
    assert variants.row_label("Reason", "fast") is None
    assert variants.row_label("Aegis", "slow") is None


def test_row_order_covers_every_label():
    """ROW_ORDER contains exactly the labels every (variant, mode) pair can produce."""
    labels = {variants.row_label(v, m)
              for v in ["Aegis", "Wild", "Mix"] + [f"{b}{s}" for b in ("Reason", "Adapt", "Hybrid")
                                                  for s in ("", "-P", "-PN")]
              for m in ("fast", "slow")}
    labels.discard(None)
    assert labels == set(variants.ROW_ORDER)
