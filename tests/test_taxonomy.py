# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Tests for the procedural taxonomy generator in data/generate_train_datasets.py.

Covers seed determinism, code/description round-tripping, distractor counts,
static vs. procedural modes, the p=0.10 exemption mechanism, and the semantic
collision filter.
"""
import random
import re

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

import generate_train_datasets as gen
from taxonomies import AEGIS_TAXONOMY, WILD_TAXONOMY, MASTER_DESC

VIOLATED = [AEGIS_TAXONOMY["S1"], AEGIS_TAXONOMY["S8"]]


def draw(seed, mode="proc", force_exempt=None, violated=VIOLATED, base="aegis"):
    """Generate one taxonomy with a fresh random state seeded by `seed`."""
    random.seed(seed)
    return gen.generate_procedural_taxonomy(violated, mode, base, force_exempt=force_exempt)


def test_same_seed_same_taxonomy():
    """Generation is fully deterministic given the same seed and mode."""
    for seed in range(20):
        assert draw(seed) == draw(seed)
        assert draw(seed, "proc_prime") == draw(seed, "proc_prime")


def test_different_seeds_vary():
    """Different seeds actually produce different taxonomies."""
    assert len({draw(s)[0] for s in range(30)}) > 20


def test_rng_stream_is_deterministic_across_calls():
    """A seeded RNG stream reproduces the same sequence of taxonomies."""
    random.seed(7)
    a = [gen.generate_procedural_taxonomy(VIOLATED, "proc_prime", "aegis") for _ in range(25)]
    random.seed(7)
    b = [gen.generate_procedural_taxonomy(VIOLATED, "proc_prime", "aegis") for _ in range(25)]
    assert a == b


def test_procedural_contains_every_violated_description_and_codes_map_back():
    """Every violated description appears, mapped codes point back into the taxonomy, no exemption without 'prime'."""
    for seed in range(40):
        tax, codes, exempted, _ = draw(seed)
        assert len(codes) == len(VIOLATED)
        for desc in VIOLATED:
            assert desc in tax
        assert exempted is None  # 'prime' not in mode
        for code in codes:
            assert code in tax
        assert re.match(r"^(RULE-|CAT_|POL-|C|S|W|OPT_)", codes[0])


def test_distractor_count_ranges():
    """Total category count stays within the configured distractor ranges."""
    sizes = set()
    for seed in range(300):
        _, codes, *_ = draw(seed)
        # no direct access to count; derive from the largest numeric suffix in the string
        tax = draw(seed)[0]
        sizes.add(len(re.findall(r"(?:RULE-|CAT_|POL-|OPT_|[CSW])\d+", tax)))
    assert min(sizes) >= len(VIOLATED) + 4
    assert max(sizes) <= len(VIOLATED) + 30


def test_static_aegis_and_wild_use_fixed_codes():
    """Static mode renders the full fixed AEGIS/WildGuard taxonomy with original codes."""
    tax, codes, exempted, _ = draw(0, "static", violated=[AEGIS_TAXONOMY["S1"]])
    assert codes == ["S1"] and exempted is None
    assert len(tax.splitlines()) == len(AEGIS_TAXONOMY)
    tax, codes, _, _ = draw(0, "static", violated=[WILD_TAXONOMY["W5"]], base="wild")
    assert codes == ["W5"]
    assert len(tax.splitlines()) == len(WILD_TAXONOMY)


def test_force_exempt_requires_prime_mode_and_violations():
    """force_exempt only takes effect in 'prime' mode on rows with violations."""
    _, codes, exempted, desc = draw(3, "proc_prime", force_exempt=True)
    assert exempted == codes[0]
    assert desc in VIOLATED
    assert draw(3, "proc", force_exempt=True)[2] is None
    assert draw(3, "proc_prime", force_exempt=True, violated=[])[2] is None
    assert draw(3, "proc_prime", force_exempt=False)[2] is None


def test_exemption_rate_is_about_ten_percent():
    """The random exemption draw fires at ~10% in 'prime' mode."""
    random.seed(0)
    n = 3000
    hits = sum(gen.generate_procedural_taxonomy(VIOLATED, "proc_prime", "aegis")[2] is not None
               for _ in range(n))
    assert 0.07 < hits / n < 0.13


def test_semantic_cluster_members_are_not_used_as_distractors():
    """No distractor is drawn from the same semantic cluster as a violated category."""
    from taxonomies import SEMANTIC_CLUSTERS
    for cluster in SEMANTIC_CLUSTERS:
        for target in sorted(cluster):
            siblings = cluster - {target}
            for seed in range(15):
                tax = draw(seed, violated=[target])[0]
                descs = {line.split(": ", 1)[1] for line in tax.splitlines() if ": " in line} \
                    if tax.startswith(("RULE-", "CAT_", "POL-", "C", "S", "W", "OPT_")) else None
                if descs is None:
                    continue  # json / paragraph rendering
                assert not (siblings & descs), (target, siblings & descs)
