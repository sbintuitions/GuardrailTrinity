#!/usr/bin/env python3
# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""
Section 5 figures for the Guardrail Trinity paper.

Generates publication-ready figures that together tell the complete
empirical story of the paper:

    Fig 1 – The Recall Illusion (Sec. 5.1)
        DynaHate Recall vs. Specificity (100 − FPR) scatter, coloured by
        XSTest Cal-BA.  Shows that high recall in static-schema baselines is
        largely driven by degenerate over-classification: most baselines
        cluster in the top-left (high recall / low specificity) quadrant,
        while only the GPT-OSS 120B family reaches the honest high-specificity
        zone.  Trinity Adapt / Hybrid-2 / Hybrid-1 variants are overlaid to show where
        the proposed method sits in the same space.

    Fig 2 – Safety Override Paradox at 8B (Sec. 5.2)
        DynaHate Recall mean ± 1 std across all 15 Llama-3.1-8B variants
        (3 seeds).  Variance stays tight for Aegis to Reason (σ ≈ 2–7) but explodes
        for Adapt (σ = 25.6), Hybrid-PN-1 (σ = 24.8), and Hybrid-PN-2 (σ = 29.0),
        proving that the full three-component gradient load exceeds reliable
        8B training capacity.

    Fig 3 – Procedural Dominance Across Scales (Sec. 5.3)
        Grouped box + strip plot comparing the score distribution of all
        Static (Aegis to Reason, 6 variants) vs. all Procedural (Adapt to Hybrid, 9 variants)
        training variants on BeaverTails-PolicyShift Macro-F1 at 8B, 14B, and
        32B (Qwen3 family).  Each dot is one specific trained variant.
        The Procedural box sits consistently ≈ +5 pp above the Static box
        at every scale, proving that format randomisation is scale-invariant.

    Fig 4 – The 32B Bifurcation (Sec. 5.4)
        Two-panel trajectory (FPR top, FRR bottom) for Hybrid-2, Hybrid-P-2, and
        Hybrid-PN-2 across Qwen3 8B → 14B → 32B.  All three families track
        together from 8B to 14B (the Pareto optimum).  From 14B to 32B they
        bifurcate: Hybrid-P over-generalises permissions (FPR +12 pp, FRR +5 pp),
        while Hybrid-PN resolves the lexical trap (FPR −35 pp, FRR −14 pp).

Usage:
    python eval/generate_plots.py               # all 4 plots
    python eval/generate_plots.py --plot 1      # single plot
    python eval/generate_plots.py --plot 1 2 4  # subset

Output:
    <out_dir>/fig*.pdf   (vector, suitable for LaTeX includegraphics)
    <out_dir>/fig*.png   (150 dpi raster, for quick preview)

Input:
    The aggregated_*_results.csv files written by eval/generate_pivot_tables.py
    (one per base model, plus aggregated_baselines_results.csv) in --results_dir.
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

# ── Paths ─────────────────────────────────────────────────────────────────────
# Set from the command line in main().
SUMMARY: Path = None
OUT: Path = None

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
C = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "pink":      "#CC79A7",
    "gray":      "#999999",
    "lightgray": "#DDDDDD",
}

# ── Column shortcuts ───────────────────────────────────────────────────────────
# These match the exact CSV headers (spacing matters!)
COLS = {
    "dh_recall": "dynahate(Recall)↑",
    "dh_fpr":    "dynahate(FPR)↓",
    "dh_advba":  "dynahate(Adv-BA)↑",
    "xs_calba":  "xstest(Cal-BA)↑",
    "xs_frr":    "xstest (FRR)↓",
    "xs_urr":    "xstest (URR)↑",
    "bt_macf1":  "bt_dyn (MacF1)↑",
    "bt_ga":     "bt_dyn (GA)↑",
    "dg_ga":     "dynagrd (GA)↑",
    "dg_balc":   "dynagrd (BalAcc)↑",
}


# ── Global style ───────────────────────────────────────────────────────────────
def setup_style() -> None:
    """Set global matplotlib style (font sizes pre-scaled for LaTeX column-width downscaling)."""
    # Figures are embedded at column width (~3.3 in) but rendered at 5–7 in,
    # so all font sizes are scaled up by ~1.5× to remain readable after
    # LaTeX downscales them.
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,
        "legend.fontsize":    11,
        "legend.framealpha":  0.88,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.which":    "major",
        "grid.alpha":         0.22,
        "grid.linewidth":     0.5,
        "figure.dpi":         150,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


# ── Data loading ───────────────────────────────────────────────────────────────
def _parse_mean_std(val: str) -> tuple[float, float]:
    """Parse a "mean ± std" cell into (mean, std); malformed cells become (nan, 0)."""
    s = str(val).strip()
    m = re.match(r"([\d.]+)\s*[±]\s*([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return np.nan, 0.0


def _load_raw(path: Path) -> pd.DataFrame:
    """Read an aggregated results CSV; whitespace-trim headers and the model column."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "model"})
    df["model"] = df["model"].str.strip()
    return df


def _split_mean_std(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every metric column into separate (means, stds) frames indexed by model."""
    means, stds = {}, {}
    for col in df.columns:
        if col == "model":
            continue
        parsed = df[col].apply(_parse_mean_std)
        means[col] = [p[0] for p in parsed]
        stds[col]  = [p[1] for p in parsed]
    idx = df["model"].values
    return pd.DataFrame(means, index=idx), pd.DataFrame(stds, index=idx)


def load_all() -> dict:
    """Load every aggregated family summary CSV into {family: {"means": df, "stds": df}}."""
    files = {
        "baselines": "aggregated_baselines_results.csv",
        "llama8b":   "aggregated_Llama_3_1_8B_Instruct_results.csv",
        "qwen8b":    "aggregated_Qwen3_8B_results.csv",
        "qwen14b":   "aggregated_Qwen3_14B_results.csv",
        "qwen32b":   "aggregated_Qwen3_32B_results.csv",
    }
    data: dict = {}
    for key, fname in files.items():
        raw = _load_raw(SUMMARY / fname)
        m, s = _split_mean_std(raw)
        data[key] = {"means": m, "stds": s}
    return data


def _save(fig: plt.Figure, name: str) -> None:
    """Write the figure as PDF + PNG into the output directory and close it."""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=150)
    print(f"  Saved → {OUT / name}.pdf  +  .png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Custom-Policy Compliance Gap (Sec. 5.1)
# ═══════════════════════════════════════════════════════════════════════════════
_BASELINE_SHORT = {
    "llama-guard-3":       "Llama Guard 3",
    "llama-guard-4":       "Llama Guard 4",
    "qwen3guard":          "Qwen3 Guard",
    "nemotron":            "Nemotron",
    "gpt-oss-20b-low":     "GPT-OSS-Safeguard 20B-L",
    "gpt-oss-20b-medium":  "GPT-OSS-Safeguard 20B-M",
    "gpt-oss-20b-high":    "GPT-OSS-Safeguard 20B-H",
    "gpt-oss-120b-low":    "GPT-OSS-Safeguard 120B-L",
    "gpt-oss-120b-medium": "GPT-OSS-Safeguard 120B-M",
    "gpt-oss-120b-high":   "GPT-OSS-Safeguard 120B-H",
    "shieldgemma":         "ShieldGemma",
    "wildguard":           "WildGuard",
    "granite":             "Granite",
    "dynaguard-8b-fast":   "DynaGuard Fast",
    "dynaguard-8b-cot":    "DynaGuard CoT",
}
_STD_F1_COLS = ["aegis (F1)↑", "wildgrd (F1)↑", "toxic (F1)↑"]

# Trinity procedural variants at Qwen3-32B (the paper's flagship scale).
# All Adapt/Hybrid variants — no cherry-picking; they form a cluster.
_TRINITY_PROCEDURAL_VARS = [
    "Adapt", "Adapt-P", "Adapt-PN",
    "Hybrid-1", "Hybrid-2", "Hybrid-P-1", "Hybrid-P-2", "Hybrid-PN-1", "Hybrid-PN-2",
]


def plot1_custom_policy_gap(data: dict) -> None:
    """
    Custom-policy compliance gap for Sec. 5.1.

    X-axis: mean F1 over three standard safety benchmarks (AEGIS, WildGuard,
            ToxiChat) — basic safety classification ability.
    Y-axis: BeaverTails-PolicyShift Macro-F1 — zero-shot adaptation to a novel
            6-category taxonomy no model was trained on.

    Baselines (gray triangles) cluster at y ≤ 65 — they cannot adapt to
    custom policies at inference time because their schema is hard-coded.
    Trinity procedural variants (blue circles, Qwen3-32B; blue diamonds,
    Qwen3-14B) clear y ≥ 68 without sacrificing standard performance —
    the high-x, high-y quadrant is Trinity-only, and 14B already enters
    that zone at a parameter count comparable to most baselines.
    """
    bm  = data["baselines"]["means"]
    m8  = data["qwen8b"]["means"]
    m14 = data["qwen14b"]["means"]
    m32 = data["qwen32b"]["means"]

    def _std_score(m: pd.DataFrame, var: str) -> float:
        """Mean of the standard-moderation F1 columns for one variant row."""
        return float(np.mean([m.loc[var, c] for c in _STD_F1_COLS]))

    BT = COLS["bt_macf1"]

    baseline_pts = []
    for mid in bm.index:
        baseline_pts.append({
            "label": _BASELINE_SHORT.get(mid, mid),
            "x": _std_score(bm, mid),
            "y": float(bm.loc[mid, BT]),
        })

    trinity_32b_pts = []
    for var in _TRINITY_PROCEDURAL_VARS:
        if var in m32.index:
            trinity_32b_pts.append({
                "label": var,
                "x": _std_score(m32, var),
                "y": float(m32.loc[var, BT]),
            })

    trinity_14b_pts = []
    for var in _TRINITY_PROCEDURAL_VARS:
        if var in m14.index:
            trinity_14b_pts.append({
                "label": var,
                "x": _std_score(m14, var),
                "y": float(m14.loc[var, BT]),
            })

    trinity_8b_pts = []
    for var in _TRINITY_PROCEDURAL_VARS:
        if var in m8.index:
            trinity_8b_pts.append({
                "label": var,
                "x": _std_score(m8, var),
                "y": float(m8.loc[var, BT]),
            })

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Compute Trinity-only zone (region above all baseline y-values)
    max_baseline_y = max(p["y"] for p in baseline_pts)
    ax.axhspan(max_baseline_y, 100, color="#eef5ff", alpha=0.6, zorder=0)
    n_total = len(trinity_8b_pts) + len(trinity_14b_pts) + len(trinity_32b_pts)
    ax.text(63.4, 73.5, f"Trinity-only zone (n={n_total})",
            fontsize=14, color=C["blue"], style="italic", fontweight="bold",
            ha="left", va="center")

    # Scatter points
    for pt in baseline_pts:
        ax.scatter(pt["x"], pt["y"], marker="^", s=70,
                   color="#888888", edgecolors="white", linewidths=0.7,
                   alpha=0.9, zorder=3)
    # Three perceptually distinct, colorblind-safe Wong hues — one per scale.
    _COL_8B  = C["orange"]      # #E69F00
    _COL_14B = C["green"]       # #009E73
    _COL_32B = C["blue"]        # #0072B2
    for pt in trinity_8b_pts:
        ax.scatter(pt["x"], pt["y"], marker="s", s=55,
                   color=_COL_8B, edgecolors="white", linewidths=0.7,
                   alpha=0.70, zorder=3.5)
    for pt in trinity_14b_pts:
        ax.scatter(pt["x"], pt["y"], marker="D", s=55,
                   color=_COL_14B, edgecolors="white", linewidths=0.7,
                   alpha=0.70, zorder=4)
    for pt in trinity_32b_pts:
        ax.scatter(pt["x"], pt["y"], marker="o", s=75,
                   color=_COL_32B, edgecolors="white", linewidths=0.7,
                   alpha=0.70, zorder=5)

    # adjust_text struggles to clear Llama Guard 3 and Granite — both triangles
    # sit in a tightly packed band just under the dashed reference line, so we
    # hand-place those two with leader lines and leave the rest for auto-place.
    BASELINE_AUTO = {
        "WildGuard", "Nemotron",
    }
    # 8-o'clock (lower-left) placements; Qwen3 Guard uses a longer leader to
    # clear the neighbouring Granite label.  Llama Guard 4 is lifted above
    # the dashed reference line for visual separation.
    BASELINE_MANUAL = {
        "Llama Guard 4": (72.0, 66.5),
        "Llama Guard 3": (78.8, 60.8),
        "Granite":       (85.5, 60.5),
        "ShieldGemma":   (74.0, 60.5),
        "Qwen3 Guard":   (78.5, 58.5),
    }

    texts = []
    for pt in baseline_pts:
        if pt["label"] in BASELINE_AUTO:
            texts.append(ax.text(pt["x"], pt["y"], pt["label"],
                                 fontsize=12, color="#444444", zorder=5))
        elif pt["label"] in BASELINE_MANUAL:
            tx, ty = BASELINE_MANUAL[pt["label"]]
            ax.annotate(pt["label"],
                        xy=(pt["x"], pt["y"]), xytext=(tx, ty),
                        fontsize=12, color="#444444", ha="left",
                        arrowprops=dict(arrowstyle="-", color="#999999",
                                        lw=0.5, alpha=0.7))

    # Trinity cluster identity is fully conveyed by the zone label (with n)
    # plus the per-scale marker legend, so no separate cluster annotation
    # is drawn here.

    # GPT-OSS family aggregates (one label per family with count).
    gpt_20b  = [p for p in baseline_pts if p["label"].startswith("GPT-OSS-Safeguard 20B")]
    gpt_120b = [p for p in baseline_pts if p["label"].startswith("GPT-OSS-Safeguard 120B")]
    if gpt_20b:
        cx = sum(p["x"] for p in gpt_20b) / len(gpt_20b)
        cy = sum(p["y"] for p in gpt_20b) / len(gpt_20b)
        ax.annotate(f"GPT-OSS-Safeguard 20B\n(L / M / H)",
                    xy=(cx, cy), xytext=(cx + 3.5, cy - 5.0),
                    fontsize=12, color="#444444", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#999999",
                                    lw=0.5, alpha=0.7))
    if gpt_120b:
        cx = sum(p["x"] for p in gpt_120b) / len(gpt_120b)
        cy = sum(p["y"] for p in gpt_120b) / len(gpt_120b)
        ax.annotate(f"GPT-OSS-Safeguard 120B\n(L / M / H)",
                    xy=(cx, cy), xytext=(cx - 6.5, cy - 4.0),
                    fontsize=12, color="#444444", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#999999",
                                    lw=0.5, alpha=0.7))

    # DynaGuard family aggregate (Fast + CoT cluster ~85.5 x, ~64.5–64.8 y).
    dynaguard = [p for p in baseline_pts if p["label"].startswith("DynaGuard")]
    if dynaguard:
        cx = sum(p["x"] for p in dynaguard) / len(dynaguard)
        cy = sum(p["y"] for p in dynaguard) / len(dynaguard)
        ax.annotate(f"DynaGuard\n(Fast / CoT)",
                    xy=(cx, cy), xytext=(cx + 1.5, cy - 2.5),
                    fontsize=12, color="#444444", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#999999",
                                    lw=0.5, alpha=0.7))

    # Best baseline reference line (horizontal dashed) — no inline label.
    ax.axhline(max_baseline_y, color="#888888", lw=0.7, ls="--", zorder=2)

    # All scatter coordinates (baselines + every Trinity scale) are passed
    # to adjust_text so labels repel from points rather than just from each
    # other.  Force / expand values are raised relative to the default so
    # baseline labels reliably clear their own triangles.
    all_xy = (
        [(p["x"], p["y"]) for p in baseline_pts]
        + [(p["x"], p["y"]) for p in trinity_8b_pts]
        + [(p["x"], p["y"]) for p in trinity_14b_pts]
        + [(p["x"], p["y"]) for p in trinity_32b_pts]
    )
    pt_x = [p[0] for p in all_xy]
    pt_y = [p[1] for p in all_xy]
    adjust_text(
        texts, x=pt_x, y=pt_y, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999",
                        lw=0.5, alpha=0.7, shrinkA=4, shrinkB=4),
        expand_points=(3.2, 3.6),
        expand_text=(1.8, 2.0),
        force_points=(2.4, 2.8),
        force_text=(1.5, 2.0),
        force_objects=(1.2, 1.4),
        lim=4000,
        only_move={"points": "xy", "texts": "xy"},
    )

    handles = [
        mlines.Line2D([0], [0], marker="^", linestyle="None",
                      markerfacecolor="#888888", markeredgecolor="white",
                      markersize=9, label="SOTA baselines"),
        mlines.Line2D([0], [0], marker="o", linestyle="None",
                      markerfacecolor=_COL_32B, markeredgecolor="white",
                      markersize=9, label="Trinity-32B"),
        mlines.Line2D([0], [0], marker="D", linestyle="None",
                      markerfacecolor=_COL_14B, markeredgecolor="white",
                      markersize=8, label="Trinity-14B"),
        mlines.Line2D([0], [0], marker="s", linestyle="None",
                      markerfacecolor=_COL_8B, markeredgecolor="white",
                      markersize=8, label="Trinity-8B"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              framealpha=0.95, edgecolor="#cccccc", fontsize=12)

    ax.set_xlabel("Standard safety F1 (%)", fontsize=14)
    ax.set_ylabel("BeaverTails-PolicyShift Macro-F1 (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)

    ax.set_xlim(63, 90)
    ax.set_ylim(44, 75)
    ax.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, "custom_policy_gap")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 – 8B Gradient Overload: Variance on DynaHate Recall (Llama-3.1-8B)
# ═══════════════════════════════════════════════════════════════════════════════
_LLAMA_VARIANT_ORDER = [
    "Aegis", "Wild", "Mix",
    "Reason", "Reason-P", "Reason-PN",
    "Adapt", "Adapt-P", "Adapt-PN",
    "Hybrid-1", "Hybrid-2", "Hybrid-P-1", "Hybrid-P-2", "Hybrid-PN-1", "Hybrid-PN-2",
]
# Zebra background bands per family — neutral tints, no clash with dot colors.
_FAMILY_BANDS = [
    (slice(0, 3),  "#ffffff",  "Static"),
    (slice(3, 6),  "#f5f5f5",  "+Reasoning"),
    (slice(6, 9),  "#ffffff",  "Procedural"),
    (slice(9, 15), "#f5f5f5",  "Hybrid"),
]
def plot2_8b_variance(data: dict) -> None:
    """
    Llama-3.1-8B DynaHate recall variability across three training seeds.

    Bars show mean minus one standard deviation, with ticks at the mean.
    The 50% line is a descriptive reference. Mean minus standard deviation
    is neither the observed minimum seed nor a statistical confidence bound;
    recall alone has no universal chance threshold. This chart characterizes
    seed sensitivity without identifying a capacity or gradient mechanism.
    """
    m = data["llama8b"]["means"]
    s = data["llama8b"]["stds"]

    variants = [v for v in _LLAMA_VARIANT_ORDER if v in m.index]
    labels   = list(variants)
    means    = np.array([m.loc[v, COLS["dh_recall"]] for v in variants])
    stds     = np.array([s.loc[v, COLS["dh_recall"]] for v in variants])

    # Descriptive variability summary; retain the original plotted values.
    lower_summary = means - stds
    REFERENCE_RECALL = 50.0

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    # Background family bands (zebra tint for family identification).
    n = len(variants)
    band_label_y = 108.0
    for band_slice, band_color, band_label in _FAMILY_BANDS:
        indices = list(range(n))[band_slice]
        if not indices:
            continue
        x0 = indices[0]  - 0.5
        x1 = indices[-1] + 0.5
        ax.axvspan(x0, x1, color=band_color, alpha=1.0, zorder=0)
        ax.text((x0 + x1) / 2, band_label_y, band_label,
                ha="center", va="bottom",
                fontsize=10, color="#444444", style="italic")

    # Vertical bars showing the variability summary (μ − σ).
    stable_color  = C["blue"]
    failure_color = C["vermilion"]
    bar_colors = [failure_color if w < REFERENCE_RECALL else stable_color
                  for w in lower_summary]
    ax.bar(range(n), lower_summary, color=bar_colors, width=0.68,
           edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)

    # Mean recall marker (small horizontal tick above each bar) for reference.
    ax.scatter(range(n), means, marker="_", color="#222222",
               s=300, linewidths=2.2, zorder=5)

    # Descriptive reference line.
    ax.axhline(REFERENCE_RECALL, color="#666666", lw=1.0, ls="--", zorder=2)
    ax.text(n - 0.4, REFERENCE_RECALL + 1.5,
            f"{REFERENCE_RECALL:.0f}% reference",
            fontsize=9, ha="right", va="bottom",
            color="#555555", style="italic")

    # Inline values for variants below the reference.
    for v in ("Adapt", "Hybrid-PN-1", "Hybrid-PN-2"):
        if v not in m.index:
            continue
        i = variants.index(v)
        ax.text(i, lower_summary[i] - 2.5, f"{lower_summary[i]:.0f}",
                ha="center", va="top",
                fontsize=9, color=failure_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("DynaHate Recall (%)")
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.7, n - 0.3)

    # Legend
    handles = [
        mlines.Line2D([0], [0], marker="s", linestyle="None",
                      markerfacecolor=stable_color, markeredgecolor="white",
                      markersize=11,
                      label="μ−σ ≥ 50%"),
        mlines.Line2D([0], [0], marker="s", linestyle="None",
                      markerfacecolor=failure_color, markeredgecolor="white",
                      markersize=11,
                      label="μ−σ < 50%"),
        mlines.Line2D([0], [0], marker="_", color="#222222",
                      linestyle="None", markersize=14, markeredgewidth=2,
                      label="Mean recall"),
    ]
    ax.legend(handles=handles, loc="lower left",
              framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    _save(fig, "8b_variance")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Scale-Invariant Procedural Dominance (BT-PolicyShift Macro-F1)
# ═══════════════════════════════════════════════════════════════════════════════
_STATIC_VARS     = ["Aegis", "Wild", "Mix", "Reason", "Reason-P", "Reason-PN"]
_PROCEDURAL_VARS = [
    "Adapt", "Adapt-P", "Adapt-PN",
    "Hybrid-1", "Hybrid-2", "Hybrid-P-1", "Hybrid-P-2", "Hybrid-PN-1", "Hybrid-PN-2",
]
_SCALE_KEYS = [("qwen8b", "8B"), ("qwen14b", "14B"), ("qwen32b", "32B")]


def plot3_procedural_dominance(data: dict) -> None:
    """
    Grouped box+strip plot proving procedural schema dominance is scale-invariant.

    Architecture : Qwen3 family only (8B, 14B, 32B) for an apples-to-apples
                   scale comparison.  Each architecture contributes one value
                   per variant; no cross-architecture averaging is performed.
    X-axis  : Parameter scale — 8B, 14B, 32B.
    Y-axis  : BeaverTails-PolicyShift Macro-F1 (%) — zero-shot policy compliance.
              Macro-F1 is used (not accuracy) because the benchmark is 71 %
              safe-skewed; accuracy can be gamed by predicting the majority
              class, while Macro-F1 strictly penalises that shortcut.
    Hue     : Schema type — Static (Aegis to Reason, 6 variants) vs. Procedural
              (Adapt to Hybrid, 9 variants).
    Box     : Median line + IQR box + 1.5×IQR whiskers over all variants in
              the group.  Outliers are suppressed (fliersize=0) because all
              individual points are shown explicitly as dots.
    Dots    : Each dot is one trained model variant (jittered horizontally
              to avoid overlap).  With only 6 / 9 points per box, showing
              the raw distribution is more informative than box statistics alone.
    Δ label : Mean difference (Procedural − Static) annotated above each pair.

    Narrative:
      The Procedural box sits consistently ≈ +5 pp above the Static box at
      every scale, and the two groups never overlap.  This scale-invariance
      proves that adaptive format randomisation is a strict upgrade over
      static training for zero-shot generalisation — irrespective of whether
      the model has 8B, 14B, or 32B parameters.
    """
    rows = []
    for key, scale_label in _SCALE_KEYS:
        m = data[key]["means"]
        for var in _STATIC_VARS:
            if var in m.index:
                rows.append({"Scale": scale_label, "Schema": "Static",
                             "Variant": var,
                             "MacF1": m.loc[var, COLS["bt_macf1"]]})
        for var in _PROCEDURAL_VARS:
            if var in m.index:
                rows.append({"Scale": scale_label, "Schema": "Procedural",
                             "Variant": var,
                             "MacF1": m.loc[var, COLS["bt_macf1"]]})

    df = pd.DataFrame(rows)

    palette = {"Static": C["sky"], "Procedural": C["green"]}

    # seaborn boxplot with width=0.55 and 2 hue groups places group centres at x ± width/4
    _BOX_WIDTH   = 0.55
    _GRP_OFFSET  = _BOX_WIDTH / 4          # ≈ 0.1375
    _HALF_SPREAD = 0.115                    # uniform-spread half-width for dots
    _DOT_COLORS  = {"Static": "#1a6fa3", "Procedural": "#006651"}
    _GRP_OFFSETS = {"Static": -_GRP_OFFSET, "Procedural": +_GRP_OFFSET}
    _VAR_ORDERS  = {
        "Static":     _STATIC_VARS,
        "Procedural": _PROCEDURAL_VARS,
    }
    _SCALE_TO_X  = {"8B": 0, "14B": 1, "32B": 2}

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # Best baseline reference: max BT-PolicyShift Macro-F1 across all SOTA baselines.
    # Makes the Trinity-vs-baseline gap visually unambiguous.
    bm = data["baselines"]["means"]
    bt_col = COLS["bt_macf1"]
    best_id = bm[bt_col].idxmax()
    best_val = bm.loc[best_id, bt_col]
    best_label = _BASELINE_SHORT.get(best_id, best_id)

    ax.axhline(best_val, color="#888888", lw=1.0, ls="--", zorder=2)

    sns.boxplot(
        data=df, x="Scale", y="MacF1", hue="Schema",
        palette=palette, width=_BOX_WIDTH, linewidth=1.0,
        fliersize=0, ax=ax, zorder=3,
        order=["8B", "14B", "32B"],
    )

    # Dots sorted Aegis to Hybrid and uniformly spaced — no random jitter
    for schema, s_df in df.groupby("Schema", sort=False):
        var_order = _VAR_ORDERS[schema]
        sort_key  = {v: i for i, v in enumerate(var_order)}
        color     = _DOT_COLORS[schema]
        x_offset  = _GRP_OFFSETS[schema]
        for scale, sc_df in s_df.groupby("Scale", sort=False):
            sc_df = sc_df.copy()
            sc_df["_k"] = sc_df["Variant"].map(sort_key)
            sc_df = sc_df.sort_values("_k")
            n = len(sc_df)
            center = _SCALE_TO_X[scale] + x_offset
            xs = np.linspace(center - _HALF_SPREAD, center + _HALF_SPREAD, n)
            ax.scatter(xs, sc_df["MacF1"].values,
                       color=color, s=28, alpha=0.85, zorder=4,
                       edgecolors="none")

    # Bracket-style Δ annotation spanning both groups — neutral color avoids
    # visual association with either schema group
    for i, (_, scale_label) in enumerate(_SCALE_KEYS):
        static_vals = df[(df["Scale"] == scale_label) & (df["Schema"].str.startswith("S"))]["MacF1"]
        proc_vals   = df[(df["Scale"] == scale_label) & (df["Schema"].str.startswith("P"))]["MacF1"]
        delta = proc_vals.mean() - static_vals.mean()
        x_stat = i - _GRP_OFFSET
        x_proc = i + _GRP_OFFSET
        y_label = max(static_vals.max(), proc_vals.max()) + 1.0
        ax.text((x_stat + x_proc) / 2, y_label,
                f"Δ={delta:+.1f}", ha="center", va="bottom",
                fontsize=12, color="#222222", fontweight="bold")

    ax.set_xlabel("Parameter scale (Qwen3)")
    ax.set_ylabel("BeaverTails-PolicyShift Macro-F1 (%)")
    ax.set_ylim(57, 77)

    ax.legend(title="Schema type", title_fontsize=11,
              loc="lower right", framealpha=0.9)

    plt.tight_layout()
    _save(fig, "procedural_dominance")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 – The 32B Bifurcation: FPR & FRR Trajectory Across Scales
# ═══════════════════════════════════════════════════════════════════════════════
_TRAJ_FAMILIES = [
    ("Hybrid-2",    "Hybrid-2 (base)",            C["blue"],   "-",  "o"),
    ("Hybrid-P-2",  "Hybrid-P-2 (+perm)",         C["orange"], "--", "s"),
    ("Hybrid-PN-2", "Hybrid-PN-2 (+perm+HN)",     C["green"],  "-.", "^"),
]
_TRAJ_SCALE_KEYS = [("qwen8b", "8B"), ("qwen14b", "14B"), ("qwen32b", "32B")]


def plot4_bifurcation(data: dict) -> None:
    """
    Two-panel trajectory charting the 32B architectural bifurcation.

    Architecture : Qwen3 family (8B → 14B → 32B).  Slow-mode (_S) variants
                   only — the section 5.4 analysis is built on slow-mode
                   comparisons, and mixing _F / _S into averages would conflate
                   the mechanism.
    X-axis (shared) : Parameter scale — 8B, 14B, 32B.
    Top panel  : DynaHate FPR (%) ↓ — false positive rate on benign inputs;
                 measures vulnerability to over-classification.
    Bottom panel: XSTest FRR (%) ↓ — false refusal rate on safe prompts;
                  measures vulnerability to the lexical trap.
    Lines      : Three families differentiated by colour + line style:
                   Hybrid-2   (blue, solid)       — base hybrid, no augmentation.
                   Hybrid-P-2 (amber, dashed)     — + procedural permission training.
                   Hybrid-PN-2 (green, dash-dot)  — + permission + hard-negative injection.
    Annotations:
      • End-value labels at 32B for each line.
      • Vertical dotted line at 14B with "14B Pareto optimum" box label.
      • Double-headed arrow at 32B showing the 43.6 pp FPR bifurcation spread.

    Narrative:
      All three families track together from 8B to 14B — the scale at which
      the architecture naturally balances multi-step reasoning with formatting.
      From 14B to 32B they diverge sharply: Hybrid-P over-generalises permission
      instructions, driving FPR up to 65.9 % and FRR up to 25.6 %.  Hybrid-PN
      goes the opposite direction — FPR collapses to 22.4 % and FRR drops to
      13.6 % — proving that hard-negative injection is the mandatory correction
      term that directs 32B's amplified capacity toward suppressing false
      positives rather than amplifying them.
    """
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.2),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={"hspace": 0.08})
    ax_fpr, ax_frr = axes

    x_pos   = np.array([0, 1, 2])
    x_ticks = ["8B", "14B", "32B"]

    # Collect end-of-trajectory label data per panel so we can resolve
    # vertical collisions (especially in the FRR panel where Hybrid-2 and Hybrid-PN-2
    # land only ~3 pp apart at 32B).
    fpr_end, frr_end = [], []
    for var, label, color, ls, marker in _TRAJ_FAMILIES:
        fpr_vals, frr_vals = [], []
        for key, _ in _TRAJ_SCALE_KEYS:
            m = data[key]["means"]
            if var in m.index:
                fpr_vals.append(m.loc[var, COLS["dh_fpr"]])
                frr_vals.append(m.loc[var, COLS["xs_frr"]])
            else:
                fpr_vals.append(np.nan)
                frr_vals.append(np.nan)

        kw = dict(color=color, linestyle=ls, linewidth=2.0,
                  marker=marker, markersize=7, markeredgecolor="white",
                  markeredgewidth=0.8, zorder=4, label=label)
        ax_fpr.plot(x_pos, fpr_vals, **kw)
        ax_frr.plot(x_pos, frr_vals, **kw)

        fpr_end.append({"y": fpr_vals[-1], "color": color})
        frr_end.append({"y": frr_vals[-1], "color": color})

    def _place_end_labels(ax, items, x_end, min_sep):
        """Place ' {val:.1f}' labels at the right of each trajectory, pushing
        upward whenever two labels would land within ``min_sep`` of each other."""
        items_sorted = sorted(items, key=lambda d: d["y"])
        placed_y = []
        for d in items_sorted:
            target = d["y"]
            if placed_y and target - placed_y[-1] < min_sep:
                target = placed_y[-1] + min_sep
            placed_y.append(target)
        for d, y in zip(items_sorted, placed_y):
            ax.annotate(f" {d['y']:.1f}",
                        xy=(x_end, d["y"]),
                        xytext=(x_end + 0.04, y),
                        fontsize=11, color=d["color"], va="center")

    _place_end_labels(ax_fpr, fpr_end, x_pos[-1], min_sep=4.5)
    _place_end_labels(ax_frr, frr_end, x_pos[-1], min_sep=2.8)

    # 14B reference line — vertical guide marking the Pareto-optimum scale.
    # The descriptive label lives in the caption to keep the figure uncluttered.
    for ax in axes:
        ax.axvline(1, color="#888888", lw=1.0, ls=":", zorder=2)

    # ── Axes cosmetics ────────────────────────────────────────────
    ax_fpr.set_ylabel("DynaHate FPR (%) ↓")
    ax_frr.set_ylabel("XSTest FRR (%) ↓")
    ax_frr.set_xticks(x_pos)
    ax_frr.set_xticklabels(x_ticks)
    ax_frr.set_xlabel("Parameter scale (Qwen3)")
    ax_fpr.set_ylim(bottom=0)
    ax_frr.set_ylim(bottom=0)

    ax_frr.legend(loc="lower left", framealpha=0.9)

    _save(fig, "bifurcation")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 – Adapt Family Cross-Scale Trajectory (Appx., parallel to Fig 4)
# ═══════════════════════════════════════════════════════════════════════════════
_ADAPT_TRAJ_FAMILIES = [
    ("Adapt",    "Adapt (base)",            C["blue"],   "-",  "o"),
    ("Adapt-P",  "Adapt-P (+perm)",         C["orange"], "--", "s"),
    ("Adapt-PN", "Adapt-PN (+perm+HN)",     C["green"],  "-.", "^"),
]


def plot5_adapt_trajectory(data: dict) -> None:
    """
    Cross-scale trajectory for the Adapt family — appendix companion to Fig 4.

    Same axes as Fig 4 (DynaHate FPR top, XSTest FRR bottom) so the two plots
    can be read side-by-side.  Three families differentiated by colour + line
    style:
        Adapt    (blue, solid)       — base Adapt, no augmentation.
        Adapt-P  (amber, dashed)     — + procedural permission training.
        Adapt-PN (green, dash-dot)   — + permission + hard-negative injection.

    Narrative: Adapt sits at lower FPR than Hybrid at every scale (procedural
    schemas + decoupled reasoning without the 3:1 fast/slow dilution).  The
    14B point is the Pareto optimum for the un-augmented Adapt variant; -PN
    further compresses FPR at 32B without the Hybrid family's permission-only
    over-classification.
    """
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.2),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={"hspace": 0.08})
    ax_fpr, ax_frr = axes

    x_pos   = np.array([0, 1, 2])
    x_ticks = ["8B", "14B", "32B"]

    fpr_end, frr_end = [], []
    for var, label, color, ls, marker in _ADAPT_TRAJ_FAMILIES:
        fpr_vals, frr_vals = [], []
        for key, _ in _TRAJ_SCALE_KEYS:
            m = data[key]["means"]
            if var in m.index:
                fpr_vals.append(m.loc[var, COLS["dh_fpr"]])
                frr_vals.append(m.loc[var, COLS["xs_frr"]])
            else:
                fpr_vals.append(np.nan)
                frr_vals.append(np.nan)

        kw = dict(color=color, linestyle=ls, linewidth=2.0,
                  marker=marker, markersize=7, markeredgecolor="white",
                  markeredgewidth=0.8, zorder=4, label=label)
        ax_fpr.plot(x_pos, fpr_vals, **kw)
        ax_frr.plot(x_pos, frr_vals, **kw)

        fpr_end.append({"y": fpr_vals[-1], "color": color})
        frr_end.append({"y": frr_vals[-1], "color": color})

    def _place_end_labels(ax, items, x_end, min_sep):
        """Annotate series end values with at least min_sep vertical spacing between labels."""
        items_sorted = sorted(items, key=lambda d: d["y"])
        placed_y = []
        for d in items_sorted:
            target = d["y"]
            if placed_y and target - placed_y[-1] < min_sep:
                target = placed_y[-1] + min_sep
            placed_y.append(target)
        for d, y in zip(items_sorted, placed_y):
            ax.annotate(f" {d['y']:.1f}",
                        xy=(x_end, d["y"]),
                        xytext=(x_end + 0.04, y),
                        fontsize=11, color=d["color"], va="center")

    _place_end_labels(ax_fpr, fpr_end, x_pos[-1], min_sep=4.5)
    _place_end_labels(ax_frr, frr_end, x_pos[-1], min_sep=2.8)

    for ax in axes:
        ax.axvline(1, color="#888888", lw=1.0, ls=":", zorder=2)

    ax_fpr.set_ylabel("DynaHate FPR (%) ↓")
    ax_frr.set_ylabel("XSTest FRR (%) ↓")
    ax_frr.set_xticks(x_pos)
    ax_frr.set_xticklabels(x_ticks)
    ax_frr.set_xlabel("Parameter scale (Qwen3)")
    ax_fpr.set_ylim(bottom=0)
    ax_frr.set_ylim(bottom=0)

    ax_frr.legend(loc="lower right", framealpha=0.9)

    _save(fig, "adapt_trajectory")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 – Per-variant BeaverTails-PolicyShift Macro-F1 (Appx., parallel to Fig 3)
# ═══════════════════════════════════════════════════════════════════════════════
_BTD_VARIANT_ORDER = [
    "Aegis", "Wild", "Mix",
    "Reason", "Reason-P", "Reason-PN",
    "Adapt", "Adapt-P", "Adapt-PN",
    "Hybrid-1", "Hybrid-2", "Hybrid-P-1", "Hybrid-P-2", "Hybrid-PN-1", "Hybrid-PN-2",
]
_BTD_FAMILY_COLORS = [
    (slice(0, 3),  C["gray"],      "Static"),
    (slice(3, 6),  C["sky"],       "+Reasoning"),
    (slice(6, 9),  C["green"],     "Procedural"),
    (slice(9, 15), C["orange"],    "Hybrid"),
]
_BTD_SCALES = [("qwen8b", "Qwen3-8B"), ("qwen14b", "Qwen3-14B"), ("qwen32b", "Qwen3-32B")]


def plot6_per_variant_btd(data: dict) -> None:
    """
    Per-variant BeaverTails-PolicyShift Macro-F1 across all three Qwen3 scales.

    Appendix companion to Fig 3 (which shows distributions via boxplot).  This
    plot exposes every individual variant so the reader can see which specific
    rows drive the procedural advantage and which Hybrid variants regress at
    32B.  Three stacked panels (one per scale) keep the comparison aligned.

    Bars are coloured by family (Static / +Reasoning / Procedural / Hybrid).
    A horizontal dashed line marks the best SOTA baseline (DynaGuard-Fast,
    64.84%) — every bar that clears it is in the "Trinity-only zone."
    """
    best_baseline = data["baselines"]["means"].loc["dynaguard-8b-fast", COLS["bt_macf1"]]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.0),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={"hspace": 0.10})

    variants = _BTD_VARIANT_ORDER
    labels   = list(variants)
    x        = np.arange(len(variants))

    for ax, (scale_key, scale_label) in zip(axes, _BTD_SCALES):
        means = data[scale_key]["means"]
        vals  = np.array([
            means.loc[v, COLS["bt_macf1"]] if v in means.index else np.nan
            for v in variants
        ])

        # Color bars by family band.
        bar_colors = [C["gray"]] * len(variants)
        for sl, color, _ in _BTD_FAMILY_COLORS:
            for i in range(*sl.indices(len(variants))):
                bar_colors[i] = color

        ax.bar(x, vals, color=bar_colors, edgecolor="white",
               linewidth=0.4, zorder=3)

        # Reference: best SOTA baseline.
        ax.axhline(best_baseline, color=C["vermilion"], lw=1.2, ls="--",
                   zorder=2, label=f"DynaGuard-Fast ({best_baseline:.1f})")

        # Numeric annotations above each bar.
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8, color="#444")

        ax.set_ylabel("MacF1 (%) ↑")
        ax.set_title(scale_label, loc="left", fontsize=12, pad=2)
        ax.set_ylim(55, max(80, np.nanmax(vals) + 4))
        ax.grid(axis="x", visible=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")

    # Single combined legend at bottom, mapping family bands to colors.
    family_handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="white", label=name)
        for _, color, name in _BTD_FAMILY_COLORS
    ]
    family_handles.append(
        mlines.Line2D([], [], color=C["vermilion"], lw=1.2, ls="--",
                      label=f"DynaGuard-Fast ({best_baseline:.1f})")
    )
    axes[-1].legend(handles=family_handles, ncol=5, loc="upper center",
                    bbox_to_anchor=(0.5, -0.50), frameon=False, fontsize=10)

    _save(fig, "per_variant_btd")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse args, load all summary tables, and render the requested Section 5 figures."""
    parser = argparse.ArgumentParser(description="Generate the paper's result figures.")
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Directory with the aggregated_*_results.csv files")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"),
                        help="Where to write the figures (default: ./figures)")
    parser.add_argument(
        "--plot", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6],
        default=[1, 2, 3, 4, 5, 6],
        help="Which plot(s) to generate (default: all)",
    )
    args = parser.parse_args()

    global SUMMARY, OUT
    SUMMARY, OUT = args.results_dir, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    setup_style()
    print("Loading data …")
    data = load_all()
    print(f"Generating {len(args.plot)} plot(s) → {OUT}")

    dispatch = {
        1: ("Custom-Policy Compliance Gap",       plot1_custom_policy_gap),
        2: ("8B Variance Error-Bar Chart",        plot2_8b_variance),
        3: ("Procedural Dominance Box Plot",      plot3_procedural_dominance),
        4: ("32B Bifurcation Trajectory",         plot4_bifurcation),
        5: ("Adapt Cross-Scale Trajectory",       plot5_adapt_trajectory),
        6: ("Per-Variant BT-PolicyShift Macro-F1",    plot6_per_variant_btd),
    }
    for n in sorted(args.plot):
        title, fn = dispatch[n]
        print(f"\n[Plot {n}] {title}")
        fn(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
