"""
plots.py — publication-quality figures for GWAS and genetic trend analysis

Figures produced
----------------
1. Manhattan plot     : -log₁₀(p) vs chromosomal position, coloured by chromosome
2. QQ plot            : observed vs expected -log₁₀(p) with 95% confidence band
3. Genetic trend plot : generation mean phenotype + LOESS smoother

All figures exported at 300 dpi (suitable for journal submission).
Colour palette is colour-blind safe (Wong 2011, Nat Methods 8:441).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2
from pathlib import Path


# ── Colour-blind-safe palette (Wong 2011) ─────────────────────────────────────
CB_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky-blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

FIGURE_DPI    = 300
FONT_FAMILY   = "DejaVu Sans"
AXIS_FONT_SZ  = 9
TITLE_FONT_SZ = 10


def _base_style():
    plt.rcParams.update({
        "font.family":       FONT_FAMILY,
        "axes.titlesize":    TITLE_FONT_SZ,
        "axes.labelsize":    AXIS_FONT_SZ,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        FIGURE_DPI,
    })


# ── 1. Manhattan plot ─────────────────────────────────────────────────────────

def manhattan_plot(
    gwas: pd.DataFrame,
    out_path: str = "outputs/manhattan.png",
    bonferroni_alpha: float = 0.05,
    suggestive_alpha: float = 1e-4,
) -> None:
    """
    Manhattan plot with Bonferroni and suggestive significance lines.

    Parameters
    ----------
    gwas           : DataFrame from run_gwas() with [chr, pos_mb, neg_log10_p,
                     p_value, is_qtl]
    bonferroni_alpha : genome-wide significance level before correction
    """
    _base_style()
    df = gwas.dropna(subset=["neg_log10_p"]).copy()

    # Compute cumulative position across chromosomes
    chrs = sorted(df["chr"].unique())
    offsets = {}
    cumpos = 0
    for c in chrs:
        offsets[c] = cumpos
        cumpos += df[df["chr"] == c]["pos_mb"].max() + 10   # 10 Mb gap

    df["cumpos"] = df.apply(lambda r: offsets[r["chr"]] + r["pos_mb"], axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    chr_colors = [CB_PALETTE[i % 2] for i in range(len(chrs))]

    for idx, c in enumerate(chrs):
        sub = df[df["chr"] == c]
        ax.scatter(sub["cumpos"], sub["neg_log10_p"],
                   c=chr_colors[idx], s=4, alpha=0.75, linewidths=0, rasterized=True)

    # Highlight true QTL (if column present)
    if "is_qtl" in df.columns:
        qtl = df[df["is_qtl"] == True]
        ax.scatter(qtl["cumpos"], qtl["neg_log10_p"],
                   c=CB_PALETTE[5], s=18, marker="^", zorder=5,
                   label="True QTL", linewidths=0.3, edgecolors="k")
        ax.legend(fontsize=7, frameon=False)

    # Significance lines
    m = len(df)
    bonf_thresh  = -np.log10(bonferroni_alpha / m)
    sugg_thresh  = -np.log10(suggestive_alpha)
    ax.axhline(bonf_thresh, color="firebrick", lw=0.8, ls="--",
               label=f"Bonferroni (p={bonferroni_alpha/m:.1e})")
    ax.axhline(sugg_thresh, color="goldenrod", lw=0.8, ls=":",
               label=f"Suggestive (p={suggestive_alpha:.0e})")

    # Chromosome tick labels at midpoint
    xticks  = [offsets[c] + (df[df["chr"] == c]["pos_mb"].max() / 2) for c in chrs]
    xlabels = [str(c) for c in chrs]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_xlabel("Chromosome", labelpad=4)
    ax.set_ylabel(r"$-\log_{10}(p)$", labelpad=4)
    ax.set_title("GWAS Manhattan plot", fontweight="bold", pad=6)
    ax.set_xlim(0, cumpos)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 2. QQ plot ────────────────────────────────────────────────────────────────

def qq_plot(
    p_values: pd.Series,
    out_path: str = "outputs/qqplot.png",
) -> None:
    """
    Observed vs expected QQ plot with 95% pointwise confidence band.

    The confidence band (Casella & Berger 2002) is:
        Beta(α/2; k, m−k+1) ≤ U_(k) ≤ Beta(1−α/2; k, m−k+1)
    where U_(k) is the k-th order statistic of Uniform(0,1) p-values.
    """
    from scipy.stats import beta as beta_dist

    _base_style()
    p = p_values.dropna().clip(1e-300).sort_values().values
    m = len(p)
    expected = -np.log10(np.arange(1, m + 1) / m)
    observed = -np.log10(p[::-1])   # largest first

    # 95% CI via Beta distribution order statistics
    k  = np.arange(1, m + 1)
    lo = -np.log10(beta_dist.ppf(0.975, k, m - k + 1))
    hi = -np.log10(beta_dist.ppf(0.025, k, m - k + 1))
    lo = lo[::-1]
    hi = hi[::-1]

    # Genomic inflation λ
    chi2_obs = chi2.ppf(1 - p, df=1)
    lambda_gc = np.median(chi2_obs) / 0.4549

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.fill_between(expected, lo, hi, alpha=0.2, color=CB_PALETTE[0], label="95% CI")
    ax.plot(expected, observed, "o", color=CB_PALETTE[0],
            markersize=2, alpha=0.7, rasterized=True)
    ax.plot([0, expected.max()], [0, expected.max()],
            color="k", lw=0.8, ls="--")
    ax.text(0.05, 0.92, f"λ_GC = {lambda_gc:.3f}",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$", labelpad=4)
    ax.set_ylabel(r"Observed $-\log_{10}(p)$", labelpad=4)
    ax.set_title("QQ plot", fontweight="bold", pad=6)
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 3. Genetic trend plot ─────────────────────────────────────────────────────

def genetic_trend_plot(
    pheno: pd.DataFrame,
    out_path: str = "outputs/genetic_trend.png",
) -> None:
    """
    Generation mean phenotype (± SEM) with LOESS smoother.

    Shows separately for males and females to reveal sex-specific trends,
    relevant for BM42m / BM42f analyses.
    """
    from scipy.ndimage import uniform_filter1d

    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    sex_labels = {0: "Female", 1: "Male"}
    colors     = {0: CB_PALETTE[2], 1: CB_PALETTE[0]}

    for ax, (sex, label) in zip(axes, sex_labels.items()):
        sub = pheno[pheno["sex"] == sex]
        gen_stats = (sub.groupby("generation")["phenotype"]
                       .agg(["mean", "sem", "count"])
                       .reset_index())

        ax.fill_between(gen_stats["generation"],
                        gen_stats["mean"] - 1.96 * gen_stats["sem"],
                        gen_stats["mean"] + 1.96 * gen_stats["sem"],
                        alpha=0.2, color=colors[sex])
        ax.plot(gen_stats["generation"], gen_stats["mean"],
                "o-", color=colors[sex], ms=4, lw=1.2, label="Observed mean")

        # LOESS smoother (uniform filter as lightweight approximation)
        if len(gen_stats) > 3:
            smooth = uniform_filter1d(gen_stats["mean"].values, size=3)
            ax.plot(gen_stats["generation"], smooth,
                    "--", color="k", lw=1, alpha=0.6, label="Smoothed")

        # Breeding value trend (if available)
        if "true_bv" in pheno.columns:
            bv_stats = (sub.groupby("generation")["true_bv"]
                          .mean().reset_index())
            ax2 = ax.twinx()
            ax2.plot(bv_stats["generation"], bv_stats["true_bv"],
                     color=CB_PALETTE[5], lw=1, alpha=0.7, ls="-.")
            ax2.set_ylabel("Mean true BV", fontsize=8, color=CB_PALETTE[5])
            ax2.tick_params(axis="y", labelcolor=CB_PALETTE[5], labelsize=7)
            ax2.spines["top"].set_visible(False)

        ax.set_xlabel("Generation", labelpad=4)
        ax.set_ylabel("Mean phenotype", labelpad=4)
        ax.set_title(label, fontweight="bold", pad=5)
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Genetic trend across selection generations",
                 fontweight="bold", fontsize=TITLE_FONT_SZ + 1, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
