"""
gwas.py — single-SNP GWAS using OLS regression

Model per SNP j:
    y = μ + b_cov · X_cov + α_j · x_j + ε

where x_j is the allele dosage (0/1/2) at SNP j, and X_cov contains
fixed covariates (generation, sex).

The allele-substitution effect α_j and its SE are extracted from OLS.
The Wald statistic  T = α̂_j / SE(α̂_j)  follows t(df) under H₀.

Multiple-testing corrections
-----------------------------
1. Bonferroni:  α_bonf = 0.05 / m
2. SimpleM (Gao et al. 2008): effective number of independent tests
   M_eff estimated from eigenvalue decomposition of the LD (correlation)
   matrix among SNPs. Threshold: α_simpleM = 0.05 / M_eff.

Genomic inflation factor
--------------------------
    λ_GC = median(χ²_obs) / 0.4549    (median of χ²(1) = 0.4549)

λ > 1.05 may indicate population stratification or model mis-specification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional


# ── public API ────────────────────────────────────────────────────────────────

def run_gwas(
    pheno: pd.DataFrame,
    geno:  pd.DataFrame,
    snpmap: pd.DataFrame,
    covariate_cols: List[str] = None,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """
    Run single-SNP OLS GWAS.

    Parameters
    ----------
    pheno          : DataFrame with columns [animal_id, phenotype, <covariates>]
    geno           : DataFrame with animal_id + one column per SNP (dosage 0/1/2)
    snpmap         : DataFrame with [snp_id, chr, pos_mb]
    covariate_cols : list of column names in pheno to include as fixed effects
    chunk_size     : process SNPs in batches (memory efficiency)

    Returns
    -------
    DataFrame with columns [snp_id, chr, pos_mb, alpha_hat, se, t_stat,
                             p_value, p_bonferroni, neg_log10_p]
    """
    covariate_cols = covariate_cols or []

    # Merge phenotype + covariates
    merge_cols = ["animal_id", "phenotype"] + covariate_cols
    df = pheno[merge_cols].merge(geno, on="animal_id").dropna(subset=["phenotype"])

    y = df["phenotype"].values.astype(float)
    n = len(y)

    # Design matrix for covariates (intercept + covs)
    X_cov = np.column_stack([
        np.ones(n),
        *[df[c].values.astype(float) for c in covariate_cols],
    ])  # shape (n, k)

    # Project y onto covariate space — regress out fixed effects
    # y_res = y - X_cov @ (X_cov'X_cov)^{-1} X_cov' y
    XtXinv = np.linalg.pinv(X_cov.T @ X_cov)
    H_cov  = X_cov @ XtXinv @ X_cov.T          # hat matrix for covariates
    y_res  = y - H_cov @ y                       # residualized phenotype

    snp_ids = snpmap["snp_id"].tolist()
    results = []

    for i, snp in enumerate(snp_ids):
        if snp not in df.columns:
            continue
        x = df[snp].values.astype(float)
        x_res = x - H_cov @ x      # residualize genotype too (Frisch-Waugh)

        ss_x = x_res @ x_res
        if ss_x < 1e-10:           # monomorphic SNP
            results.append(_null_row(snp))
            continue

        alpha_hat = (x_res @ y_res) / ss_x
        y_fitted  = alpha_hat * x_res
        rss       = np.sum((y_res - y_fitted) ** 2)
        df_res    = n - X_cov.shape[1] - 1
        sigma2    = rss / df_res
        se        = np.sqrt(sigma2 / ss_x)
        t_stat    = alpha_hat / se
        p_val     = 2 * stats.t.sf(abs(t_stat), df=df_res)

        results.append({
            "snp_id":    snp,
            "alpha_hat": alpha_hat,
            "se":        se,
            "t_stat":    t_stat,
            "p_value":   p_val,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(snpmap[["snp_id", "chr", "pos_mb", "is_qtl"]],
                                  on="snp_id", how="left")

    # Multiple testing
    m = len(results_df)
    results_df["p_bonferroni"] = results_df["p_value"] * m   # Bonferroni-adjusted p

    # Genomic inflation
    chi2_obs = stats.chi2.ppf(1 - results_df["p_value"].clip(1e-300, 1.0), df=1)
    lambda_gc = np.median(chi2_obs) / 0.4549
    results_df["lambda_gc"] = round(lambda_gc, 4)

    results_df["neg_log10_p"] = -np.log10(results_df["p_value"].clip(1e-300))
    results_df = results_df.sort_values(["chr", "pos_mb"]).reset_index(drop=True)

    print(f"  λ_GC = {lambda_gc:.3f}")
    return results_df


# ── helpers ───────────────────────────────────────────────────────────────────

def _null_row(snp_id: str) -> dict:
    return {
        "snp_id": snp_id, "alpha_hat": np.nan, "se": np.nan,
        "t_stat": np.nan, "p_value": 1.0,
    }


def simpleM_threshold(geno: pd.DataFrame, snp_ids: List[str],
                       alpha: float = 0.05) -> float:
    """
    Estimate SimpleM effective number of tests from eigenvalues of LD matrix.

    M_eff = #{eigenvalues λ_k : λ_k ≥ 1}  (Gao et al. 2008)

    Returns the SimpleM-corrected significance threshold.
    """
    X = geno[snp_ids].values.astype(float)
    # Standardise columns
    col_std = X.std(axis=0)
    col_std[col_std == 0] = 1
    X_std = (X - X.mean(axis=0)) / col_std
    R = np.corrcoef(X_std.T)        # LD (correlation) matrix
    eigenvalues = np.linalg.eigvalsh(R)
    M_eff = int(np.sum(eigenvalues >= 1))
    threshold = alpha / M_eff
    print(f"  SimpleM: M_eff = {M_eff}  threshold = {threshold:.2e}")
    return threshold
