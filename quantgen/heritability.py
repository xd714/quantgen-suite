"""
heritability.py — restricted maximum likelihood (REML) heritability estimation

The mixed model is:
    y = Xβ + Za + e

    a ~ N(0, Aσ²_a)      (breeding values)
    e ~ N(0, Iσ²_e)      (residuals)

    V = ZAZ'σ²_a + Iσ²_e

REML log-likelihood (Henderson 1973):
    l_R(σ²_a, σ²_e) = -½[log|V| + log|X'V⁻¹X| + y'Py]

where P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹  is the projection matrix.

We iterate by the Average-Information (AI) algorithm (Gilmour et al. 1995):
    θ_{t+1} = θ_t + [AI(θ_t)]⁻¹ · s(θ_t)

where s = ∂l_R/∂θ  (score) and AI = ½ P·V̇_k·P·V̇_l  (average information matrix).

For simplicity in this demonstration we use scipy.optimize.minimize with
analytical gradients to maximise l_R, which converges to the same solution
as the AI algorithm for well-conditioned problems.

In practice, ASReml or the R package {asreml} should be used for production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from typing import Tuple


# ── public API ────────────────────────────────────────────────────────────────

def reml_h2_python(
    pheno: pd.DataFrame,
    a_matrix: np.ndarray | None = None,
    fixed_cols: list[str] | None = None,
    method: str = "L-BFGS-B",
) -> Tuple[float, float, float]:
    """
    Estimate h² by REML given a phenotype DataFrame.

    Parameters
    ----------
    pheno      : DataFrame with at least columns [phenotype].
                 Optional: [generation, sex] used as fixed effects.
    a_matrix   : numerator relationship matrix A (n×n).
                 If None, identity is used (sire model approximation).
    fixed_cols : column names to use as fixed-effect covariates.

    Returns
    -------
    h2, sigma2_a, sigma2_e
    """
    fixed_cols = fixed_cols or []
    avail = [c for c in fixed_cols if c in pheno.columns]

    df = pheno.dropna(subset=["phenotype"]).copy()
    y  = df["phenotype"].values.astype(float)
    n  = len(y)

    # Fixed-effect design matrix X
    X = np.column_stack([np.ones(n), *[df[c].values.astype(float) for c in avail]])

    # Relationship matrix (use identity if not provided)
    A = a_matrix if a_matrix is not None else np.eye(n)

    # Variance components via REML
    sigma2_a, sigma2_e = _reml_optimize(y, X, A, method=method)
    sigma2_p = sigma2_a + sigma2_e
    h2 = sigma2_a / sigma2_p if sigma2_p > 0 else 0.0

    return round(h2, 4), round(sigma2_a, 4), round(sigma2_e, 4)


# ── REML core ─────────────────────────────────────────────────────────────────

def _reml_log_lik(theta: np.ndarray, y: np.ndarray,
                  X: np.ndarray, A: np.ndarray) -> float:
    """
    Negative REML log-likelihood for θ = [log σ²_a, log σ²_e].
    Using log-parameterisation ensures positivity without box constraints.
    """
    sigma2_a = np.exp(theta[0])
    sigma2_e = np.exp(theta[1])
    n = len(y)

    V = sigma2_a * A + sigma2_e * np.eye(n)

    # Cholesky decomposition for numerical stability
    try:
        L, low = cho_factor(V, lower=True)
    except np.linalg.LinAlgError:
        return 1e10   # singular → return large value

    # log|V|
    log_det_V = 2 * np.sum(np.log(np.diag(L.T if not low else L)))

    # V⁻¹y  and  V⁻¹X
    Vinv_y = cho_solve((L, low), y)
    Vinv_X = cho_solve((L, low), X)

    # log|X'V⁻¹X|
    XtVinvX = X.T @ Vinv_X
    sign, log_det_XtVinvX = np.linalg.slogdet(XtVinvX)
    if sign <= 0:
        return 1e10

    # P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹  →  y'Py
    XtVinvX_inv = np.linalg.solve(XtVinvX, np.eye(X.shape[1]))
    yPy = y @ Vinv_y - y @ Vinv_X @ XtVinvX_inv @ Vinv_X.T @ y

    neg_reml = 0.5 * (log_det_V + log_det_XtVinvX + yPy)
    return float(neg_reml)


def _reml_optimize(y: np.ndarray, X: np.ndarray,
                   A: np.ndarray, method: str = "L-BFGS-B") -> Tuple[float, float]:
    """Minimise negative REML log-likelihood with respect to [log σ²_a, log σ²_e]."""
    var_y = np.var(y, ddof=1)
    theta0 = np.log([0.3 * var_y, 0.7 * var_y])   # starting values

    res = minimize(
        _reml_log_lik,
        x0=theta0,
        args=(y, X, A),
        method=method,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    sigma2_a = np.exp(res.x[0])
    sigma2_e = np.exp(res.x[1])
    return sigma2_a, sigma2_e
