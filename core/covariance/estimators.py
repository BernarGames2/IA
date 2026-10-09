"""Covariance estimators, diagnostics and out-of-sample comparison.

All estimators return ANNUALISED covariance matrices of simple returns
(per-period covariance x periods). Each result includes diagnostics (symmetry,
eigenvalues, condition number, PSD) and documents any numerical repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

from utils.numerical import MatrixDiagnostics, matrix_diagnostics, nearest_psd
from utils.validation import InsufficientDataError


@dataclass
class CovarianceResult:
    method: str
    cov: pd.DataFrame
    diagnostics: MatrixDiagnostics
    params: dict[str, float] = field(default_factory=dict)
    repair: dict[str, float | str] | None = None
    warnings: list[str] = field(default_factory=list)


def _finish(method: str, c: np.ndarray, cols: list[str], t: int, params: dict[str, float]
            ) -> CovarianceResult:
    warnings: list[str] = []
    n = len(cols)
    if t < 2 * n:
        warnings.append(f"T={t} observações para N={n} ativos (T/N={t / n:.1f}): "
                        "covariância amostral pouco confiável; prefira shrinkage")
    repair = None
    d = matrix_diagnostics(c)
    if not d.is_psd or not d.is_symmetric:
        rep = nearest_psd(c)
        repair = {"method": rep.method, "frobenius_change": rep.frobenius_change,
                  "min_eig_before": rep.min_eigenvalue_before,
                  "min_eig_after": rep.min_eigenvalue_after}
        c = rep.matrix
        warnings.append(f"matriz reparada ({rep.method}); variação Frobenius {rep.frobenius_change:.2e}")
        d = matrix_diagnostics(c)
    if d.condition_number > 1e6:
        warnings.append(f"número de condição elevado ({d.condition_number:.2e})")
    return CovarianceResult(method, pd.DataFrame(c, index=cols, columns=cols), d, params, repair, warnings)


def sample_cov(r: pd.DataFrame, periods: int = 252) -> CovarianceResult:
    if len(r) < 2:
        raise InsufficientDataError("need >= 2 observations")
    c = np.cov(r.to_numpy(), rowvar=False, ddof=1) * periods
    return _finish("sample", np.atleast_2d(c), list(r.columns), len(r), {})


def ledoit_wolf_cov(r: pd.DataFrame, periods: int = 252) -> CovarianceResult:
    if len(r) < 2:
        raise InsufficientDataError("need >= 2 observations")
    lw = LedoitWolf().fit(r.to_numpy())
    return _finish("ledoit_wolf", lw.covariance_ * periods, list(r.columns), len(r),
                   {"shrinkage": float(lw.shrinkage_)})


def oas_cov(r: pd.DataFrame, periods: int = 252) -> CovarianceResult:
    if len(r) < 2:
        raise InsufficientDataError("need >= 2 observations")
    o = OAS().fit(r.to_numpy())
    return _finish("oas", o.covariance_ * periods, list(r.columns), len(r),
                   {"shrinkage": float(o.shrinkage_)})


def lw_constant_corr_cov(r: pd.DataFrame, periods: int = 252) -> CovarianceResult:
    """Ledoit & Wolf (2004) shrinkage toward the constant-correlation target.

    Unlike the scaled-identity target, this target keeps each asset's own sample
    variance, so it does not distort heterogeneous variances (e.g. a cash-like
    asset next to crypto). Optimal intensity
    ``delta = max(0, min(1, (pi - rho) / gamma / T))`` (Ledoit-Wolf 2004, eqs. in App. B).
    """
    x = r.to_numpy(dtype=float)
    t, n = x.shape
    if t < 3 or n < 2:
        raise InsufficientDataError("need T >= 3 and N >= 2")
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    var = np.diag(s)
    if np.any(var <= 0):
        raise InsufficientDataError("zero-variance asset: constant-correlation target undefined")
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, var)
    y = x ** 2
    phi_mat = y.T @ y / t - s ** 2
    phi = phi_mat.sum()
    theta = (x ** 3).T @ x / t - var[:, None] * s
    ratio = np.outer(1.0 / sd, sd)          # sqrt(s_jj / s_ii) at [i, j]
    off = ~np.eye(n, dtype=bool)
    rho = np.trace(phi_mat) + rbar * float(np.sum((ratio * theta)[off]))
    gamma = float(np.sum((f - s) ** 2))
    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    delta = float(max(0.0, min(1.0, kappa / t)))
    c = delta * f + (1 - delta) * s
    c = c * t / (t - 1)  # report on the same (ddof=1) scale as the sample estimator
    return _finish("lw_constant_corr", c * periods, list(r.columns), t,
                   {"shrinkage": delta, "mean_correlation": float(rbar)})


def ewma_cov(r: pd.DataFrame, periods: int = 252, lam: float = 0.94) -> CovarianceResult:
    """Exponentially weighted covariance, weights proportional to lam^(age)."""
    x = r.to_numpy()
    t = len(x)
    if t < 2:
        raise InsufficientDataError("need >= 2 observations")
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()
    m = w @ x
    xc = x - m
    c = (xc * w[:, None]).T @ xc / (1.0 - np.sum(w ** 2))
    eff_n = 1.0 / np.sum(w ** 2)
    res = _finish("ewma", c * periods, list(r.columns), t, {"lambda": lam, "effective_n": eff_n})
    if eff_n < 2 * r.shape[1]:
        res.warnings.append(f"EWMA com amostra efetiva {eff_n:.0f} pequena para {r.shape[1]} ativos")
    return res


def factor_pca_cov(r: pd.DataFrame, periods: int = 252, k: int | None = None) -> CovarianceResult:
    """Statistical factor model on the correlation matrix: C = V_k L_k V_k' + D."""
    t, n = r.shape
    if t < 3:
        raise InsufficientDataError("need >= 3 observations")
    k = k if k is not None else max(1, min(3, n - 1))
    sd = r.std(ddof=1).to_numpy()
    if np.any(sd <= 0):
        raise InsufficientDataError("zero-variance asset: factor model undefined")
    corr = np.corrcoef(r.to_numpy(), rowvar=False)
    w, v = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    common = (v[:, :k] * w[:k]) @ v[:, :k].T
    resid = np.clip(1.0 - np.diag(common), 1e-8, None)
    c_model = common + np.diag(resid)
    cov = c_model * np.outer(sd, sd) * periods
    return _finish("factor_pca", cov, list(r.columns), t,
                   {"k": float(k), "explained_variance": float(w[:k].sum() / w.sum())})


ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "lw_constant_corr": lw_constant_corr_cov,
    "oas": oas_cov,
    "ewma": ewma_cov,
    "factor_pca": factor_pca_cov,
}


def estimate_cov(r: pd.DataFrame, method: str, periods: int = 252) -> CovarianceResult:
    if method not in ESTIMATORS:
        raise ValueError(f"unknown covariance method {method!r}; options {list(ESTIMATORS)}")
    return ESTIMATORS[method](r, periods)


def _gmv_weights(c: np.ndarray) -> np.ndarray:
    ones = np.ones(len(c))
    x = np.linalg.lstsq(c, ones, rcond=None)[0]
    return x / x.sum()


def evaluate_oos(r: pd.DataFrame, lookback: int = 252, horizon: int = 63, step: int = 63,
                 periods: int = 252, methods: list[str] | None = None) -> pd.DataFrame:
    """Out-of-sample comparison via realised volatility of the unconstrained GMV portfolio.

    Lower realised GMV volatility = better covariance forecast for allocation
    (Engle & Colacito, 2006 style). The Frobenius distance to the realised sample
    covariance is also reported (noisy proxy).
    """
    methods = methods or list(ESTIMATORS)
    acc: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    for t in range(lookback, len(r) - horizon + 1, step):
        train, test = r.iloc[t - lookback:t], r.iloc[t:t + horizon]
        realised = np.cov(test.to_numpy(), rowvar=False, ddof=1) * periods
        for m in methods:
            c = estimate_cov(train, m, periods).cov.to_numpy()
            w = _gmv_weights(c)
            rv = float(np.std(test.to_numpy() @ w, ddof=1) * np.sqrt(periods))
            acc[m].append((rv, float(np.linalg.norm(c - realised))))
    rows = []
    for m, v in acc.items():
        if not v:
            continue
        a = np.array(v)
        rows.append({"method": m, "n_splits": len(v), "gmv_realized_vol": a[:, 0].mean(),
                     "gmv_realized_vol_se": a[:, 0].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan,
                     "frobenius_to_realized": a[:, 1].mean()})
    return pd.DataFrame(rows).set_index("method").sort_values("gmv_realized_vol") if rows else pd.DataFrame()
