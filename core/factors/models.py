"""Factor analysis: statistical (PCA) factors and time-series factor regressions.

Factors here are EMPIRICAL PROXIES built from available series (e.g. a market
index ETF in the universe), not academic factor datasets (Fama-French, etc.),
which are not downloaded by this version. Regressions are descriptive (full
sample) unless a rolling window is requested; descriptive betas must not be
used as if known in the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.validation import InsufficientDataError


def pca_factors(returns: pd.DataFrame, k: int = 3) -> dict[str, pd.DataFrame | pd.Series]:
    """PCA on standardised returns: explained variance and loadings."""
    if len(returns) < 3 * returns.shape[1]:
        raise InsufficientDataError("PCA needs T >= 3N")
    z = (returns - returns.mean()) / returns.std(ddof=1)
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    w, v = np.linalg.eigh(corr)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    k = min(k, len(w))
    # sign convention: loadings sum positive
    v = v * np.sign(v.sum(axis=0, keepdims=True) + 1e-15)
    loadings = pd.DataFrame(v[:, :k], index=returns.columns, columns=[f"PC{i + 1}" for i in range(k)])
    expl = pd.Series(w[:k] / w.sum(), index=loadings.columns, name="explained_variance")
    scores = pd.DataFrame(z.to_numpy() @ v[:, :k], index=returns.index, columns=loadings.columns)
    return {"loadings": loadings, "explained": expl, "scores": scores}


def _newey_west_se(x: np.ndarray, e: np.ndarray, lag: int) -> np.ndarray:
    t = len(e)
    xe = x * e[:, None]
    s = xe.T @ xe / t
    for k in range(1, lag + 1):
        g = xe[k:].T @ xe[:-k] / t
        s += (1 - k / (lag + 1)) * (g + g.T)
    xtx_inv = np.linalg.inv(x.T @ x / t)
    cov = xtx_inv @ s @ xtx_inv / t
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def factor_regression(asset: pd.Series, factors: pd.DataFrame, periods: int = 252,
                      lag: int | None = None) -> dict[str, object]:
    """OLS r_i = a + B f + e with Newey-West (HAC) standard errors."""
    df = pd.concat([asset.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < max(60, 10 * factors.shape[1]):
        raise InsufficientDataError("too few observations for factor regression")
    y = df["y"].to_numpy()
    x = np.column_stack([np.ones(len(df)), df[factors.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ beta
    lag = lag if lag is not None else int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    se = _newey_west_se(x, e, lag)
    r2 = 1 - e.var() / y.var()
    names = ["alpha"] + list(factors.columns)
    return {"coef": dict(zip(names, beta)), "se_hac": dict(zip(names, se)),
            "t_hac": dict(zip(names, beta / np.where(se > 0, se, np.nan))),
            "alpha_annual": float(beta[0] * periods), "r2": float(r2), "n_obs": len(y),
            "resid_vol_annual": float(e.std(ddof=x.shape[1]) * np.sqrt(periods))}


def factor_exposures(returns: pd.DataFrame, factors: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    rows = {}
    for c in returns.columns:
        if c in factors.columns:
            continue
        try:
            res = factor_regression(returns[c], factors, periods)
        except InsufficientDataError:
            continue
        rows[c] = {**{f"beta_{k}": v for k, v in res["coef"].items() if k != "alpha"},
                   **{f"t_{k}": v for k, v in res["t_hac"].items() if k != "alpha"},
                   "alpha_annual": res["alpha_annual"], "t_alpha": res["t_hac"]["alpha"],
                   "r2": res["r2"]}
    return pd.DataFrame(rows).T
