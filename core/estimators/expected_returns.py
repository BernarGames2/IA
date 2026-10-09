"""Expected-return estimators with a common interface and out-of-sample evaluation.

All estimators return ANNUALISED ARITHMETIC means of simple returns (the
quantity entering ``w' mu`` in mean-variance optimisation). Sample means are very
noisy; shrinkage estimators exist to reduce estimation error, not to raise the
forecast. Selection must use out-of-sample error (see :func:`evaluate_oos`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from utils.validation import InsufficientDataError

Estimator = Callable[[pd.DataFrame, int], pd.Series]


def historical_mean(r: pd.DataFrame, periods: int = 252) -> pd.Series:
    if len(r) < 2:
        raise InsufficientDataError("need >= 2 observations")
    return r.mean() * periods


def ewma_mean(r: pd.DataFrame, periods: int = 252, halflife: float = 126.0) -> pd.Series:
    if len(r) < 2:
        raise InsufficientDataError("need >= 2 observations")
    return r.ewm(halflife=halflife, adjust=True).mean().iloc[-1] * periods


def grand_mean(r: pd.DataFrame, periods: int = 252) -> pd.Series:
    """Baseline: the same mean for every asset (cross-sectional average)."""
    m = float(r.mean().mean()) * periods
    return pd.Series(m, index=r.columns)


def james_stein_mean(r: pd.DataFrame, periods: int = 252) -> pd.Series:
    """Bayes-Stein shrinkage (Jorion, 1986) toward the minimum-variance portfolio mean.

    ``mu_bs = (1-d) mu_hat + d mu_g 1`` with
    ``d = (N+2) / ((N+2) + T (mu_hat - mu_g 1)' S^{-1} (mu_hat - mu_g 1))``.
    """
    t, n = r.shape
    if t <= n + 2:
        raise InsufficientDataError("James-Stein needs T > N + 2")
    mu = r.mean().to_numpy()
    s = np.cov(r.to_numpy(), rowvar=False, ddof=1) * (t - 1) / (t - n - 2)
    s_inv = np.linalg.pinv(s)
    ones = np.ones(n)
    w_g = s_inv @ ones / (ones @ s_inv @ ones)
    mu_g = float(w_g @ mu)
    d_vec = mu - mu_g
    q = float(t * d_vec @ s_inv @ d_vec)
    delta = float(np.clip((n + 2) / ((n + 2) + q), 0.0, 1.0))
    out = pd.Series(((1 - delta) * mu + delta * mu_g) * periods, index=r.columns)
    out.attrs["shrinkage_intensity"] = delta
    out.attrs["shrinkage_target_annual"] = mu_g * periods
    return out


ESTIMATORS: dict[str, Estimator] = {
    "historical": historical_mean,
    "ewma": ewma_mean,
    "james_stein": james_stein_mean,
    "grand_mean": grand_mean,
}


def estimate(r: pd.DataFrame, method: str, periods: int = 252) -> pd.Series:
    if method not in ESTIMATORS:
        raise ValueError(f"unknown expected-return method {method!r}; options {list(ESTIMATORS)}")
    return ESTIMATORS[method](r, periods)


def evaluate_oos(r: pd.DataFrame, lookback: int = 252, horizon: int = 63, step: int = 63,
                 periods: int = 252, methods: list[str] | None = None) -> pd.DataFrame:
    """Rolling out-of-sample evaluation of expected-return estimators.

    At each split t the estimator sees ``r[t-lookback:t]`` only; the target is the
    realised annualised mean over ``r[t:t+horizon]``. Reports mean squared error,
    its standard error across splits, and the average cross-sectional rank IC.
    """
    methods = methods or list(ESTIMATORS)
    rows: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    for t in range(lookback, len(r) - horizon + 1, step):
        train, test = r.iloc[t - lookback:t], r.iloc[t:t + horizon]
        realised = test.mean() * periods
        for m in methods:
            try:
                f = estimate(train, m, periods)
            except InsufficientDataError:
                continue
            err = float(((f - realised) ** 2).mean())
            ic = float(f.rank().corr(realised.rank())) if f.nunique() > 1 else float("nan")
            rows[m].append((err, ic))
    out = []
    for m, v in rows.items():
        if not v:
            continue
        e = np.array([x[0] for x in v])
        ic = np.array([x[1] for x in v])
        out.append({"method": m, "n_splits": len(v), "mse": e.mean(),
                    "mse_se": e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan"),
                    "rank_ic_mean": float(np.nanmean(ic)) if np.isfinite(ic).any() else float("nan")})
    return pd.DataFrame(out).set_index("method").sort_values("mse") if out else pd.DataFrame()
