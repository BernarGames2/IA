"""Pure, tested performance and risk metrics.

Conventions (see docs/architecture.md, "Convenções matemáticas"):
* simple return ``r_t = P_t/P_{t-1} - 1``; log return ``log(P_t/P_{t-1})``.
* arithmetic annualised mean ``= periods * mean(r)``; geometric (CAGR)
  ``= prod(1+r)^(periods/N) - 1``. They differ by roughly ``sigma^2/2``.
* volatility ``= std(r, ddof=1) * sqrt(periods)`` -- valid under i.i.d.
  returns; the i.i.d. assumption is a documented simplification.
* risk-free rate: annual, compounded to the return frequency with
  ``(1+rf)^(1/periods) - 1``; it must be in the same currency as the returns.
* Undefined metrics (insufficient sample, zero denominator) raise
  :class:`UndefinedMetricError` with an explicit reason instead of returning junk.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.validation import InsufficientDataError, ValidationError, as_1d_float


class UndefinedMetricError(ValueError):
    """A metric is mathematically undefined for the given input."""


ArrayLike = NDArray[np.floating] | pd.Series | list[float]


# --------------------------------------------------------------------------- returns
def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns without filling gaps (a missing price yields a missing return)."""
    if (prices.to_numpy() <= 0).any():
        raise ValidationError("prices must be strictly positive")
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if (prices.to_numpy() <= 0).any():
        raise ValidationError("prices must be strictly positive")
    return np.log(prices / prices.shift(1)).iloc[1:]


def rf_per_period(rf_annual: float, periods: int = 252) -> float:
    """Convert an annual effective rate to the per-period effective rate."""
    if rf_annual <= -1:
        raise ValidationError("annual rate must be > -100%")
    return (1.0 + rf_annual) ** (1.0 / periods) - 1.0


def aggregate_returns(r: ArrayLike, h: int) -> NDArray[np.float64]:
    """Compound simple returns into NON-overlapping ``h``-period returns."""
    x = as_1d_float(r, "returns")
    if h < 1:
        raise ValidationError("h must be >= 1")
    n = (len(x) // h) * h
    if n == 0:
        raise InsufficientDataError(f"need at least {h} observations to aggregate")
    return np.prod(1.0 + x[:n].reshape(-1, h), axis=1) - 1.0


# ----------------------------------------------------------------- annualisation
def annualized_arithmetic_mean(r: ArrayLike, periods: int = 252) -> float:
    x = as_1d_float(r, "returns")
    return float(periods * x.mean())


def annualized_geometric_return(r: ArrayLike, periods: int = 252) -> float:
    x = as_1d_float(r, "returns")
    if np.any(x <= -1):
        return -1.0
    return float(np.exp(np.log1p(x).sum() * periods / len(x)) - 1.0)


def total_return(r: ArrayLike) -> float:
    x = as_1d_float(r, "returns")
    return float(np.prod(1.0 + x) - 1.0)


def annualized_volatility(r: ArrayLike, periods: int = 252) -> float:
    x = as_1d_float(r, "returns")
    if len(x) < 2:
        raise UndefinedMetricError("volatility needs at least 2 observations")
    return float(np.std(x, ddof=1) * np.sqrt(periods))


def downside_deviation(r: ArrayLike, mar_per_period: float = 0.0, periods: int = 252) -> float:
    """sqrt(mean(min(r - MAR, 0)^2)) * sqrt(periods), averaged over ALL observations."""
    x = as_1d_float(r, "returns")
    d = np.minimum(x - mar_per_period, 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(periods))


# ------------------------------------------------------------------ risk-adjusted
def sharpe_ratio(r: ArrayLike, rf_annual: float, periods: int = 252) -> float:
    """Annualised Sharpe = mean(r - rf_p)/std(r - rf_p) * sqrt(periods)."""
    x = as_1d_float(r, "returns")
    if len(x) < 2:
        raise UndefinedMetricError("Sharpe needs at least 2 observations")
    ex = x - rf_per_period(rf_annual, periods)
    sd = np.std(ex, ddof=1)
    if sd <= 1e-15:
        raise UndefinedMetricError("Sharpe undefined: zero volatility of excess returns")
    return float(ex.mean() / sd * np.sqrt(periods))


def sortino_ratio(r: ArrayLike, rf_annual: float, periods: int = 252) -> float:
    """Annualised mean excess return over annualised downside deviation (MAR = rf)."""
    x = as_1d_float(r, "returns")
    rf_p = rf_per_period(rf_annual, periods)
    dd = downside_deviation(x, rf_p, periods)
    if dd <= 1e-15:
        raise UndefinedMetricError("Sortino undefined: no returns below the MAR")
    return float((x.mean() - rf_p) * periods / dd)


def beta(r_asset: ArrayLike, r_bench: ArrayLike) -> float:
    a = as_1d_float(r_asset, "asset returns")
    b = as_1d_float(r_bench, "benchmark returns")
    if len(a) != len(b):
        raise ValidationError("asset and benchmark returns must be aligned (same length)")
    if len(a) < 3:
        raise UndefinedMetricError("beta needs at least 3 observations")
    vb = np.var(b, ddof=1)
    if vb <= 1e-18:
        raise UndefinedMetricError("beta undefined: benchmark has zero variance")
    return float(np.cov(a, b, ddof=1)[0, 1] / vb)


# -------------------------------------------------------------------- drawdowns
def wealth_index(r: ArrayLike, start: float = 1.0) -> NDArray[np.float64]:
    """Wealth path including the initial value (length N+1)."""
    x = as_1d_float(r, "returns")
    return start * np.concatenate([[1.0], np.cumprod(1.0 + x)])


def drawdown_series(r: ArrayLike) -> NDArray[np.float64]:
    """Drawdown (<= 0) of the wealth path including the starting point."""
    w = wealth_index(r)
    peak = np.maximum.accumulate(w)
    return w / peak - 1.0


def max_drawdown(r: ArrayLike) -> float:
    """Most negative drawdown (e.g. -0.25 means a 25% peak-to-trough loss)."""
    return float(drawdown_series(r).min())


def max_drawdown_from_wealth(w: NDArray[np.floating]) -> NDArray[np.float64] | float:
    """Max drawdown of wealth path(s); last axis is time."""
    arr = np.asarray(w, dtype=float)
    peak = np.maximum.accumulate(arr, axis=-1)
    dd = (arr / peak - 1.0).min(axis=-1)
    return float(dd) if np.ndim(dd) == 0 else dd


def max_drawdown_duration(r: ArrayLike) -> int:
    """Longest number of periods spent below a previous peak."""
    dd = drawdown_series(r)
    best = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def calmar_ratio(r: ArrayLike, periods: int = 252) -> float:
    mdd = max_drawdown(r)
    if mdd >= 0:
        raise UndefinedMetricError("Calmar undefined: no drawdown in sample")
    return annualized_geometric_return(r, periods) / abs(mdd)


def recovery_required(loss: float) -> float:
    """Gain needed to recover a fractional loss d in [0,1): 1/(1-d) - 1."""
    if not 0.0 <= loss < 1.0:
        raise ValidationError("loss must satisfy 0 <= d < 1 (a 100% loss is unrecoverable)")
    return 1.0 / (1.0 - loss) - 1.0


# --------------------------------------------------------------------- portfolio
def portfolio_return(w: ArrayLike, mu: ArrayLike) -> float:
    return float(np.asarray(w, float) @ np.asarray(mu, float))


def portfolio_volatility(w: ArrayLike, cov: NDArray[np.floating]) -> float:
    wv = np.asarray(w, float)
    var = float(wv @ np.asarray(cov, float) @ wv)
    if var < -1e-12:
        raise ValidationError("negative portfolio variance: covariance is not PSD")
    return float(np.sqrt(max(var, 0.0)))


def risk_contributions(w: ArrayLike, cov: NDArray[np.floating],
                       symbols: list[str] | None = None) -> pd.DataFrame:
    """Marginal (MRC_i = (Sigma w)_i / sigma_p) and total (RC_i = w_i MRC_i) contributions.

    ``sum(RC_i) == sigma_p`` (Euler decomposition, volatility is 1-homogeneous).
    """
    wv = np.asarray(w, float)
    c = np.asarray(cov, float)
    sp = portfolio_volatility(wv, c)
    if sp <= 1e-15:
        raise UndefinedMetricError("risk contributions undefined: portfolio volatility is zero")
    mrc = c @ wv / sp
    rc = wv * mrc
    idx = symbols if symbols is not None else [f"a{i}" for i in range(len(wv))]
    return pd.DataFrame({"weight": wv, "mrc": mrc, "rc": rc, "rc_pct": rc / sp}, index=idx)


def turnover(w_old: ArrayLike, w_new: ArrayLike) -> float:
    """sum |w_new - w_old| (two-way turnover)."""
    return float(np.abs(np.asarray(w_new, float) - np.asarray(w_old, float)).sum())


def transaction_cost(w_old: ArrayLike, w_new: ArrayLike, cost_bps: float) -> float:
    """Proportional cost as a fraction of portfolio value: turnover * bps/1e4."""
    return turnover(w_old, w_new) * cost_bps / 1e4


def concentration(w: ArrayLike, groups: Mapping[str, str] | None = None,
                  symbols: list[str] | None = None) -> dict[str, object]:
    """HHI, effective number of assets, max weight, and per-group exposures."""
    wv = np.asarray(w, float)
    hhi = float(np.sum(wv ** 2))
    out: dict[str, object] = {"hhi": hhi, "effective_n": 1.0 / hhi if hhi > 0 else float("nan"),
                              "max_weight": float(wv.max()), "n_nonzero": int(np.sum(wv > 1e-6))}
    if groups is not None and symbols is not None:
        exp: dict[str, float] = {}
        for s, x in zip(symbols, wv):
            g = groups.get(s, "unknown")
            exp[g] = exp.get(g, 0.0) + float(x)
        out["by_group"] = exp
    return out


def diversification_ratio(w: ArrayLike, cov: NDArray[np.floating]) -> float:
    wv = np.asarray(w, float)
    sd = np.sqrt(np.diag(np.asarray(cov, float)))
    sp = portfolio_volatility(wv, cov)
    if sp <= 1e-15:
        raise UndefinedMetricError("diversification ratio undefined: zero volatility")
    return float(wv @ sd / sp)


def performance_summary(r: pd.Series | NDArray[np.floating], rf_annual: float,
                        periods: int = 252) -> dict[str, float | str | None]:
    """Collect standard metrics; undefined ones are reported with their reason."""
    x = as_1d_float(r, "returns")
    out: dict[str, float | str | None] = {"n_obs": float(len(x))}

    def _try(name: str, fn) -> None:  # noqa: ANN001
        try:
            out[name] = float(fn())
        except (UndefinedMetricError, InsufficientDataError) as e:
            out[name] = None
            out[f"{name}_undefined_reason"] = str(e)

    _try("total_return", lambda: total_return(x))
    _try("cagr", lambda: annualized_geometric_return(x, periods))
    _try("arith_mean_ann", lambda: annualized_arithmetic_mean(x, periods))
    _try("volatility_ann", lambda: annualized_volatility(x, periods))
    _try("downside_dev_ann", lambda: downside_deviation(x, rf_per_period(rf_annual, periods), periods))
    _try("sharpe", lambda: sharpe_ratio(x, rf_annual, periods))
    _try("sortino", lambda: sortino_ratio(x, rf_annual, periods))
    _try("max_drawdown", lambda: max_drawdown(x))
    _try("max_dd_duration", lambda: max_drawdown_duration(x))
    _try("calmar", lambda: calmar_ratio(x, periods))
    return out
