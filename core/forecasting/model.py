"""The single-asset projection model of the 'Vale a pena?' check, in closed form.

Same formulas as ``core/stock_check/analyzer.py`` (``_stats`` + ``_expected`` + ``_gbm``), written on plain arrays so
the model can be refitted at many past dates for the calibration test:

    prior  = rf + beta * ERP          beta vs the benchmark, clipped to [0, 2] (1 when unknown)
    se     = vol / sqrt(years)        standard error of the historical arithmetic mean
    lambda = tau^2 / (tau^2 + se^2)
    mu_E   = prior + lambda * (mu_hist - prior)      expected ARITHMETIC annual return [E]

Under the model's GBM with continuous drift mu_c = 252 ln(1 + mu_E / 252) and the historical log volatility sigma,
the horizon return R over h years has ln(1 + R) ~ Normal(m, s^2) with m = (mu_c - sigma^2 / 2) h, s = sigma sqrt(h),
so quantiles and probabilities are exact (no simulation noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

P = 252
MIN_BETA_OBS = 120


@dataclass(frozen=True)
class Fit:
    mu_hist: float          # historical arithmetic mean, annual [H]
    vol: float              # annual volatility of simple returns [H]
    vol_log: float          # annual volatility of log returns [H]
    years: float            # years of data used
    beta: float | None      # vs the benchmark [H]
    prior: float            # rf + beta x ERP (hipótese)
    se: float               # standard error of the mean
    lam: float              # weight of the history in the expected return
    mu_e: float             # expected arithmetic annual return [E]
    rf: float
    erp: float
    tau: float

    def log_params(self, h_years: float) -> tuple[float, float]:
        mu_c = P * np.log1p(self.mu_e / P)
        return float((mu_c - 0.5 * self.vol_log ** 2) * h_years), float(self.vol_log * np.sqrt(h_years))


def fit(r: np.ndarray, rf: float, bench_r: np.ndarray | None = None, erp: float = 0.04, tau: float = 0.10) -> Fit:
    """``r``: daily simple returns of the asset (NaN-free); ``bench_r``: the benchmark's on the same days."""
    r = np.asarray(r, float)
    n = len(r)
    if n < 2:
        raise ValueError("histórico curto demais para o modelo")
    years = n / P
    mu_h = float(r.mean() * P)
    vol = float(r.std(ddof=1) * np.sqrt(P))
    vol_log = float(np.log1p(r).std(ddof=1) * np.sqrt(P))
    beta = None
    if bench_r is not None:
        b = np.asarray(bench_r, float)
        ok = np.isfinite(b) & np.isfinite(r)
        if ok.sum() >= MIN_BETA_OBS:
            c = np.cov(r[ok], b[ok], ddof=1)
            beta = float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else None
    bu = float(np.clip(beta, 0.0, 2.0)) if beta is not None else 1.0
    prior = rf + bu * erp
    se = vol / np.sqrt(max(years, 1e-9))
    lam = tau ** 2 / (tau ** 2 + se ** 2)
    return Fit(mu_h, vol, vol_log, years, beta, prior, float(se), float(lam), float(prior + lam * (mu_h - prior)),
               rf, erp, tau)


def quantiles(m: float, s: float, qs: tuple[float, ...]) -> dict[float, float]:
    return {q: float(np.expm1(m + s * norm.ppf(q))) for q in qs}


def prob_above(m: float, s: float, x: float) -> float:
    """P(R > x) for ln(1 + R) ~ N(m, s^2)."""
    if x <= -1.0:
        return 1.0
    return float(norm.sf((np.log1p(x) - m) / s)) if s > 0 else float(np.expm1(m) > x)


def pit(m: float, s: float, realised: float) -> float:
    """Probability integral transform of a realised return: uniform on [0, 1] when the forecasts are calibrated."""
    return float(norm.cdf((np.log1p(realised) - m) / s)) if s > 0 else float(np.expm1(m) <= realised)
