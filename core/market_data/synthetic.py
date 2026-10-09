"""Deterministic SYNTHETIC market-data generator for offline mode and tests.

Every series produced here is labelled ``origin=SYNTHETIC`` and uses ``SINT_``
symbols so it can never be confused with real market data. The generator has
known ground-truth parameters (drift, volatility, correlation, regime process),
which lets tests check estimators against the truth.

Model (daily, business-day calendar without holidays):
* a two-state Markov regime (calm/stress) with transition matrix ``P``;
* in each regime, correlated Student-t shocks (unit variance) with a regime
  specific correlation matrix and volatility multiplier;
* log-price increment ``(mu - 0.5*sigma_t**2)*dt + sigma_t*sqrt(dt)*z``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.market_data.base import RawSeries
from models.market_data import AssetMeta
from utils.numerical import nearest_psd, safe_cholesky


@dataclass(frozen=True)
class SyntheticAsset:
    symbol: str
    mu: float          # annual continuous drift of the price process
    sigma: float       # annual volatility in the calm regime
    meta: AssetMeta
    adv: float = 1e7   # synthetic average daily traded value (base currency)


def _m(sym: str, cls: str, sector: str, country: str = "BR") -> AssetMeta:
    return AssetMeta(sym, cls, sector, country, "BRL", "SYNTHETIC", "business_days",
                     "synthetic_generator")


DEFAULT_SYNTHETIC_UNIVERSE: tuple[SyntheticAsset, ...] = (
    SyntheticAsset("SINT_RF_POS", 0.100, 0.006, _m("SINT_RF_POS", "fixed_income", "post_fixed"), 5e8),
    SyntheticAsset("SINT_RF_INFL", 0.105, 0.060, _m("SINT_RF_INFL", "fixed_income", "inflation_linked"), 5e7),
    SyntheticAsset("SINT_ACOES_BR", 0.110, 0.220, _m("SINT_ACOES_BR", "equity", "broad_market"), 3e8),
    SyntheticAsset("SINT_ACAO_BR_A", 0.120, 0.330, _m("SINT_ACAO_BR_A", "equity", "materials"), 8e8),
    SyntheticAsset("SINT_ACAO_BR_B", 0.110, 0.270, _m("SINT_ACAO_BR_B", "equity", "financials"), 6e8),
    SyntheticAsset("SINT_ACOES_EUA", 0.120, 0.180, _m("SINT_ACOES_EUA", "equity", "broad_market", "US"), 1e8),
    SyntheticAsset("SINT_FII", 0.090, 0.140, _m("SINT_FII", "real_estate", "reit"), 2e7),
    SyntheticAsset("SINT_OURO", 0.070, 0.160, _m("SINT_OURO", "commodity", "gold", "GLOBAL"), 3e7),
    SyntheticAsset("SINT_CRIPTO", 0.250, 0.650, _m("SINT_CRIPTO", "crypto", "crypto", "GLOBAL"), 5e7),
)

# Calm-regime correlation (order as DEFAULT_SYNTHETIC_UNIVERSE).
_CALM_CORR = np.array([
    # RFPOS RFINF ACBR  ACA   ACB   EUA   FII   OURO  CRIP
    [1.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00],
    [0.10, 1.00, 0.30, 0.20, 0.25, 0.05, 0.35, 0.05, 0.05],
    [0.00, 0.30, 1.00, 0.75, 0.80, 0.30, 0.45, 0.05, 0.20],
    [0.00, 0.20, 0.75, 1.00, 0.50, 0.25, 0.30, 0.15, 0.15],
    [0.00, 0.25, 0.80, 0.50, 1.00, 0.25, 0.40, 0.00, 0.15],
    [0.00, 0.05, 0.30, 0.25, 0.25, 1.00, 0.15, 0.10, 0.25],
    [0.05, 0.35, 0.45, 0.30, 0.40, 0.15, 1.00, 0.05, 0.10],
    [0.00, 0.05, 0.05, 0.15, 0.00, 0.10, 0.05, 1.00, 0.15],
    [0.00, 0.05, 0.20, 0.15, 0.15, 0.25, 0.10, 0.15, 1.00],
])


@dataclass
class SyntheticSpec:
    assets: tuple[SyntheticAsset, ...] = DEFAULT_SYNTHETIC_UNIVERSE
    calm_corr: np.ndarray = field(default_factory=lambda: _CALM_CORR.copy())
    stress_corr_blend: float = 0.5          # stress corr = (1-b)*calm + b*ones (risky assets)
    stress_vol_multiplier: float = 2.0
    p_calm_to_stress: float = 0.01
    p_stress_to_calm: float = 0.05
    student_t_df: float = 5.0
    n_days: int = 1260                       # ~5 years of business days
    end_date: str = "2025-12-31"
    start_price: float = 100.0
    seed: int = 42
    trading_days: int = 252


@dataclass
class SyntheticResult:
    series: dict[str, RawSeries]
    meta: dict[str, AssetMeta]
    regimes: pd.Series          # ground-truth regime labels (0=calm, 1=stress)
    true_mu: pd.Series
    true_sigma_calm: pd.Series
    spec: SyntheticSpec


def generate_synthetic_market(spec: SyntheticSpec | None = None) -> SyntheticResult:
    """Generate a deterministic synthetic market (same seed -> identical output)."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = len(spec.assets)
    if spec.calm_corr.shape != (n, n):
        raise ValueError("calm_corr shape does not match the number of assets")
    dates = pd.bdate_range(end=spec.end_date, periods=spec.n_days + 1)
    dt = 1.0 / spec.trading_days

    calm = nearest_psd(spec.calm_corr).matrix
    risky = np.array([a.sigma > 0.05 for a in spec.assets])
    stress = calm.copy()
    b = spec.stress_corr_blend
    for i in range(n):
        for j in range(n):
            if i != j and risky[i] and risky[j]:
                stress[i, j] = (1 - b) * calm[i, j] + b * 1.0
    stress = nearest_psd(stress).matrix
    l_calm, _ = safe_cholesky(calm)
    l_stress, _ = safe_cholesky(stress)

    # regime path
    states = np.zeros(spec.n_days, dtype=int)
    u = rng.random(spec.n_days)
    for t in range(1, spec.n_days):
        if states[t - 1] == 0:
            states[t] = 1 if u[t] < spec.p_calm_to_stress else 0
        else:
            states[t] = 0 if u[t] < spec.p_stress_to_calm else 1

    nu = spec.student_t_df
    g = rng.standard_normal((spec.n_days, n))
    w = rng.chisquare(nu, size=(spec.n_days, 1))
    scale_t = np.sqrt((nu - 2.0) / nu)  # unit-variance multivariate t
    mu = np.array([a.mu for a in spec.assets])
    sig = np.array([a.sigma for a in spec.assets])

    log_inc = np.empty((spec.n_days, n))
    for t in range(spec.n_days):
        L = l_stress if states[t] == 1 else l_calm
        z = (L @ g[t]) / np.sqrt(w[t, 0] / nu) * scale_t
        s_t = sig * (spec.stress_vol_multiplier if states[t] == 1 else 1.0)
        s_t = np.where(risky, s_t, sig)  # low-vol cash-like asset keeps its vol
        log_inc[t] = (mu - 0.5 * s_t ** 2) * dt + s_t * np.sqrt(dt) * z

    log_p = np.vstack([np.zeros(n), np.cumsum(log_inc, axis=0)])
    prices = spec.start_price * np.exp(log_p)
    vol_noise = rng.lognormal(mean=0.0, sigma=0.35, size=(spec.n_days + 1, n))

    now = datetime.now(timezone.utc).isoformat()
    series: dict[str, RawSeries] = {}
    meta: dict[str, AssetMeta] = {}
    for i, a in enumerate(spec.assets):
        close = pd.Series(prices[:, i], index=dates, name=a.symbol)
        volume = pd.Series(a.adv * vol_noise[:, i] / prices[:, i], index=dates, name=a.symbol)
        series[a.symbol] = RawSeries(
            symbol=a.symbol, close=close, volume=volume, provider="synthetic_generator",
            price_field="synthetic_total_return_price", adjusted_for="synthetic (total return)",
            currency="BRL", frequency="1d", ingested_at_utc=now,
            notes=["DADOS SINTÉTICOS: não representam nenhum ativo real", f"seed={spec.seed}"],
        )
        meta[a.symbol] = a.meta
    return SyntheticResult(
        series=series, meta=meta,
        regimes=pd.Series(states, index=dates[1:], name="true_regime"),
        true_mu=pd.Series(mu, index=[a.symbol for a in spec.assets]),
        true_sigma_calm=pd.Series(sig, index=[a.symbol for a in spec.assets]),
        spec=spec,
    )


class SyntheticProvider:
    """Provider facade over :func:`generate_synthetic_market`."""

    name = "synthetic_generator"

    def __init__(self, spec: SyntheticSpec | None = None) -> None:
        self.result = generate_synthetic_market(spec)

    def fetch(self, symbols: list[str], period: str = "5y", interval: str = "1d") -> dict[str, RawSeries]:
        return {s: self.result.series[s] for s in symbols if s in self.result.series}
