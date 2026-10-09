"""Stress testing: hypothetical, compound, historical and correlation scenarios.

All hypothetical shock sizes are HYPOTHESES chosen to be severe-but-plausible
illustrations, not forecasts. Shocks are instantaneous price returns per asset,
derived from asset metadata (class, sector, country) by explicit rules below.
Liquidity costs are parametric (spread widening in bps) unless volume data exist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from core.metrics import recovery_required
from models.market_data import AssetMeta

ShockRule = Callable[[AssetMeta], float]

# Approximate modified durations by fixed-income sector (hypothesis, documented).
DURATION_BY_SECTOR = {"post_fixed": 0.25, "inflation_linked": 6.0, "aggregate_bonds": 6.0}


def _is_foreign(m: AssetMeta, base_country: str = "BR") -> bool:
    return m.country not in (base_country, "unknown")


@dataclass
class Scenario:
    name: str
    description: str
    rule: ShockRule
    kind: str = "hipotético"
    liquidity_spread_bps: float = 0.0
    vol_multiplier: float = 1.0
    corr_blend: float = 0.0
    assumptions: list[str] = field(default_factory=list)


def rate_shock(m: AssetMeta, bp: float) -> float:
    if m.asset_class == "fixed_income":
        return -DURATION_BY_SECTOR.get(m.sector, 5.0) * bp / 1e4
    return 0.0


def default_scenarios() -> list[Scenario]:
    """Library of hypothetical scenarios (base currency BRL)."""

    def equity_crash(m: AssetMeta) -> float:
        return {"equity": -0.30, "real_estate": -0.15, "crypto": -0.50, "commodity": 0.05,
                "fixed_income": -0.01}.get(m.asset_class, -0.20)

    def rates_up(m: AssetMeta) -> float:
        base = rate_shock(m, 300)
        return base + {"equity": -0.10, "real_estate": -0.12, "crypto": -0.15}.get(m.asset_class, 0.0)

    def brl_devaluation(m: AssetMeta) -> float:
        if _is_foreign(m):
            return 0.20
        return {"equity": -0.08, "real_estate": -0.05}.get(m.asset_class, 0.0)

    def crypto_crash(m: AssetMeta) -> float:
        return -0.70 if m.asset_class == "crypto" else 0.0

    def compound(m: AssetMeta) -> float:
        s = {"equity": -0.25, "real_estate": -0.15, "crypto": -0.60, "commodity": 0.0,
             "fixed_income": 0.0}.get(m.asset_class, -0.20)
        s += rate_shock(m, 200)
        if _is_foreign(m):
            s = (1 + s) * 1.15 - 1  # 15% BRL devaluation on foreign exposure
        return s

    return [
        Scenario("Queda de ações -30%", "Ações -30%, FII -15%, cripto -50%, ouro +5%, renda fixa -1%",
                 equity_crash),
        Scenario("Juros +300 bp", "Choque paralelo +300bp (duração modificada aproximada), "
                 "ações -10%, FII -12%, cripto -15%", rates_up,
                 assumptions=[f"durações por setor: {DURATION_BY_SECTOR}"]),
        Scenario("Desvalorização do BRL 20%", "Exposição estrangeira +20% em BRL; ações locais -8%, FII -5%",
                 brl_devaluation, assumptions=["exposição estrangeira inferida de country != BR"]),
        Scenario("Crise cripto -70%", "Criptoativos -70%; demais inalterados", crypto_crash),
        Scenario("Estresse composto", "Ações -25%, FII -15%, cripto -60%, juros +200bp, BRL -15%, "
                 "spreads de liquidez +150bp, vol x2 e correlações elevadas", compound,
                 liquidity_spread_bps=150.0, vol_multiplier=2.0, corr_blend=0.5),
    ]


def apply_scenario(weights: pd.Series, meta: Mapping[str, AssetMeta], sc: Scenario,
                   capital: float) -> dict[str, object]:
    shocks = pd.Series({s: sc.rule(meta[s]) for s in weights.index})
    if (shocks <= -1).any():
        raise ValueError("a shock below -100% is impossible for a long position")
    contrib = weights * shocks
    liq_cost = float(weights.abs().sum() * sc.liquidity_spread_bps / 1e4 / 2)  # half-spread to exit
    port = float(contrib.sum()) - liq_cost
    loss = max(0.0, -port)
    return {
        "scenario": sc.name, "kind": sc.kind, "description": sc.description,
        "portfolio_return": port, "loss_pct": loss, "loss_money": loss * capital,
        "liquidity_cost": liq_cost,
        "recovery_required": recovery_required(loss) if loss < 1 else float("inf"),
        "contributions": contrib.to_dict(), "shocks": shocks.to_dict(),
        "worst_contributor": str(contrib.idxmin()), "assumptions": sc.assumptions,
    }


def run_scenarios(weights: pd.Series, meta: Mapping[str, AssetMeta], capital: float,
                  scenarios: list[Scenario] | None = None,
                  baseline: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for sc in scenarios or default_scenarios():
        r = apply_scenario(weights, meta, sc, capital)
        if baseline is not None:
            r["baseline_return"] = apply_scenario(baseline, meta, sc, capital)["portfolio_return"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("scenario")


def historical_worst_windows(port_returns: pd.Series, windows: tuple[int, ...] = (1, 5, 21),
                             top: int = 3) -> pd.DataFrame:
    """Worst compounded returns over rolling windows (non-overlapping selection)."""
    rows = []
    lr = np.log1p(port_returns)
    for k in windows:
        roll = np.expm1(lr.rolling(k).sum()).dropna()
        taken: list[pd.Timestamp] = []
        for dt, v in roll.sort_values().items():
            pos = roll.index.get_loc(dt)
            if any(abs(pos - roll.index.get_loc(t)) < k for t in taken):
                continue
            taken.append(dt)
            start = port_returns.index[port_returns.index.get_loc(dt) - k + 1]
            rows.append({"window_days": k, "start": start.date(), "end": dt.date(), "return": float(v),
                         "recovery_required": recovery_required(-v) if v < 0 else 0.0})
            if len(taken) >= top:
                break
    return pd.DataFrame(rows)


NAMED_CRISES = {
    "Crise financeira global (2008)": ("2008-09-01", "2008-11-20"),
    "Joesley Day (2017)": ("2017-05-17", "2017-05-18"),
    "Greve dos caminhoneiros (2018)": ("2018-05-18", "2018-06-18"),
    "COVID-19 (2020)": ("2020-02-19", "2020-03-23"),
    "Choque de juros global (2022)": ("2022-01-03", "2022-06-16"),
}


def named_crisis_replay(prices: pd.DataFrame, weights: pd.Series, synthetic: bool) -> pd.DataFrame:
    """Buy-and-hold replay of named historical windows covered by REAL data only."""
    rows = []
    for name, (a, b) in NAMED_CRISES.items():
        if synthetic:
            rows.append({"crisis": name, "status": "não aplicável: dados sintéticos"})
            continue
        sub = prices.loc[a:b]
        if len(sub) < 2 or sub.index[0] > pd.Timestamp(a) + pd.Timedelta(days=7):
            rows.append({"crisis": name, "status": "fora do período coberto pelos dados"})
            continue
        rel = sub.iloc[-1] / sub.iloc[0] - 1
        rows.append({"crisis": name, "status": "ok", "start": str(sub.index[0].date()),
                     "end": str(sub.index[-1].date()), "portfolio_return": float((weights * rel).sum()),
                     "worst_asset": str(rel.idxmin())})
    return pd.DataFrame(rows)


def parametric_stress_var(weights: np.ndarray, cov_daily: np.ndarray, alpha: float,
                          corr_blends: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
                          vol_mults: tuple[float, ...] = (1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Gaussian 1-day VaR (zero mean) under stressed correlation/volatility."""
    from core.simulation.models import stress_covariance

    z = stats.norm.ppf(alpha)
    rows = []
    for b in corr_blends:
        for m in vol_mults:
            c = stress_covariance(cov_daily, b, m)
            sp = float(np.sqrt(weights @ c @ weights))
            rows.append({"corr_blend": b, "vol_multiplier": m, "vol_1d": sp, "var_1d": z * sp})
    return pd.DataFrame(rows)


def liquidity_analysis(weights: pd.Series, capital: float, adv_value: pd.Series | None,
                       participation: float = 0.10, spread_bps: Mapping[str, float] | None = None
                       ) -> pd.DataFrame:
    """Days to liquidate at a max participation of ADV (value). Parametric if ADV missing."""
    rows = []
    for s, w in weights.items():
        pos = abs(float(w)) * capital
        if adv_value is not None and s in adv_value and np.isfinite(adv_value[s]) and adv_value[s] > 0:
            days = pos / (participation * float(adv_value[s]))
            src = "volume observado (média diária em valor)"
        else:
            days = float("nan")
            src = "sem volume: não estimável"
        rows.append({"symbol": s, "position_value": pos, "days_to_liquidate": days,
                     "spread_bps_assumed": (spread_bps or {}).get(s, float("nan")), "source": src})
    return pd.DataFrame(rows).set_index("symbol")
