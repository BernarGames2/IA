"""Per-asset analysis and role explanation (historical observations vs hypotheses).

Assets are NOT ranked by individual return or Sharpe. The role of each asset is
described by its contribution to portfolio risk, its correlation with the
portfolio, its behaviour in hypothetical scenarios and binding constraints.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from core.metrics import (
    UndefinedMetricError,
    annualized_arithmetic_mean,
    annualized_geometric_return,
    annualized_volatility,
    beta,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from models.market_data import AssetMeta


def asset_statistics(returns: pd.DataFrame, rf: float, periods: int = 252,
                     benchmark: pd.Series | None = None) -> pd.DataFrame:
    rows = {}
    for c in returns.columns:
        x = returns[c].to_numpy()
        row: dict[str, float | None] = {
            "arith_mean_ann": annualized_arithmetic_mean(x, periods),
            "cagr": annualized_geometric_return(x, periods),
            "vol_ann": annualized_volatility(x, periods),
            "max_drawdown": max_drawdown(x),
            "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
        }
        for k, fn in (("sharpe", lambda: sharpe_ratio(x, rf, periods)),
                      ("sortino", lambda: sortino_ratio(x, rf, periods)),
                      ("beta_vs_benchmark", (lambda: beta(x, benchmark.to_numpy()))
                       if benchmark is not None else None)):
            if fn is None:
                row[k] = None
                continue
            try:
                row[k] = float(fn())
            except UndefinedMetricError:
                row[k] = None
        rows[c] = row
    return pd.DataFrame(rows).T


def explain_roles(weights: pd.Series, rc: pd.DataFrame, port_returns: pd.Series, returns: pd.DataFrame,
                  meta: Mapping[str, AssetMeta], upper: pd.Series, stress_shocks: pd.DataFrame | None,
                  quality_flags: Mapping[str, str], adv_value: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    port_vol = float(np.std(port_returns, ddof=1) * np.sqrt(252))
    for s in weights.index:
        w = float(weights[s])
        corr = float(np.corrcoef(returns[s], port_returns)[0, 1])
        vol = float(returns[s].std(ddof=1) * np.sqrt(252))
        rcp = float(rc.loc[s, "rc_pct"]) if s in rc.index else float("nan")
        roles = []
        if w < 1e-4:
            reason = "peso ~0: o otimizador não o incluiu"
            if upper[s] <= 1e-9:
                reason = "excluído por restrição (limite 0)"
            roles.append(reason)
        else:
            if w >= upper[s] - 1e-6:
                roles.append(f"no teto individual ({upper[s]:.0%}): restrição ativa")
            if vol < port_vol:
                roles.append("redutor de volatilidade (vol. individual < vol. da carteira)")
            if corr < 0.3:
                roles.append("diversificador (baixa correlação com a carteira)")
            if np.isfinite(rcp) and rcp > 1.5 * w:
                roles.append("concentra risco (contribuição de risco > 1,5x o peso)")
            if meta[s].country not in ("BR", "unknown"):
                roles.append("exposição a ativos/moeda estrangeiros")
        if stress_shocks is not None and s in stress_shocks.columns:
            worst = stress_shocks[s].idxmin()
            worst_v = float(stress_shocks[s].min())
        else:
            worst, worst_v = None, float("nan")
        rows.append({"symbol": s, "class": meta[s].asset_class, "weight": w, "rc_pct": rcp,
                     "corr_with_portfolio": corr, "vol_ann_hist": vol,
                     "worst_hypothetical_scenario": worst, "worst_scenario_shock": worst_v,
                     "adv_value": float(adv_value[s]) if adv_value is not None and s in adv_value else float("nan"),
                     "data_quality": quality_flags.get(s, "ok"), "role": "; ".join(roles)})
    return pd.DataFrame(rows).set_index("symbol")
