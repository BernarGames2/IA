"""Reverse stress testing: which shocks break a loss limit, and how plausible are they?

1. Gaussian "most likely breaking scenario": for shocks s ~ N(m, S) over horizon
   h and portfolio return w's, the shock of minimum Mahalanobis distance with
   w's = -L is ``s* = m - k S w`` with ``k = (L + w'm) / (w'Sw)``; its distance is
   ``d = (L + w'm) / sigma_p``. Contributions are ``w_i s*_i``.
2. Scenario scaling: for each hypothetical scenario, the multiplier of its shock
   vector needed to breach the limit (linear portfolio).
3. Empirical plausibility: how often historical h-day portfolio returns breached
   the limit. Mathematical possibility != empirical plausibility.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from core.extreme_risk.stress import Scenario, apply_scenario, default_scenarios
from models.market_data import AssetMeta


def gaussian_reverse_stress(weights: pd.Series, mean_h: pd.Series, cov_h: pd.DataFrame,
                            loss_limit: float, t_df: float | None = None) -> dict[str, object]:
    w = weights.to_numpy()
    m = mean_h.loc[weights.index].to_numpy()
    s = cov_h.loc[weights.index, weights.index].to_numpy()
    sp = float(np.sqrt(w @ s @ w))
    if sp <= 0:
        raise ValueError("zero portfolio volatility: reverse stress undefined")
    k = (loss_limit + float(w @ m)) / sp ** 2
    shock = m - k * (s @ w)
    d = (loss_limit + float(w @ m)) / sp
    out: dict[str, object] = {
        "loss_limit": loss_limit, "mahalanobis_distance": d,
        "prob_normal": float(stats.norm.sf(d)),
        "shock": dict(zip(weights.index, shock)),
        "contributions": dict(zip(weights.index, w * shock)),
        "portfolio_return_check": float(w @ shock),
    }
    if t_df is not None and t_df > 2:
        scale = np.sqrt((t_df - 2) / t_df)
        out["prob_student_t"] = float(stats.t.sf(d / scale, t_df))
        out["t_df"] = t_df
    return out


def scenario_multipliers(weights: pd.Series, meta: Mapping[str, AssetMeta], loss_limit: float,
                         scenarios: list[Scenario] | None = None) -> pd.DataFrame:
    rows = []
    for sc in scenarios or default_scenarios():
        r = apply_scenario(weights, meta, sc, 1.0)
        port = float(r["portfolio_return"])
        mult = loss_limit / -port if port < 0 else float("inf")
        feasible = all(mult * v > -1 for v in r["shocks"].values()) if np.isfinite(mult) else False
        rows.append({"scenario": sc.name, "scenario_return": port, "multiplier_to_breach": mult,
                     "mathematically_possible": bool(feasible),
                     "note": "choques escalados ultrapassariam -100%" if np.isfinite(mult) and not feasible else ""})
    return pd.DataFrame(rows).set_index("scenario")


def empirical_breach_frequency(port_returns: pd.Series, loss_limit: float, horizon: int) -> dict[str, float]:
    lr = np.log1p(port_returns)
    roll = np.expm1(lr.rolling(horizon).sum()).dropna()
    n_windows = len(roll)
    breaches = int((roll <= -loss_limit).sum())
    return {"horizon_days": horizon, "n_windows_overlapping": n_windows, "n_breaches": breaches,
            "frequency": breaches / n_windows if n_windows else float("nan"),
            "worst_observed": float(roll.min()) if n_windows else float("nan")}
