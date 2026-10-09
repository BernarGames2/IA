"""Model Selection Engine.

Rules (documented, deterministic):
* Selection uses ONLY the validation segment of the walk-forward backtest; the
  final test segment is reported afterwards and never alters the choice.
* One-standard-error rule: among strategies whose validation Sharpe is within
  one standard error of the best (SE ~ sqrt((1 + SR^2/2)/T), annualised), the
  LEAST complex is chosen. Complexity is an explicit ordinal score.
* The choice is compared with the equal-weight baseline through a HAC
  (Diebold-Mariano-type) test on daily return differences, and with the DSR,
  which accounts for the number of strategies tried.
* Covariance / volatility / expected-return models follow the same principle:
  the simplest model not significantly worse than the best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtesting.overfitting import dsr, psr
from core.backtesting.walk_forward import BacktestResult
from core.volatility.models import diebold_mariano
from utils.validation import InsufficientDataError

COMPLEXITY = {
    "equal_weight": 0, "benchmark": 0, "min_variance": 1, "risk_parity": 1, "inverse_vol": 1,
    "hrp": 2, "max_diversification": 2, "max_sharpe": 2, "min_cvar": 2,
    "robust_mean_variance": 3, "resampled_max_sharpe": 3,
}
COV_COMPLEXITY = {"sample": 0, "ledoit_wolf": 1, "oas": 1, "ewma": 1, "factor_pca": 2}
VOL_COMPLEXITY = {"rolling_hist": 0, "ewma": 1, "garch": 2, "gjr_garch": 3}
MU_COMPLEXITY = {"grand_mean": 0, "historical": 1, "ewma": 1, "james_stein": 2}


def select_strategy(bt: BacktestResult, baseline: str = "equal_weight",
                    eligible: list[str] | None = None) -> dict[str, object]:
    """Select among ``eligible`` strategies (those satisfying the investor constraints).

    Reference strategies (e.g. a single-asset benchmark or HRP violating limits)
    are backtested and reported but cannot be selected.
    """
    mv = bt.metrics_validation
    s = bt.settings
    n_val = len(bt.strategies[baseline].net_returns) * s.validation_frac
    elig_names = [k for k in mv.index if eligible is None or k in eligible]
    sharpe = mv.loc[elig_names, "sharpe"].astype(float)
    if sharpe.isna().all():
        return {"selected": baseline, "reason": "Sharpe indefinido para todas as estratégias"}
    best = sharpe.idxmax()
    se = float(np.sqrt((1 + (sharpe[best] / np.sqrt(s.periods)) ** 2 / 2) / n_val) * np.sqrt(s.periods))
    eligible_1se = sharpe[sharpe >= sharpe[best] - se].index.tolist()
    chosen = min(eligible_1se, key=lambda k: (COMPLEXITY.get(k, 5), -sharpe[k]))
    n_val_i = int(n_val)
    out: dict[str, object] = {
        "selected": chosen, "best_validation_sharpe": best, "sharpe_se": se,
        "within_one_se": eligible_1se, "eligible_for_selection": elig_names,
        "excluded_reference_strategies": [k for k in mv.index if k not in elig_names],
        "rule": "1-SE: menor complexidade entre as estratégias elegíveis a 1 EP da melhor",
    }
    if chosen != baseline:
        a = bt.strategies[chosen].net_returns.iloc[:n_val_i].to_numpy()
        b = bt.strategies[baseline].net_returns.iloc[:n_val_i].to_numpy()
        try:
            out["vs_baseline_validation"] = diebold_mariano(-a, -b)
        except InsufficientDataError as e:
            out["vs_baseline_validation"] = {"error": str(e)}
    trials = [float(bt.strategies[k].net_returns.iloc[:n_val_i].mean() /
                    bt.strategies[k].net_returns.iloc[:n_val_i].std(ddof=1))
              for k in bt.strategies]
    try:
        out["dsr_validation"] = dsr(bt.strategies[chosen].net_returns.iloc[:n_val_i].to_numpy(), trials)
    except InsufficientDataError as e:
        out["dsr_validation"] = {"error": str(e)}
    test = bt.strategies[chosen].net_returns.iloc[n_val_i:].to_numpy()
    try:
        out["psr_test"] = psr(test, 0.0)
    except InsufficientDataError as e:
        out["psr_test"] = str(e)
    out["test_metrics_selected"] = bt.metrics_test.loc[chosen].to_dict()
    out["test_metrics_baseline"] = bt.metrics_test.loc[baseline].to_dict()
    return out


def select_simplest(table: pd.DataFrame, loss_col: str, se_col: str | None,
                    complexity: dict[str, int]) -> dict[str, object]:
    """Simplest model whose loss is within one SE of the best (or best if no SE)."""
    t = table.dropna(subset=[loss_col])
    best = t[loss_col].idxmin()
    se = float(t.loc[best, se_col]) if se_col and se_col in t and np.isfinite(t.loc[best, se_col]) else 0.0
    elig = t[t[loss_col] <= t.loc[best, loss_col] + se].index.tolist()
    chosen = min(elig, key=lambda k: (complexity.get(k, 9), t.loc[k, loss_col]))
    return {"selected": chosen, "best": best, "eligible_within_1se": elig}


def select_vol_model(cmp: dict[str, object], alpha: float = 0.05) -> dict[str, object]:
    """Choose the simplest volatility model not significantly worse (DM) than the best by QLIKE."""
    table: pd.DataFrame = cmp["table"]  # type: ignore[assignment]
    best = table["qlike"].idxmin()
    dm = cmp["diebold_mariano"]
    def worse_than_best(m: str) -> bool:
        if m == best:
            return False
        if best == "ewma" and f"{m}_vs_ewma" in dm:
            d = dm[f"{m}_vs_ewma"]
            return d["p_value"] < alpha and d["dm_stat"] > 0
        if m == "ewma" and f"{best}_vs_ewma" in dm:
            d = dm[f"{best}_vs_ewma"]
            return d["p_value"] < alpha and d["dm_stat"] < 0
        return True  # no direct test available: keep only best
    elig = [m for m in table.index if not worse_than_best(m)]
    chosen = min(elig, key=lambda k: VOL_COMPLEXITY.get(k, 9))
    return {"selected": chosen, "best_qlike": best, "eligible": elig,
            "rule": "modelo mais simples não significativamente pior (DM, 5%) que o melhor em QLIKE"}
