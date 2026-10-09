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
COV_COMPLEXITY = {"sample": 0, "ledoit_wolf": 1, "lw_constant_corr": 1, "oas": 1, "ewma": 1, "factor_pca": 2}
VOL_COMPLEXITY = {"rolling_hist": 0, "ewma": 1, "garch": 2, "gjr_garch": 3}
MU_COMPLEXITY = {"grand_mean": 0, "historical": 1, "ewma": 1, "james_stein": 2}


def _dominated(m: pd.DataFrame) -> list[str]:
    """Strategies dominated in validation: another one has >= CAGR, <= vol and >= max drawdown
    (drawdowns are negative numbers), strictly better in at least one."""
    out = []
    for a in m.index:
        for b in m.index:
            if a == b:
                continue
            ge = (m.at[b, "cagr"] >= m.at[a, "cagr"] and m.at[b, "volatility_ann"] <= m.at[a, "volatility_ann"]
                  and m.at[b, "max_drawdown"] >= m.at[a, "max_drawdown"])
            gt = (m.at[b, "cagr"] > m.at[a, "cagr"] or m.at[b, "volatility_ann"] < m.at[a, "volatility_ann"]
                  or m.at[b, "max_drawdown"] > m.at[a, "max_drawdown"])
            if ge and gt:
                out.append(a)
                break
    return out


def select_strategy(bt: BacktestResult, baseline: str = "equal_weight",
                    eligible: list[str] | None = None) -> dict[str, object]:
    """Select among ``eligible`` strategies using ONLY the validation segment.

    1. Drop strategies dominated in validation (CAGR, volatility, max drawdown).
    2. Rank by excess-return Sharpe when at least one strategy beat the risk-free rate.
       When every strategy lost to the risk-free rate, Sharpe ordering is distorted
       (a riskier strategy looks *better* because a negative excess return is divided
       by a larger volatility), so the ranking uses the annualised excess return.
    3. Among strategies within one standard error of the best, choose the least complex.
    PSR/DSR use EXCESS returns over rf; Holm and Benjamini-Hochberg adjust the p-values
    of "Sharpe > 0" across all eligible strategies (multiple testing).
    Reference strategies (single-asset benchmark, constraint violators) cannot be selected.
    """
    from core.backtesting.overfitting import benjamini_hochberg, holm
    from core.metrics import rf_per_period

    mv = bt.metrics_validation
    s = bt.settings
    if baseline not in bt.strategies:
        raise ValueError(f"baseline '{baseline}' was not backtested")
    n_val_i = int(len(bt.strategies[baseline].net_returns) * s.validation_frac)
    rf_p = rf_per_period(s.rf_annual, s.periods)
    elig_names = [k for k in mv.index if eligible is None or k in eligible]

    def excess(k: str, sl: slice) -> np.ndarray:
        return bt.strategies[k].net_returns.iloc[sl].to_numpy() - rf_p

    m = mv.loc[elig_names, ["cagr", "volatility_ann", "max_drawdown", "sharpe"]].astype(float)
    dominated = _dominated(m.dropna(subset=["cagr", "volatility_ann", "max_drawdown"]))
    cand = [k for k in elig_names if k not in dominated] or elig_names
    val = slice(0, n_val_i)
    er = pd.Series({k: float(excess(k, val).mean() * s.periods) for k in cand})
    vol = m.loc[cand, "volatility_ann"]
    years = n_val_i / s.periods
    if (er > 0).any():
        metric = "sharpe_excesso"
        score = m.loc[cand, "sharpe"]
        best = score.idxmax()
        sr_p = score[best] / np.sqrt(s.periods)
        se = float(np.sqrt((1 + sr_p ** 2 / 2) / n_val_i) * np.sqrt(s.periods))
    else:
        metric = "retorno_excedente_anual"
        score = er
        best = score.idxmax()
        se = float(vol[best] / np.sqrt(years)) if years > 0 else float("nan")
    within = score[score >= score[best] - se].index.tolist()
    chosen = min(within, key=lambda k: (COMPLEXITY.get(k, 5), -score[k]))
    out: dict[str, object] = {
        "selected": chosen, "best_by_metric": best, "ranking_metric": metric, "metric_se": se,
        "scores": score.to_dict(), "within_one_se": within, "dominated_in_validation": dominated,
        "eligible_for_selection": elig_names,
        "excluded_reference_strategies": [k for k in mv.index if k not in elig_names],
        "rule": ("validação apenas: remove dominadas; ranking por " +
                 ("Sharpe excedente" if metric == "sharpe_excesso" else
                  "retorno excedente (nenhuma estratégia superou a taxa livre de risco)") +
                 "; menor complexidade a 1 EP da melhor"),
        "baseline": baseline,
    }
    if metric != "sharpe_excesso":
        out["warning"] = "nenhuma estratégia superou a taxa livre de risco na validação"
    if chosen != baseline:
        a = bt.strategies[chosen].net_returns.iloc[val].to_numpy()
        b = bt.strategies[baseline].net_returns.iloc[val].to_numpy()
        try:
            out["vs_baseline_validation"] = diebold_mariano(-a, -b)
        except InsufficientDataError as e:
            out["vs_baseline_validation"] = {"error": str(e)}
    trials = []
    for k in bt.strategies:
        x = excess(k, val)
        sd = x.std(ddof=1)
        trials.append(float(x.mean() / sd) if sd > 0 else 0.0)
    try:
        out["dsr_validation"] = dsr(excess(chosen, val), trials)
    except InsufficientDataError as e:
        out["dsr_validation"] = {"error": str(e)}
    pvals = {}
    for k in elig_names:
        try:
            pvals[k] = 1.0 - psr(excess(k, val), 0.0)
        except InsufficientDataError:
            continue
    if pvals:
        out["p_sharpe_gt_0_validation"] = pvals
        out["p_holm"] = holm(pvals)
        out["p_bh"] = benjamini_hochberg(pvals)
    try:
        out["psr_test"] = psr(excess(chosen, slice(n_val_i, None)), 0.0)
    except InsufficientDataError as e:
        out["psr_test"] = str(e)
    out["test_metrics_selected"] = bt.metrics_test.loc[chosen].to_dict()
    out["test_metrics_baseline"] = bt.metrics_test.loc[baseline].to_dict()
    out["test_metrics_equal_weight"] = bt.metrics_test.loc["equal_weight"].to_dict() \
        if "equal_weight" in bt.metrics_test.index else None
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
