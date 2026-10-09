"""Walk-forward backtesting with costs, turnover and structural look-ahead protection.

Timing convention: ``returns.iloc[t]`` is the return over (t-1, t]. A decision for
period t is made at the close of t-1 using ONLY ``returns.iloc[t-lookback:t]``
(a copy is passed to the strategy). Between rebalances weights drift with
returns. At each rebalance the two-way turnover ``sum|w_target - w_drifted|`` is
charged at ``cost_bps`` (brokerage + half-spread + slippage, parametrised).

Chronological split of the out-of-sample period: the first ``validation_frac``
is used for model selection; the remainder is a FINAL TEST used only for
reporting, never for choosing models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.metrics import performance_summary

StrategyFn = Callable[[pd.DataFrame, np.ndarray | None], np.ndarray]


@dataclass
class BacktestSettings:
    lookback: int = 252
    rebalance_every: int = 21
    cost_bps: float = 15.0          # brokerage + half-spread
    slippage_bps: float = 5.0
    validation_frac: float = 0.6
    rf_annual: float = 0.0
    periods: int = 252


@dataclass
class StrategyBacktest:
    name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame            # target weights at each rebalance date
    turnover: pd.Series
    costs: pd.Series
    failures: list[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    strategies: dict[str, StrategyBacktest]
    settings: BacktestSettings
    oos_start: pd.Timestamp
    validation_end: pd.Timestamp
    metrics_validation: pd.DataFrame
    metrics_test: pd.DataFrame
    metrics_full: pd.DataFrame
    bias_warnings: list[str]


def run_strategy(name: str, fn: StrategyFn, returns: pd.DataFrame, s: BacktestSettings) -> StrategyBacktest:
    r = returns.to_numpy()
    t_total, n = r.shape
    if t_total <= s.lookback + s.rebalance_every:
        raise ValueError("not enough data for walk-forward with this lookback")
    w = None
    net, gross, tos, costs = [], [], [], []
    w_hist: dict[pd.Timestamp, np.ndarray] = {}
    failures: list[str] = []
    cost_rate = (s.cost_bps + s.slippage_bps) / 1e4
    for t in range(s.lookback, t_total):
        cost = 0.0
        to = 0.0
        if (t - s.lookback) % s.rebalance_every == 0:
            train = returns.iloc[t - s.lookback:t].copy()
            try:
                target = np.asarray(fn(train, None if w is None else w.copy()), float)
                if target.shape != (n,) or not np.all(np.isfinite(target)) or abs(target.sum() - 1) > 1e-6:
                    raise ValueError("invalid weights returned")
            except (ValueError, np.linalg.LinAlgError) as e:
                failures.append(f"{returns.index[t].date()}: {e}; pesos anteriores mantidos")
                target = w if w is not None else np.full(n, 1.0 / n)
            to = float(np.abs(target - (w if w is not None else np.zeros(n))).sum())
            cost = to * cost_rate
            w = target
            w_hist[returns.index[t]] = target
        assert w is not None
        rp = float(w @ r[t])
        gross.append(rp)
        net.append((1 - cost) * (1 + rp) - 1)
        tos.append(to)
        costs.append(cost)
        w = w * (1 + r[t]) / (1 + rp) if rp > -1 else w
    idx = returns.index[s.lookback:]
    return StrategyBacktest(name, pd.Series(net, idx), pd.Series(gross, idx),
                            pd.DataFrame(w_hist, index=returns.columns).T,
                            pd.Series(tos, idx), pd.Series(costs, idx), failures)


def _metrics(bt: StrategyBacktest, sl: slice, s: BacktestSettings) -> dict[str, object]:
    x = bt.net_returns.iloc[sl]
    m = performance_summary(x.to_numpy(), s.rf_annual, s.periods)
    to = bt.turnover.iloc[sl]
    years = len(x) / s.periods
    m["turnover_annual"] = float(to.sum() / years) if years > 0 else None
    m["cost_drag_annual"] = float(bt.costs.iloc[sl].sum() / years) if years > 0 else None
    # only rebalances and failures that happened inside this segment (no test-period leakage)
    if len(x):
        lo, hi = x.index[0], x.index[-1]
        wh = bt.weights.loc[(bt.weights.index >= lo) & (bt.weights.index <= hi)]
        if len(wh) > 1:
            m["avg_weight_change_l1"] = float(np.abs(np.diff(wh.to_numpy(), axis=0)).sum(axis=1).mean())
        m["n_failures"] = sum(1 for f in bt.failures if lo.date() <= pd.Timestamp(f.split(":")[0]).date() <= hi.date())
    m["gross_cagr"] = performance_summary(bt.gross_returns.iloc[sl].to_numpy(), s.rf_annual, s.periods)["cagr"]
    return m


def walk_forward(returns: pd.DataFrame, strategies: dict[str, StrategyFn],
                 settings: BacktestSettings | None = None, point_in_time: bool = False,
                 synthetic: bool = False) -> BacktestResult:
    s = settings or BacktestSettings()
    res = {k: run_strategy(k, f, returns, s) for k, f in strategies.items()}
    n_oos = len(returns) - s.lookback
    n_val = int(n_oos * s.validation_frac)
    val, test = slice(0, n_val), slice(n_val, n_oos)
    mv = pd.DataFrame({k: _metrics(v, val, s) for k, v in res.items()}).T
    mt = pd.DataFrame({k: _metrics(v, test, s) for k, v in res.items()}).T
    mf = pd.DataFrame({k: _metrics(v, slice(0, n_oos), s) for k, v in res.items()}).T
    warn = []
    if synthetic:
        warn.append("backtest sobre dados SINTÉTICOS: valida a mecânica, não o desempenho real")
    if not point_in_time:
        warn.append("universo atual aplicado ao passado e sem dados point-in-time: resultados "
                    "potencialmente enviesados (survivorship/look-ahead de universo)")
    if n_oos - n_val < 252:
        warn.append(f"período de teste final curto ({n_oos - n_val} observações)")
    oos_idx = returns.index[s.lookback:]
    return BacktestResult(res, s, oos_idx[0], oos_idx[max(0, n_val - 1)], mv, mt, mf, warn)


def lookahead_invariance_check(returns: pd.DataFrame, strategy: StrategyFn,
                               settings: BacktestSettings, cut: int, seed: int = 0) -> dict[str, object]:
    """Perturb all data after ``cut`` and verify every decision made up to ``cut`` is unchanged."""
    rng = np.random.default_rng(seed)
    alt = returns.copy()
    alt.iloc[cut:] = rng.normal(0.0, 0.05, size=alt.iloc[cut:].shape)
    a = run_strategy("a", strategy, returns, settings).weights
    b = run_strategy("b", strategy, alt, settings).weights
    common = [d for d in a.index if returns.index.get_loc(d) <= cut]
    diff = float(np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy()).max()) if common else 0.0
    return {"decisions_checked": len(common), "max_abs_weight_diff": diff, "passed": diff < 1e-10}
