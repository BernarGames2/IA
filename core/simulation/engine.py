"""Portfolio wealth simulation with cash flows, fees, rebalancing and inflation.

Definitions used in every output (also written to the report):
* ``W_t``: nominal wealth including contributions/withdrawals.
* ``U_t``: time-weighted unit value (NAV per share, U_0 = 1), i.e. the strategy
  return excluding cash flows. Drawdowns and horizon VaR/ES use ``U``.
* ``p_below_initial``: P(W_T < W_0).
* ``p_below_net_invested``: P(W_T < W_0 + sum contributions - sum withdrawals).
* ``p_negative_return``: P(U_T < 1) -- strategy lost money over the horizon.
* ``p_goal``: P(W_T >= goal) for an explicitly supplied nominal goal.
* ``var_es_horizon``: VaR/ES of the horizon return ``U_T - 1``.
* ``var_es_1step``: VaR/ES of the FIRST simulated step's portfolio return,
  i.e. a one-step-ahead conditional estimate (never derived from W_T).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.extreme_risk.var_es import empirical_var_es
from core.simulation.models import ReturnModel


@dataclass
class CashflowPlan:
    monthly_contribution: float = 0.0
    monthly_withdrawal: float = 0.0
    steps_per_month: int = 21
    contribution_cost_bps: float = 0.0


@dataclass
class SimulationSettings:
    n_paths: int = 20_000
    n_steps: int = 252
    seed: int = 42
    initial_capital: float = 100_000.0
    rebalance_every: int = 21                # steps; 0 = never (buy and hold)
    rebalance_cost_bps: float = 0.0
    annual_fee_bps: float = 0.0
    inflation_annual: float = 0.0
    periods_per_year: int = 252
    batch_size: int = 5_000
    max_floats_per_batch: int = 20_000_000
    record_every: int | None = None
    goal: float | None = None
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    n_representative: int = 20


@dataclass
class SimulationResult:
    model: dict[str, object]
    settings: dict[str, object]
    time_index: np.ndarray                    # step numbers recorded
    percentiles: pd.DataFrame                 # wealth percentiles over time
    representative_paths: np.ndarray
    terminal_wealth: np.ndarray
    terminal_real_wealth: np.ndarray
    terminal_unit: np.ndarray
    max_drawdown_unit: np.ndarray
    first_step_returns: np.ndarray
    ruined: np.ndarray
    summary: dict[str, object] = field(default_factory=dict)
    convergence: pd.DataFrame | None = None


def _batch_size(s: SimulationSettings, n_assets: int) -> int:
    cap = max(100, s.max_floats_per_batch // max(1, s.n_steps * n_assets))
    return int(min(s.batch_size, cap, s.n_paths))


def simulate_portfolio(model: ReturnModel, weights: np.ndarray, settings: SimulationSettings,
                       cashflows: CashflowPlan | None = None) -> SimulationResult:
    """Simulate wealth paths for fixed target weights (processing paths in batches)."""
    s = settings
    cf = cashflows or CashflowPlan()
    w = np.asarray(weights, float)
    if w.shape != (model.n_assets,):
        raise ValueError("weights do not match model assets")
    if abs(w.sum() - 1) > 1e-6 or np.any(w < -1e-9):
        raise ValueError("weights must be long-only and sum to 1")
    rng = np.random.default_rng(s.seed)
    rec = s.record_every or max(1, s.n_steps // 252)
    rec_steps = np.unique(np.concatenate([np.arange(0, s.n_steps + 1, rec), [s.n_steps]]))
    fee = (1.0 + s.annual_fee_bps / 1e4) ** (1.0 / s.periods_per_year) - 1.0
    bsz = _batch_size(s, model.n_assets)

    paths = np.empty((s.n_paths, len(rec_steps)))
    term_u = np.empty(s.n_paths)
    mdd = np.empty(s.n_paths)
    first = np.empty(s.n_paths)
    ruined = np.zeros(s.n_paths, bool)
    net_invested = s.initial_capital

    done = 0
    while done < s.n_paths:
        b = min(bsz, s.n_paths - done)
        r = model.sample(b, s.n_steps, rng)
        hold = np.tile(s.initial_capital * w, (b, 1))
        unit = np.ones(b)
        peak = np.ones(b)
        dd = np.zeros(b)
        alive = np.ones(b, bool)
        rec_i = 0
        if rec_steps[0] == 0:
            paths[done:done + b, 0] = s.initial_capital
            rec_i = 1
        for t in range(1, s.n_steps + 1):
            v_start = hold.sum(axis=1)
            hold *= 1.0 + r[:, t - 1, :]
            hold *= 1.0 - fee
            if s.rebalance_every and t % s.rebalance_every == 0:
                tot = hold.sum(axis=1, keepdims=True)
                target = tot * w
                cost = np.abs(target - hold).sum(axis=1, keepdims=True) * s.rebalance_cost_bps / 1e4
                hold = (tot - cost) * w
            v_end = hold.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                step_ret = np.where(v_start > 0, v_end / v_start - 1.0, 0.0)
            if t == 1:
                first[done:done + b] = step_ret
            unit *= 1.0 + np.where(alive, step_ret, 0.0)
            peak = np.maximum(peak, unit)
            dd = np.minimum(dd, unit / peak - 1.0)
            if cf.steps_per_month and t % cf.steps_per_month == 0:
                flow_in = cf.monthly_contribution * (1 - cf.contribution_cost_bps / 1e4)
                hold += flow_in * w
                if cf.monthly_withdrawal:
                    tot = hold.sum(axis=1)
                    ratio = np.where(tot > 0, np.clip(1 - cf.monthly_withdrawal / np.maximum(tot, 1e-300), 0, 1), 0)
                    newly = (tot <= cf.monthly_withdrawal) & alive
                    alive &= ~newly
                    hold *= ratio[:, None]
                if done == 0:
                    net_invested += cf.monthly_contribution - cf.monthly_withdrawal
            if rec_i < len(rec_steps) and rec_steps[rec_i] == t:
                paths[done:done + b, rec_i] = hold.sum(axis=1)
                rec_i += 1
        term_u[done:done + b] = unit
        mdd[done:done + b] = dd
        ruined[done:done + b] = ~alive
        done += b

    term = paths[:, -1]
    years = s.n_steps / s.periods_per_year
    real = term / (1.0 + s.inflation_annual) ** years
    q = [5, 25, 50, 75, 95]
    pct = pd.DataFrame(np.percentile(paths, q, axis=0).T, index=rec_steps, columns=[f"p{x}" for x in q])
    pct["mean"] = paths.mean(axis=0)
    rep_idx = np.argsort(term)[np.linspace(0, s.n_paths - 1, min(s.n_representative, s.n_paths)).astype(int)]

    summary = summarize(term, real, term_u, mdd, first, ruined, s, net_invested)
    conv = convergence_table(term, term_u, s)
    return SimulationResult(
        model=model.describe(),
        settings={**s.__dict__, "confidence_levels": list(s.confidence_levels),
                  "batch_size_used": bsz, "record_every": rec, "cashflows": cf.__dict__,
                  "numpy_version": np.__version__, "python": platform.python_version()},
        time_index=rec_steps, percentiles=pct, representative_paths=paths[rep_idx],
        terminal_wealth=term, terminal_real_wealth=real, terminal_unit=term_u,
        max_drawdown_unit=mdd, first_step_returns=first, ruined=ruined,
        summary=summary, convergence=conv)


def _binom_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def summarize(term: np.ndarray, real: np.ndarray, unit: np.ndarray, mdd: np.ndarray,
              first: np.ndarray, ruined: np.ndarray, s: SimulationSettings,
              net_invested: float) -> dict[str, object]:
    n = len(term)
    out: dict[str, object] = {
        "n_paths": n, "horizon_steps": s.n_steps, "horizon_years": s.n_steps / s.periods_per_year,
        "initial_capital": s.initial_capital, "net_invested": net_invested,
        "terminal_wealth_mean": float(term.mean()),
        "terminal_wealth_mean_mc_se": float(term.std(ddof=1) / np.sqrt(n)),
        "terminal_wealth_percentiles": {f"p{q}": float(np.percentile(term, q)) for q in (5, 25, 50, 75, 95)},
        "terminal_real_wealth_median": float(np.median(real)),
        "inflation_annual_assumed": s.inflation_annual,
        "max_drawdown_unit_median": float(np.median(mdd)),
        "max_drawdown_unit_p05": float(np.percentile(mdd, 5)),
        "p_ruin": float(ruined.mean()),
    }
    for key, p in (("p_below_initial", float(np.mean(term < s.initial_capital))),
                   ("p_below_net_invested", float(np.mean(term < net_invested))),
                   ("p_negative_return", float(np.mean(unit < 1.0)))):
        out[key] = p
        out[f"{key}_mc_se"] = _binom_se(p, n)
    if s.goal is not None:
        p = float(np.mean(term >= s.goal))
        out["goal"] = s.goal
        out["p_goal"] = p
        out["p_goal_mc_se"] = _binom_se(p, n)
    var_h, var_1 = {}, {}
    for a in s.confidence_levels:
        v, e = empirical_var_es(-(unit - 1.0), a)
        var_h[f"{a:.2f}"] = {"var": v, "es": e}
        v1, e1 = empirical_var_es(-first, a)
        var_1[f"{a:.2f}"] = {"var": v1, "es": e1}
    out["var_es_horizon"] = {"variable": "retorno da carteira no horizonte, sem fluxos (U_T-1)",
                             "horizon_steps": s.n_steps, "levels": var_h}
    out["var_es_1step"] = {"variable": "retorno da carteira no 1º passo simulado (condicional)",
                           "horizon_steps": 1, "levels": var_1}
    return out


def convergence_table(term: np.ndarray, unit: np.ndarray, s: SimulationSettings) -> pd.DataFrame:
    """Statistics on nested subsets of paths (prefixes) to show Monte Carlo convergence."""
    n = len(term)
    sizes = sorted({x for x in (500, 1000, 2000, 5000, 10000, 20000, 50000, n) if x <= n})
    rows = []
    for k in sizes:
        t = term[:k]
        u = unit[:k]
        v95, e95 = empirical_var_es(-(u - 1.0), 0.95) if k >= 100 else (np.nan, np.nan)
        rows.append({"n_paths": k, "mean_terminal": t.mean(), "mc_se_mean": t.std(ddof=1) / np.sqrt(k),
                     "median_terminal": float(np.median(t)), "p_negative_return": float(np.mean(u < 1)),
                     "var95_horizon": v95, "es95_horizon": e95})
    return pd.DataFrame(rows).set_index("n_paths")
