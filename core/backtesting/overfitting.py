"""Overfitting controls: PSR, DSR, multiple-testing corrections, strategy registry.

* Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012), using the
  NON-annualised per-period Sharpe ``sr`` and kurtosis ``k`` (not excess):
  ``PSR(sr*) = Phi( (sr - sr*) sqrt(T-1) / sqrt(1 - g3 sr + (k-1)/4 sr^2) )``.
  Assumes stationary, serially-uncorrelated returns (checked loosely: the report
  states the assumption).
* Deflated Sharpe Ratio (Bailey & López de Prado, 2014): PSR against the
  expected maximum Sharpe of N trials,
  ``sr* = sqrt(V) ((1-g) Phi^-1(1-1/N) + g Phi^-1(1-1/(N e)))``.
* Holm-Bonferroni and Benjamini-Hochberg adjustments.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from utils.validation import InsufficientDataError, as_1d_float

EULER_GAMMA = 0.5772156649015329


def psr(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    x = as_1d_float(returns, "returns")
    if len(x) < 30:
        raise InsufficientDataError("PSR needs >= 30 observations")
    sd = x.std(ddof=1)
    if sd <= 0:
        raise InsufficientDataError("PSR undefined for zero volatility")
    sr = x.mean() / sd
    g3 = stats.skew(x)
    k = stats.kurtosis(x, fisher=False)
    denom = 1 - g3 * sr + (k - 1) / 4 * sr ** 2
    if denom <= 0:
        raise InsufficientDataError("PSR denominator non-positive (extreme higher moments)")
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(len(x) - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                                    + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(returns: np.ndarray, trial_sharpes: list[float]) -> dict[str, float]:
    """DSR of ``returns`` given per-period Sharpe ratios of ALL strategies tried."""
    srs = np.asarray(trial_sharpes, float)
    n = len(srs)
    v = float(np.var(srs, ddof=1)) if n > 1 else 0.0
    sr0 = expected_max_sharpe(n, v)
    return {"dsr": psr(returns, sr0), "sr_star_per_period": sr0, "n_trials": float(n),
            "var_trial_sr": v}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, items[i][1] * m / (i + 1))
        adj[i] = prev
    return {items[i][0]: adj[i] for i in range(m)}


class StrategyRegistry:
    """Append-only JSON-lines log of every strategy evaluated (reduces selection bias)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: list[dict[str, object]] = []

    def record(self, name: str, config: dict[str, object], metrics: dict[str, object],
               data_hash: str, stage: str) -> None:
        e = {"ts_utc": datetime.now(timezone.utc).isoformat(), "strategy": name, "stage": stage,
             "config": config, "metrics": metrics, "data_hash": data_hash}
        self.entries.append(e)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, default=_jsonable, ensure_ascii=False) + "\n")

    @property
    def n_trials(self) -> int:
        return len({e["strategy"] for e in self.entries})


def _jsonable(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
