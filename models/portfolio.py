"""Portfolio constraint and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class GroupConstraint:
    """sum_{i in members} w_i in [lower, upper]. ``name`` e.g. 'class:equity_br'."""

    name: str
    members: tuple[str, ...]
    lower: float = 0.0
    upper: float = 1.0


@dataclass
class PortfolioConstraints:
    """Linear constraints shared by all optimisers.

    Short selling / leverage are disabled unless ``lower`` contains negatives and
    ``allow_short`` is True. ``previous_weights`` + ``max_turnover`` express a
    turnover limit (one-way L1 distance / 2 convention is NOT used: turnover here
    is sum |w_new - w_old|).
    """

    symbols: list[str]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    groups: list[GroupConstraint] = field(default_factory=list)
    allow_short: bool = False
    previous_weights: NDArray[np.float64] | None = None
    max_turnover: float | None = None

    @classmethod
    def long_only(cls, symbols: list[str], max_weight: float | None = None,
                  min_weight: float = 0.0, groups: list[GroupConstraint] | None = None
                  ) -> "PortfolioConstraints":
        n = len(symbols)
        ub = 1.0 if max_weight is None else max_weight
        return cls(list(symbols), np.full(n, float(min_weight)), np.full(n, float(ub)),
                   list(groups or []))

    @property
    def n(self) -> int:
        return len(self.symbols)

    def group_matrix(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[str]]:
        idx = {s: i for i, s in enumerate(self.symbols)}
        rows, lo, hi, names = [], [], [], []
        for g in self.groups:
            a = np.zeros(self.n)
            for m in g.members:
                if m in idx:
                    a[idx[m]] = 1.0
            rows.append(a)
            lo.append(g.lower)
            hi.append(g.upper)
            names.append(g.name)
        if not rows:
            return np.zeros((0, self.n)), np.zeros(0), np.zeros(0), []
        return np.vstack(rows), np.asarray(lo), np.asarray(hi), names


@dataclass
class OptimizationResult:
    method: str
    weights: pd.Series
    success: bool
    status: str
    expected_return: float
    volatility: float
    sharpe: float | None
    feasible: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method, "success": self.success, "status": self.status,
            "feasible": self.feasible, "expected_return": self.expected_return,
            "volatility": self.volatility, "sharpe": self.sharpe,
            "violations": list(self.violations), "warnings": list(self.warnings),
            "weights": {k: float(v) for k, v in self.weights.items()},
        }
