"""Markowitz baseline facade (re-exports the Portfolio Construction Engine).

Portfolio expected return ``w' mu``, variance ``w' Sigma w`` and volatility
``sqrt(w' Sigma w)``; see :mod:`core.optimization.optimizers` for the solvers.
"""

from core.metrics import portfolio_return, portfolio_volatility
from core.optimization.constraints import build_constraints, check_feasibility, verify_weights
from core.optimization.optimizers import (
    efficient_frontier,
    equal_weight,
    hrp,
    max_diversification,
    max_sharpe,
    min_cvar,
    min_variance,
    return_range,
    risk_parity,
    target_return,
    target_volatility,
)
from core.optimization.robust import penalized_mean_variance, robust_mean_variance

__all__ = [
    "portfolio_return", "portfolio_volatility", "build_constraints", "check_feasibility", "verify_weights",
    "efficient_frontier", "equal_weight", "hrp", "max_diversification", "max_sharpe", "min_cvar",
    "min_variance", "return_range", "risk_parity", "target_return", "target_volatility",
    "penalized_mean_variance", "robust_mean_variance",
]
