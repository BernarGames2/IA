"""Input validation helpers raising specific exceptions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ValidationError(ValueError):
    """Raised when user or pipeline input violates a documented precondition."""


class InsufficientDataError(ValidationError):
    """Raised when a statistic is requested with too few observations."""


def require_min_obs(n: int, minimum: int, what: str) -> None:
    if n < minimum:
        raise InsufficientDataError(f"{what}: {n} observations < required minimum {minimum}")


def as_1d_float(x: Sequence[float] | NDArray[np.floating] | pd.Series, name: str = "x") -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise InsufficientDataError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or inf")
    return arr


def validate_weights(w: NDArray[np.floating], n: int | None = None, *, allow_short: bool = False,
                     tol: float = 1e-6) -> NDArray[np.float64]:
    arr = np.asarray(w, dtype=float).reshape(-1)
    if n is not None and arr.size != n:
        raise ValidationError(f"weights length {arr.size} != number of assets {n}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("weights contain NaN or inf")
    if not allow_short and np.any(arr < -tol):
        raise ValidationError("negative weights while short selling is disabled")
    if abs(arr.sum() - 1.0) > tol:
        raise ValidationError(f"weights sum to {arr.sum():.8f}, expected 1")
    return arr


def validate_covariance(cov: NDArray[np.floating], n: int | None = None) -> NDArray[np.float64]:
    from utils.numerical import matrix_diagnostics

    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValidationError("covariance must be a square matrix")
    if n is not None and c.shape[0] != n:
        raise ValidationError(f"covariance dimension {c.shape[0]} != {n}")
    d = matrix_diagnostics(c)
    if not d.is_finite:
        raise ValidationError("covariance has NaN/inf")
    if not d.is_symmetric:
        raise ValidationError(f"covariance not symmetric (max asymmetry {d.max_asymmetry:.2e})")
    if not d.is_psd:
        raise ValidationError(f"covariance not PSD (min eigenvalue {d.min_eigenvalue:.2e})")
    return c
