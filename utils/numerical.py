"""Numerical linear-algebra helpers with explicit, documented corrections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Default tolerances used across the project (documented in docs/architecture.md).
SYMMETRY_TOL = 1e-10
PSD_TOL = -1e-10          # eigenvalues above this are treated as non-negative
WEIGHT_SUM_TOL = 1e-6
BOUND_TOL = 1e-6


class NumericalError(ValueError):
    """Raised when a numerical precondition (finiteness, symmetry, PSD...) fails."""


@dataclass(frozen=True)
class MatrixDiagnostics:
    """Diagnostics of a (covariance) matrix."""

    n: int
    is_symmetric: bool
    max_asymmetry: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_psd: bool
    is_finite: bool
    rank: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return self.__dict__.copy()


def ensure_finite(x: NDArray[np.floating], name: str = "array") -> None:
    """Raise :class:`NumericalError` if ``x`` contains NaN or inf."""
    if not np.all(np.isfinite(x)):
        raise NumericalError(f"{name} contains NaN or inf values")


def matrix_diagnostics(m: FloatArray) -> MatrixDiagnostics:
    """Return symmetry, eigenvalue, conditioning and PSD diagnostics of ``m``."""
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise NumericalError("matrix must be square")
    finite = bool(np.all(np.isfinite(m)))
    if not finite:
        return MatrixDiagnostics(m.shape[0], False, float("nan"), float("nan"),
                                 float("nan"), float("inf"), False, False, 0)
    asym = float(np.max(np.abs(m - m.T))) if m.size else 0.0
    scale = max(1.0, float(np.max(np.abs(m)))) if m.size else 1.0
    sym = asym <= SYMMETRY_TOL * scale
    eig = np.linalg.eigvalsh((m + m.T) / 2.0)
    lo, hi = float(eig[0]), float(eig[-1])
    cond = float(hi / lo) if lo > 0 else float("inf")
    rank = int(np.sum(eig > max(1e-12 * max(abs(hi), 1.0), 0.0)))
    return MatrixDiagnostics(m.shape[0], sym, asym, lo, hi, cond,
                             lo >= PSD_TOL * max(1.0, abs(hi)), finite, rank)


@dataclass(frozen=True)
class PSDRepair:
    matrix: FloatArray
    was_repaired: bool
    frobenius_change: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    method: str


def nearest_psd(m: FloatArray, min_eigenvalue: float = 0.0) -> PSDRepair:
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping.

    This is the Frobenius-norm nearest PSD matrix (Higham, 1988) for symmetric
    input. The returned :class:`PSDRepair` documents the size of the change so the
    caller can report it instead of silently altering the estimate.
    """
    m = np.asarray(m, dtype=float)
    ensure_finite(m, "matrix")
    sym = (m + m.T) / 2.0
    w, v = np.linalg.eigh(sym)
    before = float(w[0])
    if before >= min_eigenvalue:
        return PSDRepair(sym, bool(np.max(np.abs(m - sym)) > 0), float(np.linalg.norm(m - sym)),
                         before, before, "symmetrize")
    w_c = np.clip(w, min_eigenvalue, None)
    fixed = (v * w_c) @ v.T
    fixed = (fixed + fixed.T) / 2.0
    return PSDRepair(fixed, True, float(np.linalg.norm(m - fixed)), before,
                     float(np.linalg.eigvalsh(fixed)[0]), "eigenvalue_clipping")


def safe_cholesky(cov: FloatArray, jitter_max: float = 1e-6) -> tuple[FloatArray, str]:
    """Return a factor ``L`` with ``L @ L.T ~= cov`` that is valid for PSD matrices.

    Tries a plain Cholesky first. For semidefinite (singular) matrices it falls
    back to an eigen-decomposition factor ``V diag(sqrt(max(w,0)))``, which is exact
    for PSD input and needs no jitter. Matrices with materially negative
    eigenvalues are rejected: they are not valid covariances.
    """
    cov = np.asarray(cov, dtype=float)
    ensure_finite(cov, "covariance")
    sym = (cov + cov.T) / 2.0
    try:
        return np.linalg.cholesky(sym), "cholesky"
    except np.linalg.LinAlgError:
        pass
    w, v = np.linalg.eigh(sym)
    scale = max(1.0, float(np.max(np.abs(w))))
    if w[0] < -jitter_max * scale:
        raise NumericalError(
            f"covariance is not PSD (min eigenvalue {w[0]:.3e}); repair it explicitly first"
        )
    return v * np.sqrt(np.clip(w, 0.0, None)), "eigen_psd"


def cov_to_corr(cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return (correlation, standard deviations). Zero-variance assets get NaN rows."""
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return corr, sd


def corr_to_cov(corr: FloatArray, sd: FloatArray) -> FloatArray:
    return np.asarray(corr) * np.outer(sd, sd)
