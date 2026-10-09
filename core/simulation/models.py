"""Interchangeable return-generating models for Monte Carlo simulation.

Every model produces SIMPLE returns per step with shape (paths, steps, assets).

GBM parametrisation: ``dS/S = mu dt + sigma dW`` discretised exactly as
``S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)``. Here ``mu`` is the
*continuous drift* (E[S_T/S_0] = exp(mu T)), which is NOT the arithmetic mean of
simple returns. :meth:`GBMModel.from_log_returns` estimates
``sigma^2 = 252 var(log r)``, ``mu = 252 mean(log r) + sigma^2/2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from core.volatility.models import fit_garch
from utils.numerical import cov_to_corr, nearest_psd, safe_cholesky


class ReturnModel(Protocol):
    name: str
    n_assets: int

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        ...

    def describe(self) -> dict[str, object]:
        ...


@dataclass
class GBMModel:
    mu: np.ndarray            # annual continuous drift per asset
    cov: np.ndarray           # annual covariance of LOG returns
    dt: float = 1.0 / 252
    name: str = "gbm"
    factor_method: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, float)
        self.cov = np.asarray(self.cov, float)
        self._L, self.factor_method = safe_cholesky(self.cov * self.dt)
        self._drift = (self.mu - 0.5 * np.diag(self.cov)) * self.dt

    @property
    def n_assets(self) -> int:
        return len(self.mu)

    @classmethod
    def from_log_returns(cls, log_r: pd.DataFrame, periods: int = 252) -> "GBMModel":
        cov = np.atleast_2d(np.cov(log_r.to_numpy(), rowvar=False, ddof=1)) * periods
        mu = log_r.mean().to_numpy() * periods + 0.5 * np.diag(cov)
        return cls(mu, cov, 1.0 / periods)

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n_paths, n_steps, self.n_assets))
        return np.expm1(self._drift + z @ self._L.T)

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "mu_continuous": self.mu.tolist(),
                "vol": np.sqrt(np.diag(self.cov)).tolist(), "dt": self.dt,
                "factorization": self.factor_method}


@dataclass
class StudentTModel:
    """Log returns = drift + L z, z multivariate Student-t scaled to unit variance.

    ``z = sqrt((df-2)/df) * y / sqrt(W/df)``, y ~ N(0, I), W ~ chi2(df), shared across
    assets (so tail events are joint). Requires df > 2. Note: E[exp(z)] is
    theoretically infinite for a t-distribution; simulated means are therefore
    less reliable than quantiles, which are the reported quantities.
    """

    mu: np.ndarray
    cov: np.ndarray
    df: float
    dt: float = 1.0 / 252
    name: str = "student_t"

    def __post_init__(self) -> None:
        if self.df <= 2:
            raise ValueError("Student-t df must be > 2 for finite variance")
        self.mu = np.asarray(self.mu, float)
        self.cov = np.asarray(self.cov, float)
        self._L, _ = safe_cholesky(self.cov * self.dt)
        self._drift = (self.mu - 0.5 * np.diag(self.cov)) * self.dt

    @property
    def n_assets(self) -> int:
        return len(self.mu)

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        y = rng.standard_normal((n_paths, n_steps, self.n_assets))
        w = rng.chisquare(self.df, size=(n_paths, n_steps, 1))
        z = y / np.sqrt(w / self.df) * np.sqrt((self.df - 2.0) / self.df)
        return np.expm1(self._drift + z @ self._L.T)

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "df": self.df, "mu_continuous": self.mu.tolist(),
                "vol": np.sqrt(np.diag(self.cov)).tolist()}


@dataclass
class BootstrapModel:
    """i.i.d. resampling of historical return VECTORS (keeps cross-asset dependence)."""

    returns: np.ndarray
    name: str = "bootstrap"

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self.returns), size=(n_paths, n_steps))
        return self.returns[idx]

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "n_hist": int(len(self.returns))}


@dataclass
class BlockBootstrapModel:
    """Circular moving-block bootstrap: preserves autocorrelation/vol clusters within blocks."""

    returns: np.ndarray
    block_length: int = 21
    name: str = "block_bootstrap"

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        t = len(self.returns)
        b = self.block_length
        n_blocks = int(np.ceil(n_steps / b))
        starts = rng.integers(0, t, size=(n_paths, n_blocks))
        offs = np.arange(b)
        idx = ((starts[:, :, None] + offs[None, None, :]) % t).reshape(n_paths, -1)[:, :n_steps]
        return self.returns[idx]

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "block_length": self.block_length, "n_hist": int(len(self.returns))}


@dataclass
class RegimeSwitchingModel:
    """Markov-switching Gaussian log returns with state-specific mean/covariance (per step)."""

    means: list[np.ndarray]        # per-step mean of log returns per state
    covs: list[np.ndarray]         # per-step covariance of log returns per state
    transition: np.ndarray         # P[i, j] = P(s_{t+1}=j | s_t=i)
    initial_probs: np.ndarray
    name: str = "regime"

    def __post_init__(self) -> None:
        self._L = [safe_cholesky(nearest_psd(c).matrix)[0] for c in self.covs]
        p = np.asarray(self.transition, float)
        if not np.allclose(p.sum(axis=1), 1.0):
            raise ValueError("transition rows must sum to 1")

    @property
    def n_assets(self) -> int:
        return len(self.means[0])

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        k = len(self.means)
        cum = np.cumsum(self.transition, axis=1)
        s = np.empty((n_paths, n_steps), dtype=int)
        s[:, 0] = rng.choice(k, size=n_paths, p=self.initial_probs)
        u = rng.random((n_paths, n_steps))
        for t in range(1, n_steps):
            s[:, t] = np.minimum((u[:, t, None] > cum[s[:, t - 1]]).sum(axis=1), k - 1)
        z = rng.standard_normal((n_paths, n_steps, self.n_assets))
        out = np.empty_like(z)
        for j in range(k):
            m = s == j
            out[m] = self.means[j] + z[m] @ self._L[j].T
        return np.expm1(out)

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "n_states": len(self.means),
                "transition": np.asarray(self.transition).tolist(),
                "initial_probs": np.asarray(self.initial_probs).tolist()}


@dataclass
class CCCGarchModel:
    """Constant-conditional-correlation GARCH(1,1): time-varying volatility per asset."""

    mu: np.ndarray                 # per-step mean of simple returns
    omega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    h0: np.ndarray                 # next-step conditional variance at simulation start
    corr: np.ndarray
    name: str = "garch_ccc"

    def __post_init__(self) -> None:
        self._L, _ = safe_cholesky(nearest_psd(self.corr).matrix)

    @property
    def n_assets(self) -> int:
        return len(self.mu)

    @classmethod
    def fit(cls, returns: pd.DataFrame) -> "CCCGarchModel":
        fits = [fit_garch(returns[c].to_numpy()) for c in returns.columns]
        mu = np.array([f.mu for f in fits])
        h = []
        std_resid = []
        for f, c in zip(fits, returns.columns):
            e = returns[c].to_numpy() - f.mu
            from core.volatility.models import garch_filter
            hh = garch_filter(e, f.omega, f.alpha, f.beta, f.gamma, float(np.var(e)))
            std_resid.append(e / np.sqrt(hh[:-1]))
            h.append(hh[-1])
        corr = np.corrcoef(np.array(std_resid))
        return cls(mu, np.array([f.omega for f in fits]), np.array([f.alpha for f in fits]),
                   np.array([f.beta for f in fits]), np.array(h), corr)

    def sample(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        n = self.n_assets
        out = np.empty((n_paths, n_steps, n))
        h = np.tile(self.h0, (n_paths, 1))
        for t in range(n_steps):
            z = rng.standard_normal((n_paths, n)) @ self._L.T
            e = np.sqrt(h) * z
            out[:, t, :] = self.mu + e
            h = self.omega + self.alpha * e ** 2 + self.beta * h
        return np.maximum(out, -0.999999)  # guard: simple return cannot go below -100%

    def describe(self) -> dict[str, object]:
        return {"model": self.name, "alpha": self.alpha.tolist(), "beta": self.beta.tolist(),
                "start_vol_ann": np.sqrt(self.h0 * 252).tolist()}


def stress_covariance(cov: np.ndarray, corr_blend: float = 0.0, vol_multiplier: float = 1.0
                      ) -> np.ndarray:
    """Raise correlations toward 1 (convex blend keeps PSD) and scale volatilities."""
    if not 0.0 <= corr_blend <= 1.0:
        raise ValueError("corr_blend must be in [0, 1]")
    corr, sd = cov_to_corr(np.asarray(cov, float))
    n = len(sd)
    stressed = (1 - corr_blend) * corr + corr_blend * np.ones((n, n))
    np.fill_diagonal(stressed, 1.0)
    sd2 = sd * vol_multiplier
    return stressed * np.outer(sd2, sd2)
