"""Regime identification: real-time volatility rule, Markov chains and Gaussian HMM.

* ``volatility_rule_regimes``: label "stress" when trailing volatility exceeds an
  EXPANDING quantile computed only with past data -> usable in real time.
* ``markov_summary``: transition matrix, expected durations, stationary law.
* ``GaussianHMM``: univariate K-state HMM fitted by Baum-Welch (scaled
  forward-backward, multiple restarts). ``filtered`` probabilities use data up to
  t (real-time); ``smoothed`` use the whole sample (RETROSPECTIVE, never to be
  used as information available at t). States are ordered by variance.
* ``select_n_states``: compares K = 1 (no regimes, baseline), 2, 3 by BIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.validation import InsufficientDataError, as_1d_float


def volatility_rule_regimes(r: pd.Series, window: int = 21, quantile: float = 0.80,
                            min_history: int = 252) -> pd.Series:
    """1 = stress, 0 = calm, NaN before enough history. No look-ahead."""
    vol = r.rolling(window).std(ddof=1)
    thr = vol.expanding(min_periods=min_history).quantile(quantile).shift(1)
    lab = (vol > thr).astype(float)
    lab[thr.isna() | vol.isna()] = np.nan
    return lab.rename("regime_vol_rule")


def markov_summary(labels: pd.Series | np.ndarray, k: int | None = None) -> dict[str, object]:
    x = pd.Series(labels).dropna().astype(int).to_numpy()
    if len(x) < 10:
        raise InsufficientDataError("need >= 10 labelled observations")
    k = k or int(x.max()) + 1
    counts = np.zeros((k, k))
    for a, b in zip(x[:-1], x[1:]):
        counts[a, b] += 1
    rows = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, rows, out=np.full_like(counts, np.nan), where=rows > 0)
    dur = [float(1 / (1 - p[i, i])) if np.isfinite(p[i, i]) and p[i, i] < 1 else float("inf")
           for i in range(k)]
    try:
        w, v = np.linalg.eig(np.nan_to_num(p).T)
        st = np.real(v[:, np.argmin(np.abs(w - 1))])
        st = st / st.sum()
    except np.linalg.LinAlgError:
        st = np.full(k, np.nan)
    return {"transition": p, "counts": counts, "expected_duration": dur, "stationary": st,
            "occupancy": np.bincount(x, minlength=k) / len(x)}


@dataclass
class HMMFit:
    k: int
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    loglik: float
    n_obs: int
    converged: bool
    n_iter: int
    filtered: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    smoothed: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def n_params(self) -> int:
        return self.k * (self.k - 1) + (self.k - 1) + 2 * self.k

    @property
    def bic(self) -> float:
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    @property
    def expected_durations(self) -> list[float]:
        return [float(1 / (1 - self.transition[i, i])) if self.transition[i, i] < 1 else float("inf")
                for i in range(self.k)]


def _emission(x: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (x[:, None] - means) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _forward_backward(b: np.ndarray, a: np.ndarray, pi: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t, k = b.shape
    alpha = np.empty((t, k))
    c = np.empty(t)
    alpha[0] = pi * b[0]
    c[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= c[0]
    for i in range(1, t):
        alpha[i] = (alpha[i - 1] @ a) * b[i]
        c[i] = max(alpha[i].sum(), 1e-300)
        alpha[i] /= c[i]
    beta = np.empty((t, k))
    beta[-1] = 1.0
    for i in range(t - 2, -1, -1):
        beta[i] = (a @ (b[i + 1] * beta[i + 1])) / c[i + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = a * (alpha[:-1].T @ (b[1:] * beta[1:] / c[1:, None]))
    return alpha, gamma, xi, float(np.log(c).sum())


def fit_hmm(r: np.ndarray | pd.Series, k: int = 2, n_restarts: int = 5, max_iter: int = 300,
            tol: float = 1e-6, seed: int = 42, var_floor: float = 1e-10) -> HMMFit:
    x = as_1d_float(r, "returns")
    if len(x) < 100 * k:
        raise InsufficientDataError(f"HMM with {k} states needs >= {100 * k} observations")
    rng = np.random.default_rng(seed)
    best: HMMFit | None = None
    v0 = float(x.var())
    for rs in range(n_restarts):
        qs = np.sort(rng.uniform(0.2, 0.8, k)) if rs else np.linspace(0.25, 0.75, k)
        means = np.quantile(x, qs) * 0.5
        var = v0 * np.linspace(0.5, 2.0, k) * (rng.uniform(0.8, 1.25, k) if rs else 1)
        a = np.full((k, k), 0.05 / max(k - 1, 1))
        np.fill_diagonal(a, 0.95)
        if k == 1:
            a = np.ones((1, 1))
        pi = np.full(k, 1.0 / k)
        prev = -np.inf
        conv = False
        it = 0
        for it in range(1, max_iter + 1):
            b = _emission(x, means, var)
            alpha, gamma, xi, ll = _forward_backward(b, a, pi)
            pi = gamma[0] / gamma[0].sum()
            a = xi / xi.sum(axis=1, keepdims=True)
            w = gamma.sum(axis=0)
            means = (gamma * x[:, None]).sum(axis=0) / w
            var = np.maximum((gamma * (x[:, None] - means) ** 2).sum(axis=0) / w, var_floor)
            if ll - prev < tol * max(1.0, abs(ll)):
                conv = True
                break
            prev = ll
        order = np.argsort(var)
        b = _emission(x, means[order], var[order])
        a_o = a[np.ix_(order, order)]
        alpha, gamma, _, ll = _forward_backward(b, a_o, pi[order])
        fit = HMMFit(k, means[order], var[order], a_o, pi[order], ll, len(x), conv, it, alpha, gamma)
        if best is None or fit.loglik > best.loglik:
            best = fit
    assert best is not None
    return best


def select_n_states(r: np.ndarray | pd.Series, ks: tuple[int, ...] = (1, 2, 3), seed: int = 42
                    ) -> tuple[pd.DataFrame, dict[int, HMMFit]]:
    fits: dict[int, HMMFit] = {}
    rows = []
    for k in ks:
        try:
            f = fit_hmm(r, k, seed=seed)
        except InsufficientDataError as e:
            rows.append({"k": k, "error": str(e)})
            continue
        fits[k] = f
        rows.append({"k": k, "loglik": f.loglik, "bic": f.bic, "converged": f.converged,
                     "vol_ann_by_state": np.sqrt(f.variances * 252).round(4).tolist(),
                     "expected_duration": np.round(f.expected_durations, 1).tolist()})
    return pd.DataFrame(rows).set_index("k"), fits


def state_conditional_moments(returns: pd.DataFrame, probs: np.ndarray, use_log: bool = True
                              ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Probability-weighted mean/covariance of asset (log) returns per state."""
    x = np.log1p(returns.to_numpy()) if use_log else returns.to_numpy()
    means, covs = [], []
    for j in range(probs.shape[1]):
        w = probs[:, j] / probs[:, j].sum()
        m = w @ x
        xc = x - m
        c = (xc * w[:, None]).T @ xc / max(1 - np.sum(w ** 2), 1e-12)
        means.append(m)
        covs.append(c)
    return means, covs
