"""Volatility models: historical, EWMA (RiskMetrics) and GARCH(1,1)/GJR-GARCH(1,1).

All forecasts are one-step-ahead *variance* forecasts made with information up
to t-1. Out-of-sample comparison uses MSE against squared returns and QLIKE
(``log h + r^2/h``), which is robust to the noise of the squared-return proxy
(Patton, 2011). A Diebold-Mariano test (Newey-West variance) checks whether the
difference in losses is statistically significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from utils.validation import InsufficientDataError, as_1d_float


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    gamma: float           # 0 for symmetric GARCH; leverage term for GJR
    mu: float
    loglik: float
    n_obs: int
    converged: bool
    model: str
    last_variance: float   # h_{T+1|T} (forecast for the next period)
    warnings: list[str] = field(default_factory=list)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def unconditional_variance(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1 else float("inf")

    @property
    def half_life(self) -> float:
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    @property
    def aic(self) -> float:
        k = 4 + (1 if self.model == "gjr" else 0)
        return 2 * k - 2 * self.loglik

    @property
    def bic(self) -> float:
        k = 4 + (1 if self.model == "gjr" else 0)
        return k * np.log(self.n_obs) - 2 * self.loglik

    def forecast(self, steps: int) -> np.ndarray:
        """Multi-step variance forecasts h_{T+1..T+steps}."""
        p = self.persistence
        hu = self.unconditional_variance
        k = np.arange(steps)
        if p >= 1:
            return self.last_variance + self.omega * k
        return hu + p ** k * (self.last_variance - hu)


def garch_filter(e: np.ndarray, omega: float, alpha: float, beta: float, gamma: float,
                 h0: float) -> np.ndarray:
    """Return h_1..h_{T+1}: h_t is the variance of e_t given e_1..e_{t-1}."""
    t = len(e)
    h = np.empty(t + 1)
    h[0] = h0
    for i in range(t):
        h[i + 1] = omega + (alpha + gamma * (e[i] < 0)) * e[i] ** 2 + beta * h[i]
    return h


def _negloglik(theta: np.ndarray, e: np.ndarray, h0: float, gjr: bool) -> float:
    omega, alpha, beta = theta[0], theta[1], theta[2]
    gamma = theta[3] if gjr else 0.0
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta + 0.5 * gamma) >= 0.9999 or alpha + gamma < 0:
        return 1e10
    h = garch_filter(e, omega, alpha, beta, gamma, h0)[:-1]
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e10
    return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(h) + e ** 2 / h))


def fit_garch(returns: np.ndarray | pd.Series, model: str = "garch", min_obs: int = 250) -> GarchFit:
    """Gaussian quasi-maximum-likelihood fit (constant mean). Scales to percent internally."""
    x = as_1d_float(returns, "returns")
    if len(x) < min_obs:
        raise InsufficientDataError(f"GARCH needs >= {min_obs} observations (got {len(x)})")
    gjr = model == "gjr"
    scale = 100.0
    y = x * scale
    mu = float(y.mean())
    e = y - mu
    v = float(e.var())
    h0 = v
    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95), (0.15, 0.70)):
        th0 = [v * (1 - a0 - b0), a0, b0] + ([0.05] if gjr else [])
        r = minimize(_negloglik, np.array(th0), args=(e, h0, gjr), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    th = best.x
    omega, alpha, beta = th[0], th[1], th[2]
    gamma = th[3] if gjr else 0.0
    h = garch_filter(e, omega, alpha, beta, gamma, h0)
    warnings = []
    if alpha + beta + 0.5 * gamma > 0.995:
        warnings.append("persistência próxima de 1 (quase IGARCH): previsões de longo prazo instáveis")
    if alpha < 1e-4:
        warnings.append("alpha ~ 0: sem evidência de clusters de volatilidade; GARCH degenera para constante")
    s2 = scale ** 2
    return GarchFit(omega=omega / s2, alpha=float(alpha), beta=float(beta), gamma=float(gamma),
                    mu=mu / scale, loglik=float(-best.fun + len(e) * np.log(scale)),
                    n_obs=len(e), converged=bool(best.success), model=model,
                    last_variance=float(h[-1] / s2), warnings=warnings)


def ewma_variance_forecasts(r: np.ndarray, lam: float = 0.94, init: int = 30) -> np.ndarray:
    """h_t (forecast of var(r_t) made at t-1) for t = 0..T-1; first ``init`` are NaN."""
    x = as_1d_float(r, "returns")
    h = np.full(len(x), np.nan)
    if len(x) <= init:
        raise InsufficientDataError("not enough data to initialise EWMA")
    cur = float(np.mean(x[:init] ** 2))
    for t in range(init, len(x)):
        h[t] = cur
        cur = lam * cur + (1 - lam) * x[t] ** 2
    return h


def rolling_variance_forecasts(r: np.ndarray, window: int = 63) -> np.ndarray:
    x = as_1d_float(r, "returns")
    s = pd.Series(x).rolling(window).var(ddof=1).shift(1)
    return s.to_numpy()


def garch_forecasts_oos(r: np.ndarray, lookback: int = 750, refit_every: int = 63,
                        model: str = "garch") -> np.ndarray:
    """One-step forecasts with periodic re-estimation on a rolling window (no look-ahead)."""
    x = as_1d_float(r, "returns")
    h = np.full(len(x), np.nan)
    for t0 in range(lookback, len(x), refit_every):
        fit = fit_garch(x[t0 - lookback:t0], model=model)
        e_all = x[t0 - lookback:min(len(x), t0 + refit_every)] - fit.mu
        hh = garch_filter(e_all, fit.omega, fit.alpha, fit.beta, fit.gamma,
                          h0=float(np.var(x[t0 - lookback:t0])))
        # hh[k] is the variance of e_all[k] given e_all[:k]
        h[t0:min(len(x), t0 + refit_every)] = hh[lookback:lookback + min(refit_every, len(x) - t0)]
    return h


def qlike(r2: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.log(h) + r2 / h


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None) -> dict[str, float]:
    """DM test of equal predictive accuracy; negative stat => model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        raise InsufficientDataError("DM test needs >= 30 paired losses")
    lag = lag if lag is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    gamma0 = float(dc @ dc / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        g = float(dc[k:] @ dc[:-k] / n)
        lrv += 2 * (1 - k / (lag + 1)) * g
    stat = float(d.mean() / np.sqrt(max(lrv, 1e-300) / n))
    p = float(2 * (1 - stats.norm.cdf(abs(stat))))
    return {"dm_stat": stat, "p_value": p, "mean_loss_diff": float(d.mean()), "n": float(n)}


def compare_volatility_models(r: np.ndarray, lam: float = 0.94, window: int = 63,
                              garch_lookback: int = 750, refit_every: int = 63,
                              include_gjr: bool = True) -> dict[str, object]:
    """Out-of-sample comparison of variance forecasts on a common evaluation sample."""
    x = as_1d_float(r, "returns")
    if len(x) < garch_lookback + 100:
        raise InsufficientDataError(
            f"need >= {garch_lookback + 100} observations for OOS volatility comparison")
    fc = {
        "rolling_hist": rolling_variance_forecasts(x, window),
        "ewma": ewma_variance_forecasts(x, lam),
        "garch": garch_forecasts_oos(x, garch_lookback, refit_every, "garch"),
    }
    if include_gjr:
        fc["gjr_garch"] = garch_forecasts_oos(x, garch_lookback, refit_every, "gjr")
    mask = np.all([np.isfinite(v) & (v > 0) for v in fc.values()], axis=0)
    r2 = x[mask] ** 2
    table = []
    losses = {}
    for k, h in fc.items():
        hm = h[mask]
        ql = qlike(r2, hm)
        losses[k] = ql
        table.append({"model": k, "n_eval": int(mask.sum()), "mse": float(np.mean((r2 - hm) ** 2)),
                      "qlike": float(np.mean(ql))})
    df = pd.DataFrame(table).set_index("model").sort_values("qlike")
    dm = {}
    for k in losses:
        if k != "ewma":
            dm[f"{k}_vs_ewma"] = diebold_mariano(losses[k], losses["ewma"])
    return {"table": df, "diebold_mariano": dm, "baseline": "ewma",
            "eval_start_index": int(np.argmax(mask))}
