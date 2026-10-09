"""Tail dependence: empirical coefficients and a Student-t copula fit.

* Empirical lower-tail dependence at level q: P(U1 <= q, U2 <= q) / q on
  rank-based pseudo-observations.
* t-copula: correlation from Kendall's tau (rho = sin(pi tau / 2)), degrees of
  freedom by profile likelihood on a grid, compared with the Gaussian copula
  (no tail dependence). Theoretical coefficient
  lambda = 2 t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from utils.numerical import nearest_psd
from utils.validation import InsufficientDataError


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return stats.rankdata(x, axis=0) / (n + 1.0)


def empirical_lower_tail_dependence(returns: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    u = pseudo_observations(returns.to_numpy())
    n = u.shape[1]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(np.mean((u[:, i] <= q) & (u[:, j] <= q)) / q)
            out[i, j] = out[j, i] = v
    return pd.DataFrame(out, index=returns.columns, columns=returns.columns)


def _t_copula_loglik(u: np.ndarray, r: np.ndarray, nu: float) -> float:
    x = stats.t.ppf(u, nu)
    d = r.shape[0]
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0:
        return -np.inf
    rinv = np.linalg.inv(r)
    q = np.einsum("ij,jk,ik->i", x, rinv, x)
    log_mv = (gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi)
              - 0.5 * logdet - (nu + d) / 2 * np.log1p(q / nu))
    log_marg = stats.t.logpdf(x, nu).sum(axis=1)
    return float(np.sum(log_mv - log_marg))


def _gauss_copula_loglik(u: np.ndarray, r: np.ndarray) -> float:
    x = stats.norm.ppf(u)
    sign, logdet = np.linalg.slogdet(r)
    rinv = np.linalg.inv(r)
    q = np.einsum("ij,jk,ik->i", x, rinv - np.eye(len(r)), x)
    return float(np.sum(-0.5 * logdet - 0.5 * q))


@dataclass
class TCopulaFit:
    nu: float
    corr: pd.DataFrame
    loglik_t: float
    loglik_gauss: float
    lambda_theoretical: pd.DataFrame
    profile: dict[float, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def lr_stat(self) -> float:
        return 2 * (self.loglik_t - self.loglik_gauss)


def fit_t_copula(returns: pd.DataFrame, min_obs: int = 250,
                 grid: tuple[float, ...] = (2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)) -> TCopulaFit:
    if len(returns) < min_obs:
        raise InsufficientDataError(f"t-copula needs >= {min_obs} observations (got {len(returns)})")
    x = returns.to_numpy()
    n = x.shape[1]
    tau = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau[i, j] = tau[j, i] = stats.kendalltau(x[:, i], x[:, j]).statistic
    r = np.sin(np.pi * tau / 2)
    rep = nearest_psd(r, min_eigenvalue=1e-6)
    r = rep.matrix
    d = np.sqrt(np.diag(r))
    r = r / np.outer(d, d)
    u = pseudo_observations(x)
    prof = {nu: _t_copula_loglik(u, r, nu) for nu in grid}
    nu_best = max(prof, key=prof.get)
    lg = _gauss_copula_loglik(u, r)
    lam = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho = min(r[i, j], 0.999999)
            lam[i, j] = 2 * stats.t.cdf(-np.sqrt((nu_best + 1) * (1 - rho) / (1 + rho)), nu_best + 1)
    w = []
    if rep.was_repaired:
        w.append(f"matriz de Kendall invertida não era PSD; reparada (Frobenius {rep.frobenius_change:.2e})")
    if nu_best == grid[-1]:
        w.append("graus de liberdade no limite superior da grade: dependência de cauda fraca/indistinguível da Gaussiana")
    cols = returns.columns
    return TCopulaFit(float(nu_best), pd.DataFrame(r, cols, cols), prof[nu_best], lg,
                      pd.DataFrame(lam, cols, cols), prof, w)
