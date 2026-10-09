"""Robust portfolio construction and sensitivity analysis (requirement 9.1).

Implemented:
* ``robust_mean_variance``: worst-case mean-variance with an ellipsoidal
  uncertainty set on expected returns, ``max w'mu - kappa sqrt(w'Omega w) - (l/2) w'Sigma w``
  (Ceria & Stubbs, 2006; Goldfarb & Iyengar, 2003). Concave objective, linear
  constraints -> convex problem solved with SLSQP. ``Omega = diag(Sigma) * periods / T``
  (independent estimation error of each mean; cross-correlation of errors ignored,
  documented). ``kappa = z_{(1+c)/2}`` so that, for a fixed portfolio, the expected
  return lies in ``w'mu +/- kappa*se`` with confidence ``c``. NOTE: with
  Omega proportional to Sigma the robust max-Sharpe portfolio would equal the
  nominal one, which is why a diagonal Omega is used.
* ``penalized_mean_variance``: L2 concentration penalty and L1 turnover cost.
* ``resampled_weights``: Michaud-style resampling (block bootstrap of the return
  history -> re-estimate -> re-optimise) giving the DISTRIBUTION of weights.
* ``sensitivity_analysis``: estimator, window and expected-return perturbations.

A portfolio is never labelled "robust" because of weight limits alone: robustness
claims must cite the dispersion statistics produced here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from core.covariance.estimators import estimate_cov
from core.estimators.expected_returns import estimate as estimate_mu
from core.optimization.optimizers import SolverSettings, _prep, solve
from models.portfolio import OptimizationResult, PortfolioConstraints

Optimizer = Callable[[pd.Series, pd.DataFrame, PortfolioConstraints], OptimizationResult]


def robust_mean_variance(mu, cov, pc: PortfolioConstraints, n_obs: int, rf: float = 0.0,
                         risk_aversion: float = 4.0, confidence: float = 0.90, periods: int = 252,
                         settings: SolverSettings | None = None) -> OptimizationResult:
    m, c = _prep(mu, cov, pc)
    assert m is not None
    kappa = float(stats.norm.ppf((1 + confidence) / 2))
    omega_diag = np.diag(c) * periods / n_obs

    def f(w: np.ndarray) -> float:
        unc = float(np.sqrt(max(np.sum(omega_diag * w ** 2), 1e-18)))
        return -(float(w @ m) - kappa * unc - 0.5 * risk_aversion * float(w @ c @ w))

    def g(w: np.ndarray) -> np.ndarray:
        unc = float(np.sqrt(max(np.sum(omega_diag * w ** 2), 1e-18)))
        return -(m - kappa * omega_diag * w / unc - risk_aversion * c @ w)

    res = solve("robust_mean_variance", f, g, pc, m, c, rf, settings)
    if res.success:
        w = res.weights.to_numpy()
        res.diagnostics.update({
            "kappa": kappa, "confidence": confidence, "risk_aversion": risk_aversion,
            "worst_case_return": float(w @ m - kappa * np.sqrt(np.sum(omega_diag * w ** 2))),
            "nominal_return": float(w @ m)})
    return res


def penalized_mean_variance(mu, cov, pc: PortfolioConstraints, rf: float = 0.0,
                            risk_aversion: float = 4.0, l2_penalty: float = 0.0,
                            turnover_cost: float = 0.0, settings: SolverSettings | None = None
                            ) -> OptimizationResult:
    """max w'mu - (l/2) w'Sw - g ||w||^2 - c sum|w - w_prev| (cost needs previous_weights)."""
    m, c = _prep(mu, cov, pc)
    assert m is not None

    def f(w: np.ndarray) -> float:
        return -(float(w @ m) - 0.5 * risk_aversion * float(w @ c @ w) - l2_penalty * float(w @ w))

    def g(w: np.ndarray) -> np.ndarray:
        return -(m - risk_aversion * c @ w - 2 * l2_penalty * w)

    return solve("penalized_mean_variance", f, g, pc, m, c, rf, settings,
                 turnover_penalty=turnover_cost if pc.previous_weights is not None else 0.0)


def _block_resample(r: pd.DataFrame, block: int, rng: np.random.Generator) -> pd.DataFrame:
    t = len(r)
    n_blocks = int(np.ceil(t / block))
    starts = rng.integers(0, t, n_blocks)
    idx = ((starts[:, None] + np.arange(block)[None, :]) % t).reshape(-1)[:t]
    return pd.DataFrame(r.to_numpy()[idx], columns=r.columns)


@dataclass
class ResamplingResult:
    base: OptimizationResult
    weights: pd.DataFrame                  # one row per successful resample
    summary: pd.DataFrame                  # mean/std/p5/p95 per asset
    average_portfolio: pd.Series           # Michaud average (feasible: convex combination)
    stability_l1: float                    # mean L1 distance to base weights
    n_failed: int
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def resampled_weights(returns: pd.DataFrame, optimizer: Optimizer, pc: PortfolioConstraints,
                      mu_method: str, cov_method: str, n_resamples: int = 100, block: int = 21,
                      seed: int = 42, periods: int = 252) -> ResamplingResult:
    rng = np.random.default_rng(seed)
    mu0 = estimate_mu(returns, mu_method, periods)
    cov0 = estimate_cov(returns, cov_method, periods).cov
    base = optimizer(mu0, cov0, pc)
    rows, mets, failed = [], [], 0
    for _ in range(n_resamples):
        rb = _block_resample(returns, block, rng)
        try:
            mu_b = estimate_mu(rb, mu_method, periods)
            cov_b = estimate_cov(rb, cov_method, periods).cov
            res = optimizer(mu_b, cov_b, pc)
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        if not (res.success and res.feasible):
            failed += 1
            continue
        rows.append(res.weights)
        w = res.weights.to_numpy()
        # evaluate the resampled portfolio under the ORIGINAL estimates
        mets.append({"ret_under_base": float(w @ mu0.to_numpy()),
                     "vol_under_base": float(np.sqrt(w @ cov0.to_numpy() @ w))})
    wdf = pd.DataFrame(rows).reset_index(drop=True)
    if wdf.empty:
        summ = pd.DataFrame()
        avg = pd.Series(np.nan, index=pc.symbols)
        stab = float("nan")
    else:
        summ = pd.DataFrame({"base": base.weights, "mean": wdf.mean(), "std": wdf.std(ddof=1),
                             "p05": wdf.quantile(0.05), "p95": wdf.quantile(0.95)})
        avg = wdf.mean()
        avg = avg / avg.sum()
        stab = float(np.mean(np.abs(wdf.to_numpy() - base.weights.to_numpy()).sum(axis=1))) \
            if base.success else float("nan")
    return ResamplingResult(base, wdf, summ, avg, stab, failed, pd.DataFrame(mets))


def sensitivity_analysis(returns: pd.DataFrame, optimizer: Optimizer, pc: PortfolioConstraints,
                         mu_method: str, cov_method: str, rf: float, periods: int = 252,
                         n_mu_draws: int = 50, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Re-optimise under alternative estimators, windows and perturbed expected returns."""
    out: dict[str, pd.DataFrame] = {}
    base_mu = estimate_mu(returns, mu_method, periods)
    base_cov = estimate_cov(returns, cov_method, periods).cov
    base = optimizer(base_mu, base_cov, pc)
    bw = base.weights.to_numpy() if base.success else None

    def _row(label: str, res: OptimizationResult) -> dict[str, object]:
        w = res.weights.to_numpy()
        d = float(np.abs(w - bw).sum()) if (bw is not None and res.success) else float("nan")
        return {"variant": label, "success": res.success and res.feasible,
                "exp_ret_base_inputs": float(w @ base_mu.to_numpy()) if res.success else np.nan,
                "vol_base_inputs": float(np.sqrt(w @ base_cov.to_numpy() @ w)) if res.success else np.nan,
                "l1_vs_base": d, **{f"w_{k}": float(v) for k, v in res.weights.items()}}

    rows = []
    for cm in ("sample", "ledoit_wolf", "oas", "ewma", "factor_pca"):
        rows.append(_row(f"cov={cm}", optimizer(base_mu, estimate_cov(returns, cm, periods).cov, pc)))
    for mm in ("historical", "ewma", "james_stein", "grand_mean"):
        rows.append(_row(f"mu={mm}", optimizer(estimate_mu(returns, mm, periods), base_cov, pc)))
    out["estimators"] = pd.DataFrame(rows).set_index("variant")

    rows = []
    n = len(returns)
    windows = {"primeira metade": returns.iloc[: n // 2], "segunda metade": returns.iloc[n // 2:],
               "últimos 252": returns.iloc[-252:] if n > 300 else returns}
    for lab, sub in windows.items():
        try:
            res = optimizer(estimate_mu(sub, mu_method, periods),
                            estimate_cov(sub, cov_method, periods).cov, pc)
            rows.append(_row(f"janela={lab}", res))
        except ValueError:
            rows.append({"variant": f"janela={lab}", "success": False})
    out["windows"] = pd.DataFrame(rows).set_index("variant")

    rng = np.random.default_rng(seed)
    se = np.sqrt(np.diag(base_cov.to_numpy()) * periods / n)
    rows = []
    for i in range(n_mu_draws):
        mu_p = base_mu + rng.standard_normal(len(se)) * se
        rows.append(_row(f"mu_perturbado_{i}", optimizer(mu_p, base_cov, pc)))
    pert = pd.DataFrame(rows).set_index("variant")
    out["mu_perturbation"] = pert
    wcols = [c for c in pert.columns if c.startswith("w_")]
    out["mu_perturbation_summary"] = pd.DataFrame({
        "base": [float(base.weights.get(c[2:], np.nan)) for c in wcols],
        "mean": pert[wcols].mean().to_numpy(), "std": pert[wcols].std(ddof=1).to_numpy(),
        "min": pert[wcols].min().to_numpy(), "max": pert[wcols].max().to_numpy()},
        index=[c[2:] for c in wcols])
    return out
