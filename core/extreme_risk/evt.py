"""Extreme Value Theory: Peaks Over Threshold with a Generalized Pareto tail.

For losses L above a threshold u, excesses Y = L - u ~ GPD(xi, beta).
With N observations and N_u exceedances, for a > 1 - N_u/N:

    VaR_a = u + beta/xi * ( (N/N_u * (1-a))^(-xi) - 1 )          (xi != 0)
    ES_a  = (VaR_a + beta - xi*u) / (1 - xi)                     (xi < 1)

The estimate is REFUSED (``EVTRefusal``) when exceedances are too few, and every
result carries threshold sensitivity, bootstrap uncertainty, a goodness-of-fit
statistic and extrapolation warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from utils.validation import as_1d_float


class EVTRefusal(ValueError):
    """EVT estimate refused because its preconditions are not met."""


@dataclass
class GPDFit:
    threshold: float
    threshold_quantile: float
    xi: float
    beta: float
    n_total: int
    n_exceed: int
    ks_stat: float
    ks_pvalue_approx: float
    warnings: list[str] = field(default_factory=list)


def fit_pot(losses: np.ndarray, threshold_quantile: float = 0.95, min_exceedances: int = 50) -> GPDFit:
    x = as_1d_float(losses, "losses")
    u = float(np.quantile(x, threshold_quantile))
    exc = x[x > u] - u
    if len(exc) < min_exceedances:
        raise EVTRefusal(
            f"apenas {len(exc)} excedências acima do limiar (quantil {threshold_quantile:.1%}); "
            f"mínimo exigido {min_exceedances}. Estimativa EVT recusada.")
    xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    ks = stats.kstest(exc, "genpareto", args=(xi, 0.0, beta))
    w = ["p-valor KS aproximado: parâmetros estimados na mesma amostra (teste otimista)"]
    if xi >= 0.5:
        w.append(f"xi = {xi:.2f} >= 0.5: variância da cauda infinita; ES muito incerto")
    if ks.pvalue < 0.05:
        w.append("ajuste GPD rejeitado a 5% pelo teste KS (aproximado)")
    return GPDFit(u, threshold_quantile, float(xi), float(beta), len(x), len(exc),
                  float(ks.statistic), float(ks.pvalue), w)


def gpd_var_es(fit: GPDFit, alpha: float) -> tuple[float, float | None]:
    tail_prob = fit.n_exceed / fit.n_total
    if 1 - alpha >= tail_prob:
        raise EVTRefusal(f"nível {alpha:.3f} não está além do limiar (quantil "
                         f"{fit.threshold_quantile:.3f}); use o estimador empírico")
    ratio = fit.n_total / fit.n_exceed * (1 - alpha)
    if abs(fit.xi) < 1e-8:
        var = fit.threshold - fit.beta * np.log(ratio)
    else:
        var = fit.threshold + fit.beta / fit.xi * (ratio ** (-fit.xi) - 1)
    es = (var + fit.beta - fit.xi * fit.threshold) / (1 - fit.xi) if fit.xi < 1 else None
    return float(var), (None if es is None else float(es))


@dataclass
class EVTResult:
    alpha: float
    var: float | None
    es: float | None
    fit: GPDFit | None
    ci_var: tuple[float, float] | None
    ci_es: tuple[float, float] | None
    sensitivity: list[dict[str, float | str]]
    empirical_var: float
    empirical_es: float
    warnings: list[str]
    refused: bool = False
    refusal_reason: str = ""


def evt_analysis(losses: np.ndarray, alpha: float = 0.99, threshold_quantile: float = 0.95,
                 min_exceedances: int = 50, n_boot: int = 200, seed: int = 42,
                 sensitivity_quantiles: tuple[float, ...] = (0.90, 0.925, 0.95, 0.975)) -> EVTResult:
    """Full POT analysis with refusal, sensitivity and bootstrap uncertainty."""
    from core.extreme_risk.var_es import empirical_var_es

    x = as_1d_float(losses, "losses")
    ev, ee = empirical_var_es(x, alpha)
    warnings: list[str] = []
    try:
        fit = fit_pot(x, threshold_quantile, min_exceedances)
        var, es = gpd_var_es(fit, alpha)
    except EVTRefusal as e:
        return EVTResult(alpha, None, None, None, None, None, [], ev, ee, [str(e)], True, str(e))
    warnings += fit.warnings
    if (1 - alpha) * len(x) < 1:
        warnings.append("nível de confiança além da maior observação: extrapolação sem suporte empírico")
    sens = []
    for q in sensitivity_quantiles:
        try:
            f2 = fit_pot(x, q, min_exceedances)
            v2, e2 = gpd_var_es(f2, alpha)
            sens.append({"threshold_quantile": q, "n_exceed": f2.n_exceed, "xi": f2.xi, "var": v2,
                         "es": float("nan") if e2 is None else e2})
        except EVTRefusal as e:
            sens.append({"threshold_quantile": q, "refused": str(e)})
    vs = [s["var"] for s in sens if "var" in s]
    if len(vs) >= 2 and var and (max(vs) - min(vs)) / abs(var) > 0.25:
        warnings.append("VaR EVT varia mais de 25% conforme o limiar: estimativa sensível")
    rng = np.random.default_rng(seed)
    bv, be = [], []
    for _ in range(n_boot):
        xb = x[rng.integers(0, len(x), len(x))]
        try:
            fb = fit_pot(xb, threshold_quantile, min_exceedances)
            v, e = gpd_var_es(fb, alpha)
            bv.append(v)
            if e is not None:
                be.append(e)
        except EVTRefusal:
            continue
    ci_v = (float(np.percentile(bv, 5)), float(np.percentile(bv, 95))) if len(bv) > 20 else None
    ci_e = (float(np.percentile(be, 5)), float(np.percentile(be, 95))) if len(be) > 20 else None
    return EVTResult(alpha, var, es, fit, ci_v, ci_e, sens, ev, ee, warnings)
