"""Value-at-Risk and Expected Shortfall (historical, parametric, simulated).

Convention: loss ``L = -R`` (R = simple return over the stated horizon).
``VaR_a = inf{l : P(L <= l) >= a}`` (lower a-quantile of L, numpy method
``inverted_cdf``). ``ES_a`` is the Acerbi-Tasche tail mean, which is coherent and
handles atoms/ties in discrete distributions exactly:

    ES_a = ( E[L 1{L > VaR}] + VaR * (P(L <= VaR) - a) ) / (1 - a)

Every estimate carries method, confidence level, horizon and sample size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from utils.validation import InsufficientDataError, ValidationError, as_1d_float


@dataclass
class RiskEstimate:
    method: str
    confidence: float
    horizon: str
    var: float | None
    es: float | None
    n_obs: int | None = None
    params: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValidationError("confidence must be in (0, 1)")


def empirical_var_es(losses: NDArray[np.floating], alpha: float) -> tuple[float, float]:
    """Exact VaR/ES of the empirical distribution of ``losses`` (equal weights)."""
    _check_alpha(alpha)
    x = as_1d_float(losses, "losses")
    var = float(np.quantile(x, alpha, method="inverted_cdf"))
    f = float(np.mean(x <= var))
    es = (float(np.mean(x * (x > var))) + var * (f - alpha)) / (1.0 - alpha)
    return var, float(es)


def historical_var_es(returns: NDArray[np.floating], alpha: float, horizon: str = "1 período",
                      min_tail_obs: int = 10) -> RiskEstimate:
    x = as_1d_float(returns, "returns")
    n_tail = len(x) * (1 - alpha)
    if n_tail < 1:
        raise InsufficientDataError(
            f"historical VaR at {alpha:.1%} needs >= {int(np.ceil(1 / (1 - alpha)))} observations")
    var, es = empirical_var_es(-x, alpha)
    w = []
    if n_tail < min_tail_obs:
        w.append(f"apenas {n_tail:.1f} observações na cauda: estimativa muito imprecisa")
    return RiskEstimate("historical", alpha, horizon, var, es, len(x), warnings=w)


def normal_var_es(mu: float, sigma: float, alpha: float, horizon: str = "1 período") -> RiskEstimate:
    """Gaussian VaR/ES for returns ~ N(mu, sigma^2)."""
    _check_alpha(alpha)
    if sigma < 0:
        raise ValidationError("sigma must be >= 0")
    z = stats.norm.ppf(alpha)
    var = -mu + sigma * z
    es = -mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    return RiskEstimate("normal", alpha, horizon, float(var), float(es),
                        params={"mu": mu, "sigma": sigma})


def t_scale_from_std(std: float, df: float) -> float:
    """Scale s of a Student-t with given standard deviation: Var = s^2 df/(df-2)."""
    if df <= 2:
        raise ValidationError("variance of Student-t is finite only for df > 2")
    return float(std * np.sqrt((df - 2.0) / df))


def student_t_var_es(loc: float, scale: float, df: float, alpha: float,
                     horizon: str = "1 período") -> RiskEstimate:
    """VaR/ES for returns R = loc + scale*T_df (scale is NOT the std. dev.)."""
    _check_alpha(alpha)
    if df <= 1:
        raise ValidationError("ES of Student-t is finite only for df > 1")
    if scale < 0:
        raise ValidationError("scale must be >= 0")
    q = stats.t.ppf(alpha, df)
    es_std = stats.t.pdf(q, df) / (1 - alpha) * (df + q ** 2) / (df - 1)
    var = -loc + scale * q
    es = -loc + scale * es_std
    w = [] if df > 2 else ["df <= 2: variância infinita"]
    return RiskEstimate("student_t", alpha, horizon, float(var), float(es),
                        params={"loc": loc, "scale": scale, "df": df}, warnings=w)


def fit_student_t(returns: NDArray[np.floating], min_df: float = 2.05,
                  start_dfs: tuple[float, ...] = (3.0, 6.0, 15.0)) -> dict[str, float]:
    """Maximum-likelihood fit of a location-scale Student-t (df bounded below by ``min_df``).

    ``scipy.stats.t.fit`` alone can stop at non-optimal points on daily-scale data
    (it returned df < 2 where the true optimum was df ~ 3), so the log-likelihood is
    maximised directly with Nelder-Mead over (log(df - min_df), loc, log scale)
    from several starting dfs seeded with robust moments; scipy's own fit is kept
    as one more candidate and the best log-likelihood wins.
    """
    from scipy.optimize import minimize

    x = as_1d_float(returns, "returns")
    if len(x) < 50:
        raise InsufficientDataError("Student-t fit needs at least 50 observations")
    med = float(np.median(x))
    sd = float(x.std(ddof=1))
    if sd <= 0:
        raise InsufficientDataError("Student-t fit undefined for zero variance")

    def nll(theta: NDArray[np.float64]) -> float:
        df = min_df + float(np.exp(theta[0]))
        val = -float(np.sum(stats.t.logpdf(x, df, theta[1], float(np.exp(theta[2])))))
        return val if np.isfinite(val) else 1e300

    candidates: list[tuple[float, float, float]] = []
    for df0 in start_dfs:
        s0 = sd * np.sqrt((df0 - 2.0) / df0) if df0 > 2 else 0.7 * sd
        r = minimize(nll, np.array([np.log(df0 - min_df), med, np.log(s0)]), method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000, "maxfev": 8000})
        candidates.append((min_df + float(np.exp(r.x[0])), float(r.x[1]), float(np.exp(r.x[2]))))
    df_s, loc_s, scale_s = stats.t.fit(x)
    if df_s >= min_df:
        candidates.append((float(df_s), float(loc_s), float(scale_s)))
    lls = [float(np.sum(stats.t.logpdf(x, d, l, sc))) for d, l, sc in candidates]
    k = int(np.argmax(lls))
    df, loc, scale = candidates[k]
    return {"df": df, "loc": loc, "scale": scale, "loglik": lls[k]}


def cornish_fisher_var_es(returns: NDArray[np.floating], alpha: float,
                          horizon: str = "1 período", grid: int = 2000) -> RiskEstimate:
    """Cornish-Fisher (skew/kurtosis-adjusted) VaR; ES by averaging tail quantiles."""
    x = as_1d_float(returns, "returns")
    if len(x) < 100:
        raise InsufficientDataError("Cornish-Fisher needs at least 100 observations")
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    s, k = float(stats.skew(x)), float(stats.kurtosis(x))  # excess kurtosis

    def q_loss(p: NDArray[np.float64]) -> NDArray[np.float64]:
        z = stats.norm.ppf(1 - p)  # left tail of returns
        zcf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
               - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        return -(mu + sd * zcf)

    var = float(q_loss(np.array([alpha]))[0])
    ps = alpha + (1 - alpha) * (np.arange(grid) + 0.5) / grid
    qs = q_loss(ps)
    w = []
    if np.any(np.diff(qs) < 0):
        w.append("expansão de Cornish-Fisher não monotônica: assimetria/curtose fora do domínio válido")
    return RiskEstimate("cornish_fisher", alpha, horizon, var, float(qs.mean()), len(x),
                        params={"mu": mu, "sigma": sd, "skew": s, "excess_kurtosis": k}, warnings=w)


def simulated_var_es(sim_returns: NDArray[np.floating], alpha: float, horizon: str,
                     model: str) -> RiskEstimate:
    x = as_1d_float(sim_returns, "simulated returns")
    var, es = empirical_var_es(-x, alpha)
    return RiskEstimate(f"simulated:{model}", alpha, horizon, var, es, len(x))


def sqrt_time_scaled(est: RiskEstimate, h: int) -> RiskEstimate:
    """Scale a 1-period Gaussian VaR/ES by sqrt(h).

    Only valid for i.i.d. zero-mean normal returns; the result is labelled with
    that assumption. For other models aggregate returns or simulate instead.
    """
    if est.method != "normal":
        raise ValidationError("sqrt-time scaling is only allowed for the normal model")
    mu = est.params["mu"] * h
    sigma = est.params["sigma"] * np.sqrt(h)
    out = normal_var_es(mu, sigma, est.confidence, horizon=f"{h} períodos (escala sqrt(h))")
    out.warnings.append("escala sqrt(h) assume retornos i.i.d. normais; ignora autocorrelação e "
                        "clusters de volatilidade")
    return out


def var_es_table(returns: NDArray[np.floating], levels: list[float], horizon: str = "1 dia"
                 ) -> list[RiskEstimate]:
    """Historical, normal, Student-t and Cornish-Fisher estimates for each level."""
    x = as_1d_float(returns, "returns")
    out: list[RiskEstimate] = []
    tfit = None
    try:
        tfit = fit_student_t(x)
    except InsufficientDataError:
        pass
    for a in levels:
        try:
            out.append(historical_var_es(x, a, horizon))
        except InsufficientDataError as e:
            out.append(RiskEstimate("historical", a, horizon, None, None, len(x), warnings=[str(e)]))
        out.append(normal_var_es(float(x.mean()), float(x.std(ddof=1)), a, horizon))
        if tfit is not None:
            e = student_t_var_es(tfit["loc"], tfit["scale"], tfit["df"], a, horizon)
            e.n_obs = len(x)
            out.append(e)
        try:
            out.append(cornish_fisher_var_es(x, a, horizon))
        except InsufficientDataError as e2:
            out.append(RiskEstimate("cornish_fisher", a, horizon, None, None, len(x), warnings=[str(e2)]))
    return out
