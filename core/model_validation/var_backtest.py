"""Out-of-sample VaR/ES backtesting (Kupiec, Christoffersen, Acerbi-Szekely Z2).

Rolling one-step-ahead forecasts are produced with data up to t-1 only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from core.extreme_risk.var_es import empirical_var_es, fit_student_t, student_t_var_es
from utils.validation import InsufficientDataError, as_1d_float

Forecaster = Callable[[np.ndarray, float], tuple[float, float]]


def kupiec_pof(breaches: np.ndarray, alpha: float) -> dict[str, float]:
    """Unconditional coverage LR test; H0: breach probability = 1 - alpha."""
    i = np.asarray(breaches, bool)
    n, x = len(i), int(i.sum())
    p = 1 - alpha
    phat = x / n
    def ll(q: float) -> float:
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)
    lr = -2 * (ll(p) - ll(phat))
    return {"n": n, "breaches": x, "expected": n * p, "rate": phat, "lr_uc": float(lr),
            "p_value": float(stats.chi2.sf(lr, 1))}


def christoffersen(breaches: np.ndarray, alpha: float) -> dict[str, float]:
    """Independence and conditional-coverage LR tests (first-order Markov)."""
    i = np.asarray(breaches, int)
    n00 = int(np.sum((i[:-1] == 0) & (i[1:] == 0)))
    n01 = int(np.sum((i[:-1] == 0) & (i[1:] == 1)))
    n10 = int(np.sum((i[:-1] == 1) & (i[1:] == 0)))
    n11 = int(np.sum((i[:-1] == 1) & (i[1:] == 1)))
    def lg(q: float, a: int, b: int) -> float:
        q = min(max(q, 1e-12), 1 - 1e-12)
        return a * np.log(1 - q) + b * np.log(q)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (lg(p1, n00 + n10, n01 + n11) - lg(p01, n00, n01) - lg(p11, n10, n11))
    uc = kupiec_pof(i.astype(bool), alpha)
    lr_cc = uc["lr_uc"] + lr_ind
    return {"lr_ind": float(lr_ind), "p_value_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_value_cc": float(stats.chi2.sf(lr_cc, 2)),
            "p_breach_after_breach": p11, "p_breach_after_calm": p01}


def acerbi_szekely_z2(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Z2 = 1 - sum(L_t I_t / ES_t) / (T (1-alpha)). E[Z2]=0 if ES is correct; Z2<0 => ES too low."""
    l = np.asarray(losses)
    ind = l > np.asarray(var)
    return float(1 - np.sum(l * ind / np.asarray(es)) / (len(l) * (1 - alpha)))


def _hist(x: np.ndarray, a: float) -> tuple[float, float]:
    return empirical_var_es(-x, a)


def _normal(x: np.ndarray, a: float) -> tuple[float, float]:
    mu, sd = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(a)
    return float(-mu + sd * z), float(-mu + sd * stats.norm.pdf(z) / (1 - a))


def _ewma_normal(x: np.ndarray, a: float, lam: float = 0.94) -> tuple[float, float]:
    w = lam ** np.arange(len(x) - 1, -1, -1)
    w /= w.sum()
    sd = float(np.sqrt(w @ (x ** 2)))
    z = stats.norm.ppf(a)
    return float(sd * z), float(sd * stats.norm.pdf(z) / (1 - a))


def _student_t(x: np.ndarray, a: float) -> tuple[float, float]:
    f = fit_student_t(x)
    e = student_t_var_es(f["loc"], f["scale"], f["df"], a)
    return float(e.var), float(e.es)


FORECASTERS: dict[str, Forecaster] = {
    "historical": _hist, "normal": _normal, "ewma_normal": _ewma_normal, "student_t": _student_t,
}


def backtest_var_models(returns: pd.Series | np.ndarray, alpha: float = 0.99, window: int = 500,
                        refit_every: int = 5, models: list[str] | None = None) -> pd.DataFrame:
    """Rolling OOS VaR/ES forecasts and coverage tests for each model."""
    x = as_1d_float(returns, "returns")
    if len(x) < window + 250:
        raise InsufficientDataError(f"VaR backtest needs >= {window + 250} observations")
    models = models or list(FORECASTERS)
    rows = []
    for m in models:
        f = FORECASTERS[m]
        var_f = np.empty(len(x) - window)
        es_f = np.empty(len(x) - window)
        cur = (np.nan, np.nan)
        for j, t in enumerate(range(window, len(x))):
            if j % refit_every == 0 or m == "ewma_normal":
                cur = f(x[t - window:t], alpha)
            var_f[j], es_f[j] = cur
        losses = -x[window:]
        br = losses > var_f
        uc = kupiec_pof(br, alpha)
        cc = christoffersen(br, alpha)
        rows.append({"model": m, "alpha": alpha, "n": uc["n"], "breaches": uc["breaches"],
                     "expected": uc["expected"], "kupiec_p": uc["p_value"], "christoffersen_cc_p": cc["p_value_cc"],
                     "independence_p": cc["p_value_ind"], "z2_es": acerbi_szekely_z2(losses, var_f, es_f, alpha),
                     "avg_var": float(var_f.mean())})
    return pd.DataFrame(rows).set_index("model")
