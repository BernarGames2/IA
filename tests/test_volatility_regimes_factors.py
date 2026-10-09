import numpy as np
import pandas as pd
import pytest

from core.factors.models import factor_regression, pca_factors
from core.model_validation.var_backtest import backtest_var_models, christoffersen, kupiec_pof
from core.regimes.models import fit_hmm, markov_summary, select_n_states, volatility_rule_regimes
from core.volatility.models import (
    diebold_mariano,
    ewma_variance_forecasts,
    fit_garch,
    garch_filter,
    garch_forecasts_oos,
)


def sim_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    h = omega / (1 - alpha - beta)
    x = np.empty(n)
    for t in range(n):
        x[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * x[t] ** 2 + beta * h
    return x


def test_garch_recovers_parameters():
    x = sim_garch(6000, 2e-6, 0.08, 0.90)
    f = fit_garch(x)
    assert f.alpha == pytest.approx(0.08, abs=0.03)
    assert f.beta == pytest.approx(0.90, abs=0.04)
    assert f.unconditional_variance == pytest.approx(1e-4, rel=0.3)
    fc = f.forecast(1000)
    assert fc[-1] == pytest.approx(f.unconditional_variance, rel=0.05)


def test_garch_filter_and_ewma_recursions():
    e = np.array([0.01, -0.02, 0.005])
    h = garch_filter(e, 1e-6, 0.1, 0.8, 0.0, 1e-4)
    assert h[1] == pytest.approx(1e-6 + 0.1 * 1e-4 + 0.8 * 1e-4)
    assert len(h) == 4
    x = np.full(40, 0.01)
    hh = ewma_variance_forecasts(x, 0.94, init=30)
    assert np.isnan(hh[:30]).all() and hh[30] == pytest.approx(1e-4)


def test_garch_insufficient_data_and_oos_no_lookahead():
    from utils.validation import InsufficientDataError

    with pytest.raises(InsufficientDataError):
        fit_garch(np.random.default_rng(0).normal(size=100))
    x = sim_garch(1200, 2e-6, 0.08, 0.9, seed=1)
    h1 = garch_forecasts_oos(x, lookback=750, refit_every=100)
    y = x.copy()
    y[1000:] *= 10  # future perturbation
    h2 = garch_forecasts_oos(y, lookback=750, refit_every=100)
    np.testing.assert_allclose(h1[:1001], h2[:1001], equal_nan=True)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.1, 500)
    b = a + 0.05 + rng.normal(0, 0.01, 500)
    r = diebold_mariano(a, b)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.01


def test_hmm_recovers_two_regimes():
    rng = np.random.default_rng(3)
    n = 3000
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() > (0.01 if s[t - 1] == 0 else 0.04) else 1 - s[t - 1]
    x = rng.normal(0, np.where(s == 0, 0.005, 0.02))
    f = fit_hmm(x, 2)
    assert np.sqrt(f.variances) == pytest.approx([0.005, 0.02], rel=0.15)
    acc = np.mean(f.smoothed.argmax(axis=1) == s)
    assert acc > 0.9
    tbl, fits = select_n_states(x, (1, 2))
    assert tbl.loc[2, "bic"] < tbl.loc[1, "bic"]  # regimes beat the single-state baseline


def test_hmm_filtered_is_real_time():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.01, 1000)
    f1 = fit_hmm(x, 2, n_restarts=1)
    # filtered probability at t uses only x[:t+1]: recompute with truncated data and same params
    from core.regimes.models import _emission, _forward_backward

    b = _emission(x[:500], f1.means, f1.variances)
    alpha, *_ = _forward_backward(b, f1.transition, f1.initial)
    np.testing.assert_allclose(alpha[-1], f1.filtered[499], atol=1e-10)


def test_vol_rule_uses_past_only_and_markov_summary():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 800), index=pd.bdate_range("2020-01-01", periods=800))
    lab1 = volatility_rule_regimes(r)
    r2 = r.copy()
    r2.iloc[600:] *= 5
    lab2 = volatility_rule_regimes(r2)
    pd.testing.assert_series_equal(lab1.iloc[:600], lab2.iloc[:600])
    m = markov_summary(np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0]))
    assert np.allclose(np.nansum(m["transition"], axis=1), 1)


def test_var_coverage_tests():
    br = np.zeros(1000, bool)
    br[::100] = True  # 10 breaches, 1% rate
    k = kupiec_pof(br, 0.99)
    assert k["breaches"] == 10 and k["p_value"] > 0.9
    k2 = kupiec_pof(np.r_[np.ones(60, bool), np.zeros(940, bool)], 0.99)
    assert k2["p_value"] < 1e-6
    c = christoffersen(np.r_[np.ones(10, int), np.zeros(990, int)], 0.99)
    assert c["p_value_ind"] < 0.01  # clustered breaches
    x = np.random.default_rng(6).normal(0, 0.01, 1000)
    t = backtest_var_models(x, 0.99, window=500, refit_every=20, models=["historical", "normal"])
    assert set(t.index) == {"historical", "normal"} and (t["n"] == 500).all()


def test_factor_regression_recovers_beta():
    rng = np.random.default_rng(7)
    f = pd.DataFrame({"MKT": rng.normal(0, 0.01, 1000)})
    y = pd.Series(0.0001 + 1.3 * f["MKT"] + rng.normal(0, 0.005, 1000))
    res = factor_regression(y, f)
    assert res["coef"]["MKT"] == pytest.approx(1.3, abs=0.05)
    assert res["t_hac"]["MKT"] > 10
    p = pca_factors(pd.DataFrame(rng.normal(size=(500, 4))), 2)
    assert len(p["explained"]) == 2 and p["explained"].sum() < 1


def test_vectorised_garch_filter_matches_reference_loop():
    from core.volatility.models import garch_filter_loop

    e = np.random.default_rng(9).normal(0, 0.01, 3000)
    for gamma in (0.0, 0.06):
        np.testing.assert_allclose(garch_filter(e, 1e-6, 0.07, 0.9, gamma, 1e-4),
                                   garch_filter_loop(e, 1e-6, 0.07, 0.9, gamma, 1e-4), rtol=1e-12)
