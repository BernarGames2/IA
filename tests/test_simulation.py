import numpy as np
import pandas as pd
import pytest

from core.simulation.engine import CashflowPlan, SimulationSettings, simulate_portfolio
from core.simulation.models import (
    BlockBootstrapModel,
    BootstrapModel,
    GBMModel,
    RegimeSwitchingModel,
    StudentTModel,
    stress_covariance,
)
from utils.numerical import matrix_diagnostics


def one(mu=0.08, sig=0.2):
    return GBMModel(np.array([mu]), np.array([[sig ** 2]]))


def run(model, w, **kw):
    s = SimulationSettings(**{"n_paths": 4000, "n_steps": 63, "initial_capital": 1.0, "rebalance_every": 0, **kw})
    return simulate_portfolio(model, np.asarray(w, float), s)


def test_reproducible_with_seed_and_shapes():
    a = run(one(), [1.0], seed=1)
    b = run(one(), [1.0], seed=1)
    c = run(one(), [1.0], seed=2)
    np.testing.assert_array_equal(a.terminal_wealth, b.terminal_wealth)
    assert not np.array_equal(a.terminal_wealth, c.terminal_wealth)
    r = one().sample(7, 11, np.random.default_rng(0))
    assert r.shape == (7, 11, 1)
    assert a.percentiles.shape[0] == len(a.time_index)


def test_batches_do_not_change_results_distributionally():
    a = run(one(), [1.0], seed=3, n_paths=6000, batch_size=6000)
    b = run(one(), [1.0], seed=3, n_paths=6000, batch_size=500)
    assert a.terminal_wealth.mean() == pytest.approx(b.terminal_wealth.mean(), rel=0.01)


def test_zero_volatility_is_deterministic():
    m = GBMModel(np.array([0.10]), np.array([[0.0]]))
    res = run(m, [1.0], n_steps=252)
    np.testing.assert_allclose(res.terminal_wealth, np.exp(0.10), rtol=1e-10)
    assert res.max_drawdown_unit.min() == pytest.approx(0.0)


def test_gbm_positive_and_matches_theory():
    mu, sig, steps = 0.08, 0.25, 252
    res = run(one(mu, sig), [1.0], n_paths=40000, n_steps=steps)
    w = res.terminal_wealth
    assert (w > 0).all()
    se = w.std(ddof=1) / np.sqrt(len(w))
    assert abs(w.mean() - np.exp(mu)) < 4 * se                      # E[S_T] = S0 e^{mu T}
    assert np.median(w) == pytest.approx(np.exp(mu - sig ** 2 / 2), rel=0.01)
    lr = np.log(w)
    assert lr.std(ddof=1) == pytest.approx(sig, rel=0.02)


def test_gbm_from_log_returns_parametrisation():
    rng = np.random.default_rng(0)
    lr = pd.DataFrame(rng.normal(0.0004, 0.01, size=(5000, 2)))
    g = GBMModel.from_log_returns(lr)
    np.testing.assert_allclose(g.mu, lr.mean() * 252 + 0.5 * np.diag(g.cov))


def test_multiasset_correlation_and_semidefinite_cov():
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])  # perfectly correlated, singular
    m = GBMModel(np.array([0.05, 0.05]), cov)
    x = np.log1p(m.sample(2000, 50, np.random.default_rng(1)).reshape(-1, 2))
    assert np.corrcoef(x.T)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_student_t_unit_variance_scaling():
    m = StudentTModel(np.array([0.0]), np.array([[0.04]]), df=5)
    x = np.log1p(m.sample(20000, 50, np.random.default_rng(2)).ravel())
    assert x.std() == pytest.approx(0.2 * np.sqrt(1 / 252), rel=0.03)
    from scipy import stats

    assert stats.kurtosis(x) > 2.0  # fat tails (theory: 6 for df=5)
    with pytest.raises(ValueError):
        StudentTModel(np.array([0.0]), np.array([[0.04]]), df=2)


def test_bootstrap_uses_only_historical_vectors():
    hist = np.array([[0.01, 0.02], [-0.03, 0.01], [0.0, -0.02]])
    s = BootstrapModel(hist).sample(50, 20, np.random.default_rng(3)).reshape(-1, 2)
    assert all(any(np.array_equal(row, h) for h in hist) for row in s)


def test_block_bootstrap_contiguous_blocks():
    hist = np.arange(100, dtype=float)[:, None]
    s = BlockBootstrapModel(hist, block_length=10).sample(5, 30, np.random.default_rng(4))[:, :, 0]
    for path in s:
        for b in range(3):
            blk = path[b * 10:(b + 1) * 10]
            assert np.all((np.diff(blk) == 1) | (np.diff(blk) == -99))


def test_stress_covariance_raises_correlation_keeps_psd():
    c = np.array([[0.04, 0.0], [0.0, 0.09]])
    s = stress_covariance(c, 0.5, 1.5)
    assert s[0, 1] / np.sqrt(s[0, 0] * s[1, 1]) == pytest.approx(0.5)
    assert np.sqrt(s[0, 0]) == pytest.approx(0.3)
    assert matrix_diagnostics(s).is_psd


def test_regime_model_transitions():
    m = RegimeSwitchingModel([np.zeros(1), np.zeros(1)], [np.eye(1) * 1e-4, np.eye(1) * 9e-4],
                             np.array([[0.99, 0.01], [0.05, 0.95]]), np.array([1.0, 0.0]))
    x = np.log1p(m.sample(500, 500, np.random.default_rng(5)))
    assert x.std() > 0.01  # mixture includes high-vol regime
    with pytest.raises(ValueError):
        RegimeSwitchingModel([np.zeros(1)] * 2, [np.eye(1)] * 2, np.array([[0.5, 0.4], [0.5, 0.5]]), np.array([1, 0]))


def test_contributions_fees_and_probabilities_accounting():
    m = GBMModel(np.array([0.0]), np.array([[0.0]]))  # zero return
    s = SimulationSettings(n_paths=200, n_steps=252, initial_capital=1000, rebalance_every=0, annual_fee_bps=0,
                           goal=12_000)
    res = simulate_portfolio(m, np.array([1.0]), s, CashflowPlan(monthly_contribution=1000))
    months = 252 // 21
    np.testing.assert_allclose(res.terminal_wealth, 1000 + 1000 * months)
    assert res.summary["net_invested"] == 1000 + 1000 * months
    assert res.summary["p_goal"] == 1.0 and res.summary["p_negative_return"] == 0.0
    fee = simulate_portfolio(m, np.array([1.0]), SimulationSettings(n_paths=10, n_steps=252, initial_capital=1000,
                                                                     rebalance_every=0, annual_fee_bps=100))
    np.testing.assert_allclose(fee.terminal_wealth, 1000 * 0.99, rtol=1e-9)


def test_withdrawals_can_ruin():
    m = GBMModel(np.array([0.0]), np.array([[0.0]]))
    res = simulate_portfolio(m, np.array([1.0]), SimulationSettings(n_paths=10, n_steps=252, initial_capital=1000,
                                                                    rebalance_every=0),
                             CashflowPlan(monthly_withdrawal=200))
    assert res.summary["p_ruin"] == 1.0 and (res.terminal_wealth == 0).all()


def test_var_es_not_from_terminal_wealth_and_labelled():
    res = run(one(), [1.0], n_paths=5000)
    sm = res.summary
    assert sm["var_es_1step"]["horizon_steps"] == 1 and sm["var_es_horizon"]["horizon_steps"] == 63
    v1 = sm["var_es_1step"]["levels"]["0.99"]
    vh = sm["var_es_horizon"]["levels"]["0.99"]
    assert v1["es"] >= v1["var"] and vh["es"] >= vh["var"] and vh["var"] > v1["var"]
    assert list(res.convergence.index)[-1] == 5000


def test_rebalancing_weights_validation():
    with pytest.raises(ValueError):
        run(GBMModel(np.zeros(2), np.eye(2) * 0.04), [0.7, 0.7])
