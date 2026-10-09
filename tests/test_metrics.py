import numpy as np
import pandas as pd
import pytest

from core import metrics as m
from utils.validation import ValidationError

TOL = 1e-12


def test_simple_and_log_returns_manual():
    p = pd.Series([100.0, 110.0, 99.0])
    np.testing.assert_allclose(m.simple_returns(p).to_numpy(), [0.10, -0.10], atol=TOL)
    np.testing.assert_allclose(m.log_returns(p).to_numpy(), [np.log(1.1), np.log(0.9)], atol=TOL)


def test_returns_reject_nonpositive_prices():
    with pytest.raises(ValidationError):
        m.simple_returns(pd.Series([100.0, 0.0, 10.0]))


def test_missing_price_yields_missing_return_not_zero():
    p = pd.Series([100.0, np.nan, 110.0, 121.0])
    r = m.simple_returns(p)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])
    assert r.iloc[2] == pytest.approx(0.10)


def test_annualization_conventions():
    r = np.array([0.01, -0.005, 0.002, 0.0])
    assert m.annualized_arithmetic_mean(r, 252) == pytest.approx(252 * r.mean())
    geo = np.prod(1 + r) ** (252 / 4) - 1
    assert m.annualized_geometric_return(r, 252) == pytest.approx(geo, rel=1e-12)
    assert m.annualized_volatility(r, 252) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert m.total_return(r) == pytest.approx(np.prod(1 + r) - 1)


def test_rf_per_period_compounds_back():
    rp = m.rf_per_period(0.10, 252)
    assert (1 + rp) ** 252 - 1 == pytest.approx(0.10, rel=1e-12)


def test_sharpe_manual_and_undefined():
    r = np.array([0.01, 0.02, -0.01, 0.005, 0.0])
    rf = 0.05
    ex = r - m.rf_per_period(rf, 252)
    assert m.sharpe_ratio(r, rf) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    with pytest.raises(m.UndefinedMetricError):
        m.sharpe_ratio(np.full(10, 0.001), 0.0)
    with pytest.raises(m.UndefinedMetricError):
        m.sharpe_ratio(np.array([0.01]), 0.0)


def test_sortino_manual():
    r = np.array([0.02, -0.01, 0.03, -0.02])
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    assert m.downside_deviation(r, 0.0, 252) == pytest.approx(dd)
    assert m.sortino_ratio(r, 0.0) == pytest.approx(r.mean() * 252 / dd)
    with pytest.raises(m.UndefinedMetricError):
        m.sortino_ratio(np.array([0.01, 0.02, 0.03]), 0.0)


def test_beta_known_value():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 0.01, 500)
    a = 1.5 * b  # exact linear relation
    assert m.beta(a, b) == pytest.approx(1.5, abs=1e-12)
    with pytest.raises(m.UndefinedMetricError):
        m.beta(a, np.zeros(500))
    with pytest.raises(ValidationError):
        m.beta(a, b[:-1])


def test_drawdown_manual():
    r = np.array([0.10, -0.50, 0.20])   # wealth 1, 1.1, 0.55, 0.66
    dd = m.drawdown_series(r)
    np.testing.assert_allclose(dd, [0, 0, 0.55 / 1.1 - 1, 0.66 / 1.1 - 1], atol=TOL)
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    assert m.max_drawdown_duration(r) == 2
    w = np.array([[1, 2, 1, 3], [1, 1, 1, 1]], float)
    np.testing.assert_allclose(m.max_drawdown_from_wealth(w), [-0.5, 0.0])


def test_recovery_required_domain():
    assert m.recovery_required(0.5) == pytest.approx(1.0)
    assert m.recovery_required(0.0) == 0.0
    with pytest.raises(ValidationError):
        m.recovery_required(1.0)
    with pytest.raises(ValidationError):
        m.recovery_required(-0.1)


def test_risk_contribution_identity(rng):
    a = rng.normal(size=(5, 5))
    cov = a @ a.T / 10
    w = rng.dirichlet(np.ones(5))
    rc = m.risk_contributions(w, cov)
    sp = np.sqrt(w @ cov @ w)
    assert rc["rc"].sum() == pytest.approx(sp, rel=1e-12)
    assert rc["rc_pct"].sum() == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(m.UndefinedMetricError):
        m.risk_contributions(w, np.zeros((5, 5)))


def test_turnover_costs_concentration():
    w0, w1 = np.array([0.5, 0.5, 0]), np.array([0.2, 0.5, 0.3])
    assert m.turnover(w0, w1) == pytest.approx(0.6)
    assert m.transaction_cost(w0, w1, 10) == pytest.approx(0.6 * 0.001)
    c = m.concentration(np.full(4, 0.25), {"a": "x", "b": "x", "c": "y", "d": "y"}, list("abcd"))
    assert c["hhi"] == pytest.approx(0.25) and c["effective_n"] == pytest.approx(4)
    assert c["by_group"] == {"x": 0.5, "y": 0.5}


def test_aggregate_returns_non_overlapping():
    r = np.array([0.1, 0.1, -0.1, 0.2, 0.05])
    np.testing.assert_allclose(m.aggregate_returns(r, 2), [1.1 * 1.1 - 1, 0.9 * 1.2 - 1])


def test_performance_summary_reports_undefined():
    out = m.performance_summary(np.full(30, 0.001), 0.0)
    assert out["sharpe"] is None and "sharpe_undefined_reason" in out
