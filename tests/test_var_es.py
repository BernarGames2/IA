import numpy as np
import pytest
from scipy import integrate, stats

from core.extreme_risk import var_es as v
from utils.validation import InsufficientDataError, ValidationError


def test_empirical_var_es_hand_computed():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    var, es = v.empirical_var_es(losses, 0.95)
    assert var == 95.0
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_es_handles_ties_and_atoms():
    losses = np.array([0.0] * 90 + [5.0] * 10)
    var, es = v.empirical_var_es(losses, 0.95)
    assert var == 5.0 and es == pytest.approx(5.0)
    losses = np.array([1.0] * 96 + [10.0] * 4)   # VaR inside atom at 1.0
    var, es = v.empirical_var_es(losses, 0.95)
    # tail of mass 5%: 4% at 10 and 1% at 1 -> ES = (0.04*10 + 0.01*1)/0.05
    assert var == 1.0 and es == pytest.approx((0.4 + 0.01) / 0.05)


def test_es_geq_var_property(rng):
    for _ in range(20):
        x = rng.standard_t(3, size=rng.integers(50, 500))
        for a in (0.9, 0.95, 0.99):
            var, es = v.empirical_var_es(x, a)
            assert es >= var - 1e-12


def test_normal_closed_form():
    e = v.normal_var_es(0.001, 0.02, 0.99)
    z = stats.norm.ppf(0.99)
    assert e.var == pytest.approx(-0.001 + 0.02 * z)
    assert e.es == pytest.approx(-0.001 + 0.02 * stats.norm.pdf(z) / 0.01)


def test_student_t_matches_numerical_integration():
    loc, scale, df, a = 0.0005, 0.01, 4.0, 0.975
    e = v.student_t_var_es(loc, scale, df, a)
    q = stats.t.ppf(1 - a, df, loc, scale)       # return quantile (left tail)
    assert e.var == pytest.approx(-q, rel=1e-10)
    tail, _ = integrate.quad(lambda x: x * stats.t.pdf(x, df, loc, scale), -np.inf, q)
    assert e.es == pytest.approx(-tail / (1 - a), rel=1e-6)


def test_t_scale_vs_std():
    s = v.t_scale_from_std(0.02, 5)
    assert stats.t.std(5, scale=s) == pytest.approx(0.02)
    with pytest.raises(ValidationError):
        v.t_scale_from_std(0.02, 2)


def test_cornish_fisher_gaussian_sample_close_to_normal(rng):
    x = rng.normal(0, 0.01, 200_000)
    cf = v.cornish_fisher_var_es(x, 0.99)
    n = v.normal_var_es(x.mean(), x.std(ddof=1), 0.99)
    assert cf.var == pytest.approx(n.var, rel=0.02)


def test_historical_requires_enough_obs():
    with pytest.raises(InsufficientDataError):
        v.historical_var_es(np.zeros(50), 0.99)
    e = v.historical_var_es(np.linspace(-0.05, 0.05, 200), 0.99)
    assert e.warnings  # only 2 tail observations -> imprecision warning


def test_sqrt_time_only_for_normal():
    n = v.normal_var_es(0.0, 0.01, 0.99)
    s = v.sqrt_time_scaled(n, 10)
    assert s.var == pytest.approx(n.var * np.sqrt(10))
    assert "sqrt" in s.horizon and s.warnings
    with pytest.raises(ValidationError):
        v.sqrt_time_scaled(v.historical_var_es(np.linspace(-1, 1, 1000), 0.95), 10)


def test_fit_student_t_recovers_df(rng):
    x = stats.t.rvs(4, loc=0, scale=0.01, size=20000, random_state=1)
    f = v.fit_student_t(x)
    assert 3.3 < f["df"] < 4.8
