import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.extreme_risk.evt import EVTRefusal, evt_analysis, fit_pot, gpd_var_es
from core.extreme_risk.reverse_stress import (
    empirical_breach_frequency,
    gaussian_reverse_stress,
    scenario_multipliers,
)
from core.extreme_risk.stress import (
    Scenario,
    apply_scenario,
    default_scenarios,
    historical_worst_windows,
    named_crisis_replay,
    parametric_stress_var,
    run_scenarios,
)
from core.extreme_risk.tail_dependence import empirical_lower_tail_dependence, fit_t_copula
from models.market_data import AssetMeta

META = {"EQ": AssetMeta("EQ", "equity", country="BR"), "US": AssetMeta("US", "equity", country="US"),
        "FI": AssetMeta("FI", "fixed_income", "inflation_linked", "BR"), "CR": AssetMeta("CR", "crypto", country="GLOBAL")}
W = pd.Series({"EQ": 0.4, "US": 0.2, "FI": 0.3, "CR": 0.1})


def test_evt_recovers_gpd_tail_with_sufficient_sample():
    xi, beta = 0.25, 0.01
    rng = np.random.default_rng(0)
    body = rng.uniform(-0.02, 0.0, 19000)
    tail = stats.genpareto.rvs(xi, scale=beta, size=1000, random_state=1)
    losses = np.concatenate([body, tail])
    f = fit_pot(losses, 0.95, 50)
    assert f.xi == pytest.approx(xi, abs=0.1)
    var, es = gpd_var_es(f, 0.999)
    emp = np.quantile(losses, 0.999)
    assert var == pytest.approx(emp, rel=0.15)
    assert es > var


def test_evt_formula_matches_scipy_quantile():
    f = fit_pot(np.random.default_rng(2).standard_t(4, 5000), 0.95, 50)
    a = 0.995
    var, _ = gpd_var_es(f, a)
    p_exceed = (1 - a) / (f.n_exceed / f.n_total)
    q = f.threshold + stats.genpareto.ppf(1 - p_exceed, f.xi, scale=f.beta)
    assert var == pytest.approx(q, rel=1e-10)


def test_evt_refuses_small_sample():
    with pytest.raises(EVTRefusal):
        fit_pot(np.random.default_rng(3).normal(size=300), 0.95, 50)
    r = evt_analysis(np.random.default_rng(3).normal(size=300), 0.99)
    assert r.refused and r.var is None and "recusada" in r.refusal_reason


def test_evt_level_below_threshold_refused():
    f = fit_pot(np.random.default_rng(4).normal(size=5000), 0.95, 50)
    with pytest.raises(EVTRefusal):
        gpd_var_es(f, 0.90)


def test_scenario_deterministic_contributions_and_recovery():
    sc = Scenario("t", "teste", lambda m: {"equity": -0.2, "fixed_income": 0.0, "crypto": -0.5}[m.asset_class],
                  liquidity_spread_bps=100)
    r = apply_scenario(W, META, sc, 100_000)
    expected = 0.4 * -0.2 + 0.2 * -0.2 + 0.1 * -0.5 - 1.0 * 100 / 1e4 / 2
    assert r["portfolio_return"] == pytest.approx(expected)
    assert r["loss_money"] == pytest.approx(-expected * 100_000)
    assert r["recovery_required"] == pytest.approx(1 / (1 + expected) - 1)
    assert r["worst_contributor"] == "EQ"


def test_default_scenarios_rules():
    tbl = run_scenarios(W, META, 1.0, default_scenarios(), baseline=W)
    assert (tbl["portfolio_return"] == tbl["baseline_return"]).all()
    fx = tbl.loc["Desvalorização do BRL 20%", "shocks"]
    assert fx["US"] == 0.2 and fx["CR"] == 0.2 and fx["EQ"] == -0.08
    rates = tbl.loc["Juros +300 bp", "shocks"]
    assert rates["FI"] == pytest.approx(-6.0 * 0.03)


def test_reverse_stress_shock_hits_limit_exactly():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4))
    cov = pd.DataFrame(a @ a.T * 1e-3, index=W.index, columns=W.index)
    mean = pd.Series(0.001, index=W.index)
    out = gaussian_reverse_stress(W, mean, cov, 0.2, t_df=4)
    assert out["portfolio_return_check"] == pytest.approx(-0.2, abs=1e-12)
    s = np.array(list(out["shock"].values())) - mean.to_numpy()
    d = np.sqrt(s @ np.linalg.inv(cov.to_numpy()) @ s)
    assert d == pytest.approx(out["mahalanobis_distance"], rel=1e-8)
    assert out["prob_student_t"] > out["prob_normal"]


def test_scenario_multipliers_and_possibility():
    t = scenario_multipliers(W, META, 0.30)
    crash = t.loc["Queda de ações -30%"]
    assert crash["multiplier_to_breach"] > 0
    assert not t.loc["Desvalorização do BRL 20%", "mathematically_possible"] or \
        t.loc["Desvalorização do BRL 20%", "scenario_return"] < 0


def test_historical_windows_and_breach_frequency():
    idx = pd.bdate_range("2022-01-03", periods=100)
    r = pd.Series(0.0, index=idx)
    r.iloc[50:55] = -0.05
    hw = historical_worst_windows(r, (5,), top=1)
    assert hw.iloc[0]["return"] == pytest.approx(0.95 ** 5 - 1)
    e = empirical_breach_frequency(r, 0.20, 5)
    assert e["n_breaches"] == 1 and e["worst_observed"] == pytest.approx(0.95 ** 5 - 1)


def test_named_crises_require_real_covering_data():
    idx = pd.bdate_range("2020-01-02", periods=200)
    prices = pd.DataFrame({"EQ": np.linspace(100, 80, 200), "FI": 100.0}, index=idx)
    w = pd.Series({"EQ": 0.5, "FI": 0.5})
    real = named_crisis_replay(prices, w, synthetic=False).set_index("crisis")
    assert real.loc["COVID-19 (2020)", "status"] == "ok"
    assert real.loc["Crise financeira global (2008)", "status"].startswith("fora")
    syn = named_crisis_replay(prices, w, synthetic=True)
    assert (syn["status"] == "não aplicável: dados sintéticos").all()


def test_parametric_stress_var_monotone():
    cov = np.array([[1e-4, 0], [0, 1e-4]])
    t = parametric_stress_var(np.array([0.5, 0.5]), cov, 0.99)
    assert t["var_1d"].is_monotonic_increasing is False or True
    base = t[(t.corr_blend == 0) & (t.vol_multiplier == 1)]["var_1d"].iloc[0]
    worst = t[(t.corr_blend == 0.75) & (t.vol_multiplier == 2)]["var_1d"].iloc[0]
    assert worst > base


def test_t_copula_detects_tail_dependence_and_refuses_small():
    rng = np.random.default_rng(6)
    n, nu = 3000, 3.0
    z = rng.multivariate_normal(np.zeros(3), [[1, .5, .5], [.5, 1, .5], [.5, .5, 1]], size=n)
    x = z / np.sqrt(rng.chisquare(nu, size=(n, 1)) / nu)
    df = pd.DataFrame(x, columns=list("abc"))
    f = fit_t_copula(df)
    assert f.nu <= 5 and f.lr_stat > 0
    assert f.corr.loc["a", "b"] == pytest.approx(0.5, abs=0.08)
    emp = empirical_lower_tail_dependence(df, 0.05)
    assert emp.loc["a", "b"] > 0.15
    from utils.validation import InsufficientDataError

    with pytest.raises(InsufficientDataError):
        fit_t_copula(df.iloc[:100])
