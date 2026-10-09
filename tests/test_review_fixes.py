"""Regression tests for defects found by the adversarial review workflow."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.backtesting.walk_forward import BacktestSettings, walk_forward
from core.extreme_risk.reverse_stress import gaussian_reverse_stress
from core.extreme_risk.var_es import fit_student_t
from core.market_data.base import RawSeries
from core.market_data.cache import SeriesCache
from core.model_validation.model_selection import select_strategy
from core.optimization import optimizers as o
from core.optimization.constraints import check_feasibility
from core.simulation.engine import SimulationSettings, level_key, simulate_portfolio
from core.simulation.models import GBMModel
from models.portfolio import PortfolioConstraints


def test_student_t_fit_is_at_least_as_good_as_scipy_and_accurate():
    x = stats.t.rvs(3.0, loc=2e-4, scale=3.5e-3, size=1260, random_state=7)
    f = fit_student_t(x)
    df_s, loc_s, sc_s = stats.t.fit(x)
    assert f["loglik"] >= float(np.sum(stats.t.logpdf(x, df_s, loc_s, sc_s))) - 1e-6
    assert 2.3 < f["df"] < 4.2
    # the optimum is interior: perturbing df lowers the likelihood
    for d in (f["df"] * 0.8, f["df"] * 1.25):
        assert np.sum(stats.t.logpdf(x, d, f["loc"], f["scale"])) < f["loglik"]


def test_selection_does_not_reward_volatility_when_all_lose_to_rf():
    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (700, 1))
    r = pd.DataFrame({"A": 0.0001 + 0.002 * base[:, 0], "B": 0.0001 + 0.002 * base[:, 0]}, index=idx)
    safe = lambda t, p: np.array([1.0, 0.0])          # noqa: E731  (low vol, same mean)
    risky = lambda t, p: np.array([0.0, 1.0])         # noqa: E731
    r["B"] = 0.00005 + 0.01 * rng.normal(0, 1, 700)  # lower mean AND higher vol: dominated
    bt = walk_forward(r, {"equal_weight": lambda t, p: np.array([0.5, 0.5]), "min_variance": safe,
                          "max_sharpe": risky},
                      BacktestSettings(lookback=100, rebalance_every=20, cost_bps=0, slippage_bps=0, rf_annual=0.10))
    sel = select_strategy(bt, "equal_weight")
    assert sel["ranking_metric"] == "retorno_excedente_anual"
    assert sel["selected"] != "max_sharpe"
    assert "warning" in sel and "p_holm" in sel


def test_reverse_stress_flags_impossible_shocks():
    w = pd.Series({"a": 0.5, "b": 0.5})
    cov = pd.DataFrame([[0.25, 0.0], [0.0, 0.0001]], index=w.index, columns=w.index)
    out = gaussian_reverse_stress(w, pd.Series(0.0, index=w.index), cov, 0.6)
    assert out["impossible_components"] and not out["plausible_as_price_shock"] and "warning" in out


def test_level_keys_never_collide():
    assert level_key(0.975) == "0.975" and level_key(0.995) != level_key(0.99)
    m = GBMModel(np.array([0.05]), np.array([[0.04]]))
    res = simulate_portfolio(m, np.array([1.0]), SimulationSettings(n_paths=300, n_steps=20, initial_capital=1.0,
                                                                    confidence_levels=(0.975, 0.995)))
    assert set(res.summary["var_es_horizon"]["levels"]) == {"0.975", "0.995"}


def test_turnover_conflict_is_diagnosed():
    pc = PortfolioConstraints.long_only(["a", "b", "c"], max_weight=0.5)
    pc.previous_weights = np.array([1.0, 0.0, 0.0])   # violates max 50%: must trade at least 100%
    pc.max_turnover = 0.2
    f = check_feasibility(pc)
    assert not f.feasible and any("turnover" in c for c in f.conflicts)
    assert f.nearest_weights is not None and any("turnover" in v for v in f.violations)


def test_min_cvar_and_return_range_respect_turnover():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, (400, 3)) * np.array([1, 2, 3])
    pc = PortfolioConstraints.long_only(["a", "b", "c"])
    pc.previous_weights = np.array([0.0, 0.0, 1.0])
    pc.max_turnover = 0.3
    res = o.min_cvar(r, pc, 0.95)
    assert res.success and np.abs(res.weights.to_numpy() - pc.previous_weights).sum() <= 0.3 + 1e-6
    lo, hi = o.return_range(np.array([0.01, 0.02, 0.03]), pc)
    assert hi == pytest.approx(0.03) and lo >= 0.03 - 0.3 / 2 * 0.02 - 1e-9


def test_equal_weight_without_covariance_has_no_fabricated_vol():
    res = o.equal_weight(PortfolioConstraints.long_only(["a", "b"]))
    assert np.isnan(res.volatility) and res.sharpe is None


def test_max_sharpe_warns_when_no_feasible_portfolio_beats_rf():
    mu = pd.Series({"a": 0.02, "b": 0.20})
    cov = pd.DataFrame(np.diag([0.01, 0.09]), index=mu.index, columns=mu.index)
    pc = PortfolioConstraints(["a", "b"], np.array([0.0, 0.0]), np.array([1.0, 0.1]))
    res = o.max_sharpe(mu, cov, pc, rf=0.10)   # best attainable = 0.9*0.02+0.1*0.2 = 3.8% < 10%
    assert any("máximo atingível" in w for w in res.warnings)


def test_corrupted_cache_entry_is_a_miss(tmp_path):
    c = SeriesCache(tmp_path, ttl_hours=1)
    s = RawSeries("A", pd.Series([1.0, 2.0], index=pd.bdate_range("2022-01-03", periods=2)), None, "t", "close",
                  "none", "BRL", "1d")
    c.put(s, "1y", "1d")
    meta = next(tmp_path.glob("*.json"))
    meta.write_text("{not json")
    assert c.get("t", "A", "1y", "1d") is None
    c.put(s, "1y", "1d")
    d = json.loads(meta.read_text())
    del d["provider"]
    meta.write_text(json.dumps(d))
    assert c.get("t", "A", "1y", "1d") is None


def test_offline_run_uses_estimator_drift_and_correct_beta(tmp_path):
    from config import AppConfig
    from core.portfolio_engine import run_analysis
    from core.profiler import demo_profile

    cfg = AppConfig(offline=True, run_heavy_analyses=False, n_simulations=500, make_charts=False,
                    output_dir=tmp_path, profile_horizon_simulation=False)
    res = run_analysis(cfg, demo_profile())
    dr = res.sections["mc_drift"]
    assert dr["mode"] == "estimator"
    w = res.selected.weights.to_numpy()
    assert dr["portfolio_expected_return_E"] == pytest.approx(float(w @ res.mu.to_numpy()))
    # continuous drift implied by the [E] mean: P*log1p(mu/P) ~ mu
    assert dr["portfolio_drift_continuous"] == pytest.approx(dr["portfolio_expected_return_E"], abs=2e-3)


def test_analyst_budget_is_per_question(tmp_path):
    from config import AppConfig
    from core.ai_analyst.analyst import ResearchAnalyst
    from core.ai_analyst.tools import AnalysisContext
    from core.portfolio_engine import run_analysis
    from core.profiler import demo_profile

    cfg = AppConfig(offline=True, run_heavy_analyses=False, n_simulations=500, make_charts=False,
                    output_dir=tmp_path, profile_horizon_simulation=False)
    a = ResearchAnalyst(AnalysisContext.from_result(run_analysis(cfg, demo_profile())), max_tool_calls=2)
    for _ in range(3):   # would raise ToolError on the 2nd question before the fix
        ans = a.ask("Qual o risco, o estresse e a simulação? Compare as carteiras.")
        assert ans.verified
    assert any("não executada" in x for x in ans.limitations)
