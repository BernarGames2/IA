import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from core.optimization import optimizers as o
from core.optimization.constraints import build_constraints, check_feasibility, verify_weights
from core.optimization.robust import penalized_mean_variance, robust_mean_variance
from models.investor import ResearchLimits
from models.market_data import AssetMeta
from models.portfolio import GroupConstraint, PortfolioConstraints

SYMS = ["A", "B", "C", "D"]
MU = pd.Series([0.08, 0.10, 0.12, 0.05], index=SYMS)
SD = np.array([0.10, 0.15, 0.25, 0.05])
CORR = np.array([[1, 0.3, 0.2, 0.0], [0.3, 1, 0.5, 0.1], [0.2, 0.5, 1, 0.0], [0.0, 0.1, 0.0, 1]])
COV = pd.DataFrame(CORR * np.outer(SD, SD), index=SYMS, columns=SYMS)
RF = 0.03


def lo(max_w=1.0):
    return PortfolioConstraints.long_only(SYMS, max_weight=max_w)


def _valid(res, pc):
    w = res.weights.to_numpy()
    assert res.success and res.feasible
    assert abs(w.sum() - 1) < 1e-8 and np.all(w >= -1e-9)
    assert not verify_weights(w, pc)


def test_gmv_two_asset_closed_form():
    s1, s2, rho = 0.2, 0.1, 0.3
    c = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    w1 = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    pc = PortfolioConstraints.long_only(["x", "y"])
    res = o.min_variance(c, pc)
    assert res.weights["x"] == pytest.approx(w1, abs=1e-6)


def test_min_variance_long_only_valid_and_optimal():
    pc = lo()
    res = o.min_variance(COV, pc, MU, RF)
    _valid(res, pc)
    rng = np.random.default_rng(0)
    for _ in range(500):
        w = rng.dirichlet(np.ones(4))
        assert w @ COV.to_numpy() @ w >= res.volatility ** 2 - 1e-10


def test_max_sharpe_matches_convex_reformulation():
    pc = lo()
    res = o.max_sharpe(MU, COV, pc, RF)
    _valid(res, pc)
    ex = MU.to_numpy() - RF
    c = COV.to_numpy()
    r = minimize(lambda y: y @ c @ y, np.full(4, 1.0), method="SLSQP", bounds=[(0, None)] * 4,
                 constraints=[{"type": "eq", "fun": lambda y: ex @ y - 1}], options={"ftol": 1e-15})
    w_ref = r.x / r.x.sum()
    np.testing.assert_allclose(res.weights.to_numpy(), w_ref, atol=1e-4)


def test_bounds_and_groups_respected():
    g = [GroupConstraint("grp", ("B", "C"), 0.2, 0.4)]
    pc = PortfolioConstraints.long_only(SYMS, max_weight=0.5, groups=g)
    for res in (o.min_variance(COV, pc, MU, RF), o.max_sharpe(MU, COV, pc, RF),
                o.risk_parity(COV, pc, MU, RF), o.max_diversification(COV, pc, MU, RF),
                robust_mean_variance(MU, COV, pc, 1000, RF)):
        _valid(res, pc)
        assert 0.2 - 1e-6 <= res.weights[["B", "C"]].sum() <= 0.4 + 1e-6


def test_infeasible_max_weight_detected_with_nearest_and_relaxation():
    pc = lo(max_w=0.2)  # 4 * 0.2 < 1
    f = check_feasibility(pc)
    assert not f.feasible
    assert any("máximos" in c for c in f.conflicts) and f.suggestions
    res = o.min_variance(COV, pc, MU, RF)
    assert not res.success and res.status == "infeasible_constraints" and not res.feasible
    assert res.violations  # nearest allocation reports its violations
    assert abs(res.weights.sum() - 1) < 1e-9 and (res.weights >= -1e-12).all()


def test_infeasible_group_minimum_detected():
    meta = {s: AssetMeta(s, "equity") for s in SYMS}
    lim = ResearchLimits(max_weight_per_asset=0.5, min_class_weight={"fixed_income": 0.4},
                         target_volatility_band=(0.05, 0.1), max_drawdown_reference=0.2)
    pc = build_constraints(SYMS, meta, lim)
    f = check_feasibility(pc)
    assert not f.feasible and any("fixed_income" in c for c in f.conflicts)


def test_target_return_and_volatility():
    pc = lo()
    lo_r, hi_r = o.return_range(MU.to_numpy(), pc)
    assert lo_r == pytest.approx(0.05) and hi_r == pytest.approx(0.12)
    res = o.target_return(MU, COV, pc, 0.09, RF)
    _valid(res, pc)
    assert res.expected_return == pytest.approx(0.09, abs=1e-7)
    bad = o.target_return(MU, COV, pc, 0.20, RF)
    assert not bad.success and bad.status == "target_out_of_range"
    tv = o.target_volatility(MU, COV, pc, 0.10, RF)
    _valid(tv, pc)
    assert tv.volatility <= 0.10 + 1e-6
    low = o.target_volatility(MU, COV, pc, 0.001, RF)
    assert low.status == "target_below_min_vol"


def test_frontier_only_valid_monotone_points():
    pc = lo()
    fr = o.efficient_frontier(MU, COV, pc, RF, n_points=15)
    assert len(fr) >= 10
    assert fr["expected_return"].is_monotonic_increasing
    assert fr["volatility"].is_monotonic_increasing
    w = fr[[c for c in fr.columns if c.startswith("w_")]]
    np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-8)
    ms = o.max_sharpe(MU, COV, pc, RF)
    assert fr["sharpe"].max() <= ms.sharpe + 1e-6


def test_risk_parity_equal_contributions():
    pc = lo()
    res = o.risk_parity(COV, pc)
    w = res.weights.to_numpy()
    c = COV.to_numpy()
    rc = w * (c @ w)
    np.testing.assert_allclose(rc / rc.sum(), 0.25, atol=1e-6)


def test_max_diversification_equals_correlation_gmv_rescaled():
    pc = lo()
    res = o.max_diversification(COV, pc)
    y = o.min_variance(CORR, lo()).weights.to_numpy()
    w = (y / SD) / np.sum(y / SD)
    np.testing.assert_allclose(res.weights.to_numpy(), w, atol=1e-4)


def test_hrp_two_assets_inverse_variance():
    c = np.diag([0.04, 0.01])
    res = o.hrp(c, PortfolioConstraints.long_only(["x", "y"]))
    np.testing.assert_allclose(res.weights.to_numpy(), [0.2, 0.8], atol=1e-12)


def test_min_cvar_lp_equals_empirical_es(rng):
    r = rng.standard_t(4, size=(500, 4)) * SD / 16 + MU.to_numpy() / 252
    pc = lo()
    res = o.min_cvar(r, pc, 0.95)
    assert res.success
    from core.extreme_risk.var_es import empirical_var_es

    _, es = empirical_var_es(-(r @ res.weights.to_numpy()), 0.95)
    assert res.diagnostics["cvar"] == pytest.approx(es, rel=1e-6)
    for _ in range(300):  # no random portfolio has lower empirical ES
        w = rng.dirichlet(np.ones(4))
        assert empirical_var_es(-(r @ w), 0.95)[1] >= es - 1e-10


def test_singular_covariance_still_feasible():
    v = np.array([[0.1], [0.2], [0.3], [0.1]])
    c = v @ v.T + np.diag([0, 0, 0, 1e-12])
    res = o.min_variance(c, lo())
    assert res.success and abs(res.weights.sum() - 1) < 1e-8


def test_non_psd_covariance_rejected():
    with pytest.raises(ValueError):
        o.min_variance(np.array([[1.0, 2.0], [2.0, 1.0]]), PortfolioConstraints.long_only(["x", "y"]))


def test_convergence_failure_is_reported():
    c = COV.to_numpy()
    res = o.solve("broken", lambda w: float("nan"), lambda w: np.full(4, np.nan), lo(), MU.to_numpy(), c, RF)
    assert not res.success and not res.feasible
    assert "nenhum ponto inicial" in res.warnings[0]
    assert res.weights.isna().all()  # no weights are fabricated


def test_turnover_constraint_respected():
    pc = lo()
    pc.previous_weights = np.array([0.25, 0.25, 0.25, 0.25])
    pc.max_turnover = 0.10
    res = o.max_sharpe(MU, COV, pc, RF)
    _valid(res, pc)
    assert np.abs(res.weights.to_numpy() - pc.previous_weights).sum() <= 0.10 + 1e-6


def test_turnover_penalty_reduces_trading():
    pc = lo()
    pc.previous_weights = np.array([0.25, 0.25, 0.25, 0.25])
    free = penalized_mean_variance(MU, COV, pc, RF, 4.0, 0.0, 0.0)
    pen = penalized_mean_variance(MU, COV, pc, RF, 4.0, 0.0, 0.05)
    to = lambda r: np.abs(r.weights.to_numpy() - pc.previous_weights).sum()  # noqa: E731
    assert to(pen) < to(free)


def test_robust_mv_more_conservative_than_nominal():
    pc = lo()
    nominal = penalized_mean_variance(MU, COV, pc, RF, 4.0)
    robust = robust_mean_variance(MU, COV, pc, 250, RF, risk_aversion=4.0, confidence=0.95)
    assert robust.diagnostics["worst_case_return"] < robust.diagnostics["nominal_return"]
    assert robust.volatility < nominal.volatility
    # robust solution maximises the worst-case objective: evaluate both portfolios on it
    kappa = robust.diagnostics["kappa"]
    om = np.diag(COV.to_numpy()) * 252 / 250
    c, m = COV.to_numpy(), MU.to_numpy()

    def wc(w):
        return w @ m - kappa * np.sqrt(np.sum(om * w ** 2)) - 2.0 * w @ c @ w

    assert wc(robust.weights.to_numpy()) >= wc(nominal.weights.to_numpy()) - 1e-9
