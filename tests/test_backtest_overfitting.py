import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.backtesting.overfitting import (
    StrategyRegistry,
    benjamini_hochberg,
    dsr,
    expected_max_sharpe,
    holm,
    psr,
)
from core.backtesting.walk_forward import (
    BacktestSettings,
    lookahead_invariance_check,
    run_strategy,
    walk_forward,
)
from core.model_validation.model_selection import select_simplest, select_strategy


def rets(n=600, k=3, seed=0):
    r = np.random.default_rng(seed).normal(0.0003, 0.01, size=(n, k))
    return pd.DataFrame(r, index=pd.bdate_range("2020-01-01", periods=n), columns=list("ABC")[:k])


S = BacktestSettings(lookback=100, rebalance_every=20, cost_bps=10, slippage_bps=0, validation_frac=0.6)


def test_strategy_sees_only_past_data():
    r = rets()
    seen = []

    def spy(train, prev):
        seen.append(train.index.max())
        return np.full(3, 1 / 3)

    bt = run_strategy("spy", spy, r, S)
    for decision_date, last_seen in zip(bt.weights.index, seen):
        assert last_seen < decision_date  # information strictly before the trading day


def test_lookahead_strategy_detected_by_invariance_check():
    r = rets()

    def honest(train, prev):
        m = train.mean().to_numpy()
        w = np.exp(m * 100)
        return w / w.sum()

    assert lookahead_invariance_check(r, honest, S, cut=350)["passed"]
    full = r.copy()

    def cheater(train, prev):  # peeks at the global frame -> future leakage
        nxt = full.loc[full.index > train.index.max()].iloc[:20].mean().to_numpy()
        w = np.exp(nxt * 1000)
        return w / w.sum()

    a = run_strategy("c", cheater, r, S).weights
    full.iloc[351:] = 0.0  # perturb the future
    b = run_strategy("c", cheater, r, S).weights
    common = [d for d in a.index if r.index.get_loc(d) <= 350]
    assert np.abs(a.loc[common].to_numpy() - b.loc[common].to_numpy()).max() > 1e-6


def test_costs_and_turnover_accounting():
    r = pd.DataFrame(0.0, index=pd.bdate_range("2020-01-01", periods=200), columns=list("AB"))
    flip = {"i": 0}

    def alternate(train, prev):
        flip["i"] += 1
        return np.array([1.0, 0.0]) if flip["i"] % 2 else np.array([0.0, 1.0])

    bt = run_strategy("alt", alternate, r, BacktestSettings(lookback=100, rebalance_every=10, cost_bps=10,
                                                             slippage_bps=0))
    assert bt.turnover.iloc[0] == pytest.approx(1.0)          # initial purchase
    assert bt.turnover.iloc[10] == pytest.approx(2.0)         # full switch
    assert bt.net_returns.iloc[10] == pytest.approx(-2.0 * 0.001)
    assert bt.gross_returns.abs().sum() == 0


def test_weights_drift_between_rebalances():
    idx = pd.bdate_range("2020-01-01", periods=140)
    r = pd.DataFrame({"A": 0.01, "B": 0.0}, index=idx)
    bt = run_strategy("ew", lambda t, p: np.array([0.5, 0.5]), r,
                      BacktestSettings(lookback=100, rebalance_every=30, cost_bps=0, slippage_bps=0))
    # day 2: drifted weight of A = 0.5*1.01/(1.005)
    assert bt.gross_returns.iloc[1] == pytest.approx(0.01 * 0.5 * 1.01 / 1.005)


def test_walk_forward_split_and_selection_prefers_simple_when_equal():
    r = rets(800)
    strats = {"equal_weight": lambda t, p: np.full(3, 1 / 3),
              "min_variance": lambda t, p: np.array([0.34, 0.33, 0.33])}
    bt = walk_forward(r, strats, S, synthetic=True)
    assert bt.oos_start < bt.validation_end < bt.strategies["equal_weight"].net_returns.index[-1]
    assert any("SINTÉTICOS" in w for w in bt.bias_warnings) and any("point-in-time" in w for w in bt.bias_warnings)
    sel = select_strategy(bt, "equal_weight")
    assert sel["selected"] == "equal_weight"
    sel2 = select_strategy(bt, "equal_weight", eligible=["equal_weight"])
    assert sel2["excluded_reference_strategies"] == ["min_variance"]


def test_psr_matches_formula_and_dsr_penalises_trials():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 1000)
    sr = x.mean() / x.std(ddof=1)
    g3, k = stats.skew(x), stats.kurtosis(x, fisher=False)
    expected = stats.norm.cdf(sr * np.sqrt(999) / np.sqrt(1 - g3 * sr + (k - 1) / 4 * sr ** 2))
    assert psr(x) == pytest.approx(expected)
    one = dsr(x, [sr])
    many = dsr(x, list(rng.normal(0, 0.05, 100)))
    assert many["dsr"] < psr(x) and one["sr_star_per_period"] == 0.0
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0


def test_multiple_testing_corrections():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    h = holm(p)
    assert h["a"] == pytest.approx(0.04) and h["c"] == pytest.approx(0.09) and h["b"] == pytest.approx(0.09)
    bh = benjamini_hochberg(p)
    assert bh["a"] == pytest.approx(0.04) and bh["d"] == pytest.approx(0.20)
    assert all(bh[k] <= h[k] + 1e-12 for k in p)


def test_registry_logs_every_trial(tmp_path):
    reg = StrategyRegistry(tmp_path / "r.jsonl")
    reg.record("a", {"x": 1}, {"sharpe": np.float64(0.5)}, "hash", "validation")
    reg.record("b", {"x": 2}, {"sharpe": 0.1}, "hash", "validation")
    assert reg.n_trials == 2 and len((tmp_path / "r.jsonl").read_text().splitlines()) == 2


def test_select_simplest_one_se_rule():
    t = pd.DataFrame({"loss": [1.00, 0.98, 1.20], "se": [0.05, 0.05, 0.05]}, index=["sample", "factor_pca", "ewma"])
    out = select_simplest(t, "loss", "se", {"sample": 0, "factor_pca": 2, "ewma": 1})
    assert out["best"] == "factor_pca" and out["selected"] == "sample"
