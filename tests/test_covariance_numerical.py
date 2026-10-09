import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from core.covariance import estimators as ce
from utils.numerical import NumericalError, matrix_diagnostics, nearest_psd, safe_cholesky


def _rets(rng, t=400, n=5):
    a = rng.normal(size=(n, n))
    return pd.DataFrame(rng.normal(size=(t, n)) @ a * 0.01, columns=list("ABCDE")[:n])


@pytest.mark.parametrize("method", list(ce.ESTIMATORS))
def test_all_estimators_symmetric_psd(rng, method):
    r = _rets(rng)
    res = ce.estimate_cov(r, method)
    d = matrix_diagnostics(res.cov.to_numpy())
    assert d.is_symmetric and d.is_psd and d.is_finite


def test_sample_matches_numpy(rng):
    r = _rets(rng)
    np.testing.assert_allclose(ce.sample_cov(r, 252).cov.to_numpy(),
                               np.cov(r.to_numpy(), rowvar=False) * 252, rtol=1e-12)


def test_ledoit_wolf_matches_sklearn(rng):
    r = _rets(rng)
    np.testing.assert_allclose(ce.ledoit_wolf_cov(r, 1).cov.to_numpy(),
                               LedoitWolf().fit(r.to_numpy()).covariance_, rtol=1e-12)


def test_lw_constant_corr_matches_loop_formula(rng):
    r = _rets(rng, t=120, n=4)
    x = r.to_numpy() - r.to_numpy().mean(axis=0)
    t, n = x.shape
    s = x.T @ x / t
    sd = np.sqrt(np.diag(s))
    rbar = sum(s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else rbar * sd[i] * sd[j] for j in range(n)] for i in range(n)])
    pi = sum(np.mean((x[:, i] * x[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))
    rho = sum(np.mean((x[:, i] * x[:, i] - s[i, i]) ** 2) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            th_ii = np.mean((x[:, i] ** 2 - s[i, i]) * (x[:, i] * x[:, j] - s[i, j]))
            th_jj = np.mean((x[:, j] ** 2 - s[j, j]) * (x[:, i] * x[:, j] - s[i, j]))
            rho += rbar / 2 * (np.sqrt(s[j, j] / s[i, i]) * th_ii + np.sqrt(s[i, i] / s[j, j]) * th_jj)
    gamma = np.sum((f - s) ** 2)
    delta = max(0, min(1, (pi - rho) / gamma / t))
    expected = (delta * f + (1 - delta) * s) * t / (t - 1)
    res = ce.lw_constant_corr_cov(r, 1)
    assert res.params["shrinkage"] == pytest.approx(delta, rel=1e-10)
    np.testing.assert_allclose(res.cov.to_numpy(), expected, rtol=1e-10)


def test_singular_sample_covariance_flagged_and_psd(rng):
    r = _rets(rng, t=4, n=5)  # T < N -> singular
    res = ce.sample_cov(r)
    assert res.diagnostics.rank < 5
    assert any("T=" in w for w in res.warnings)
    lw = ce.ledoit_wolf_cov(r)
    assert lw.diagnostics.min_eigenvalue > 0  # shrinkage restores positive definiteness


def test_nearest_psd_repairs_and_documents():
    m = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.9], [0.7, -0.9, 1.0]])  # not PSD
    assert matrix_diagnostics(m).min_eigenvalue < 0
    rep = nearest_psd(m)
    assert rep.was_repaired and rep.frobenius_change > 0
    assert matrix_diagnostics(rep.matrix).is_psd


def test_safe_cholesky_semidefinite_and_rejects_indefinite():
    v = np.array([[1.0], [2.0]])
    m = v @ v.T  # rank-1 PSD, plain Cholesky fails
    l, method = safe_cholesky(m)
    assert method == "eigen_psd"
    np.testing.assert_allclose(l @ l.T, m, atol=1e-12)
    with pytest.raises(NumericalError):
        safe_cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_oos_evaluation_runs(rets):
    t = ce.evaluate_oos(rets.iloc[:600], lookback=252, horizon=63, step=126)
    assert set(t.index) == set(ce.ESTIMATORS) and (t["gmv_realized_vol"] > 0).all()


def test_expected_return_estimators(rets):
    from core.estimators import expected_returns as er

    js = er.james_stein_mean(rets)
    assert 0 <= js.attrs["shrinkage_intensity"] <= 1
    h = er.historical_mean(rets)
    # shrinkage moves every estimate toward the common target
    tgt = js.attrs["shrinkage_target_annual"]
    assert np.all(np.abs(js - tgt) <= np.abs(h - tgt) + 1e-12)
    ev = er.evaluate_oos(rets, step=126)
    assert "grand_mean" in ev.index and (ev["mse"] > 0).all()
