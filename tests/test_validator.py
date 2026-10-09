import numpy as np
import pandas as pd

from core.model_validation import independent_validator as iv


def test_detects_weight_and_constraint_violations():
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["a", "b"], columns=["a", "b"])
    bad = pd.Series({"a": 0.8, "b": 0.3})
    checks = iv.check_portfolio("x", bad, cov, np.zeros(2), np.ones(2) * 0.7, [], None, None)
    assert checks[0].status == iv.Status.FAIL and checks[0].critical


def test_detects_wrong_reported_volatility_and_rc():
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["a", "b"], columns=["a", "b"])
    w = pd.Series({"a": 0.5, "b": 0.5})
    ok = iv.check_portfolio("x", w, cov, np.zeros(2), np.ones(2), [], float(np.sqrt(0.02)),
                            pd.Series({"a": np.sqrt(0.02) / 2, "b": np.sqrt(0.02) / 2}))
    assert all(c.status == iv.Status.PASS for c in ok)
    wrong = iv.check_portfolio("x", w, cov, np.zeros(2), np.ones(2), [], 0.5, None)
    assert wrong[1].status == iv.Status.FAIL


def test_non_psd_covariance_fails():
    c = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]])
    assert iv.check_covariance(c).status == iv.Status.FAIL


def test_var_es_incoherence_fails():
    t = pd.DataFrame([{"method": "h", "confidence": 0.95, "horizon": "1d", "var": 0.02, "es": 0.01}])
    assert iv.check_var_es(t).status == iv.Status.FAIL


def test_overall_gating():
    rep = iv.ValidationReport([iv.Check("a", iv.Status.PASS, True, ""), iv.Check("b", iv.Status.WARNING, False, "")])
    assert rep.overall == iv.Status.WARNING and rep.validated
    rep.checks.append(iv.Check("c", iv.Status.FAIL, True, "boom"))
    assert rep.overall == iv.Status.FAIL and not rep.validated
    rep2 = iv.ValidationReport([iv.Check("c", iv.Status.FAIL, False, "non critical")])
    assert rep2.validated


def test_mc_reproducibility_check():
    def ok(seed, n_paths):
        return np.random.default_rng(seed).normal(size=n_paths)

    assert iv.check_mc_reproducibility(ok).status == iv.Status.PASS
    state = {"n": 0}

    def broken(seed, n_paths):
        state["n"] += 1
        return np.random.default_rng(state["n"]).normal(size=n_paths)

    assert iv.check_mc_reproducibility(broken).status == iv.Status.FAIL


def test_data_checks_flag_synthetic_and_currency():
    p = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.1]}, index=pd.bdate_range("2020-01-01", periods=2))
    cs = iv.check_data(p, True, "ok", {"a": "BRL", "b": "USD"}, "BRL", 1)
    by = {c.name: c for c in cs}
    assert by["dados: unidades/moeda consistentes"].status == iv.Status.FAIL
    assert by["dados: origem"].status == iv.Status.WARNING
