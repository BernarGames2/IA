import pytest
from pydantic import ValidationError

from config import AppConfig
from core.profiler import assess_profile, demo_profile, score_band
from models.investor import (
    Experience,
    InvestorProfileInput,
    LossCapacity,
    Objective,
    RiskProfile,
)


def base(**kw):
    d = dict(investor_id="p1", horizon_years=10, objective=Objective.LONG_TERM_GROWTH,
             risk_tolerance_score=80, loss_capacity=LossCapacity.HIGH, capital=1000.0,
             experience=Experience.ADVANCED, max_tolerable_drawdown=0.4, liquidity_need_fraction_12m=0.0)
    d.update(kw)
    return InvestorProfileInput(**d)


@pytest.mark.parametrize("score,expected", [(0, "conservador"), (33, "conservador"), (33.9, "conservador"),
                                            (34, "moderado"), (66, "moderado"), (67, "arrojado"), (100, "arrojado")])
def test_score_bands(score, expected):
    assert score_band(score).value == expected


def test_score_alone_does_not_decide():
    a = assess_profile(base(loss_capacity=LossCapacity.LOW))
    assert a.score_band_suggestion == RiskProfile.ARROJADO
    assert a.final_profile == RiskProfile.CONSERVADOR
    assert a.conflicts and a.caps_applied


@pytest.mark.parametrize("kw,expected", [
    ({"horizon_years": 1}, RiskProfile.CONSERVADOR),
    ({"horizon_years": 3}, RiskProfile.MODERADO),
    ({"liquidity_need_fraction_12m": 0.6}, RiskProfile.CONSERVADOR),
    ({"max_tolerable_drawdown": 0.15}, RiskProfile.MODERADO),
    ({"objective": Objective.CAPITAL_PRESERVATION}, RiskProfile.CONSERVADOR),
    ({"experience": Experience.NONE}, RiskProfile.MODERADO),
    ({}, RiskProfile.ARROJADO),
])
def test_independent_caps(kw, expected):
    assert assess_profile(base(**kw)).final_profile == expected


def test_missing_information_explicit_and_declared_drawdown_used():
    a = assess_profile(base(max_tolerable_drawdown=None, liquidity_need_fraction_12m=None, experience=None))
    assert len(a.missing_information) == 3
    b = assess_profile(base(max_tolerable_drawdown=0.25))
    assert b.research_limits.max_drawdown_reference == pytest.approx(0.25)


def test_pseudonym_enforced():
    with pytest.raises(ValidationError):
        base(investor_id="joao@example.com")
    with pytest.raises(ValidationError):
        base(investor_id="12345678901")
    with pytest.raises(ValidationError):
        base(unexpected_field=1)


def test_demo_profiles_are_marked_synthetic():
    for p in RiskProfile:
        d = demo_profile(p)
        assert d.is_synthetic_example and d.capital == 100_000 and d.monthly_contribution == 1_000
        assert assess_profile(d).final_profile == p


def test_config_validation_and_rf_fallback_flagged():
    c = AppConfig(offline=True)
    assert c.rf_is_fallback and c.risk_free_rate_source == "demo_fallback"
    c2 = AppConfig(risk_free_rate=0.105)
    assert not c2.rf_is_fallback
    with pytest.raises(ValidationError):
        AppConfig(tickers=["ONLY1"])
    with pytest.raises(ValidationError):
        AppConfig(tickers=["A", "A"])
    with pytest.raises(ValidationError):
        AppConfig(min_weight=-0.1)
    with pytest.raises(ValidationError):
        AppConfig(confidence_levels=[1.2])
    with pytest.raises(ValidationError):
        AppConfig(base_currency="JPY")  # no demo fallback: rf must be explicit
    assert AppConfig(base_currency="JPY", risk_free_rate=0.001).risk_free_rate == 0.001
