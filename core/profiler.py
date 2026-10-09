"""Investor profiling: initial, configurable quantitative classification.

This is NOT regulatory suitability (e.g. CVM Res. 30 in Brazil). It produces a
research classification and research limits. The psychological tolerance score
only *suggests* a band; the final profile is the most restrictive of that band and
independent caps from loss capacity, horizon, liquidity need, declared drawdown,
objective and experience. Every cap that binds is reported as a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.investor import (
    Experience,
    InvestorProfileInput,
    LossCapacity,
    Objective,
    ProfileAssessment,
    ResearchLimits,
    RiskProfile,
)

DISCLAIMER = (
    "Classificação quantitativa inicial para pesquisa e simulação. Não constitui "
    "suitability regulatório, recomendação de investimento nem garantia de resultado."
)


@dataclass
class ProfilerSettings:
    """Configurable thresholds (documented defaults, not regulatory rules)."""

    band_edges: tuple[float, float] = (34.0, 67.0)
    horizon_cap_conservative_years: float = 2.0
    horizon_cap_moderate_years: float = 5.0
    liquidity_cap_conservative: float = 0.50
    liquidity_cap_moderate: float = 0.20
    drawdown_cap_conservative: float = 0.10
    drawdown_cap_moderate: float = 0.20
    limits: dict[RiskProfile, ResearchLimits] = field(default_factory=lambda: {
        RiskProfile.CONSERVADOR: ResearchLimits(
            max_weight_per_asset=0.35,
            max_class_weight={"equity": 0.30, "crypto": 0.0, "real_estate": 0.15,
                              "commodity": 0.10},
            min_class_weight={"fixed_income": 0.40},
            target_volatility_band=(0.02, 0.08), max_drawdown_reference=0.10),
        RiskProfile.MODERADO: ResearchLimits(
            max_weight_per_asset=0.30,
            max_class_weight={"equity": 0.60, "crypto": 0.05, "real_estate": 0.20,
                              "commodity": 0.15},
            min_class_weight={"fixed_income": 0.15},
            target_volatility_band=(0.06, 0.14), max_drawdown_reference=0.20),
        RiskProfile.ARROJADO: ResearchLimits(
            max_weight_per_asset=0.40,
            max_class_weight={"equity": 0.90, "crypto": 0.10, "real_estate": 0.30,
                              "commodity": 0.20},
            min_class_weight={},
            target_volatility_band=(0.10, 0.25), max_drawdown_reference=0.35),
    })


def score_band(score: float, settings: ProfilerSettings | None = None) -> RiskProfile:
    """Map a 0-100 tolerance score to a *suggested* band (0-33, 34-66, 67-100)."""
    s = settings or ProfilerSettings()
    if not 0.0 <= score <= 100.0:
        raise ValueError("score must be in [0, 100]")
    lo, hi = s.band_edges
    if score < lo:
        return RiskProfile.CONSERVADOR
    if score < hi:
        return RiskProfile.MODERADO
    return RiskProfile.ARROJADO


def assess_profile(inp: InvestorProfileInput,
                   settings: ProfilerSettings | None = None) -> ProfileAssessment:
    """Classify an investor and derive research limits, reporting conflicts."""
    s = settings or ProfilerSettings()
    suggested = score_band(inp.risk_tolerance_score, s)
    caps: list[tuple[RiskProfile, str]] = []

    if inp.loss_capacity == LossCapacity.LOW:
        caps.append((RiskProfile.CONSERVADOR, "capacidade financeira de perda baixa"))
    elif inp.loss_capacity == LossCapacity.MEDIUM:
        caps.append((RiskProfile.MODERADO, "capacidade financeira de perda média"))

    if inp.horizon_years < s.horizon_cap_conservative_years:
        caps.append((RiskProfile.CONSERVADOR,
                     f"horizonte {inp.horizon_years:g} anos < {s.horizon_cap_conservative_years:g}"))
    elif inp.horizon_years < s.horizon_cap_moderate_years:
        caps.append((RiskProfile.MODERADO,
                     f"horizonte {inp.horizon_years:g} anos < {s.horizon_cap_moderate_years:g}"))

    liq = inp.liquidity_need_fraction_12m
    if liq is not None:
        if liq >= s.liquidity_cap_conservative:
            caps.append((RiskProfile.CONSERVADOR, f"necessidade de liquidez em 12m de {liq:.0%}"))
        elif liq >= s.liquidity_cap_moderate:
            caps.append((RiskProfile.MODERADO, f"necessidade de liquidez em 12m de {liq:.0%}"))

    dd = inp.max_tolerable_drawdown
    if dd is not None:
        if dd < s.drawdown_cap_conservative:
            caps.append((RiskProfile.CONSERVADOR, f"drawdown tolerável declarado de {dd:.0%}"))
        elif dd < s.drawdown_cap_moderate:
            caps.append((RiskProfile.MODERADO, f"drawdown tolerável declarado de {dd:.0%}"))

    if inp.objective == Objective.CAPITAL_PRESERVATION:
        caps.append((RiskProfile.CONSERVADOR, "objetivo de preservação de capital"))
    elif inp.objective == Objective.INCOME:
        caps.append((RiskProfile.MODERADO, "objetivo de renda"))

    if inp.experience == Experience.NONE:
        caps.append((RiskProfile.MODERADO, "nenhuma experiência declarada em investimentos"))

    final_rank = min([suggested.rank] + [c[0].rank for c in caps])
    final = RiskProfile.from_rank(final_rank)
    caps_applied = [f"{c[1]} -> limite {c[0].value}" for c in caps if c[0].rank < suggested.rank]
    conflicts = [
        f"Tolerância psicológica sugere '{suggested.value}', mas {c[1]} limita a '{c[0].value}'."
        for c in caps if c[0].rank < suggested.rank
    ]
    if inp.objective == Objective.SPECULATION and final != RiskProfile.ARROJADO:
        conflicts.append("Objetivo especulativo é incompatível com as restrições declaradas.")
    if inp.monthly_withdrawal > 0 and inp.monthly_withdrawal * 12 > 0.10 * inp.capital:
        conflicts.append("Retiradas anuais hipotéticas superam 10% do capital: risco de "
                         "esgotamento deve ser avaliado na simulação.")

    missing = []
    if inp.max_tolerable_drawdown is None:
        missing.append("drawdown tolerável não informado: usado limite de referência do perfil")
    if inp.liquidity_need_fraction_12m is None:
        missing.append("necessidade de liquidez não informada: não foi aplicado limite de liquidez")
    if inp.experience is None:
        missing.append("experiência não informada")

    base = s.limits[final]
    max_dd = base.max_drawdown_reference
    if dd is not None and dd < max_dd:
        max_dd = dd
    limits = base.model_copy(update={"max_drawdown_reference": max_dd})

    return ProfileAssessment(
        investor_id=inp.investor_id, score_band_suggestion=suggested, final_profile=final,
        caps_applied=caps_applied, conflicts=conflicts, missing_information=missing,
        research_limits=limits, disclaimer=DISCLAIMER,
        is_synthetic_example=inp.is_synthetic_example,
    )


def demo_profile(profile: RiskProfile = RiskProfile.MODERADO) -> InvestorProfileInput:
    """Explicitly synthetic example (capital R$100.000, 5 anos, aporte R$1.000/mês)."""
    score = {RiskProfile.CONSERVADOR: 20.0, RiskProfile.MODERADO: 50.0,
             RiskProfile.ARROJADO: 80.0}[profile]
    capacity = {RiskProfile.CONSERVADOR: LossCapacity.LOW, RiskProfile.MODERADO: LossCapacity.MEDIUM,
                RiskProfile.ARROJADO: LossCapacity.HIGH}[profile]
    dd = {RiskProfile.CONSERVADOR: 0.08, RiskProfile.MODERADO: 0.20,
          RiskProfile.ARROJADO: 0.35}[profile]
    horizon = {RiskProfile.CONSERVADOR: 5.0, RiskProfile.MODERADO: 5.0,
               RiskProfile.ARROJADO: 10.0}[profile]
    obj = {RiskProfile.CONSERVADOR: Objective.CAPITAL_PRESERVATION,
           RiskProfile.MODERADO: Objective.BALANCED_GROWTH,
           RiskProfile.ARROJADO: Objective.LONG_TERM_GROWTH}[profile]
    return InvestorProfileInput(
        investor_id=f"demo-{profile.value}", horizon_years=horizon, objective=obj,
        risk_tolerance_score=score, loss_capacity=capacity, max_tolerable_drawdown=dd,
        liquidity_need_fraction_12m=0.05, capital=100_000.0, monthly_contribution=1_000.0,
        base_currency="BRL", experience=Experience.INTERMEDIATE, is_synthetic_example=True,
    )
