"""Typed investor profile, objectives and constraints.

Only data needed for a quantitative *research* classification is modelled. No name,
document number, e-mail or address is requested or stored: ``investor_id`` is a
pseudonym. Missing information is represented explicitly with ``None`` and surfaced
as a limitation by :mod:`core.profiler` instead of being guessed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskProfile(str, Enum):
    CONSERVADOR = "conservador"
    MODERADO = "moderado"
    ARROJADO = "arrojado"

    @property
    def rank(self) -> int:
        return {"conservador": 0, "moderado": 1, "arrojado": 2}[self.value]

    @classmethod
    def from_rank(cls, r: int) -> "RiskProfile":
        return [cls.CONSERVADOR, cls.MODERADO, cls.ARROJADO][max(0, min(2, r))]


class Objective(str, Enum):
    CAPITAL_PRESERVATION = "preservacao_capital"
    INCOME = "renda"
    BALANCED_GROWTH = "crescimento_equilibrado"
    LONG_TERM_GROWTH = "crescimento_longo_prazo"
    SPECULATION = "especulacao"


class LossCapacity(str, Enum):
    """Financial capacity to absorb losses (distinct from psychological tolerance)."""

    LOW = "baixa"
    MEDIUM = "media"
    HIGH = "alta"


class Experience(str, Enum):
    NONE = "nenhuma"
    BASIC = "basica"
    INTERMEDIATE = "intermediaria"
    ADVANCED = "avancada"


class AgeBand(str, Enum):
    UNDER_30 = "<30"
    B30_45 = "30-45"
    B45_60 = "45-60"
    OVER_60 = "60+"


class InvestorProfileInput(BaseModel):
    """Raw investor answers. All monetary values are hypothetical unless stated."""

    model_config = ConfigDict(extra="forbid")

    investor_id: str = Field(..., min_length=1, max_length=64,
                             description="Pseudonymous, non-sensitive identifier")
    age_band: AgeBand | None = Field(None, description="Optional; only used if relevant")
    horizon_years: float = Field(..., gt=0, le=60)
    objective: Objective
    risk_tolerance_score: float = Field(..., ge=0, le=100,
                                        description="Psychological risk tolerance, 0-100")
    loss_capacity: LossCapacity
    max_tolerable_drawdown: float | None = Field(
        None, gt=0, lt=1, description="Declared tolerable peak-to-trough loss as a fraction")
    liquidity_need_fraction_12m: float | None = Field(
        None, ge=0, le=1, description="Fraction of capital that may be withdrawn within 12 months")
    capital: float = Field(..., gt=0)
    monthly_contribution: float = Field(0.0, ge=0)
    monthly_withdrawal: float = Field(0.0, ge=0)
    base_currency: str = Field("BRL", min_length=3, max_length=3)
    preferred_markets: list[str] = Field(default_factory=list)
    excluded_assets: list[str] = Field(default_factory=list)
    excluded_asset_classes: list[str] = Field(default_factory=list)
    experience: Experience | None = None
    is_synthetic_example: bool = Field(
        False, description="True when the profile is a hypothetical demo, never a real person")

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("investor_id")
    @classmethod
    def _no_obvious_pii(cls, v: str) -> str:
        if "@" in v or sum(ch.isdigit() for ch in v) >= 11:
            raise ValueError("investor_id must be a pseudonym (no e-mail or document numbers)")
        return v


class ResearchLimits(BaseModel):
    """Research configuration derived from a profile. NOT promises or guarantees."""

    max_weight_per_asset: float = Field(..., gt=0, le=1)
    max_class_weight: dict[str, float] = Field(default_factory=dict)
    min_class_weight: dict[str, float] = Field(default_factory=dict)
    target_volatility_band: tuple[float, float]
    max_drawdown_reference: float = Field(..., gt=0, lt=1)

    @field_validator("target_volatility_band")
    @classmethod
    def _band(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not 0 < v[0] <= v[1] < 2:
            raise ValueError("invalid volatility band")
        return v


class ProfileAssessment(BaseModel):
    """Output of the profiler: classification + reasoning + conflicts."""

    investor_id: str
    score_band_suggestion: RiskProfile
    final_profile: RiskProfile
    caps_applied: list[str]
    conflicts: list[str]
    missing_information: list[str]
    research_limits: ResearchLimits
    disclaimer: str
    is_synthetic_example: bool
