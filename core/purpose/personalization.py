"""Investor personalisation: from the answers given to the limits, preferences and risk aversion
the optimiser uses. Every rule is documented here, and every assumption made for an answer that
was not given is returned so it can be shown to the investor — nothing is presumed silently.

Rules (documented defaults, not regulation; CVM suitability is out of scope):

* Profile — the profiler (``core/profiler.py``) caps the declared profile with the loss capacity,
  horizon, liquidity need, declared drawdown and objective; the final profile sets the base class
  limits, the maximum weight per asset and the volatility band.
* Risk aversion (lambda in max E[r] - lambda/2 * variance): conservador 8, moderado 4, arrojado 2;
  x1.5 for "preservação", x1.25 for "renda".
* Horizon — capacity to wait out a fall and matching fixed income duration to the horizon:
  < 1 year: equity <= 10%, real estate <= 5%, no crypto, no long-duration bonds, cash >= 50%;
  1-3 years: equity <= 30%, real estate <= 10%, no crypto, no long-duration bonds, cash >= 20%;
  3-5 years: equity <= 50%, long-duration bonds <= 20%; 5+ years: profile limits. Horizon is never
  used to INCREASE risk beyond the profile (time diversification does not remove the risk of loss).
* Objective — renda: income-producing instruments (FIIs, REITs, dividend ETFs, fixed income that
  pays out and cash) >= 50%; preservação: cash >= 20% (and the profiler caps at conservador);
  crescimento: profile limits.
* Loss tolerance D (maximum drawdown the investor accepts): volatility cap = min(profile cap, D/2) —
  premise: the historical maximum drawdown of diversified portfolios is usually 2-3x their annual
  volatility.
* Liquidity need L (share of the money that may be needed within 12 months): cash >= L.
* Concentration — per equity sector: 25% / 35% / 45%; foreign currency for the reference currency:
  30% / 50% / 70% (conservador / moderado / arrojado). The currency counted is the EXPOSURE, not the
  quote: an S&P 500 ETF quoted in reais (IVVB11), gold and crypto move with the dollar (assets of
  country US or GLOBAL count as dollar exposure; the catalogue and the B3 ETF rules record the country).
  When the investor chooses only foreign markets, the foreign-currency limit does not apply (it would
  force the rest into cash in reais); the rule is shown.
* Sectors are matched by key (``core.purpose.catalog.canonical_sector``: "Petróleo e gás", "oil" and
  "petroleo_gas" are the same sector); a sector name that is not recognised is reported, never ignored
  silently. B3 shares outside the catalogue have no sector in the source ("nao_classificado"): the sector
  limit does not reach them (each still has the per-asset limit and the correlation screening).
* Crypto — only when allowed explicitly; then at most 0% / 5% / 10% by profile.
* Cash-like instruments (daily-liquidity CDI cash, short Treasury ETFs) count as cash only in the investor's
  currency (a US Treasury ETF is not cash for someone who spends reais: it carries the dollar). They may go
  above the per-asset maximum (up to 100%) only when the investor's own rules ask for cash (short horizon,
  liquidity need, preservation); otherwise the profile's per-asset maximum applies to them as to any asset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.profiler import assess_profile
from core.purpose.catalog import SECTOR_LABEL, CatalogAsset, canonical_sector
from models.investor import (
    InvestorProfileInput,
    LossCapacity,
    Objective,
    ResearchLimits,
    RiskProfile,
)
from models.portfolio import GroupConstraint

OBJECTIVES = ("crescimento", "renda", "preservacao")
RISK_AVERSION = {RiskProfile.CONSERVADOR: 8.0, RiskProfile.MODERADO: 4.0, RiskProfile.ARROJADO: 2.0}
SECTOR_CAP = {RiskProfile.CONSERVADOR: 0.25, RiskProfile.MODERADO: 0.35, RiskProfile.ARROJADO: 0.45}
FOREIGN_CAP = {RiskProfile.CONSERVADOR: 0.30, RiskProfile.MODERADO: 0.50, RiskProfile.ARROJADO: 0.70}
CRYPTO_CAP = {RiskProfile.CONSERVADOR: 0.0, RiskProfile.MODERADO: 0.05, RiskProfile.ARROJADO: 0.10}
CLASS_PT = {"fixed_income": "renda fixa", "equity": "ações", "real_estate": "imóveis", "commodity": "commodities",
            "crypto": "cripto"}
COUNTRY_CCY = {"BR": "BRL", "US": "USD", "GLOBAL": "USD"}     # currency of the exposure by country of the asset
HOME_MARKET = {"BRL": "B3", "USD": "US"}
NO_SECTOR = ("indice", "etf", "nao_classificado")

# instruments whose role follows from what they are (documented, by ticker or name)
CASH_TICKERS = {"CAIXA_CDI", "SINT_SELIC", "SHY", "BIL", "SGOV", "LFTS11.SA"}
LONG_DURATION = {"IMAB11.SA", "TLT", "IMBB11.SA", "IB5M11.SA"}
SHORT_DURATION = {"B5P211.SA", "SHY", "BIL", "SGOV", "CAIXA_CDI", "SINT_SELIC", "LFTS11.SA", "IRFM11.SA"}
DIVIDEND_FUNDS = {"DIVO11.SA", "SCHD", "VYM", "NSDV11.SA"}
DISTRIBUTING_BONDS = {"AGG", "BND", "IEF", "TLT", "TIP", "LQD", "HYG", "SHY", "BIL", "SGOV"}


@dataclass(frozen=True)
class Traits:
    ticker: str
    asset_class: str
    macro_class: str
    market: str                  # B3 | US | CAIXA (daily-liquidity cash in reais)
    currency: str                # quote currency
    exposure: str                # currency the value actually follows (IVVB11: quoted in BRL, exposure USD)
    sector: str
    cash: bool
    income: bool
    duration: str | None         # curta | media | longa (fixed income only)
    crypto: bool


def traits(a: CatalogAsset) -> Traits:
    t = a.ticker
    cash = t in CASH_TICKERS or a.macro_class == "caixa"
    fixed = a.asset_class == "fixed_income"
    if not fixed:
        duration = None
    elif t in LONG_DURATION or re.search(r"longo|20\+|long", a.business, re.IGNORECASE):
        duration = "longa"
    elif t in SHORT_DURATION or cash or re.search(r"1 a 3|curto|short|0-?5|até 5", a.business, re.IGNORECASE):
        duration = "curta"
    else:
        duration = "media"
    income = (cash or a.macro_class == "fii" or t in DIVIDEND_FUNDS or t in DISTRIBUTING_BONDS
              or (a.asset_class == "real_estate" and a.macro_class in ("acoes_eua", "etf_eua")))
    if cash and a.currency == "BRL" and not t.endswith(".SA"):
        market = "CAIXA"            # CDI cash (or its synthetic stand-in): not an exchange listing
    elif t.endswith(".SA") or a.currency == "BRL":
        market = "B3"
    else:
        market = "US"
    return Traits(t, a.asset_class, a.macro_class, market, a.currency, COUNTRY_CCY.get(a.country, a.currency),
                  a.sector, cash, income, duration, a.asset_class == "crypto")


@dataclass
class InvestorPolicy:
    """Everything personal the engine needs, with the reasons, ready to be shown."""

    profile: RiskProfile
    profile_notes: list[str]
    profile_input: InvestorProfileInput
    limits: ResearchLimits
    risk_aversion: float
    vol_cap: float
    objective: str
    allow_crypto: bool
    markets: set[str]
    excluded_tickers: set[str]
    excluded_sectors: set[str]
    min_cash: float
    min_income: float
    long_duration_cap: float | None
    sector_cap: float
    foreign_cap: float
    base_currency: str
    assumptions: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_cash(self, a: CatalogAsset) -> bool:
        """Cash for THIS investor: a cash-like instrument in the investor's own currency."""
        tr = traits(a)
        return tr.cash and tr.currency == self.base_currency and tr.exposure == self.base_currency

    def allows(self, a: CatalogAsset) -> str | None:
        """None when the asset may enter the universe, otherwise the reason it may not."""
        tr = traits(a)
        if a.ticker in self.excluded_tickers or a.ticker.removesuffix(".SA") in self.excluded_tickers:
            return "excluído por você"
        if a.sector in self.excluded_sectors:
            return f"setor '{SECTOR_LABEL.get(a.sector, a.sector)}' excluído por você"
        if tr.crypto and not self.allow_crypto:
            return "cripto não permitida (marque a opção para incluir)"
        if tr.market != "CAIXA" and tr.market not in self.markets:
            return f"mercado {'EUA' if tr.market == 'US' else 'Brasil'} fora das suas preferências"
        if tr.duration == "longa" and self.long_duration_cap == 0.0:
            return "renda fixa de prazo longo incompatível com o seu horizonte (risco de taxa)"
        return None

    def groups(self, assets: list[CatalogAsset]) -> list[GroupConstraint]:
        """Linear group limits for the optimiser over the given universe."""
        trs = [traits(a) for a in assets]
        out: list[GroupConstraint] = []
        cash = tuple(a.ticker for a in assets if self.is_cash(a))
        if self.min_cash > 0:
            out.append(GroupConstraint("caixa (mínimo)", cash, self.min_cash, 1.0))
        if self.min_income > 0:
            out.append(GroupConstraint("renda (mínimo)", tuple(t.ticker for t in trs if t.income),
                                       self.min_income, 1.0))
        if self.long_duration_cap is not None and self.long_duration_cap < 1.0:
            longs = tuple(t.ticker for t in trs if t.duration == "longa")
            if longs:
                out.append(GroupConstraint("renda fixa longa (máximo)", longs, 0.0, self.long_duration_cap))
        sectors: dict[str, list[str]] = {}
        for t in trs:
            if t.asset_class == "equity" and t.sector not in NO_SECTOR:
                sectors.setdefault(t.sector, []).append(t.ticker)
        for s, members in sorted(sectors.items()):
            # one member is already bounded by the per-asset limit unless that limit is above the sector's
            if len(members) > 1 or self.limits.max_weight_per_asset > self.sector_cap:
                out.append(GroupConstraint(f"setor {s} (máximo)", tuple(members), 0.0, self.sector_cap))
        foreign = tuple(t.ticker for t in trs if t.exposure != self.base_currency)
        if foreign and self.foreign_cap < 1.0:
            out.append(GroupConstraint("moeda estrangeira (máximo)", foreign, 0.0, self.foreign_cap))
        return out

    def upper_overrides(self, assets: list[CatalogAsset]) -> dict[str, float]:
        """Cash above the per-asset maximum only when a cash minimum applies: without it, risk-based
        optimisers would park most of the money in the near-riskless instrument whatever the profile."""
        if self.min_cash <= 0:
            return {}
        return {a.ticker: 1.0 for a in assets if self.is_cash(a)}

    def required_groups(self, assets: list[CatalogAsset]) -> list[tuple[str, list[str], int]]:
        """(label, members, how many) that must reach the optimiser so the limits can be met:
        a cash instrument for a cash minimum, income payers for an income minimum, and members of
        every class with a minimum weight (e.g. fixed income for conservative profiles)."""
        trs = [traits(a) for a in assets]
        out: list[tuple[str, list[str], int]] = []
        if self.min_cash > 0:
            out.append(("o caixa mínimo", [a.ticker for a in assets if self.is_cash(a)], 1))
        if self.min_income > 0:
            out.append(("a renda mínima", [t.ticker for t in trs if t.income], 4))
        for cls, lo in sorted(self.limits.min_class_weight.items()):
            if lo > 0:
                out.append((f"o mínimo de {CLASS_PT.get(cls, cls)} do seu perfil",
                            [t.ticker for t in trs if t.asset_class == cls], 2))
        return out

    def as_dict(self) -> dict[str, object]:
        return {"perfil_final": self.profile.value, "notas_perfil": self.profile_notes,
                "aversao_ao_risco": self.risk_aversion, "teto_volatilidade": self.vol_cap,
                "objetivo": self.objective, "cripto_permitida": self.allow_crypto,
                "mercados": sorted(self.markets), "excluidos": sorted(self.excluded_tickers),
                "setores_excluidos": sorted(self.excluded_sectors), "caixa_minimo": self.min_cash,
                "renda_minima": self.min_income, "renda_fixa_longa_maxima": self.long_duration_cap,
                "teto_por_setor": self.sector_cap, "teto_moeda_estrangeira": self.foreign_cap,
                "avisos": self.warnings,
                "limites_por_classe": dict(self.limits.max_class_weight),
                "minimos_por_classe": dict(self.limits.min_class_weight),
                "peso_maximo_por_ativo": self.limits.max_weight_per_asset,
                "hipoteses": self.assumptions, "regras": self.rules}


def build_policy(*, profile: RiskProfile, horizon_years: float, base_currency: str, objective: str | None = None,
                 loss_tolerance: float | None = None, liquidity_need: float | None = None,
                 markets: list[str] | None = None, allow_crypto: bool = False,
                 excluded_tickers: list[str] | None = None, excluded_sectors: list[str] | None = None,
                 capital: float = 0.0, monthly_contribution: float = 0.0, synthetic: bool = False) -> InvestorPolicy:
    assumptions: list[str] = []
    rules: list[str] = []
    if objective is None:
        objective = "crescimento"
        assumptions.append("Objetivo não informado: tratado como crescimento do patrimônio (sem exigência de renda "
                           "periódica nem de preservação estrita).")
    if objective not in OBJECTIVES:
        raise ValueError(f"objetivo deve ser um de {OBJECTIVES}")
    obj_map = {"renda": Objective.INCOME, "preservacao": Objective.CAPITAL_PRESERVATION}
    default_obj = {RiskProfile.CONSERVADOR: Objective.CAPITAL_PRESERVATION, RiskProfile.MODERADO:
                   Objective.BALANCED_GROWTH, RiskProfile.ARROJADO: Objective.LONG_TERM_GROWTH}[profile]
    inp = InvestorProfileInput(
        investor_id="proposito", horizon_years=horizon_years, objective=obj_map.get(objective, default_obj),
        risk_tolerance_score={RiskProfile.CONSERVADOR: 20.0, RiskProfile.MODERADO: 50.0,
                              RiskProfile.ARROJADO: 80.0}[profile],
        loss_capacity={RiskProfile.CONSERVADOR: LossCapacity.LOW, RiskProfile.MODERADO: LossCapacity.MEDIUM,
                       RiskProfile.ARROJADO: LossCapacity.HIGH}[profile],
        max_tolerable_drawdown=loss_tolerance, liquidity_need_fraction_12m=liquidity_need,
        capital=capital, monthly_contribution=max(monthly_contribution, 0.0),
        base_currency=base_currency, is_synthetic_example=synthetic)
    prof = assess_profile(inp)
    final = prof.final_profile
    lim = prof.research_limits.model_copy(deep=True)       # profile defaults + declared drawdown
    max_cls = dict(lim.max_class_weight)
    min_cls = dict(lim.min_class_weight)
    vol_lo, vol_cap = lim.target_volatility_band
    if loss_tolerance is None:
        assumptions.append(f"Tolerância a perdas não informada: usada a referência do perfil {final.value} (queda de "
                           f"até {lim.max_drawdown_reference:.0%}).")
    else:
        cap = max(0.02, loss_tolerance / 2.0)
        if cap < vol_cap:
            rules.append(f"Tolerância a perdas de {loss_tolerance:.0%} → volatilidade máxima {cap:.1%} ao ano (premissa: "
                         "queda máxima histórica ≈ 2× a volatilidade anual).")
            vol_cap = cap
    if liquidity_need is None:
        assumptions.append("Necessidade de liquidez não informada: nenhum mínimo em caixa exigido por esse motivo.")
    min_cash = float(liquidity_need or 0.0)
    if min_cash:
        rules.append(f"Necessidade de liquidez de {min_cash:.0%} em 12 meses → pelo menos {min_cash:.0%} em caixa de "
                     "liquidez diária.")
    long_cap: float | None = None
    eq = max_cls.get("equity", 1.0)
    if horizon_years < 1:
        eq, long_cap = min(eq, 0.10), 0.0
        max_cls["real_estate"] = min(max_cls.get("real_estate", 1.0), 0.05)
        min_cash = max(min_cash, 0.50)
        rules.append("Horizonte abaixo de 1 ano → ações até 10%, imóveis até 5%, sem cripto nem renda fixa longa, "
                     "caixa de pelo menos 50% (pouco tempo para recuperar quedas).")
        max_cls["crypto"] = 0.0
    elif horizon_years < 3:
        eq, long_cap = min(eq, 0.30), 0.0
        max_cls["real_estate"] = min(max_cls.get("real_estate", 1.0), 0.10)
        max_cls["crypto"] = 0.0
        min_cash = max(min_cash, 0.20)
        rules.append("Horizonte de 1 a 3 anos → ações até 30%, imóveis até 10%, sem cripto nem renda fixa longa, "
                     "caixa de pelo menos 20%.")
    elif horizon_years < 5:
        eq, long_cap = min(eq, 0.50), 0.20
        rules.append("Horizonte de 3 a 5 anos → ações até 50% e renda fixa longa até 20% (risco de taxa de juros "
                     "maior que o prazo).")
    max_cls["equity"] = eq
    min_income = 0.0
    if objective == "renda":
        min_income = 0.50
        rules.append("Objetivo de renda → pelo menos 50% em instrumentos que pagam renda (FIIs, REITs, ETFs de "
                     "dividendos, renda fixa que distribui e caixa).")
    elif objective == "preservacao":
        min_cash = max(min_cash, 0.20)
        rules.append("Objetivo de preservação → pelo menos 20% em caixa de liquidez diária.")
    if not allow_crypto:
        max_cls["crypto"] = 0.0
    else:
        max_cls["crypto"] = min(max_cls.get("crypto", 0.0), CRYPTO_CAP[final])
        rules.append(f"Cripto permitida por você: até {max_cls['crypto']:.0%} no perfil {final.value}.")
    lam = RISK_AVERSION[final] * (1.5 if objective == "preservacao" else 1.25 if objective == "renda" else 1.0)
    if markets is None:
        markets = ["BR", "US"]
        assumptions.append("Mercados não informados: Brasil e EUA (ativos dos EUA supõem conta em corretora "
                           "internacional e exposição ao dólar).")
    mk = {"B3" if m.upper() in ("BR", "B3", "BRASIL") else "US" for m in markets}
    foreign_cap = FOREIGN_CAP[final]
    home = HOME_MARKET.get(base_currency)
    if home is not None and home not in mk:
        foreign_cap = 1.0
        rules.append(f"Você escolheu só {'EUA' if home == 'B3' else 'Brasil'}: o limite de moeda estrangeira "
                     f"({FOREIGN_CAP[final]:.0%} no perfil {final.value}) não se aplica — a parte investida fica exposta "
                     f"à outra moeda; o caixa em {base_currency} continua disponível.")
    warnings: list[str] = []
    sectors: set[str] = set()
    for name in excluded_sectors or []:
        key = canonical_sector(name)
        if key is None:
            warnings.append(f"Setor '{name}' não reconhecido (nada foi excluído por ele). Setores disponíveis: "
                            + ", ".join(sorted(k for k in SECTOR_LABEL if k not in NO_SECTOR + ("caixa",))) + ".")
        else:
            sectors.add(key)
    if sectors:
        rules.append("Setores excluídos por você: " + ", ".join(SECTOR_LABEL[k] for k in sorted(sectors))
                     + " (ações, FIIs e ETFs setoriais desses setores ficam fora; ETFs amplos não são filtrados por "
                     "setor porque a composição deles não está nas fontes).")
    lim = lim.model_copy(update={"max_class_weight": max_cls, "min_class_weight": min_cls,
                                 "target_volatility_band": (min(vol_lo, vol_cap), vol_cap)})
    return InvestorPolicy(final, list(prof.conflicts), inp, lim, lam, vol_cap, objective, allow_crypto, mk,
                          {t.upper() for t in (excluded_tickers or [])}, sectors,
                          min_cash, min_income, long_cap, SECTOR_CAP[final], foreign_cap, base_currency,
                          assumptions, rules, warnings)
