"""Purpose-aligned portfolio selection engine (steps 1-9, in the specified order).

1. required CAGR (before any optimisation result);
2. eligible universe of INDIVIDUAL tickers with validated series (>= 252 observations;
   assets that fail are excluded and listed, never proxied);
3. currency sleeves: per-asset and per-sleeve statistics in their own currency with
   their own risk-free rate (one rf per currency, never a single rf);
4. quantitative optimisation (profile-constrained candidates, walk-forward out-of-sample
   validation, one-standard-error rule) — the existing research engine;
5. governance filter: applied as a HARD exclusion before the optimiser runs (equivalent
   to the constraint w_i = 0; removing assets after optimising and renormalising would
   break the profile limits) and verified on the final weights;
6. themes of ``motivo_investidor`` (deterministic, evidence quoted verbatim);
7. alignment score per eligible asset (0..1, one-sentence reason);
8. final weights = (1-a) w_quant + a w_alignment, where w_alignment respects every
   constraint, the profile volatility cap and a tracking-error budget (alignment never
   replaces the optimiser and never adds an asset that breaks risk/governance limits);
9. dual justification per asset: financial thesis and purpose thesis, kept separate.

Outputs follow the required format: required vs expected CAGR, weight table, theses,
optimistic/base/pessimistic scenarios, P(goal) with Monte Carlo standard error and
validation alerts. All language is probabilistic (checked by :mod:`core.purpose.language`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import APP_VERSION, DEMO_RISK_FREE_FALLBACK, AppConfig
from core.market_data.base import DataUnavailableError, DownloadError, EmptyDataError
from core.market_data.cache import SeriesCache
from core.market_data.cash import (
    CASH_ASSET,
    CASH_TICKER,
    WithExtraSeries,
    fetch_cdi_cash,
)
from core.market_data.logos import logo_path
from core.market_data.quality import QualitySettings
from core.market_data.risk_free import RiskFreeQuote, fetch_live_rates
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider
from core.optimization import optimizers as opt
from core.optimization.constraints import (
    build_constraints,
    check_feasibility,
    semicontinuous_weights,
    verify_weights,
)
from core.portfolio_engine import AnalysisResult, OptimizationPolicy, run_analysis
from core.purpose.alignment import (
    THEME_LABEL,
    THEME_TO_TAGS,
    AlignmentResult,
    CombinedPortfolio,
    combine,
    score_assets,
)
from core.purpose.catalog import (
    REAL_CATALOG,
    SECTOR_LABEL,
    SYNTHETIC_CATALOG,
    TAG_SOURCE_REAL,
    CatalogAsset,
    business_sector,
    expand,
    is_etf,
)
from core.purpose.changes import (
    PRESENT,
    ChangeReport,
    compare,
    fingerprint,
    stability_test,
)
from core.purpose.concentration import ConcentrationReport, analyse
from core.purpose.feasibility import (
    ExecutionPlan,
    contribution_plan,
    executable_universe,
    is_money,
    lot_size,
    min_weight,
    plan,
)
from core.purpose.governance import GovernanceResult, apply_governance
from core.purpose.language import guarantee_phrases
from core.purpose.personalization import (
    InvestorPolicy,
    build_policy,
    traits,
)
from core.purpose.required_cagr import PERIODS_PER_YEAR, RequiredCagr, required_cagr
from core.purpose.sleeves import Sleeve, currency_sleeves, native_asset_stats
from core.purpose.themes import ThemeExtraction, extract_themes
from core.simulation.engine import (
    CashflowPlan,
    SimulationResult,
    SimulationSettings,
    simulate_portfolio,
)
from core.simulation.models import BlockBootstrapModel, GBMModel
from core.universe.listings import load_market_listings
from core.universe.scoring import MIN_CANDIDATE_OBS, ScreenResult, screen
from core.universe.screen import Funnel, listing_to_catalog, preselect
from models.investor import RiskProfile
from models.market_data import PriceDataset
from models.portfolio import PortfolioConstraints

MIN_OBS = 252
WEIGHT_FLOOR = 0.005          # weights below 0.5% are reported as zero in the final table
POOL_FACTOR = 2.0             # stage A downloads up to 2x the candidates the optimiser receives...
POOL_MAX = 100                # ...but never more than 100 series per request
MAX_FAILED_SHARE = 0.5        # more than half of the pool without quotes: no portfolio is built
PRICE_CACHE_HOURS = 4.0       # downloaded series are reused for at most 4 hours
_B3_CODE = re.compile(r"([A-Z][A-Z0-9]{3}\d{1,2})F?")        # PETR4, B3SA3, B5P211, TAEE11; F = fractional
_US_CLASS = re.compile(r"([A-Z]{1,5})[./]([A-Z])")             # BRK.B, BRK/B -> BRK-B (Yahoo's code)


class AmountTooSmall(ValueError):
    """The money cannot buy a portfolio within the investor's limits: said plainly, never computed around."""


def norm_ticker(t: str) -> str:
    """'petr4' -> 'PETR4.SA', 'b3sa3' -> 'B3SA3.SA', 'PETR4F' (fractional market) -> 'PETR4.SA', 'BRK.B' ->
    'BRK-B'; other codes upper-cased."""
    t = t.strip().upper()
    if m := _B3_CODE.fullmatch(t):
        return f"{m.group(1)}.SA"
    if m := _US_CLASS.fullmatch(t):
        return f"{m.group(1)}-{m.group(2)}"
    return t


class PurposeRequest(BaseModel):
    """Form inputs (names follow the product specification)."""

    model_config = ConfigDict(extra="forbid")

    capital_inicial: float = Field(0.0, ge=0, le=1e10,
                                   description="quanto já tem hoje para investir de uma vez; 0 = começa só com aportes")
    aporte_periodico: float = Field(0.0, ge=0, le=1e9)
    periodicidade: Literal["mensal", "trimestral", "semestral", "anual", "nenhum"] = "mensal"
    valor_alvo: float = Field(..., gt=0, le=1e12)
    moeda_objetivo: str = Field("BRL", min_length=3, max_length=3)
    horizonte_anos: float = Field(..., gt=0, le=50)
    perfil_risco: RiskProfile = RiskProfile.MODERADO
    moeda: str | None = Field(None, min_length=3, max_length=3,
                              description="moeda do capital e dos aportes (deve ser a moeda do objetivo)")
    taxa_livre_risco: dict[str, float] = Field(default_factory=dict,
                                               description="uma rf anual (decimal) por moeda, ex. {'BRL':0.105,'USD':0.043}")
    restricoes_governanca: dict[str, dict[str, str]] = Field(
        default_factory=dict, max_length=200,
        description="flags por ticker: {'status': 'sob_revisao'|'excluido', 'reason': ..., 'source': ...}")
    incluir_sob_revisao: bool = False
    motivo_investidor: str = Field("", max_length=2000)
    peso_alinhamento: float = Field(0.30, ge=0.0, le=1.0, description="slider: 0.30 = 70% quant / 30% alinhamento")
    orcamento_tracking_error: float = Field(0.05, gt=0.0, le=0.30)
    classes: list[str] | None = None
    universo: Literal["completo", "curado"] = "completo"   # completo: exchange listings + curated catalogue
    max_ativos: int = Field(60, ge=10, le=400)            # candidates handed to the optimiser
    n_simulacoes: int = Field(10_000, ge=500, le=200_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    offline: bool = True
    # personalisation (core/purpose/personalization.py documents every rule; None = not informed)
    objetivo_tipo: Literal["crescimento", "renda", "preservacao"] | None = None
    tolerancia_perda: float | None = Field(None, gt=0, lt=1, description="maior queda aceitável (fração, ex. 0.15)")
    necessidade_liquidez: float | None = Field(None, ge=0, le=1,
                                               description="fração do dinheiro que pode precisar em 12 meses")
    mercados: list[Literal["BR", "US"]] | None = Field(None, min_length=1)
    permitir_cripto: bool = False
    ativos_excluidos: list[str] = Field(default_factory=list, max_length=1000)
    setores_excluidos: list[str] = Field(default_factory=list, max_length=60)
    carteira_atual: dict[str, float] = Field(default_factory=dict, max_length=200,
                                             description="o que você já tem: ticker -> valor, na moeda do objetivo")
    carteira_anterior: dict[str, float] = Field(default_factory=dict, max_length=200,
                                                description="pesos da recomendação anterior (comparação e estabilidade)")
    carteira_anterior_parametros: str | None = Field(None, max_length=64,
                                                     description="impressão digital das entradas da recomendação anterior")
    fracionario_eua: bool = False
    valor_minimo_posicao: float = Field(100.0, ge=0, le=1e7, description="menor compra que vale a pena fazer")
    # rules from an external rulebook (e.g. an investment competition, core/competition); None/0 = no such rule
    preco_minimo: float | None = Field(None, gt=0, le=1e5, description="menor preço por ação/cota aceito")
    peso_maximo_ativo: float | None = Field(None, gt=0, le=1, description="teto por ativo imposto pela regra")
    min_posicoes: int | None = Field(None, ge=1, le=100)
    max_posicoes: int | None = Field(None, ge=2, le=200)
    min_etfs: int = Field(0, ge=0, le=20, description="quantos ETFs a carteira precisa ter, no mínimo")
    etfs_permitidos: list[str] | None = Field(None, max_length=600, description="lista aprovada; None = qualquer ETF")
    min_setores: int | None = Field(None, ge=1, le=15, description="setores diferentes com pelo menos uma posição")

    @field_validator("moeda_objetivo")
    @classmethod
    def _ccy(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("ativos_excluidos")
    @classmethod
    def _excl(cls, v: list[str]) -> list[str]:
        return sorted({norm_ticker(t) for t in v if t.strip()})

    @field_validator("etfs_permitidos")
    @classmethod
    def _etfs(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else sorted({norm_ticker(t) for t in v if t.strip()})

    @field_validator("carteira_atual", "carteira_anterior")
    @classmethod
    def _positions(cls, v: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, x in v.items():
            if not k.strip():
                raise ValueError("há uma posição sem código de ativo: informe TICKER e valor")
            if not np.isfinite(x) or x < 0:
                raise ValueError(f"valor inválido para {k}: use números >= 0")
            if x > 0:
                t = norm_ticker(k)
                out[t] = out.get(t, 0.0) + float(x)
        return out

    @field_validator("taxa_livre_risco")
    @classmethod
    def _rf(cls, v: dict[str, float]) -> dict[str, float]:
        out = {}
        for k, x in v.items():
            if not -0.05 <= float(x) <= 1.0:
                raise ValueError(f"taxa livre de risco de {k} fora de [-5%, 100%]")
            out[k.strip().upper()] = float(x)
        return out

    @model_validator(mode="after")
    def _consistency(self) -> PurposeRequest:
        if self.periodicidade == "nenhum" and self.aporte_periodico > 0:
            raise ValueError("aporte_periodico > 0 exige periodicidade diferente de 'nenhum'")
        m = PERIODS_PER_YEAR[self.periodicidade]
        if self.capital_inicial == 0 and not self.carteira_atual and (
                self.aporte_periodico <= 0 or not m or int(np.floor(m * self.horizonte_anos + 1e-9)) < 1):
            raise ValueError("sem capital inicial, informe um aporte periódico (por exemplo, mensal) com pelo menos "
                             "um aporte dentro do horizonte")
        if self.offline and self.moeda_objetivo != "BRL":
            raise ValueError("o universo sintético offline é cotado em BRL: use moeda_objetivo BRL")
        if self.moeda is not None and self.moeda.upper() != self.moeda_objetivo:
            raise ValueError("capital/aportes em moeda diferente da moeda do objetivo não são convertidos "
                             "automaticamente: informe-os na moeda do objetivo")
        if self.min_posicoes and self.max_posicoes and self.min_posicoes > self.max_posicoes:
            raise ValueError("o número mínimo de posições passa do máximo")
        if self.peso_maximo_ativo and self.max_posicoes and self.peso_maximo_ativo * self.max_posicoes < 1 - 1e-9:
            raise ValueError(f"com no máximo {self.max_posicoes} posições de até {self.peso_maximo_ativo:.0%} cada, só "
                             f"{self.peso_maximo_ativo * self.max_posicoes:.0%} do dinheiro pode ser investido")
        if self.etfs_permitidos is not None and self.min_etfs > len(self.etfs_permitidos):
            raise ValueError("a regra pede mais ETFs do que a lista aprovada tem")
        return self

    @property
    def counting_rules(self) -> bool:
        return bool(self.min_posicoes or self.max_posicoes or self.min_etfs or self.min_setores)


@dataclass
class Alert:
    category: str      # dados | rf | governanca | backtest | concentracao | sensibilidade | objetivo | ...
    severity: str      # INFO | WARNING | FAIL
    message: str


@dataclass
class PurposeResult:
    request: PurposeRequest
    steps: list[dict[str, object]]
    required: RequiredCagr
    universe: dict[str, object]
    rf_by_currency: dict[str, float]
    rf_source: dict[str, str]
    native_stats: pd.DataFrame
    governance: GovernanceResult
    analysis: AnalysisResult
    themes: ThemeExtraction
    alignment: AlignmentResult
    combined: CombinedPortfolio
    table: pd.DataFrame
    theses: list[dict[str, str]]
    expected: dict[str, object]
    scenarios: pd.DataFrame
    goal: dict[str, object]
    sensitivity: pd.DataFrame
    sleeves_final: list[Sleeve]
    alerts: list[Alert] = field(default_factory=list)
    is_synthetic: bool = True
    allocation: list[dict[str, object]] = field(default_factory=list)   # "onde investir e por quê"
    strategy_text: str = ""
    policy: dict[str, object] = field(default_factory=dict)             # personal rules, hypotheses
    screening: ScreenResult | None = None                                # stage B: every measured asset
    execution: dict[str, object] = field(default_factory=dict)          # purchase plans, lots, minimums
    concentration: ConcentrationReport | None = None
    changes: ChangeReport | None = None
    sources: list[dict[str, object]] = field(default_factory=list)      # provenance of every dataset
    reproducibility: dict[str, object] = field(default_factory=dict)

    @property
    def is_final_portfolio(self) -> bool:
        """A synthetic demonstration is never a final portfolio."""
        return not self.is_synthetic


# ----------------------------------------------------------------------------- helpers
def subset_dataset(ds: PriceDataset, keep: list[str]) -> PriceDataset:
    cols = [s for s in ds.symbols if s in keep]
    return dataclasses.replace(
        ds, prices=ds.prices[cols], raw_prices={s: ds.raw_prices[s] for s in cols},
        provenance={s: ds.provenance[s] for s in cols if s in ds.provenance},
        meta={s: ds.meta[s] for s in cols},
        volumes=ds.volumes.reindex(columns=cols) if ds.volumes is not None else None,   # cash has no volume
        native_prices={s: ds.native_prices[s] for s in cols if s in ds.native_prices},
        native_currency={s: ds.native_currency[s] for s in cols if s in ds.native_currency})


def _resolve_rf(req: PurposeRequest, currencies: set[str],
                live: dict[str, RiskFreeQuote] | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Priority: rate typed by the user > rate fetched from a public source > flagged demo fallback."""
    rf, src = {}, {}
    live = live or {}
    for c in sorted(currencies | {req.moeda_objetivo}):
        if c in req.taxa_livre_risco:
            rf[c], src[c] = req.taxa_livre_risco[c], "informada pelo usuário"
        elif c in live:
            rf[c], src[c] = live[c].rate, f"{live[c].source}, em {live[c].as_of}"
        elif c in DEMO_RISK_FREE_FALLBACK:
            rf[c], src[c] = DEMO_RISK_FREE_FALLBACK[c], "FALLBACK DE DEMONSTRAÇÃO (não é cotação)"
        else:
            raise ValueError(f"taxa livre de risco ausente para {c}: informe-a (nunca é assumida igual a zero)")
    return rf, src


def _start_capital(req: PurposeRequest) -> float:
    """What is invested or available today: new capital + current holdings (never the contributions)."""
    return req.capital_inicial + sum(req.carteira_atual.values())


def _policy(req: PurposeRequest) -> InvestorPolicy:
    m = PERIODS_PER_YEAR[req.periodicidade]
    pol = build_policy(profile=req.perfil_risco, horizon_years=req.horizonte_anos, base_currency=req.moeda_objetivo,
                       objective=req.objetivo_tipo, loss_tolerance=req.tolerancia_perda,
                       liquidity_need=req.necessidade_liquidez, markets=req.mercados,
                       allow_crypto=req.permitir_cripto, excluded_tickers=req.ativos_excluidos,
                       excluded_sectors=req.setores_excluidos, capital=_start_capital(req),
                       monthly_contribution=req.aporte_periodico * m / 12.0 if m else 0.0, synthetic=req.offline)
    cap = req.peso_maximo_ativo
    if cap is None:
        return pol
    rules = list(pol.rules)
    if cap < pol.limits.max_weight_per_asset:
        rules.append(f"Regra externa: no máximo {cap:.0%} por ativo (o perfil {pol.profile.value} permitiria "
                     f"{pol.limits.max_weight_per_asset:.0%}); vale também para o caixa.")
    else:
        rules.append(f"Regra externa: no máximo {cap:.0%} por ativo (o limite do perfil, "
                     f"{pol.limits.max_weight_per_asset:.0%}, já é menor); vale também para o caixa.")
    return dataclasses.replace(
        pol, limits=pol.limits.model_copy(update={"max_weight_per_asset": min(cap, pol.limits.max_weight_per_asset)}),
        hard_cap=cap, rules=rules)


def _fmt(x: float, ccy: str = "") -> str:
    v = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{ {'BRL': 'R$', 'USD': 'US$', 'EUR': '€'}.get(ccy, ccy)} {v}".strip()


def _reference_amount(req: PurposeRequest) -> tuple[float, str]:
    """The amount the target portfolio is sized for (minimum purchase, lots): what is invested today plus
    the contributions of the next 12 months (within the horizon). The target is reached over that year:
    the initial money and each contribution buy the positions furthest below the target first."""
    now = _start_capital(req)
    ccy = req.moeda_objetivo
    m = PERIODS_PER_YEAR[req.periodicidade]
    n = min(m, int(np.floor(m * req.horizonte_anos + 1e-9))) if m and req.aporte_periodico > 0 else 0
    year = req.aporte_periodico * n
    months = round(n * 12 / m) if n else 0
    span = "do próximo mês" if months == 1 else f"dos próximos {months} meses"
    today = ("capital inicial + carteira atual" if req.carteira_atual and req.capital_inicial > 0 else
             "carteira atual" if req.carteira_atual else "capital inicial")
    if year <= 0:
        return now, f"carteira dimensionada para {_fmt(now, ccy)} ({today})"
    if now <= 0:
        return year, (f"sem capital inicial: carteira dimensionada para os aportes {span} "
                      f"({_fmt(year, ccy)}); cada aporte compra as posições mais distantes do alvo")
    return now + year, (f"carteira dimensionada para {_fmt(now + year, ccy)}: {today} ({_fmt(now, ccy)}) + aportes "
                        f"{span} ({_fmt(year, ccy)}); o dinheiro de hoje e cada aporte compram "
                        "primeiro as posições mais distantes do alvo")


_SMALL_FIX = ("Aumente o capital ou o aporte, reduza a 'Compra mínima por ativo' (em 'Mais sobre você') ou use a seção "
              "Ação do dia, que aponta uma ação para o valor que você tem.")


def _buyable(symbols: list[str], floor: dict[str, float], meta: dict, cat: dict[str, CatalogAsset],
             policy: InvestorPolicy, cfg: AppConfig) -> bool:
    """Whether at least two positions, each at no less than its floor (the minimum purchase as a weight of the
    amount), make a portfolio within every limit of the investor: the linear constraints the optimiser gets in step 8
    (per asset, class, sector, currency, cash...)."""
    if len(symbols) < 2:
        return False
    assets = [cat[s] for s in symbols]
    pi = policy.profile_input
    pc = build_constraints(symbols, meta, policy.limits, max_weight=cfg.max_weight, min_weight=cfg.min_weight,
                           excluded=pi.excluded_assets, excluded_classes=pi.excluded_asset_classes)
    pc.groups.extend(g for g in policy.groups(assets) if g.members or g.lower <= 0)
    over = policy.upper_overrides(assets)
    for i, s in enumerate(pc.symbols):
        if s in over and pc.upper[i] > 0:
            pc.upper[i] = over[s]
    return semicontinuous_weights(pc, np.array([floor[s] for s in symbols]), min_held=2) is not None


def _least_amount(ok: Callable[[float], bool], start: float, top: float = 1e9) -> float | None:
    """Smallest amount (within 0.5%) for which ``ok`` holds, knowing it fails at ``start``: more money only lowers
    each position's minimum weight, so ``ok`` is monotone. None when not even ``top`` is enough."""
    lo, hi = start, max(start, 1.0) * 2
    while not ok(hi):
        lo, hi = hi, hi * 2
        if hi > top:
            return None
    while hi - lo > max(1.0, 0.005 * hi):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def _default_provider(cache_dir: Path, progress: Callable[[str], None] | None = None,
                      budget_s: float | None = None) -> YahooChartProvider:
    """Real prices: Yahoo chart endpoint through the standard library, with a file cache,
    parallel downloads and an optional total time budget."""
    return YahooChartProvider(cache=SeriesCache(cache_dir, PRICE_CACHE_HOURS), progress=progress,
                              time_budget_s=budget_s)


STRATEGY_PT = {
    "max_sharpe": "máximo Sharpe (melhor retorno esperado por unidade de risco)",
    "min_variance": "mínima variância (menor oscilação possível dentro do perfil)",
    "risk_parity": "paridade de risco (cada ativo contribui com o mesmo risco)",
    "max_diversification": "máxima diversificação",
    "hrp": "paridade de risco hierárquica (HRP)",
    "min_cvar": "mínimo CVaR (menor perda média nos piores dias)",
    "robust_mean_variance": "média-variância robusta a erros de estimativa",
    "target_volatility": "volatilidade-alvo do perfil",
    "equal_weight": "pesos iguais ajustados aos limites do perfil",
    "utility": "utilidade pessoal (retorno esperado menos uma penalidade de risco calibrada para o seu perfil, "
               "objetivo e horizonte, dentro do seu teto de volatilidade)",
}


def _role(w: float, risk_share: float, ret_share: float | None) -> str:
    """Plain-language role of a position from its shares of capital, risk and expected return [E]."""
    parts = f"{w:.0%} da carteira, {risk_share:.0%} do risco" + (
        f" e {ret_share:.0%} do retorno esperado [E]" if ret_share is not None else "")
    if risk_share < 0.5 * w:
        label = "estabilizador"
    elif ret_share is not None and ret_share > 1.2 * max(risk_share, w):
        label = "motor de retorno"
    elif risk_share > 1.5 * w:
        label = "crescimento (assume mais risco do que o seu peso)"
    else:
        label = "diversificador"
    return f"{label}: {parts}"


def _load_universe(req: PurposeRequest, assets: list[CatalogAsset], cfg: AppConfig,
                   provider: object | None) -> tuple[PriceDataset, object | None]:
    from core.purpose.synthetic_universe import asset_meta, build_synthetic_universe

    if req.offline:
        ds, truth = build_synthetic_universe(seed=req.seed)
        return subset_dataset(ds, [a.ticker for a in assets]), truth
    from core.data_loader import load_price_dataset

    prov = provider or _default_provider(cfg.cache_dir)
    ds = load_price_dataset([a.ticker for a in assets], prov, period=cfg.period, interval="1d",  # type: ignore[arg-type]
                            base_currency=req.moeda_objetivo, meta={a.ticker: asset_meta(a, False) for a in assets},
                            quality=QualitySettings(min_history=MIN_OBS))
    return ds, None


SIM_STEPS_PER_YEAR = 12     # the goal simulation runs on MONTHLY steps (see _simulate)


def _gbm_estimator(res: AnalysisResult, cols: list[str], drift_shift: float = 0.0) -> GBMModel:
    """Multivariate GBM on monthly steps, centred on the optimiser's expected returns [E]."""
    P = res.config.trading_days
    hist = GBMModel.from_log_returns(res.log_returns[cols], P)
    mu_vec = res.mu.loc[cols].to_numpy()
    mu_c = P * np.log1p(mu_vec / P) + drift_shift
    return GBMModel(mu_c, hist.cov, 1.0 / SIM_STEPS_PER_YEAR)


def _monthly_bootstrap(res: AnalysisResult, cols: list[str]) -> BlockBootstrapModel:
    """Bootstrap of overlapping 21-day compounded returns, shifted so that the mean monthly return
    matches the optimiser's expected return [E] (same centring as the GBM)."""
    P = res.config.trading_days
    g = np.log1p(res.returns[cols].to_numpy())
    cs = np.vstack([np.zeros((1, g.shape[1])), np.cumsum(g, axis=0)])
    monthly = np.expm1(cs[21:] - cs[:-21])
    target = (1.0 + res.mu.loc[cols].to_numpy() / P) ** 21 - 1.0
    return BlockBootstrapModel(monthly + (target - monthly.mean(axis=0)), 1)


def _equiv_cagr(req: PurposeRequest, wealth: float) -> float | None:
    """Annual rate that turns the same capital + contributions into ``wealth`` (same FV formula
    as the required CAGR, so both numbers are directly comparable)."""
    r = required_cagr(_start_capital(req), req.aporte_periodico, req.periodicidade, max(wealth, 1e-9),
                      req.horizonte_anos)
    return r.rate


def _simulate(req: PurposeRequest, res: AnalysisResult, model: object, w: np.ndarray, n_paths: int,
              seed: int) -> SimulationResult:
    """Monthly steps: the portfolio is rebalanced monthly and contributions are monthly,
    quarterly, semiannual or annual, so nothing happens between month ends. For a GBM the
    monthly lognormal increment is EXACT (same terminal distribution as daily steps), at 1/21
    of the cost; intra-month drawdowns are not resolved (they are not reported here)."""
    cfg = res.config
    S = SIM_STEPS_PER_YEAR
    m = PERIODS_PER_YEAR[req.periodicidade]
    cf = CashflowPlan(monthly_contribution=req.aporte_periodico if m else 0.0,
                      steps_per_month=(S // m) if m else 0, contribution_cost_bps=cfg.transaction_cost_bps)
    ss = SimulationSettings(n_paths=n_paths, n_steps=round(req.horizonte_anos * S), seed=seed,
                            initial_capital=_start_capital(req), rebalance_every=1,
                            rebalance_cost_bps=cfg.transaction_cost_bps + cfg.spread_bps,
                            inflation_annual=cfg.inflation_annual, periods_per_year=S,
                            batch_size=cfg.simulation_batch_size, goal=req.valor_alvo, record_every=1)
    return simulate_portfolio(model, w, ss, cf)  # type: ignore[arg-type]


def _restate(comb: CombinedPortfolio, w: pd.Series, cov: pd.DataFrame, mu: pd.Series, al: AlignmentResult,
             note: str) -> CombinedPortfolio:
    """The combined portfolio with new final weights; every reported metric is recomputed on them."""
    c = cov.loc[w.index, w.index].to_numpy()
    wv, wq = w.to_numpy(), comb.quant.loc[w.index].to_numpy()
    te = float(np.sqrt(max((wv - wq) @ c @ (wv - wq), 0.0)))
    sc = np.array([al.scores[t].score for t in w.index], float) if al.computed else None
    return dataclasses.replace(comb, final=w, tracking_error=te, vol_final=float(np.sqrt(max(wv @ c @ wv, 0.0))),
                               ret_final=float(wv @ mu.loc[w.index]),
                               alignment_final=float(sc @ wv) if sc is not None else comb.alignment_final,
                               notes=[*comb.notes, note])


@dataclass(frozen=True)
class CountRules:
    """Rules on HOW MANY positions an external rulebook (e.g. an investment competition) asks for: held positions
    between ``min_held`` and ``max_held``, at least ``k`` of each (label, members, k) group, and holdings in at least
    ``least_sets`` of the member sets in ``sets`` (e.g. one per sector)."""

    min_held: int = 2
    max_held: int | None = None
    groups: tuple[tuple[str, frozenset[str], int], ...] = ()
    sets: tuple[frozenset[str], ...] = ()
    least_sets: int = 0

    def solver_args(self, syms: list[str]) -> dict[str, object]:
        pos = {s: i for i, s in enumerate(syms)}
        idx = [[pos[s] for s in g if s in pos] for g in self.sets]
        return {"min_held": self.min_held, "max_held": self.max_held,
                "count_groups": [([pos[s] for s in mem if s in pos], k) for _, mem, k in self.groups],
                "distinct": ([g for g in idx if g], self.least_sets) if self.least_sets else None}


def _count_rules(req: PurposeRequest, cat: dict[str, CatalogAsset]) -> CountRules | None:
    if not req.counting_rules:
        return None
    etfs = frozenset(s for s, a in cat.items() if is_etf(a))
    sectors: dict[str, set[str]] = {}
    for s, a in cat.items():
        if (sec := business_sector(a)) is not None:
            sectors.setdefault(sec, set()).add(s)
    return CountRules(min_held=max(2, req.min_posicoes or 2), max_held=req.max_posicoes,
                      groups=((f"pelo menos {req.min_etfs} ETF(s)", etfs, req.min_etfs),) if req.min_etfs else (),
                      sets=tuple(frozenset(v) for _, v in sorted(sectors.items())) if req.min_setores else (),
                      least_sets=req.min_setores or 0)


def _make_executable(comb: CombinedPortfolio, pc: PortfolioConstraints, cov: pd.DataFrame, mu: pd.Series,
                     al: AlignmentResult, floor: pd.Series, vol_cap: float, te_limit: float | None = None,
                     counts: CountRules | None = None) -> tuple[CombinedPortfolio, dict[str, str], bool]:
    """Every position must be at least its floor (``WEIGHT_FLOOR``, or the smallest sensible purchase — one
    lot or the minimum position — as a weight of the reference amount). Greedy semi-continuous rule: hold
    the positions whose theoretical weight reaches the floor; if the limits cannot be met with them alone,
    add back the next largest positions (raised to their floor) until they can; if that order never meets them,
    the held set is chosen exactly (mixed-integer, closest to the theoretical weights). The weights of the held
    set are then the executable portfolio closest to the theoretical one (minimum tracking error, plus a
    small distance-in-weights term so near-riskless assets are not shuffled), within every limit, the
    volatility cap and, when the alignment tilt was applied, its tracking-error budget against the quant
    portfolio (dropped, and said, only if the limits cannot be met with it). With ``counts`` (how many positions,
    ETFs, sectors an external rulebook asks for) the held set is always chosen exactly, with those counts as
    constraints. Returns the portfolio, the positions left out with the reason, and False when no held set meets the
    limits (weights unchanged)."""
    syms = pc.symbols
    n = len(syms)
    target = comb.final.loc[syms].to_numpy()
    c = cov.loc[syms, syms].to_numpy()
    m = mu.loc[syms].to_numpy()
    fl = floor.reindex(syms).fillna(WEIGHT_FLOOR).clip(lower=WEIGHT_FLOOR).to_numpy()
    order = [int(i) for i in np.argsort(-target, kind="stable") if target[i] > 1e-10]
    held = [i for i in order if target[i] >= fl[i] - 1e-12]
    cand = [i for i in order if i not in held]
    if not cand and counts is None:
        return comb, {}, True
    cap = max(vol_cap, float(np.sqrt(max(target @ c @ target, 0.0))))
    eps = 1e-4 * float(np.trace(c)) / n
    wq = comb.quant.loc[syms].to_numpy()
    vol_con = [{"type": "ineq", "fun": lambda x: cap ** 2 - float(x[:n] @ c @ x[:n]),
                "jac": lambda x: np.concatenate([-2 * c @ x[:n], np.zeros(len(x) - n)])}]
    te_con = [] if te_limit is None else [
        {"type": "ineq", "fun": lambda x: te_limit ** 2 - float((x[:n] - wq) @ c @ (x[:n] - wq)),
         "jac": lambda x: np.concatenate([-2 * c @ (x[:n] - wq), np.zeros(len(x) - n)])}]

    def project(pc2: PortfolioConstraints, cons: list[dict]) -> object:
        hint = np.clip(np.where(pc2.upper > 0, target, 0.0), pc2.lower, pc2.upper)
        return opt.solve("executavel",
                         lambda x: float((x - target) @ c @ (x - target)) + eps * float((x - target) @ (x - target)),
                         lambda x: 2 * c @ (x - target) + 2 * eps * (x - target), pc2, m, c, 0.0,
                         opt.SolverSettings(n_starts=3), extra_constraints=cons, start_hint=[hint / hint.sum()])

    def bounds(h: list[int]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.array(h, int)
        lower, upper = np.zeros(n), np.zeros(n)
        upper[idx] = pc.upper[idx]
        lower[idx] = np.minimum(np.maximum(pc.lower[idx], fl[idx]), upper[idx])
        return lower, upper

    def attempt(h: list[int]) -> np.ndarray | None:
        nonlocal te_relaxed
        lower, upper = bounds(h)
        pc2 = dataclasses.replace(pc, lower=lower, upper=upper)
        if lower.sum() > 1 + 1e-9 or not check_feasibility(pc2).feasible:
            return None
        r = project(pc2, vol_con + te_con)
        if not r.success and te_con:
            r = project(pc2, vol_con)
            te_relaxed = r.success
        return r.weights.to_numpy() if r.success else None

    w, te_relaxed = None, False
    extra = counts.solver_args(syms) if counts is not None else {}
    while counts is None and bounds(held)[0].sum() <= 1 + 1e-9:
        w = attempt(held)
        if w is not None or not cand:
            break
        held.append(cand.pop(0))
    if w is None:
        # the greedy order can spend the money on large lots before a group minimum (e.g. fixed income) is met: pick
        # the held set exactly instead (each position out or at least its floor, every linear limit, closest to the
        # theoretical weights), among the theoretical positions first, then the whole universe; a held set the
        # volatility cap rules out is not tried again
        for pool in (order, [i for i in range(n) if pc.upper[i] > 0]):
            tried: list[list[int]] = []
            while w is None and len(tried) < 5:
                x = semicontinuous_weights(pc, fl, target, pool, tried, **extra)  # type: ignore[arg-type]
                if x is None:
                    break
                tried.append([int(i) for i in np.flatnonzero(x > 0)])
                w = attempt(tried[-1])
            if w is not None:
                held = tried[-1]
                break
    dropped = {syms[i]: (f"peso calculado {target[i]:.2%} menor que uma compra mínima ({fl[i]:.2%} do valor)"
                         if fl[i] > WEIGHT_FLOOR + 1e-12 else
                         f"peso calculado {target[i]:.2%} abaixo de {WEIGHT_FLOOR:.1%} (resíduo numérico)")
               for i in order if i not in held}
    if w is None:
        return comb, dropped, False
    w = np.where(w < 1e-10, 0.0, w)
    w = w / w.sum()
    raised = [syms[i] for i in held if target[i] < fl[i] - 1e-12 and w[i] >= fl[i] - 1e-9]
    below = [syms[i] for i in held if 0.0 < w[i] < fl[i] - 1e-9]       # the per-asset limit is under one purchase
    ws = pd.Series(w, index=syms)
    return (_restate(comb, ws, cov, mu, al,
                     f"{len(dropped)} posição(ões) pequena(s) demais para comprar ficaram de fora e os pesos foram "
                     "recalculados dentro de todos os limites (carteira executável mais próxima da teórica)"
                     + (f"; {', '.join(raised)} subiu(ram) até a compra mínima para os limites fecharem" if raised else "")
                     + (f"; {', '.join(below)} ficou(aram) abaixo de uma compra mínima porque o limite por ativo é "
                        "menor que ela" if below else "")
                     + ("; para fechar os limites, o orçamento de tracking error do alinhamento foi excedido"
                        if te_relaxed else "")
                     + ("; as posições foram escolhidas com as regras de contagem (quantas posições, ETFs e setores)"
                        if counts is not None else "")),
            dropped, True)


def _frontier_gap(frontier: pd.DataFrame, vol: float, ret: float) -> float | None:
    """Expected-return shortfall vs the efficient frontier at the same volatility (>= 0 up to the
    linear interpolation between frontier points, which slightly understates the frontier)."""
    if frontier is None or frontier.empty or not np.isfinite(vol):
        return None
    f = frontier.sort_values("volatility")
    v, r = f["volatility"].to_numpy(), f["expected_return"].to_numpy()
    if vol < v[0] - 1e-9:
        return None
    best = float(np.interp(min(vol, v[-1]), v, r))
    return max(best - ret, 0.0)


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Monte Carlo proportion (well defined also at p = 0 or 1)."""
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _pct(x: float | None, d: int = 2) -> str:
    return "indisponível" if x is None or not np.isfinite(x) else f"{x:.{d}%}"


# ----------------------------------------------------------------------------- theses
def _financial_thesis(a: CatalogAsset, s: str, res: AnalysisResult, w: pd.Series, rc: pd.Series,
                      synthetic: bool) -> str:
    st = res.asset_stats.loc[s] if s in res.asset_stats.index else None
    mu = float(res.mu.get(s, np.nan))
    se = res.sections.get("mu_uncertainty")
    t = float(se.loc[s, "t_stat"]) if isinstance(se, pd.DataFrame) and s in se.index else np.nan
    ds = res.dataset
    span = f"{ds.prices.index[0].date()} a {ds.prices.index[-1].date()}"
    parts = []
    if synthetic:
        parts.append("[ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real]")
    parts.append(f"Modelo de negócio: {a.business.replace('[sintético] ', '')}.")
    if st is not None:
        parts.append(f"Métrica-chave [H] ({span}, em {res.config.base_currency}): retorno médio anual "
                     f"{_pct(float(st.get('arith_mean_ann', np.nan)))}, volatilidade {_pct(float(st.get('vol_ann', np.nan)))}, "
                     f"drawdown máximo {_pct(float(st.get('max_drawdown', np.nan)))}.")
    parts.append(f"Retorno esperado usado pelo otimizador [E]: {_pct(mu)}"
                 + (f" (estatística t = {t:.2f}; " + ("estatisticamente indistinto de zero" if abs(t) < 2
                                                      else "distinto de zero a ~95%") + ")" if np.isfinite(t) else "")
                 + f". Na carteira: peso {w[s]:.1%}, contribuição de {rc[s]:.1%} do risco total.")
    parts.append("Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não "
                 "possui; nada é inferido).")
    parts.append(f"Risco principal: {a.class_risk}.")
    return " ".join(parts)


def _purpose_thesis(a: CatalogAsset, s: str, al: AlignmentResult, themes: ThemeExtraction) -> str:
    sc = al.scores.get(s)
    if not al.computed or sc is None or sc.score is None:
        return ("Tese de propósito não elaborada: score de alinhamento não calculado "
                + ("por falta de input (motivo_investidor vazio)." if themes.empty
                   else "(nenhum tema setorial identificado no texto; nada é inferido)."))
    if not sc.matched:
        declared = ", ".join(THEME_LABEL.get(t, t) for t in al.mapped_themes)
        return (f"Sem conexão temática direta com o que você declarou ({declared}); score 0,00. O ativo está na "
                "carteira pela função financeira descrita na tese financeira, não pelo propósito.")
    ev = {t.name: "\", \"".join(t.evidence[:2]) for t in themes.themes}
    lab_to_name = {v: k for k, v in THEME_LABEL.items()}
    quotes = "; ".join(f"{lab} — você escreveu \"{ev[lab_to_name[lab]]}\"" for lab in sc.matched
                       if lab_to_name.get(lab) in ev)
    return (f"Conexão com o seu motivo: {quotes}. Atividade do ativo: {a.business.replace('[sintético] ', '')}. "
            f"Score de alinhamento {sc.score:.2f} (fração dos seus temas setoriais atendidos).")


def _param_fingerprint(req: PurposeRequest) -> str:
    """Fingerprint of everything the investor typed that shapes the portfolio — not the previous
    recommendation itself, nor the number of simulated paths (it changes only the goal probability).
    Canonical: the order in which lists and holdings were typed does not matter."""
    d = req.model_dump(mode="json", exclude={"carteira_anterior", "carteira_anterior_parametros", "n_simulacoes"})
    d = {k: sorted(v) if isinstance(v, list) else v for k, v in d.items()}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _stability(req: PurposeRequest, comb: CombinedPortfolio, res: AnalysisResult, floor: pd.Series, vol_cap: float,
               risk_aversion: float, cost: float) -> dict[str, object]:
    """May the previous recommendation be kept? Only for the same inputs (new prices only), when it still
    meets every current limit and every position is still a sensible purchase; then the new portfolio must
    beat it by more than estimation error + trading cost (``core.purpose.changes``)."""
    if req.carteira_anterior_parametros != _param_fingerprint(req):
        return {"manter_anterior": False,
                "explicacao": "suas entradas mudaram desde a recomendação anterior (ou ela veio sem a impressão "
                              "digital dos parâmetros): a carteira foi recalculada para as entradas atuais"}
    prev = pd.Series(req.carteira_anterior, dtype=float)
    prev = prev / prev.sum()
    pc = res.constraints
    missing = [t for t in prev.index if t not in pc.symbols and prev[t] >= PRESENT]
    why = []
    w0 = prev.reindex(pc.symbols).fillna(0.0)
    if missing:
        why.append(f"{', '.join(missing)} não está(ão) entre os ativos analisados agora")
    else:
        w0 = w0 / w0.sum()
        viol = verify_weights(w0.to_numpy(), pc)
        if viol:
            why.append("não respeita os limites atuais (" + "; ".join(viol[:3]) + ")")
        c = res.cov.cov.loc[pc.symbols, pc.symbols].to_numpy()
        v0 = float(np.sqrt(max(w0.to_numpy() @ c @ w0.to_numpy(), 0.0)))
        cap = max(vol_cap, comb.vol_final)      # when no portfolio fits the cap, the new one is the reference
        if v0 > cap + 1e-6:
            why.append(f"volatilidade estimada {v0:.1%} acima do teto {cap:.1%}")
        small = [t for t in pc.symbols if 0 < w0[t] < float(floor.get(t, 0.0)) - 1e-12]
        if small:
            why.append("posições menores que uma compra mínima com o valor atual: " + ", ".join(small))
    if why:
        return {"manter_anterior": False,
                "explicacao": "a recomendação anterior não pode ser mantida: " + "; ".join(why)}
    out = stability_test(w0, comb.final, res.mu, res.cov.cov, risk_aversion,
                         len(res.returns) / res.config.trading_days, cost)
    out["_pesos"] = w0.to_dict()
    return out


def _change_reasons(w_f: pd.Series, scr: ScreenResult, not_analysed: dict[str, str], excluded_pref: dict[str, str],
                    excluded_data: dict[str, str], gov: GovernanceResult, not_exec: dict[str, str],
                    dropped: dict[str, str], allocation: list[dict[str, object]], analysed: list[str],
                    last_quote: str, context: str = "") -> dict[str, str]:
    """Why each asset is (or is not) in the portfolio, from the first pipeline step that decided it;
    ``context`` says what changed since the previous recommendation (inputs or only the data)."""
    why: dict[str, str] = {}
    for a in allocation:
        why[str(a["ticker"])] = ((context + ": " if context else "")
                                 + f"{a['motivo_selecao'] or 'candidato analisado'}; peso {float(a['peso']):.1%} com "
                                 f"dados até {last_quote} ({a['papel']})")
    for src, label in ((not_analysed, ""), (excluded_pref, "fora das suas preferências: "),
                       (excluded_data, "dados insuficientes: "), (gov.removed, "governança: "),
                       (not_exec, "tamanho mínimo de compra: "), (dropped, "tamanho mínimo de compra: ")):
        for t, r_ in src.items():
            why.setdefault(t, label + r_)
    for t in scr.table.index:
        if scr.table.at[t, "decisao"] == "descartado":
            why.setdefault(t, "triagem: " + str(scr.table.at[t, "motivo"]))
    for t in analysed:
        if float(w_f.get(t, 0.0)) < PRESENT:
            why.setdefault(t, "o otimizador não deu peso: com as estimativas atuais não melhora a relação retorno/"
                              "risco dentro dos seus limites")
    return why


def _sources(req: PurposeRequest, ds: PriceDataset, funnel: Funnel | None, rf_src: dict[str, str], cash_note: str,
             last_quote: str, fetched: str) -> list[dict[str, object]]:
    """Where every dataset came from, when, and its limits."""
    if req.offline:
        return [{"dado": "preços", "fonte": "gerador sintético determinístico (semente fixa)", "data": last_quote,
                 "limite": "demonstração do método: não descreve nenhum ativo real"},
                {"dado": "taxa livre de risco", "fonte": "; ".join(f"{k}: {v}" for k, v in rf_src.items())}]
    prices = {s: p for s, p in ds.provenance.items() if s != CASH_TICKER}
    providers = sorted({p.provider for p in prices.values()})
    cached = sum(1 for p in prices.values() if p.cache_hit)
    cdi = ds.provenance.get(CASH_TICKER)
    out: list[dict[str, object]] = [
        {"dado": "preços diários ajustados", "fonte": "Yahoo Finance (" + ", ".join(providers) + ")",
         "data": last_quote, "obtido_em_utc": fetched,
         "limite": f"série diária com atraso (o último ponto pode ser o pregão em andamento), não tempo real; "
                   f"{cached} de {len(prices)} séries vieram do cache local (até {PRICE_CACHE_HOURS:g} h)"},
        {"dado": "taxa livre de risco", "fonte": "; ".join(f"{k}: {v}" for k, v in rf_src.items())},
        {"dado": "caixa em reais", "fonte": cash_note or "não usado (classes escolhidas)",
         **({"data": str(cdi.last_observation), "obtido_em_utc": cdi.ingested_at_utc,
             "limite": "o Banco Central publica o CDI com 1 dia útil de atraso; o painel usa só as datas em que todas "
                       "as séries têm valor, então a última data usada pode ser o dia útil anterior"} if cdi else {})},
        {"dado": "classificação e temas dos ativos", "fonte": TAG_SOURCE_REAL},
    ]
    if funnel is not None:
        out.insert(1, {"dado": "listagens de ativos", "fonte": "; ".join(f"{k}: {v}" for k, v in funnel.sources.items())
                       or "catálogo curado",
                       "limite": "; ".join(f"{k}: {v}" for k, v in funnel.errors.items()) or ""})
    return out


# ----------------------------------------------------------------------------- main
def run_purpose(req: PurposeRequest, output_dir: Path | None = None, provider: object | None = None,
                governance_path: Path | None = None, cache_dir: Path | None = None,
                rf_fetch: FetchFn | None = None, progress: Callable[[str], None] | None = None,
                fast: bool = False, download_budget_s: float | None = None) -> PurposeResult:
    """``fast`` (web): quarterly instead of monthly rebalancing in the walk-forward backtest and
    fewer paths in the sensitivity runs; ``download_budget_s`` bounds the total download time.
    ``rf_fetch`` is the HTTP function for the public non-price sources (Selic at the BCB and the
    exchange listings); None = standard library."""
    steps: list[dict[str, object]] = []
    alerts: list[Alert] = []

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    # 1 ---------------------------------------------------------------- required CAGR
    ccy = req.moeda_objetivo
    held_value = sum(req.carteira_atual.values())
    start = _start_capital(req)
    reqc = required_cagr(start, req.aporte_periodico, req.periodicidade, req.valor_alvo, req.horizonte_anos)
    steps.append({"etapa": 1, "titulo": "CAGR necessário",
                  "resultado": reqc.explanation + (f" (inclui {_fmt(held_value, ccy)} da carteira atual)"
                                                   if held_value else "")})
    if reqc.status == "inatingivel":
        alerts.append(Alert("objetivo", "FAIL", reqc.explanation))

    # 2 ---------------------------------------------------------------- personal rules
    policy = _policy(req)
    steps.append({"etapa": 2, "titulo": "Perfil e regras pessoais",
                  "resultado": f"perfil final {policy.profile.value}; aversão ao risco {policy.risk_aversion:g}; teto de "
                               f"volatilidade {policy.vol_cap:.0%} a.a.; objetivo {policy.objective}; "
                               + ("; ".join(policy.rules) if policy.rules else "limites padrão do perfil")})
    for w_ in policy.warnings:
        alerts.append(Alert("perfil", "WARNING", w_))
    for a_ in policy.assumptions:
        alerts.append(Alert("perfil", "INFO", "Hipótese (informação não fornecida): " + a_))
    for r_ in policy.rules:
        alerts.append(Alert("perfil", "INFO", "Regra aplicada: " + r_))
    # a portfolio needs at least two positions of at least the minimum purchase each, none above its cap (only cash
    # may hold everything, under a cash minimum): below that no portfolio is executable, said before any data is
    # fetched (step 7 checks today's prices)
    ref_amount, ref_note = _reference_amount(req)
    w_top = min(1.0, policy.hard_cap or 1.0) if policy.min_cash > 0 else policy.limits.max_weight_per_asset
    k_min = max(2, math.ceil(1 / w_top - 1e-9))
    if req.valor_minimo_posicao > 0 and ref_amount < k_min * req.valor_minimo_posicao - 1e-9:
        why = (f"o perfil {policy.profile.value} não deixa nenhum ativo passar de {w_top:.0%} da carteira, então "
               if w_top < 1 else "")
        raise AmountTooSmall(
            f"Valor pequeno demais para montar uma carteira: {why}são necessárias pelo menos {k_min} posições de "
            f"{_fmt(req.valor_minimo_posicao, ccy)} ou mais cada (a 'Compra mínima por ativo'), e com "
            f"{_fmt(ref_amount, ccy)} entre o que você tem hoje e os aportes dos próximos 12 meses isso não fecha. Uma "
            f"carteira assim pede pelo menos {_fmt(k_min * req.valor_minimo_posicao, ccy)}, e com os preços de hoje "
            "pode pedir mais. " + _SMALL_FIX)

    # 3 ---------------------------------------------------------------- discovery + stage A
    allowed_etfs = set(req.etfs_permitidos) if req.etfs_permitidos is not None else None

    def allows(a: CatalogAsset) -> str | None:
        """The investor's preferences, then the approved-ETF list of an external rulebook (when one is given)."""
        if why := policy.allows(a):
            return why
        if allowed_etfs is not None and is_etf(a) and a.ticker not in allowed_etfs:
            return "ETF fora da lista aprovada pela regra"
        return None

    cache_root = cache_dir or Path(".cache") / "market_data"
    funnel: Funnel | None = None
    pre = extract_themes(req.motivo_investidor)          # themes only steer the theme slots
    pre_tags = {tag for t in pre.themes if t.name in THEME_TO_TAGS for tag in THEME_TO_TAGS[t.name]}
    mine = set(req.carteira_atual) | set(req.carteira_anterior)
    not_analysed: dict[str, str] = {}      # holdings / previous positions that cannot be analysed, and why
    want_cash = req.classes is None or "caixa" in req.classes
    classes = None if req.classes is None else [c for c in req.classes if c != "caixa"]   # caixa = the CDI model
    if req.offline:
        base = expand(SYNTHETIC_CATALOG, classes)
        if want_cash and classes is not None:      # the synthetic stand-in for the CDI cash
            base += [a for a in SYNTHETIC_CATALOG if traits(a).market == "CAIXA" and a not in base]
        excluded_pref = {a.ticker: why for a in base if (why := allows(a))}
        pool = [a for a in base if a.ticker not in excluded_pref]
        known = {a.ticker: a for a in SYNTHETIC_CATALOG}
        for t in sorted(mine):
            if t not in known:
                not_analysed[t] = "não existe no universo sintético de demonstração"
            elif t in excluded_pref:
                not_analysed[t] = f"fora das suas preferências: {excluded_pref[t]}"
            elif all(a.ticker != t for a in pool):
                not_analysed[t] = "fora das classes de ativos escolhidas"
        stage_a = (f"{len(base)} ativos sintéticos; {len(excluded_pref)} fora das suas preferências → {len(pool)} "
                   "analisados")
    else:
        listings = None
        if req.universo == "completo":
            say("lendo as listagens completas da B3 e dos EUA…")
            listings = load_market_listings(cache_root, rf_fetch)
        known = {a.ticker: a for a in REAL_CATALOG}
        listed = {la.ticker: la for la in listings.all()} if listings is not None and listings.available else {}
        always = []
        for t in sorted(mine - {CASH_TICKER}):
            a_ = known.get(t) or (listing_to_catalog(listed[t]) if t in listed else None)
            if a_ is None:
                not_analysed[t] = "não reconhecido: não está no catálogo curado nem nas listagens consultadas"
            elif why := allows(a_):
                not_analysed[t] = f"fora das suas preferências: {why}"
            elif classes is not None and a_.macro_class not in classes:
                not_analysed[t] = "fora das classes de ativos escolhidas"
            else:
                always.append(a_)
        pool_cap = min(max(int(POOL_FACTOR * req.max_ativos), req.max_ativos + 10), POOL_MAX)
        pool, funnel = preselect(expand(REAL_CATALOG, classes), listings, pre_tags, pool_cap, classes,
                                 allows=allows, always=always)
        excluded_pref = dict(funnel.excluded_by_preference)
        listed_txt = (f"{funnel.listed_b3} na B3 e {funnel.listed_us} nos EUA listados + catálogo curado"
                      if funnel.used_listings else "catálogo curado (listagens completas indisponíveis)")
        stage_a = (f"{listed_txt} → {funnel.eligible} elegíveis (classe identificada, suas preferências, liquidez "
                   f"mínima, uma classe de ação por empresa) → {funnel.selected} para baixar, amostra estratificada "
                   "por classe e setor (desempenho passado NÃO usado nesta etapa)")
    for t, why in not_analysed.items():
        whose = "da sua carteira atual" if t in req.carteira_atual else "da recomendação anterior"
        alerts.append(Alert("dados", "WARNING", f"{t} ({whose}) não entra na análise: {why}"))
    cash_note = ""
    if not req.offline:
        provider = WithExtraSeries(provider or _default_provider(cache_root, progress, download_budget_s))
        cash_why = policy.allows(CASH_ASSET)
        if cash_why and want_cash:
            excluded_pref[CASH_TICKER] = cash_why
            cash_note = f"caixa (CDI) não usado: {cash_why}"
        elif want_cash:
            try:
                provider.extra[CASH_TICKER] = fetch_cdi_cash(rf_fetch)  # type: ignore[arg-type]
                pool.append(CASH_ASSET)
                cash_note = "caixa pós-fixado modelado pelo CDI diário (Banco Central, SGS 12)"
            except (DownloadError, EmptyDataError, OSError, ValueError) as e:
                cash_note = f"caixa (CDI) indisponível: {e}"
                alerts.append(Alert("dados", "WARNING", cash_note + " — a carteira foi montada sem esse instrumento; "
                                    "mínimos de caixa (se houver) podem ficar inviáveis e aparecem nos alertas."))
    steps.append({"etapa": 3, "titulo": "Descoberta e filtro do universo", "resultado": stage_a})
    live: dict[str, RiskFreeQuote] = {}
    live_errors: dict[str, str] = {}
    if not req.offline:
        say(f"baixando cotações reais de {len(pool)} ativos e as taxas livres de risco…")
        need = ({a.currency for a in pool} | {ccy}) - set(req.taxa_livre_risco)
        live, live_errors = fetch_live_rates(need, provider, rf_fetch)  # type: ignore[arg-type]
    rf0, rf0_src = _resolve_rf(req, set(), live)    # objective-currency rf (never silently 0)
    cfg = AppConfig(offline=req.offline, base_currency=ccy, cache_dir=cache_root,
                    risk_free_rate=rf0[ccy],
                    risk_free_rate_source="demo_fallback" if rf0_src[ccy].startswith("FALLBACK") else "user",
                    run_heavy_analyses=False, run_backtest=True, n_simulations=1000, horizon_days=21,
                    profile_horizon_simulation=False, make_charts=False, n_resamples=0, evt_bootstrap=0,
                    seed=req.seed, rebalance_every=126 if fast else 21, backtest_light=fast,
                    output_dir=output_dir or Path("outputs") / "proposito")

    # 4 ---------------------------------------------------------------- data + stage B screening
    try:
        ds_pool, truth = _load_universe(req, pool, cfg, provider)
    except DataUnavailableError as e:
        wanted = {a.ticker for a in pool}
        fails = {k: v for k, v in (getattr(provider, "failures", {}) or {}).items() if k in wanted}
        if fails:
            first = next(iter(fails.values()))
            raise DataUnavailableError(f"{len(fails)} de {len(pool)} tickers sem cotação; exemplo — {first}") from e
        raise
    excluded_data = dict(ds_pool.quality.excluded_symbols)
    failed = {k: v for k, v in excluded_data.items() if v.startswith("download failure")}
    # series left untried because the download time ran out are not failures of the source: the pool was larger
    # than the time allowed, and the screening works with what was downloaded (said in the alerts)
    untried = {k: v for k, v in failed.items() if "tempo de download esgotado" in v}
    tried = len(pool) - len(untried)
    if not req.offline and len(failed) - len(untried) > MAX_FAILED_SHARE * max(tried, 1):
        first = next(v for k, v in failed.items() if k not in untried)
        raise DataUnavailableError(f"{len(failed) - len(untried)} de {tried} séries sem cotação agora (exemplo — "
                                   f"{first}); uma carteira montada só com o restante não seria confiável")
    cat = {a.ticker: a for a in pool}
    # governance is a hard exclusion applied BEFORE the screening: a flagged asset takes no candidate slot and is
    # never offered as an alternative
    gov = apply_governance(ds_pool.symbols, req.incluir_sob_revisao, governance_path, req.restricoes_governanca)
    series = {s: ds_pool.raw_prices[s] for s in gov.kept}
    say(f"medindo risco, qualidade, liquidez e correlação de {len(series)} ativos…")
    need = policy.required_groups([cat[s] for s in series])
    if req.min_etfs:
        need.append((f"a regra de pelo menos {req.min_etfs} ETF(s)", [s for s in series if is_etf(cat[s])],
                     max(2, req.min_etfs + 1)))
    if req.min_setores:
        by_sector: dict[str, list[str]] = {}
        for s in series:
            if (sec := business_sector(cat[s])) is not None:
                by_sector.setdefault(sec, []).append(s)
        need.extend((f"a regra de pelo menos {req.min_setores} setores (setor {SECTOR_LABEL.get(k, k)})", v, 1)
                    for k, v in sorted(by_sector.items()))
    scr = screen(series, ds_pool.volumes, {s: cat[s] for s in series}, policy.profile, policy.objective, rf0[ccy],
                 req.max_ativos, required=need,
                 themed={s for s in series if set(cat[s].tags) & pre_tags},
                 keep={s: ("você já tem este ativo" if s in req.carteira_atual else "estava na recomendação anterior")
                       + ": entra na análise para ser comparado com as alternativas" for s in series if s in mine})
    for t in sorted(mine & set(scr.short_history)):
        not_analysed[t] = scr.short_history[t]
        whose = "da sua carteira atual" if t in req.carteira_atual else "da recomendação anterior"
        alerts.append(Alert("dados", "WARNING", f"{t} ({whose}) não entra na análise: {scr.short_history[t]}"))
    if len(scr.selected) < 2:
        raise DataUnavailableError(f"menos de dois ativos com histórico suficiente (>= {MIN_CANDIDATE_OBS} preços "
                                   "diários) depois das suas preferências e da governança: amplie as classes ou os "
                                   "mercados")
    if req.offline:
        ds_all = subset_dataset(ds_pool, scr.selected)
    else:       # re-aligned on the candidates only: a recent listing outside them does not shorten their history
        ds_all, _ = _load_universe(req, [cat[s] for s in scr.selected], cfg, provider)
    price_prov = {s_: p for s_, p in ds_all.provenance.items() if s_ != CASH_TICKER}    # the CDI is listed apart
    providers = sorted({p.provider for p in price_prov.values()})
    last_quote = str(ds_all.prices.index[-1].date())
    fetched = max((p.ingested_at_utc for p in price_prov.values() if p.ingested_at_utc), default="")
    universe = {"origem": "SINTÉTICA (demonstração offline)" if ds_all.is_synthetic
                else "real — Yahoo Finance (" + ", ".join(providers) + ")",
                "ultima_cotacao": last_quote, "obtido_em_utc": fetched,
                "baixados": [a.ticker for a in pool], "candidatos": list(scr.selected), "validados": ds_all.symbols,
                "excluidos_por_dados": excluded_data, "excluidos_por_preferencia": excluded_pref,
                "fora_da_analise": not_analysed, "min_observacoes": MIN_OBS,
                "min_observacoes_candidato": MIN_CANDIDATE_OBS, "historico_curto": scr.short_history,
                "n_datas": len(ds_all.prices), "periodo": f"{ds_all.prices.index[0].date()} a "
                                                         f"{ds_all.prices.index[-1].date()}",
                "cobertura": ("B3: ações, FIIs e ETFs negociados no dia de referência (listagem oficial da B3); EUA: "
                              "ações do screener da Nasdaq (NYSE, Nasdaq e AMEX) e ETFs do catálogo curado; caixa em "
                              "reais pelo CDI. Não cobre outros países, fundos não listados, CDBs/LCIs específicos nem "
                              "títulos do Tesouro individuais (representados por ETFs e pelo CDI)." if not req.offline
                              else "universo sintético de demonstração (14 ativos fictícios)")}
    if funnel is not None:
        universe["funil"] = funnel.as_dict()
    steps.append({"etapa": 4, "titulo": "Dados e triagem",
                  "resultado": f"{len(ds_pool.symbols)} de {len(pool)} com série validada (>= {MIN_OBS} observações); "
                               f"{len(gov.removed)} fora por governança antes da triagem; triagem para o perfil "
                               f"{policy.profile.value} (candidato precisa de >= {MIN_CANDIDATE_OBS} preços diários): "
                               f"{scr.notes[0]}; excluídos por dados: {len(excluded_data)}"})
    if funnel is not None:
        if funnel.used_listings:
            alerts.append(Alert("dados", "INFO",
                                f"Universo: todas as ações listadas lidas ({funnel.listed_b3} na B3, {funnel.listed_us} "
                                f"nos EUA; fontes: {', '.join(funnel.sources.values())}) → {funnel.eligible} elegíveis "
                                f"→ {funnel.selected} baixados → {len(scr.selected)} candidatos após a triagem. "
                                f"Critérios: {funnel.criteria}."))
        else:
            why = "; ".join(f"{k}: {v}" for k, v in funnel.errors.items()) or "modo curado escolhido"
            alerts.append(Alert("dados", "WARNING" if funnel.errors else "INFO",
                                f"Listagens completas da B3/EUA não usadas ({why}): universo = catálogo curado de "
                                f"{len(REAL_CATALOG)} ativos → {funnel.selected} baixados → {len(scr.selected)} "
                                f"candidatos. Critérios: {funnel.criteria}."))
        alerts.append(Alert("dados", "INFO", "Cobertura consultada: " + str(universe["cobertura"])))
    if ds_all.is_synthetic:
        alerts.append(Alert("dados", "FAIL",
                            "DADOS SINTÉTICOS: os tickers SINT_* não existem no mercado. Este resultado é uma "
                            "DEMONSTRAÇÃO do método e NÃO é uma carteira final; para tickers reais rode no modo "
                            "online com acesso ao provedor."))
    else:
        alerts.append(Alert("dados", "INFO",
                            f"DADOS REAIS: {len(ds_all.symbols)} candidatos com cotação do Yahoo Finance (preço ajustado "
                            f"por desdobramentos e dividendos), {universe['periodo']}; última cotação {last_quote}; "
                            f"obtidos em {fetched[:16].replace('T', ' ')} UTC. Fechamentos diários com atraso, não "
                            "cotações em tempo real; o provedor pode revisar séries; as métricas são históricas [H] e "
                            "estimativas [E], não previsões."))
    if len(failed) > len(untried):
        n_bad = len(failed) - len(untried)
        alerts.append(Alert("dados", "WARNING" if n_bad > 0.2 * tried else "INFO",
                            f"{n_bad} de {tried} ativos sem cotação nesta consulta (excluídos, sem proxy)."))
    if untried:
        alerts.append(Alert("dados", "INFO", f"{len(untried)} de {len(pool)} ativos não foram baixados porque o tempo "
                                             f"de download acabou; a triagem usou os {len(ds_pool.symbols)} com dados "
                                             "(nada foi preenchido no lugar deles)."))
    for s_, why in excluded_data.items():
        if s_ not in failed:
            alerts.append(Alert("dados", "WARNING", f"{s_} excluído (sem proxy): {why}"))

    # 5 ---------------------------------------------------------------- currency sleeves
    ccys = {ds_all.native_currency.get(s_, ccy) for s_ in ds_all.symbols}
    rf, rf_src = _resolve_rf(req, ccys, live)
    for c, err in sorted(live_errors.items()):
        if c in ccys or c == ccy:
            alerts.append(Alert("rf", "WARNING", f"rf de {c}: fonte automática indisponível ({err})"))
    nstats = native_asset_stats(ds_all, rf, ccy, cfg.trading_days)
    steps.append({"etapa": 5, "titulo": "Sleeves por moeda",
                  "resultado": "; ".join(f"{c}: {int((nstats['moeda'] == c).sum())} ativos, rf {rf[c]:.2%} "
                                         f"({rf_src[c]})" for c in sorted(ccys))})
    for c in sorted(set(rf) | ccys):
        if rf_src[c].startswith("FALLBACK"):
            alerts.append(Alert("rf", "WARNING", f"rf de {c} = {rf[c]:.2%} é fallback de demonstração, não cotação; "
                                                 "informe a taxa real de cada moeda."))

    # 6 ---------------------------------------------------------------- governance (hard exclusion, applied in 4)
    ds = subset_dataset(ds_all, [s_ for s_ in ds_all.symbols if s_ in gov.kept])
    if len(ds.symbols) < 2:
        raise ValueError("menos de dois ativos elegíveis após dados e governança")
    steps.append({"etapa": 6, "titulo": "Filtro de governança",
                  "resultado": ("removidos: " + "; ".join(f"{k} — {v}" for k, v in gov.removed.items())
                                if gov.removed else "nenhum ativo removido")
                               + ("; incluídos sob revisão por opção explícita: "
                                  + ", ".join(gov.included_under_review) if gov.included_under_review else "")})
    for s_, why in gov.removed.items():
        alerts.append(Alert("governanca", "INFO", f"{s_}: {why}"))
    for s_, why in gov.included_under_review.items():
        alerts.append(Alert("governanca", "WARNING", f"{s_}: {why}"))
    for t, why in gov.rejected_entries.items():
        alerts.append(Alert("governanca", "WARNING", f"flag de governança de {t} ignorada: {why}"))

    # 7 ---------------------------------------------------------------- executable universe
    last_px = ds.prices.iloc[-1]
    lim = policy.limits
    over = policy.upper_overrides([cat[s_] for s_ in ds.symbols])
    caps = {s_: over.get(s_, min(lim.max_weight_per_asset, lim.max_class_weight.get(cat[s_].asset_class, 1.0)))
            for s_ in ds.symbols}
    not_exec = {s_: f"a classe {cat[s_].asset_class} tem limite de 0% para o seu perfil e horizonte"
                for s_, cp in caps.items() if cp <= 0}
    if req.preco_minimo is not None:      # e.g. "stocks priced at $5 or higher": the last close is the reference
        for s_ in ds.symbols:
            if s_ not in not_exec and not is_money(cat[s_]) and float(last_px[s_]) < req.preco_minimo:
                not_exec[s_] = (f"último preço {_fmt(float(last_px[s_]), ccy)} abaixo do mínimo de "
                                f"{_fmt(req.preco_minimo, ccy)} da regra")
                caps[s_] = 0.0
    keep_x, too_big = executable_universe({s_: cat[s_] for s_ in ds.symbols if s_ not in not_exec}, last_px, caps,
                                          ref_amount, req.valor_minimo_posicao, req.fracionario_eua, ccy)
    not_exec.update(too_big)
    cands = [s_ for s_ in ds.symbols if caps[s_] > 0]

    def buyable(amount: float) -> bool:
        return _buyable(cands, {s_: max(WEIGHT_FLOOR, min_weight(cat[s_], float(last_px[s_]), amount,
                                                                 req.valor_minimo_posicao, req.fracionario_eua))
                                for s_ in cands}, ds.meta, cat, policy, cfg)

    if not buyable(ref_amount):
        # the money cannot buy a portfolio within the limits: said with the least amount that can, never computed
        # around with positions the money cannot buy
        least = _least_amount(buyable, ref_amount)
        frac = ("Se a sua corretora compra frações de ações dos EUA, marque essa opção. "
                if not req.fracionario_eua and any(cat[s_].macro_class in ("acoes_eua", "etf_eua") for s_ in too_big)
                else "")
        if least is not None:
            raise AmountTooSmall(
                f"Valor pequeno demais para montar uma carteira dentro dos limites do seu perfil: com "
                f"{_fmt(ref_amount, ccy)} entre o que você tem hoje e os aportes dos próximos 12 meses, não há "
                "combinação de ativos em que cada posição compre pelo menos 1 ação ou cota e "
                f"{_fmt(req.valor_minimo_posicao, ccy)} sem passar dos limites do perfil {policy.profile.value} (peso "
                "máximo por ativo, por classe e por setor, e os mínimos do perfil). Com os preços de hoje, a menor "
                f"carteira possível pede pelo menos {_fmt(math.ceil(least / 10) * 10, ccy)}. " + frac + _SMALL_FIX)
        if len(keep_x) < 2:
            raise AmountTooSmall(
                f"com {_fmt(ref_amount, ccy)} entre o que você tem hoje e os aportes dos próximos 12 meses, menos de dois "
                "ativos do universo escolhido cabem numa compra mínima, e nem com mais dinheiro os ativos escolhidos "
                f"fecham uma carteira dentro das regras do perfil {policy.profile.value}: amplie os mercados e as "
                "classes. " + frac + _SMALL_FIX)
    ds = subset_dataset(ds, keep_x)
    exec_txt = (f"{len(not_exec)} ativo(s) fora: " + "; ".join(f"{k} — {v}" for k, v in list(not_exec.items())[:6])
                if not_exec else "todos os candidatos cabem numa compra mínima")
    steps.append({"etapa": 7, "titulo": "Tamanho mínimo de compra", "resultado": ref_note + "; " + exec_txt})

    # 8 ---------------------------------------------------------------- quant optimisation
    held_w = ({t: v / start for t, v in req.carteira_atual.items() if t in ds.symbols}
              if req.carteira_atual and start > 0 else None)
    assets_ds = [cat[s_] for s_ in ds.symbols]
    cost_turnover = (cfg.transaction_cost_bps + cfg.spread_bps + cfg.slippage_bps) / 1e4
    groups = policy.groups(assets_ds)
    for g in [g for g in groups if g.lower > 0 and not g.members]:     # a minimum nothing available can meet
        groups.remove(g)
        if g.name.startswith("caixa"):
            alerts.append(Alert("perfil", "WARNING",
                                f"Regra '{g.name}' de {g.lower:.0%} não aplicada: nenhum instrumento de caixa em {ccy} "
                                "está disponível agora" + (f" ({cash_note})" if cash_note else "") + ". A carteira "
                                f"abaixo não tem essa reserva: separe {g.lower:.0%} em caixa por conta própria (por "
                                "exemplo, Tesouro Selic ou CDB de liquidez diária) e aplique os pesos ao restante, ou "
                                "tente de novo mais tarde."))
        else:
            alerts.append(Alert("perfil", "WARNING",
                                f"Regra '{g.name}' de {g.lower:.0%} não aplicada: nenhum ativo que a cumpra passou pelos "
                                "filtros agora; a carteira abaixo não atende a essa regra."))
    opt_policy = OptimizationPolicy(lim, groups, policy.upper_overrides(assets_ds),
                                    policy.risk_aversion, held_w, cost_turnover)
    say(f"otimizando e validando no backtest walk-forward ({len(ds.symbols)} ativos)…")
    res = run_analysis(cfg, policy.profile_input, dataset=(ds, truth), policy=opt_policy)
    if fast:
        alerts.append(Alert("backtest", "INFO", "modo rápido do site: backtest walk-forward com rebalanceamento "
                                                "semestral, 7 estratégias e menos pontos de partida no otimizador "
                                                "(a linha de comando usa rebalanceamento mensal e 9 estratégias)"))
    w_q = res.selected.weights
    sel = res.selection
    how = (f"escolhida por {sel.get('rule', 'regra 1-EP')} no walk-forward fora da amostra" if res.backtest is not None
           else f"backtest NÃO executado ({sel.get('rule', 'sem seleção')})")
    steps.append({"etapa": 8, "titulo": "Otimização quantitativa",
                  "resultado": f"estratégia '{res.selected_name}' {how}; perfil final "
                               f"{res.profile.final_profile.value}; [E] retorno {res.selected.expected_return:.2%}, "
                               f"vol {res.selected.volatility:.2%}"})
    if not res.feasibility.get("feasible", True):
        alerts.append(Alert("validacao", "FAIL", "restrições do perfil INVIÁVEIS para o universo elegível ("
                            + "; ".join(map(str, res.feasibility.get("conflicts", []))) + "): a carteira não "
                            "respeita o perfil; amplie as classes do universo ou revise o perfil."))
    if res.backtest is not None and isinstance(sel, dict):
        tv = sel.get("test_metrics_selected") or {}
        te = sel.get("test_metrics_equal_weight") or {}
        if tv and te:
            alerts.append(Alert("backtest", "INFO",
                                f"comparação com a linha de base 1/N (pesos iguais) no período de teste [H]: Sharpe "
                                f"{float(tv.get('sharpe', np.nan)):.2f} (escolhida) vs {float(te.get('sharpe', np.nan)):.2f} "
                                "(1/N); 1/N fica como referência e não é escolhida automaticamente."))

    # 9-10 ------------------------------------------------------------- themes + alignment
    say("lendo o seu motivo e pontuando o alinhamento…")
    themes = extract_themes(req.motivo_investidor)
    elig_assets = [cat[s_] for s_ in ds.symbols]
    al = score_assets(themes, elig_assets)
    steps.append({"etapa": 9, "titulo": "Temas do motivo",
                  "resultado": ("motivo vazio: score de alinhamento não calculado por falta de input"
                                if themes.empty else
                                f"temas: {[THEME_LABEL.get(t.name, t.name) for t in themes.themes] or 'nenhum'}; "
                                f"confiança {themes.confidence}" + (f"; porquê: \"{themes.why}\"" if themes.why else ""))})
    steps.append({"etapa": 10, "titulo": "Score de alinhamento",
                  "resultado": "calculado para todos os ativos elegíveis" if al.computed else "não calculado"})
    if themes.empty:
        alerts.append(Alert("alinhamento", "WARNING", "motivo_investidor vazio: score de alinhamento não calculado por "
                                                      "falta de input; carteira apenas quantitativa."))
    elif themes.confidence == "baixa":
        alerts.append(Alert("alinhamento", "WARNING", "confiança BAIXA na extração de temas: " + "; ".join(al.notes)))
    for n in al.notes:
        if "sem correspondência" in n:
            alerts.append(Alert("alinhamento", "INFO", n))

    # 11 --------------------------------------------------------------- combination + executable weights
    vol_cap = res.profile.research_limits.target_volatility_band[1]
    comb = combine(w_q, al, res.mu, res.cov.cov, res.constraints, req.peso_alinhamento,
                   req.orcamento_tracking_error, vol_cap)
    if comb.violations:
        alerts.append(Alert("validacao", "FAIL", "carteira combinada violou limites (" + "; ".join(comb.violations)
                            + "): usados os pesos quantitativos"))
        comb = dataclasses.replace(comb, final=comb.quant, vol_final=comb.vol_quant, ret_final=comb.ret_quant,
                                   alignment_final=comb.alignment_quant, tracking_error=0.0)
    floor = pd.Series({s_: min_weight(cat[s_], float(last_px[s_]), ref_amount, req.valor_minimo_posicao,
                                      req.fracionario_eua) for s_ in res.constraints.symbols})
    comb, dropped, viable = _make_executable(comb, res.constraints, res.cov.cov, res.mu, al, floor,
                                             max(comb.vol_quant, vol_cap),
                                             req.peso_alinhamento * req.orcamento_tracking_error
                                             if comb.alignment_target is not None else None,
                                             _count_rules(req, {s_: cat[s_] for s_ in res.constraints.symbols}))
    if not viable:
        alerts.append(Alert("execucao", "WARNING", "sem as posições pequenas demais para comprar ("
                            + ", ".join(dropped) + ") os limites do perfil não fecham: a carteira-alvo foi mantida e "
                            + ("essas posições entram aos poucos pelos aportes." if req.aporte_periodico > 0 else
                               "essas posições ficam fora da compra com o valor informado.")))
        dropped = {}
    # stability: the previous recommendation is kept unless the new one is better beyond error + cost
    stability = None
    if req.carteira_anterior:
        stability = _stability(req, comb, res, floor, vol_cap, policy.risk_aversion, cost_turnover)
        if stability.get("manter_anterior"):
            prev_in = pd.Series(stability.pop("_pesos"), dtype=float)
            comb = _restate(comb, prev_in.reindex(comb.final.index).fillna(0.0), res.cov.cov, res.mu, al,
                            "recomendação anterior mantida pela regra de estabilidade")
            dropped = {t: why for t, why in dropped.items() if float(comb.final.get(t, 0.0)) <= 0.0}
        stability.pop("_pesos", None)
    w_f = comb.final
    gap_q = _frontier_gap(res.frontier, comb.vol_quant, comb.ret_quant)
    gap_f = _frontier_gap(res.frontier, comb.vol_final, comb.ret_final)
    steps.append({"etapa": 11, "titulo": "Combinação quant + alinhamento",
                  "resultado": f"{1 - req.peso_alinhamento:.0%} quant / {req.peso_alinhamento:.0%} alinhamento; "
                               f"tracking error vs quant {comb.tracking_error:.2%} (orçamento "
                               f"{req.peso_alinhamento * req.orcamento_tracking_error:.2%}); vol "
                               f"{comb.vol_quant:.2%} -> {comb.vol_final:.2%}; distância à fronteira eficiente "
                               f"(retorno [E] abaixo da fronteira na mesma volatilidade): {_pct(gap_q)} (quant) -> "
                               f"{_pct(gap_f)} (final)"
                               + ("; " + " ".join(comb.notes) if comb.notes else "")})
    leaked = [s_ for s_ in gov.removed if s_ in w_f.index and w_f[s_] > 0]
    if leaked:   # defensive: cannot happen because the assets are not in the optimised universe
        raise RuntimeError(f"ativos removidos por governança com peso positivo: {leaked}")

    # 12 --------------------------------------------------------------- table, theses, explanations
    cov = res.cov.cov.loc[w_f.index, w_f.index].to_numpy()
    wv = w_f.to_numpy()
    tot = float(wv @ cov @ wv)
    rc = pd.Series(wv * (cov @ wv) / tot if tot > 0 else np.zeros(len(wv)), index=w_f.index)
    rows, theses = [], []
    for s in w_f.sort_values(ascending=False).index:
        a = cat[s]
        sc = al.scores[s].score if al.computed else None
        rows.append({"ticker": s, "classe": a.macro_class, "peso": float(w_f[s]),
                     "score_alinhamento": sc, "moeda": ds.native_currency.get(s, ccy),
                     "peso_quant": float(comb.quant[s]), "motivo_score": al.scores[s].reason})
        if w_f[s] > 0:
            theses.append({"ticker": s, "nome": a.name,
                           "tese_financeira": _financial_thesis(a, s, res, w_f, rc, ds.is_synthetic),
                           "tese_proposito": _purpose_thesis(a, s, al, themes)})
    table = pd.DataFrame(rows).set_index("ticker").sort_values("peso", ascending=False)
    # where to invest and why: amount, role, link to the motive, evidence and alternatives
    m_per = PERIODS_PER_YEAR[req.periodicidade]
    mu_f = res.mu.loc[w_f.index]
    port_mu = float(w_f @ mu_f)
    held_now = [s for s in w_f.index if w_f[s] > 0]
    corr_h = res.returns[held_now].corr() if len(held_now) > 1 else None
    mu_unc = res.sections.get("mu_uncertainty")
    allocation = []
    for s in table.index[table["peso"] > 0]:
        a = cat[s]
        w_s = float(w_f[s])
        ret_share = float(w_s * mu_f[s] / port_mu) if port_mu > 0 else None
        role = _role(w_s, float(rc[s]), ret_share)
        if al.computed and al.scores[s].matched:
            purpose = "atende ao(s) tema(s) do seu motivo: " + ", ".join(al.scores[s].matched)
        elif al.computed:
            purpose = "sem ligação com os temas do seu motivo (está pela função financeira)"
        else:
            purpose = "alinhamento não calculado (motivo vazio)" if themes.empty else "alinhamento não calculado"
        st = res.asset_stats.loc[s]
        pv = ds.provenance.get(s)
        avg_c = (float(corr_h.loc[s, [x for x in held_now if x != s]].mean()) if corr_h is not None else None)
        t_stat = (float(mu_unc.loc[s, "t_stat"]) if isinstance(mu_unc, pd.DataFrame) and s in mu_unc.index
                  else float("nan"))
        alts = [{"ticker": x, "nome": cat[x].name if x in cat else x,
                 "pontuacao": float(scr.table.at[x, "pontuacao"]), "motivo": str(scr.table.at[x, "motivo"])}
                for x in scr.alternatives(s)]
        mk = traits(a).market
        allocation.append({
            "ticker": s, "nome": a.name, "logo": None if ds.is_synthetic else logo_path(s),
            "classe": a.macro_class, "moeda": table.loc[s, "moeda"],
            "mercado": {"B3": "B3 (Brasil)", "US": "EUA", "CAIXA": "caixa em reais (CDI)"}[mk], "peso": w_s,
            "valor_inicial": req.capital_inicial * w_s,
            "valor_alvo": start * w_s, "valor_atual": float(req.carteira_atual.get(s, 0.0)),
            "aporte_por_periodo": req.aporte_periodico * w_s if m_per else 0.0,
            "papel": role, "proposito": purpose, "por_que": f"{role}; {purpose}",
            "negocio": a.business.replace("[sintético] ", ""), "riscos": a.class_risk,
            "motivo_selecao": str(scr.table.at[s, "motivo"]) if s in scr.table.index else "",
            "diversificacao": (f"{float(rc[s]):.0%} do risco com {w_s:.0%} do dinheiro"
                               + (f"; correlação média de {avg_c:.2f} com as demais posições" if avg_c is not None
                                  else "")),
            "alternativas": alts,
            "fatos_H": (f"{float(st.get('arith_mean_ann', np.nan)):.1%} de retorno médio anual, volatilidade "
                        f"{float(st.get('vol_ann', np.nan)):.1%} e queda máxima {float(st.get('max_drawdown', np.nan)):.1%} "
                        f"de {universe['periodo']} (em {ccy})"),
            "estimativas_E": (f"retorno esperado {float(mu_f[s]):.1%} ao ano"
                              + (f" (t = {t_stat:.2f}: {'indistinto de zero' if abs(t_stat) < 2 else 'distinto de zero'})"
                                 if np.isfinite(t_stat) else "") + f"; {float(rc[s]):.0%} do risco da carteira"),
            "hipoteses": ("supõe que volatilidades e correlações futuras fiquem parecidas com as do histórico; "
                          "o retorno passado pode não se repetir"),
            "dados": (f"{pv.provider}, {pv.n_observations} observações até {pv.last_observation}"
                      + (" (cache)" if pv.cache_hit else "") if pv is not None else "sintético")})
    strat = STRATEGY_PT.get(res.selected_name, res.selected_name)
    chosen_how = ("pela validação fora da amostra (walk-forward; a carteira de utilidade do seu perfil fica quando "
                  "está a 1 erro-padrão da melhor) entre as estratégias que respeitam o perfil "
                  if res.backtest is not None and res.selection.get("preferred") else
                  "pela validação fora da amostra (walk-forward, regra de 1 erro-padrão) entre as estratégias que "
                  "respeitam o perfil " if res.backtest is not None else "sem backtest (restrições inviáveis) para o perfil ")
    strategy_text = (f"Carteira-base quantitativa: {strat}, escolhida {chosen_how}{res.profile.final_profile.value}. "
                     + (f"Depois, {req.peso_alinhamento:.0%} da alocação foi inclinada para os ativos ligados ao seu "
                        "motivo, sem sair dos limites de risco, governança e tracking error."
                        if al.computed and req.peso_alinhamento > 0 else "Sem ajuste de propósito."))
    only_alignment = [s for s in w_f.index if w_f[s] > 0 and comb.quant[s] < WEIGHT_FLOOR]
    for s in only_alignment:
        sc_s = al.scores[s].score if al.computed else None
        if sc_s:
            msg = (f"{s} entrou pela componente de alinhamento (score {sc_s:.2f}; peso quant ~0), dentro de todos "
                   "os limites de risco, governança e tracking error")
        else:
            msg = (f"{s} (score 0) entrou como contrapeso de risco da componente de alinhamento: mantém a volatilidade "
                   "e o tracking error dentro dos limites ao abrir espaço para os ativos alinhados")
        alerts.append(Alert("alinhamento", "INFO", msg))

    # purchase plans: what the money buys today (lots, minimums) and what the next contribution buys
    current = dict(req.carteira_atual)
    contributions = req.aporte_periodico > 0 and bool(m_per)
    buy_now: ExecutionPlan | None = None
    if req.capital_inicial > 0:      # all at once only when nothing else will be added; otherwise gaps first
        buy_now = (plan(w_f, last_px, cat, req.capital_inicial, req.fracionario_eua, ccy)
                   if not current and not contributions else
                   contribution_plan(w_f, current, last_px, cat, req.capital_inicial, req.valor_minimo_posicao,
                                     req.fracionario_eua, more_coming=contributions, ccy=ccy))
    after = buy_now.holdings_after(current) if buy_now else dict(current)
    next_c = (contribution_plan(w_f, after, last_px, cat, req.aporte_periodico, req.valor_minimo_posicao,
                                req.fracionario_eua, ccy=ccy) if contributions else None)
    context = ""
    if stability is not None:
        context = ("recalculado para as suas entradas atuais (mudaram desde a recomendação anterior)"
                   if req.carteira_anterior_parametros != _param_fingerprint(req) else
                   f"recalculado com os mesmos parâmetros e dados até {last_quote}")
    why = _change_reasons(w_f, scr, not_analysed, excluded_pref, excluded_data, gov, not_exec, dropped, allocation,
                          res.constraints.symbols, last_quote, context)
    sells = []
    if current:
        total = held_value + req.capital_inicial
        for t, v in sorted(after.items(), key=lambda kv: -kv[1]):
            if t == CASH_TICKER or (t in cat and is_money(cat[t])):
                continue        # cash above the target is not sold: the plans redeem it (shown in each plan)
            tgt = float(w_f.get(t, 0.0)) * total
            if v - tgt > max(req.valor_minimo_posicao, 0.01 * total):
                sells.append({"ticker": t, "valor_atual": v, "valor_alvo": tgt, "excesso": v - tgt,
                              "motivo": why.get(t, "")})
    # no new money is coming: the target is reached only by selling the excess (and using the cash above the
    # target), so say where that money would go
    rebalance: ExecutionPlan | None = None
    if current and not contributions:
        sold = dict(after)
        for v_ in sells:
            sold[v_["ticker"]] = sold.get(v_["ticker"], 0.0) - float(v_["excesso"])
        rb = contribution_plan(w_f, sold, last_px, cat, float(sum(v_["excesso"] for v_ in sells)),
                               req.valor_minimo_posicao, req.fracionario_eua, more_coming=False, ccy=ccy)
        rebalance = rb if sells or rb.rows else None
    execution = {
        "valor_de_referencia": ref_amount, "nota_referencia": ref_note,
        "precos": (f"último preço da série diária ({last_quote}), em {ccy} — dado com atraso (pode ser o pregão "
                   "ainda em andamento), não cotação em tempo real; confira o preço na corretora antes de comprar"),
        "regras_de_lote": {s: lot_size(cat[s], req.fracionario_eua)[1] for s in w_f.index if w_f[s] > 0},
        "valor_minimo_posicao": req.valor_minimo_posicao, "fracionario_eua": req.fracionario_eua,
        "fora_por_tamanho_minimo": not_exec, "retirados_apos_otimizar": dropped,
        "compra_inicial": buy_now.as_dict() if buy_now else None,
        "proximo_aporte": next_c.as_dict() if next_c else None,
        "vendas_sugeridas": sells,
        "reequilibrio": rebalance.as_dict() if rebalance else None,
        "nota_vendas": ("vender tem custo e pode gerar imposto sobre ganho de capital (não calculado aqui); "
                        + ("com aportes regulares, dá para reequilibrar comprando o que está abaixo do alvo, sem vender."
                           if contributions else "sem aportes, só vendendo o excesso a carteira chega ao alvo; o plano "
                           "'se vender o excesso' mostra onde colocar esse dinheiro. Manter como está também é uma opção.")
                        if sells else ""),
    }
    if buy_now is not None and not buy_now.viable:
        alerts.append(Alert("execucao", "WARNING", "com o capital informado nem toda posição-alvo compra um lote "
                            "inteiro: " + "; ".join(f"{k} — {v}" for k, v in buy_now.skipped.items())
                            + (". Essas posições entram pelos próximos aportes." if contributions else
                               ". Com este valor elas ficam fora da compra e a carteira comprada se afasta do alvo.")))
    if buy_now:
        plan_txt = (f"compra inicial de {_fmt(req.capital_inicial, ccy)} em {len(buy_now.rows)} posição(ões), sobra "
                    f"{_fmt(buy_now.cash_left, ccy)}")
    elif current:
        plan_txt = "sem capital novo: " + ("os aportes completam as posições abaixo do alvo" if contributions else
                                           "" if rebalance else
                                           "a carteira atual já está perto do alvo; nada a comprar ou vender")
    else:
        plan_txt = "sem compra inicial (sem capital)"
    if rebalance:
        plan_txt += (f"{'; ' if buy_now else ''}se vender o excesso ({_fmt(rebalance.amount, ccy)}), o dinheiro compra "
                     f"{len(rebalance.rows)} posição(ões) abaixo do alvo")
    steps.append({"etapa": 12, "titulo": "Carteira final, plano de compra e dupla justificativa",
                  "resultado": f"{int((w_f > 0).sum())} ativos com peso; teses financeira e de propósito separadas; "
                               + plan_txt + (f"; próximo aporte: {len(next_c.rows)} compra(s)" if next_c else "")})

    # 13 --------------------------------------------------------------- changes + real diversification
    conc = analyse(w_f[w_f > 0], res.cov.cov, res.returns, cat)
    for w_ in conc.warnings:
        alerts.append(Alert("concentracao", "WARNING", w_))
    changes: ChangeReport | None = None
    if req.carteira_anterior:
        changes = compare(req.carteira_anterior, w_f, why, "recomendação anterior")
        changes.stability = stability
    elif current:
        changes = compare(current, w_f, why, "carteira atual")
    if changes is not None:
        moved = [c for c in changes.changes if c.kind != "mantido"]
        changes.notes.append("nenhuma mudança material" if changes.same else
                             f"{len(moved)} mudança(s) material(is); giro de {changes.turnover:.0%} do dinheiro")
    params = _param_fingerprint(req)
    reproducibility = {"impressao_digital_carteira": fingerprint(w_f), "parametros": params,
                       "dados": ds.content_hash()[:12], "versao": APP_VERSION, "semente": req.seed,
                       "nota": "mesmos dados + mesmos parâmetros + mesma versão = mesma carteira (cálculo determinístico)"}
    steps.append({"etapa": 13, "titulo": "Mudanças e diversificação real",
                  "resultado": (f"{conc.n_positions} posições ≈ {conc.enb:.1f} apostas independentes; "
                                f"N efetivo pelos pesos {conc.effective_n:.1f}; "
                                + ("; ".join(changes.notes) + (f"; regra de estabilidade: {stability['explicacao']}"
                                                               if stability else "") if changes else
                                   "sem carteira anterior para comparar")
                                + f"; impressão digital {reproducibility['impressao_digital_carteira']}")})

    # simulations ---------------------------------------------------- scenarios + P(goal)
    # only assets held by the quant or the final portfolio are simulated (zero weights add cost, not risk)
    cols = [c for c in res.returns.columns if w_f.get(c, 0.0) > 0 or comb.quant.get(c, 0.0) > 1e-10]

    def _vec(w: pd.Series) -> np.ndarray:
        v = np.clip(w.reindex(cols).fillna(0.0).to_numpy(), 0.0, None)
        return v / v.sum()

    wf_np, wq_np = _vec(w_f), _vec(comb.quant)
    say("simulando o patrimônio até o objetivo (Monte Carlo)…")
    gbm = _gbm_estimator(res, cols)
    mc_f = _simulate(req, res, gbm, wf_np, req.n_simulacoes, req.seed)
    mc_q = _simulate(req, res, gbm, wq_np, req.n_simulacoes, req.seed)     # common random numbers
    tw = mc_f.terminal_wealth
    tr = mc_f.terminal_real_wealth
    scen = []
    for name, q in (("otimista (P90)", 90), ("base (P50)", 50), ("pessimista (P10)", 10)):
        v = float(np.percentile(tw, q))
        scen.append({"cenario": name, "patrimonio_final": v, "patrimonio_final_real": float(np.percentile(tr, q)),
                     "cagr_equivalente": _equiv_cagr(req, v), "atinge_objetivo": v >= req.valor_alvo})
    scenarios = pd.DataFrame(scen).set_index("cenario")
    sm = mc_f.summary
    goal = {"valor_alvo": req.valor_alvo, "p_objetivo": sm["p_goal"], "erro_padrao": sm["p_goal_mc_se"],
            "ic95": _wilson(sm["p_goal"], sm["n_paths"]),
            "n_trajetorias": sm["n_paths"], "modelo": "GBM multivariado em passos mensais, centrado no retorno esperado [E] do "
                                                      "otimizador; rebalanceamento mensal com custos, aportes no fim "
                                                      "de cada período",
            "p_objetivo_carteira_quant": mc_q.summary["p_goal"], "erro_padrao_quant": mc_q.summary["p_goal_mc_se"],
            "p_abaixo_do_aportado": sm["p_below_net_invested"], "total_aportado": sm["net_invested"],
            "inflacao_hipotetica": res.config.inflation_annual}
    # model / estimation-error sensitivity
    P = res.config.trading_days
    n_sens = min(req.n_simulacoes, 1000 if fast else 2000)
    vol_p = float(np.sqrt(wf_np @ (gbm.cov) @ wf_np))
    se_p = vol_p * np.sqrt(P / len(res.returns))
    hist_gbm = GBMModel.from_log_returns(res.log_returns[cols], P)
    sens_models = {
        "GBM, retorno [E] (principal)": gbm,
        "GBM, retorno [E] − 1 erro-padrão": _gbm_estimator(res, cols, -se_p),
        "GBM, retorno [E] + 1 erro-padrão": _gbm_estimator(res, cols, +se_p),
        "bootstrap de retornos mensais históricos, centrado em [E]": _monthly_bootstrap(res, cols),
        "GBM, deriva histórica": GBMModel(hist_gbm.mu, hist_gbm.cov, 1.0 / SIM_STEPS_PER_YEAR),
    }
    srows = []
    for name, mdl in sens_models.items():
        r_ = _simulate(req, res, mdl, wf_np, n_sens, req.seed + 1)
        s_ = r_.summary
        srows.append({"modelo": name, "p_objetivo": s_["p_goal"], "erro_padrao": s_["p_goal_mc_se"],
                      "mediana_final": s_["terminal_wealth_percentiles"]["p50"],
                      "cagr_equivalente_mediano": _equiv_cagr(req, s_["terminal_wealth_percentiles"]["p50"])})
    sensitivity = pd.DataFrame(srows).set_index("modelo")

    med = float(np.percentile(tw, 50))
    exp_cagr = _equiv_cagr(req, med)
    expected = {"cagr_necessario": reqc.rate, "cagr_esperado_mediano_simulado": exp_cagr,
                "distancia_fronteira_quant_E": gap_q, "distancia_fronteira_final_E": gap_f,
                "retorno_esperado_aritmetico_E": comb.ret_final,
                "retorno_esperado_efetivo_E": float((1 + comb.ret_final / P) ** P - 1),
                "volatilidade_E": comb.vol_final,
                "erro_padrao_retorno_carteira": se_p,
                "gap": (exp_cagr - reqc.rate) if (exp_cagr is not None and reqc.rate is not None) else None}

    # alerts ------------------------------------------------------------------
    sp = res.sections.get("estimator_selection", {})
    if res.backtest is not None:
        for b in res.backtest.bias_warnings:
            alerts.append(Alert("backtest", "WARNING", b))
    alerts.append(Alert("backtest", "WARNING",
                        "viés de sobrevivência e de seleção: o universo é escolhido com dados até hoje (listagens "
                        "atuais e triagem com o histórico completo); empresas deslistadas ou que fracassaram não "
                        "aparecem, e o backtest sobre esse universo tende a ser otimista."))
    if isinstance(sel, dict) and sel.get("warning"):
        alerts.append(Alert("backtest", "WARNING", str(sel["warning"])))
    hhi = float((wv ** 2).sum())
    held_tbl = table[table["peso"] > 0]
    by_ccy = held_tbl["peso"].groupby(held_tbl.index.map(lambda t: traits(cat[t]).exposure)).sum()
    by_cls = held_tbl.groupby("classe")["peso"].sum()
    alerts.append(Alert("concentracao", "WARNING" if wv.max() > 0.25 else "INFO",
                        f"maior peso {wv.max():.1%} ({w_f.idxmax()}); N efetivo {1 / hhi:.1f}; por moeda de exposição: "
                        + ", ".join(f"{k} {v:.0%}" for k, v in by_ccy.items()) + "; por classe: "
                        + ", ".join(f"{k} {v:.0%}" for k, v in by_cls.items())))
    lo = sensitivity.iloc[1]["p_objetivo"]
    hi = sensitivity.iloc[2]["p_objetivo"]
    sensitive = hi - lo > 0.15
    alerts.append(Alert("sensibilidade", "WARNING" if sensitive else "INFO",
                        f"P(objetivo) varia de {lo:.1%} a {hi:.1%} quando o retorno esperado da carteira muda ±1 "
                        f"erro-padrão (±{se_p:.2%} a.a.)" + (": resultado SENSÍVEL a erro de estimativa." if sensitive
                                                             else "; o erro-padrão do retorno esperado em si é grande, "
                                                                  "o que limita a precisão de qualquer estimativa.")))
    for w_ in res.warnings:
        if "Nenhum retorno esperado" in w_ or "INVIÁVEIS" in w_:
            alerts.append(Alert("sensibilidade", "WARNING", w_))
        elif w_.startswith(("Teto de volatilidade", "Nenhuma estratégia")):
            alerts.append(Alert("validacao", "WARNING", w_))
    for c in res.profile.conflicts:
        alerts.append(Alert("perfil", "INFO", c))
    if reqc.rate is not None and exp_cagr is not None and exp_cagr < reqc.rate:
        alerts.append(Alert("objetivo", "WARNING",
                            f"no cenário base da simulação o CAGR equivalente ({exp_cagr:.2%}) fica abaixo do necessário "
                            f"({reqc.rate:.2%}); considerar prazo maior, aportes maiores ou objetivo menor."))
    if res.validation is not None:
        alerts.append(Alert("validacao", "INFO" if res.validation.validated else "FAIL",
                            f"validador independente da etapa quantitativa: {res.validation.overall.value}"))
    if comb.alignment_quant is not None and comb.alignment_final is not None:
        alerts.append(Alert("alinhamento", "INFO",
                            f"alinhamento médio ponderado {comb.alignment_quant:.2f} (quant) -> {comb.alignment_final:.2f} "
                            f"(final); P(objetivo) {goal['p_objetivo_carteira_quant']:.1%} (quant) -> "
                            f"{goal['p_objetivo']:.1%} (final), mesmas trajetórias aleatórias."))
    if sp and isinstance(sp, dict) and sp.get("cov_used"):
        alerts.append(Alert("estimacao", "INFO", f"covariância escolhida fora da amostra: {sp['cov_used']}"))

    sleeves_final = currency_sleeves(ds, w_f[w_f > 0], rf, rf_src, req.moeda_objetivo, P)
    sources = _sources(req, ds_all, funnel, rf_src, cash_note, last_quote, fetched)
    out = PurposeResult(req, steps, reqc, universe, rf, rf_src, nstats, gov, res, themes, al, comb, table,
                        theses, expected, scenarios, goal, sensitivity, sleeves_final, alerts, ds.is_synthetic,
                        allocation, strategy_text, policy.as_dict(), scr, execution, conc, changes, sources,
                        reproducibility)
    from core.purpose.report import render_markdown

    say("montando o relatório…")
    text = render_markdown(out)
    for quote in [req.motivo_investidor, *themes.why_clauses, *(e for t in themes.themes for e in t.evidence)]:
        if quote:
            text = text.replace(quote, "")      # the investor's own words are quoted, not generated
    hits = guarantee_phrases(text)
    if hits:
        out.alerts.append(Alert("linguagem", "FAIL", f"linguagem de garantia detectada no relatório: {hits}"))
    return out
