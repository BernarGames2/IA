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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import DEMO_RISK_FREE_FALLBACK, AppConfig
from core.market_data.base import DataUnavailableError
from core.market_data.cache import SeriesCache
from core.market_data.quality import QualitySettings
from core.market_data.risk_free import RiskFreeQuote, fetch_live_rates
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider
from core.optimization.constraints import verify_weights
from core.portfolio_engine import AnalysisResult, run_analysis
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
    SYNTHETIC_CATALOG,
    CatalogAsset,
    expand,
)
from core.purpose.governance import GovernanceResult, apply_governance
from core.purpose.language import guarantee_phrases
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
from core.universe.screen import Funnel, preselect
from models.investor import InvestorProfileInput, LossCapacity, Objective, RiskProfile
from models.market_data import PriceDataset
from models.portfolio import PortfolioConstraints

MIN_OBS = 252
WEIGHT_FLOOR = 0.005          # weights below 0.5% are reported as zero in the final table


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

    @field_validator("moeda_objetivo")
    @classmethod
    def _ccy(cls, v: str) -> str:
        return v.strip().upper()

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
        if self.capital_inicial == 0 and (self.aporte_periodico <= 0 or not m
                                          or int(np.floor(m * self.horizonte_anos + 1e-9)) < 1):
            raise ValueError("sem capital inicial, informe um aporte periódico (por exemplo, mensal) com pelo menos "
                             "um aporte dentro do horizonte")
        if self.offline and self.moeda_objetivo != "BRL":
            raise ValueError("o universo sintético offline é cotado em BRL: use moeda_objetivo BRL")
        if self.moeda is not None and self.moeda.upper() != self.moeda_objetivo:
            raise ValueError("capital/aportes em moeda diferente da moeda do objetivo não são convertidos "
                             "automaticamente: informe-os na moeda do objetivo")
        return self


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
        volumes=ds.volumes[cols] if ds.volumes is not None else None,
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


def _profile_input(req: PurposeRequest) -> InvestorProfileInput:
    rp = req.perfil_risco
    m = PERIODS_PER_YEAR[req.periodicidade]
    return InvestorProfileInput(
        investor_id="proposito",
        horizon_years=req.horizonte_anos,
        objective={RiskProfile.CONSERVADOR: Objective.CAPITAL_PRESERVATION,
                   RiskProfile.MODERADO: Objective.BALANCED_GROWTH,
                   RiskProfile.ARROJADO: Objective.LONG_TERM_GROWTH}[rp],
        risk_tolerance_score={RiskProfile.CONSERVADOR: 20.0, RiskProfile.MODERADO: 50.0,
                              RiskProfile.ARROJADO: 80.0}[rp],
        loss_capacity={RiskProfile.CONSERVADOR: LossCapacity.LOW, RiskProfile.MODERADO: LossCapacity.MEDIUM,
                       RiskProfile.ARROJADO: LossCapacity.HIGH}[rp],
        capital=req.capital_inicial,
        monthly_contribution=req.aporte_periodico * m / 12.0 if m else 0.0,
        base_currency=req.moeda_objetivo,
        is_synthetic_example=req.offline,
    )


def _default_provider(cache_dir: Path, progress: Callable[[str], None] | None = None,
                      budget_s: float | None = None) -> YahooChartProvider:
    """Real prices: Yahoo chart endpoint through the standard library, with a file cache,
    parallel downloads and an optional total time budget."""
    return YahooChartProvider(cache=SeriesCache(cache_dir, 12.0), progress=progress, time_budget_s=budget_s)


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
    r = required_cagr(req.capital_inicial, req.aporte_periodico, req.periodicidade, max(wealth, 1e-9),
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
                            initial_capital=req.capital_inicial, rebalance_every=1,
                            rebalance_cost_bps=cfg.transaction_cost_bps + cfg.spread_bps,
                            inflation_annual=cfg.inflation_annual, periods_per_year=S,
                            batch_size=cfg.simulation_batch_size, goal=req.valor_alvo, record_every=1)
    return simulate_portfolio(model, w, ss, cf)  # type: ignore[arg-type]


def _drop_dust(comb: CombinedPortfolio, pc: PortfolioConstraints, cov: pd.DataFrame, mu: pd.Series,
               al: AlignmentResult, te_limit: float | None, vol_cap: float) -> CombinedPortfolio:
    """Zero the weights below WEIGHT_FLOOR (solver dust) and renormalise. The cleaned portfolio is
    kept only if it still satisfies every constraint, the volatility cap and the tracking-error
    budget; all reported metrics are recomputed on the weights actually shown."""
    w = comb.final.copy()
    if not ((w > 0) & (w < WEIGHT_FLOOR)).any():
        return comb
    w[w < WEIGHT_FLOOR] = 0.0
    w = w / w.sum()
    c = cov.loc[w.index, w.index].to_numpy()
    wv, wq = w.to_numpy(), comb.quant.loc[w.index].to_numpy()
    te = float(np.sqrt(max((wv - wq) @ c @ (wv - wq), 0.0)))
    vol = float(np.sqrt(wv @ c @ wv))
    if verify_weights(wv, pc) or vol > vol_cap + 1e-9 or (te_limit is not None and te > te_limit + 1e-9):
        return comb
    sc = np.array([al.scores[t].score for t in w.index], float) if al.computed else None
    return dataclasses.replace(comb, final=w, tracking_error=te, vol_final=vol, ret_final=float(wv @ mu.loc[w.index]),
                               alignment_final=float(sc @ wv) if sc is not None else comb.alignment_final,
                               notes=[*comb.notes, (f"pesos abaixo de {WEIGHT_FLOOR:.1%} zerados e renormalizados "
                                                    "(limites, teto de volatilidade e tracking error reverificados)")])


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
    reqc = required_cagr(req.capital_inicial, req.aporte_periodico, req.periodicidade, req.valor_alvo,
                         req.horizonte_anos)
    steps.append({"etapa": 1, "titulo": "CAGR necessário", "resultado": reqc.explanation})
    if reqc.status == "inatingivel":
        alerts.append(Alert("objetivo", "FAIL", reqc.explanation))

    # 2 ---------------------------------------------------------------- eligible universe
    cache_root = cache_dir or Path(".cache") / "market_data"
    funnel: Funnel | None = None
    if req.offline:
        assets = expand(SYNTHETIC_CATALOG, req.classes)
    else:
        listings = None
        if req.universo == "completo":
            say("lendo as listagens completas da B3 e dos EUA…")
            listings = load_market_listings(cache_root, rf_fetch)
        pre = extract_themes(req.motivo_investidor)          # themes only steer the theme slots
        pre_tags = {tag for t in pre.themes if t.name in THEME_TO_TAGS for tag in THEME_TO_TAGS[t.name]}
        assets, funnel = preselect(expand(REAL_CATALOG, req.classes), listings, pre_tags, req.max_ativos,
                                   req.classes)
    live: dict[str, RiskFreeQuote] = {}
    live_errors: dict[str, str] = {}
    if not req.offline:
        say(f"baixando cotações reais de {len(assets)} ativos pré-selecionados e as taxas livres de risco…")
        provider = provider or _default_provider(cache_root, progress, download_budget_s)
        need = ({a.currency for a in assets} | {req.moeda_objetivo}) - set(req.taxa_livre_risco)
        live, live_errors = fetch_live_rates(need, provider, rf_fetch)  # type: ignore[arg-type]
    rf0, rf0_src = _resolve_rf(req, set(), live)    # objective-currency rf (never silently 0)
    cfg = AppConfig(offline=req.offline, base_currency=req.moeda_objetivo, cache_dir=cache_root,
                    risk_free_rate=rf0[req.moeda_objetivo],
                    risk_free_rate_source="demo_fallback" if rf0_src[req.moeda_objetivo].startswith("FALLBACK")
                    else "user",
                    run_heavy_analyses=False, run_backtest=True, n_simulations=1000, horizon_days=21,
                    profile_horizon_simulation=False, make_charts=False, n_resamples=0, evt_bootstrap=0,
                    seed=req.seed, rebalance_every=126 if fast else 21, backtest_light=fast,
                    output_dir=output_dir or Path("outputs") / "proposito")
    try:
        ds_all, truth = _load_universe(req, assets, cfg, provider)
    except DataUnavailableError as e:
        wanted = {a.ticker for a in assets}
        fails = {k: v for k, v in (getattr(provider, "failures", {}) or {}).items() if k in wanted}
        if fails:
            first = next(iter(fails.values()))
            raise DataUnavailableError(f"{len(fails)} de {len(assets)} tickers sem cotação; exemplo — {first}") from e
        raise
    excluded_data = dict(ds_all.quality.excluded_symbols)
    cat = {a.ticker: a for a in assets}
    providers = sorted({p.provider for p in ds_all.provenance.values()})
    last_quote = str(ds_all.prices.index[-1].date())
    fetched = max((p.ingested_at_utc for p in ds_all.provenance.values() if p.ingested_at_utc), default="")
    universe = {"origem": "SINTÉTICA (demonstração offline)" if ds_all.is_synthetic
                else "real — Yahoo Finance (" + ", ".join(providers) + ")",
                "ultima_cotacao": last_quote, "obtido_em_utc": fetched,
                "candidatos": [a.ticker for a in assets], "validados": ds_all.symbols,
                "excluidos_por_dados": excluded_data, "min_observacoes": MIN_OBS,
                "n_datas": len(ds_all.prices), "periodo": f"{ds_all.prices.index[0].date()} a "
                                                         f"{ds_all.prices.index[-1].date()}"}
    funnel_txt = ""
    if funnel is not None:
        universe["funil"] = funnel.as_dict()
        listed = (f"{funnel.listed_b3} na B3 e {funnel.listed_us} nos EUA listados → " if funnel.used_listings
                  else "catálogo curado (listagens completas indisponíveis) → ")
        funnel_txt = (f"{listed}{funnel.eligible} elegíveis por liquidez/tamanho → {funnel.selected} "
                      f"pré-selecionados → ")
    steps.append({"etapa": 2, "titulo": "Universo elegível",
                  "resultado": f"{funnel_txt}{len(ds_all.symbols)} de {len(assets)} tickers com série validada "
                               f"(>= {MIN_OBS} observações); excluídos: {excluded_data or 'nenhum'}"})
    if funnel is not None:
        if funnel.used_listings:
            alerts.append(Alert("dados", "INFO",
                                f"Universo: todas as ações listadas lidas ({funnel.listed_b3} na B3, {funnel.listed_us} "
                                f"nos EUA; fontes: {', '.join(funnel.sources.values())}) → {funnel.eligible} elegíveis "
                                f"→ {funnel.selected} levados à otimização. Critérios: {funnel.criteria}."))
        else:
            why = "; ".join(f"{k}: {v}" for k, v in funnel.errors.items()) or "modo curado escolhido"
            alerts.append(Alert("dados", "WARNING" if funnel.errors else "INFO",
                                f"Listagens completas da B3/EUA não usadas ({why}): universo = catálogo curado de "
                                f"{len(REAL_CATALOG)} ativos → {funnel.selected} levados à otimização. "
                                f"Critérios: {funnel.criteria}."))
    if ds_all.is_synthetic:
        alerts.append(Alert("dados", "FAIL",
                            "DADOS SINTÉTICOS: os tickers SINT_* não existem no mercado. Este resultado é uma "
                            "DEMONSTRAÇÃO do método e NÃO é uma carteira final; para tickers reais rode no modo "
                            "online com acesso ao provedor."))
    else:
        alerts.append(Alert("dados", "INFO",
                            f"DADOS REAIS: {len(ds_all.symbols)} tickers do Yahoo Finance (preço ajustado por "
                            f"desdobramentos e dividendos), {universe['periodo']}; última cotação {last_quote}; "
                            f"obtidos em {fetched[:16].replace('T', ' ')} UTC. Cotações podem ter atraso e o provedor "
                            "pode revisar séries; as métricas são históricas [H] e estimativas [E], não previsões."))
    for s, why in excluded_data.items():
        alerts.append(Alert("dados", "WARNING", f"{s} excluído (sem proxy): {why}"))

    # 3 ---------------------------------------------------------------- currency sleeves
    ccys = {ds_all.native_currency.get(s, req.moeda_objetivo) for s in ds_all.symbols}
    rf, rf_src = _resolve_rf(req, ccys, live)
    for c, err in sorted(live_errors.items()):
        if c in ccys or c == req.moeda_objetivo:
            alerts.append(Alert("rf", "WARNING", f"rf de {c}: fonte automática indisponível ({err})"))
    nstats = native_asset_stats(ds_all, rf, req.moeda_objetivo, cfg.trading_days)
    steps.append({"etapa": 3, "titulo": "Sleeves por moeda",
                  "resultado": "; ".join(f"{c}: {int((nstats['moeda'] == c).sum())} ativos, rf {rf[c]:.2%} "
                                         f"({rf_src[c]})" for c in sorted(ccys))})
    for c in sorted(set(rf) | ccys):
        if rf_src[c].startswith("FALLBACK"):
            alerts.append(Alert("rf", "WARNING", f"rf de {c} = {rf[c]:.2%} é fallback de demonstração, não cotação; "
                                                 "informe a taxa real de cada moeda."))

    # 5 (applied as hard exclusion before 4, see module docstring) ---- governance
    gov = apply_governance(ds_all.symbols, req.incluir_sob_revisao, governance_path, req.restricoes_governanca)
    ds = subset_dataset(ds_all, gov.kept)
    if len(ds.symbols) < 2:
        raise ValueError("menos de dois ativos elegíveis após dados e governança")

    # 4 ---------------------------------------------------------------- quant optimisation
    say(f"otimizando e validando no backtest walk-forward ({len(ds.symbols)} ativos)…")
    res = run_analysis(cfg, _profile_input(req), dataset=(ds, truth))
    if fast:
        alerts.append(Alert("backtest", "INFO", "modo rápido do site: backtest walk-forward com rebalanceamento "
                                                "semestral, 6 estratégias e menos pontos de partida no otimizador "
                                                "(a linha de comando usa rebalanceamento mensal e 8 estratégias)"))
    w_q = res.selected.weights
    sel = res.selection
    how = (f"escolhida por {sel.get('rule', 'regra 1-EP')} no walk-forward fora da amostra" if res.backtest is not None
           else f"backtest NÃO executado ({sel.get('rule', 'sem seleção')})")
    steps.append({"etapa": 4, "titulo": "Otimização quantitativa",
                  "resultado": f"estratégia '{res.selected_name}' {how}; perfil final "
                               f"{res.profile.final_profile.value}; [E] retorno {res.selected.expected_return:.2%}, "
                               f"vol {res.selected.volatility:.2%}"})
    if not res.feasibility.get("feasible", True):
        alerts.append(Alert("validacao", "FAIL", "restrições do perfil INVIÁVEIS para o universo elegível ("
                            + "; ".join(map(str, res.feasibility.get("conflicts", []))) + "): a carteira não "
                            "respeita o perfil; amplie as classes do universo ou revise o perfil."))
    steps.append({"etapa": 5, "titulo": "Filtro de governança",
                  "resultado": ("removidos: " + "; ".join(f"{k} — {v}" for k, v in gov.removed.items())
                                if gov.removed else "nenhum ativo removido")
                               + ("; incluídos sob revisão por opção explícita: "
                                  + ", ".join(gov.included_under_review) if gov.included_under_review else "")})
    for s, why in gov.removed.items():
        alerts.append(Alert("governanca", "INFO", f"{s}: {why}"))
    for s, why in gov.included_under_review.items():
        alerts.append(Alert("governanca", "WARNING", f"{s}: {why}"))
    for t, why in gov.rejected_entries.items():
        alerts.append(Alert("governanca", "WARNING", f"flag de governança de {t} ignorada: {why}"))

    # 6-7 -------------------------------------------------------------- themes + alignment
    say("lendo o seu motivo e pontuando o alinhamento…")
    themes = extract_themes(req.motivo_investidor)
    elig_assets = [cat[s] for s in ds.symbols]
    al = score_assets(themes, elig_assets)
    steps.append({"etapa": 6, "titulo": "Temas do motivo",
                  "resultado": ("motivo vazio: score de alinhamento não calculado por falta de input"
                                if themes.empty else
                                f"temas: {[THEME_LABEL.get(t.name, t.name) for t in themes.themes] or 'nenhum'}; "
                                f"confiança {themes.confidence}" + (f"; porquê: \"{themes.why}\"" if themes.why else ""))})
    steps.append({"etapa": 7, "titulo": "Score de alinhamento",
                  "resultado": "calculado para todos os ativos elegíveis" if al.computed else "não calculado"})
    if themes.empty:
        alerts.append(Alert("alinhamento", "WARNING", "motivo_investidor vazio: score de alinhamento não calculado por "
                                                      "falta de input; carteira apenas quantitativa."))
    elif themes.confidence == "baixa":
        alerts.append(Alert("alinhamento", "WARNING", "confiança BAIXA na extração de temas: " + "; ".join(al.notes)))
    for n in al.notes:
        if "sem correspondência" in n:
            alerts.append(Alert("alinhamento", "INFO", n))

    # 8 ---------------------------------------------------------------- combination
    vol_cap = res.profile.research_limits.target_volatility_band[1]
    comb = combine(w_q, al, res.mu, res.cov.cov, res.constraints, req.peso_alinhamento,
                   req.orcamento_tracking_error, vol_cap)
    if comb.violations:
        alerts.append(Alert("validacao", "FAIL", "carteira combinada violou limites (" + "; ".join(comb.violations)
                            + "): usados os pesos quantitativos"))
        comb = dataclasses.replace(comb, final=comb.quant, vol_final=comb.vol_quant, ret_final=comb.ret_quant,
                                   alignment_final=comb.alignment_quant, tracking_error=0.0)
    applied = comb.alignment_target is not None
    comb = _drop_dust(comb, res.constraints, res.cov.cov, res.mu, al,
                      req.peso_alinhamento * req.orcamento_tracking_error if applied else None,
                      max(comb.vol_quant, vol_cap))
    w_f = comb.final
    gap_q = _frontier_gap(res.frontier, comb.vol_quant, comb.ret_quant)
    gap_f = _frontier_gap(res.frontier, comb.vol_final, comb.ret_final)
    steps.append({"etapa": 8, "titulo": "Combinação quant + alinhamento",
                  "resultado": f"{1 - req.peso_alinhamento:.0%} quant / {req.peso_alinhamento:.0%} alinhamento; "
                               f"tracking error vs quant {comb.tracking_error:.2%} (orçamento "
                               f"{req.peso_alinhamento * req.orcamento_tracking_error:.2%}); vol "
                               f"{comb.vol_quant:.2%} -> {comb.vol_final:.2%}; distância à fronteira eficiente "
                               f"(retorno [E] abaixo da fronteira na mesma volatilidade): {_pct(gap_q)} (quant) -> "
                               f"{_pct(gap_f)} (final)"
                               + ("; " + " ".join(comb.notes) if comb.notes else "")})
    leaked = [s for s in gov.removed if s in w_f.index and w_f[s] > 0]
    if leaked:   # defensive: cannot happen because the assets are not in the optimised universe
        raise RuntimeError(f"ativos removidos por governança com peso positivo: {leaked}")

    # 9 ---------------------------------------------------------------- table + dual theses
    cov = res.cov.cov.loc[w_f.index, w_f.index].to_numpy()
    wv = w_f.to_numpy()
    tot = float(wv @ cov @ wv)
    rc = pd.Series(wv * (cov @ wv) / tot if tot > 0 else np.zeros(len(wv)), index=w_f.index)
    rows, theses = [], []
    for s in w_f.sort_values(ascending=False).index:
        a = cat[s]
        sc = al.scores[s].score if al.computed else None
        rows.append({"ticker": s, "classe": a.macro_class, "peso": float(w_f[s]),
                     "score_alinhamento": sc, "moeda": ds.native_currency.get(s, req.moeda_objetivo),
                     "peso_quant": float(comb.quant[s]), "motivo_score": al.scores[s].reason})
        if w_f[s] > 0:
            theses.append({"ticker": s, "nome": a.name,
                           "tese_financeira": _financial_thesis(a, s, res, w_f, rc, ds.is_synthetic),
                           "tese_proposito": _purpose_thesis(a, s, al, themes)})
    table = pd.DataFrame(rows).set_index("ticker").sort_values("peso", ascending=False)
    # where to invest and why: amount per position, role in the portfolio, link to the motive
    m_per = PERIODS_PER_YEAR[req.periodicidade]
    mu_f = res.mu.loc[w_f.index]
    port_mu = float(w_f @ mu_f)
    allocation = []
    for s in table.index[table["peso"] > 0]:
        w_s = float(w_f[s])
        ret_share = float(w_s * mu_f[s] / port_mu) if port_mu > 0 else None
        role = _role(w_s, float(rc[s]), ret_share)
        if al.computed and al.scores[s].matched:
            purpose = "atende ao(s) tema(s) do seu motivo: " + ", ".join(al.scores[s].matched)
        elif al.computed:
            purpose = "sem ligação com os temas do seu motivo (está pela função financeira)"
        else:
            purpose = "alinhamento não calculado (motivo vazio)" if themes.empty else "alinhamento não calculado"
        allocation.append({"ticker": s, "nome": cat[s].name, "classe": cat[s].macro_class,
                           "moeda": table.loc[s, "moeda"], "peso": w_s,
                           "valor_inicial": req.capital_inicial * w_s,
                           "aporte_por_periodo": req.aporte_periodico * w_s if m_per else 0.0,
                           "papel": role, "proposito": purpose, "por_que": f"{role}; {purpose}",
                           "negocio": cat[s].business.replace("[sintético] ", "")})
    strat = STRATEGY_PT.get(res.selected_name, res.selected_name)
    strategy_text = (f"Carteira-base quantitativa: {strat}, escolhida "
                     + ("pela validação fora da amostra (walk-forward, regra de 1 erro-padrão a favor da mais "
                        "simples) entre as estratégias que respeitam o perfil " if res.backtest is not None
                        else "sem backtest (restrições inviáveis) para o perfil ")
                     + f"{res.profile.final_profile.value}. Depois, {req.peso_alinhamento:.0%} da alocação foi "
                     "inclinada para os ativos ligados ao seu motivo, sem sair dos limites de risco, governança e "
                     "tracking error." if al.computed and req.peso_alinhamento > 0 else
                     f"Carteira-base quantitativa: {strat}, escolhida "
                     + ("pela validação fora da amostra (walk-forward, regra de 1 erro-padrão) "
                        if res.backtest is not None else "sem backtest (restrições inviáveis) ")
                     + f"para o perfil {res.profile.final_profile.value}; sem ajuste de propósito.")
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
    steps.append({"etapa": 9, "titulo": "Carteira final com dupla justificativa",
                  "resultado": f"{int((w_f > 0).sum())} ativos com peso; teses financeira e de propósito separadas"})

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
                        "viés de sobrevivência/seleção: o catálogo é uma lista estática escolhida hoje; ativos "
                        "deslistados ou que fracassaram não estão no histórico."))
    if isinstance(sel, dict) and sel.get("warning"):
        alerts.append(Alert("backtest", "WARNING", str(sel["warning"])))
    hhi = float((wv ** 2).sum())
    held_tbl = table[table["peso"] > 0]
    by_ccy = held_tbl.groupby("moeda")["peso"].sum()
    by_cls = held_tbl.groupby("classe")["peso"].sum()
    alerts.append(Alert("concentracao", "WARNING" if wv.max() > 0.25 else "INFO",
                        f"maior peso {wv.max():.1%} ({w_f.idxmax()}); N efetivo {1 / hhi:.1f}; por moeda: "
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
    out = PurposeResult(req, steps, reqc, universe, rf, rf_src, nstats, gov, res, themes, al, comb, table,
                        theses, expected, scenarios, goal, sensitivity, sleeves_final, alerts, ds.is_synthetic,
                        allocation, strategy_text)
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
