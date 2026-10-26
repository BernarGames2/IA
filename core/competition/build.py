"""From the client case and the competition rulebook to orders the team can type into the competition's simulator.

The portfolio comes from the purpose engine (``core/purpose/engine.py``) — the same discovery, screening,
walk-forward-validated optimisation and alignment behind "Montar carteira" — with the client's profile as the
investor's and the rulebook's limits as hard constraints (minimum share price, approved ETFs, per-asset cap, how many
positions, ETFs and sectors). This module adds what a competition needs on top of it:

* the client's goal at the competition's scale: with no fixed costs, multiplying the client's money, contributions
  and goal by the same factor leaves the chance of reaching the goal unchanged, so the goal is reproduced on the
  competition's capital (or set from a required return, or as keeping purchasing power — always said);
* the order list (whole shares unless the rules allow fractions), trimmed share by share so no position starts above
  the per-asset cap at the purchase prices;
* a rule-by-rule check measured on those orders (never assumed), the client-fit map, the distribution of results
  until the end of the trading window, and an Investment Policy Statement draft (``core/competition/ips.py``).

Exclusions the client asks for (fossil fuels, tobacco, weapons, alcohol, gambling) use the catalogue's sectors, a
documented editorial list of companies whose main business is the excluded activity, and patterns on the industry
and the company name of the listed stocks. Nothing is executed and no data is invented: missing data is reported.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.stats import norm

from core.competition.ips import (
    EXCLUSION_EN,
    EXCLUSION_PT,
    PROFILE_EN,
    THEME_PT,
    english_promises,
    ips_english,
    money_pt,
    summary_pt,
)
from core.market_data.yahoo_chart import FetchFn
from core.purpose.catalog import (
    SECTOR_LABEL,
    CatalogAsset,
    business_sector,
    canonical_sector,
    is_etf,
)
from core.purpose.engine import (
    STRATEGY_PT,
    AmountTooSmall,
    PurposeRequest,
    PurposeResult,
    norm_ticker,
    run_purpose,
)
from core.purpose.feasibility import plan
from core.purpose.language import guarantee_phrases
from core.purpose.personalization import CASH_TICKERS, traits
from core.purpose.themes import _fold
from core.universe.listings import load_market_listings
from utils.br_format import num, pct

ExclusionKey = Literal["combustiveis_fosseis", "tabaco", "armas", "alcool", "jogos"]
ThemeKey = Literal["sustentabilidade", "saude", "tecnologia", "educacao", "comunidade", "familia_legado"]

# companies whose main business is the excluded activity (editorial, documented; review periodically)
EXCLUSION_TICKERS: dict[str, tuple[str, ...]] = {
    "tabaco": ("MO", "PM", "BTI", "VGR", "TPB", "UVV", "IMBBY", "JAPAY"),
    "armas": ("LMT", "NOC", "GD", "RTX", "LHX", "HII", "BA", "KTOS", "AVAV", "RGR", "SWBI", "AOUT"),
    "alcool": ("BUD", "TAP", "STZ", "DEO", "BF-B", "BF-A", "SAM", "MGPI", "ABEV", "ABEV3.SA"),
    "jogos": ("LVS", "WYNN", "MGM", "CZR", "DKNG", "PENN", "CHDN", "BYD", "RSI", "FLUT", "LNW", "IGT", "MCRI"),
    "combustiveis_fosseis": ("BTU", "CNR", "ARCH", "CEIX", "AMR", "HCC", "NRP"),     # coal (oil and gas: by sector)
}
# patterns on the Nasdaq industry + company name of listed stocks outside the catalogue
EXCLUSION_PATTERNS: dict[str, str] = {
    "tabaco": r"tobacco|cigar",
    "armas": r"military|ordnance|small arms|firearm|ammunition|defen[cs]e",
    "alcool": r"brew|beer|winer|wines\b|distill|liquor|spirits|vineyard",
    "jogos": r"casino|gaming|gambling|lottery|betting|sportsbook",
    "combustiveis_fosseis": r"\bcoal\b",
}
EXCLUSION_SECTORS = {"combustiveis_fosseis": ("petroleo_gas",)}
THEME_WORDS = {"sustentabilidade": "sustentabilidade", "saude": "saúde", "tecnologia": "tecnologia",
               "educacao": "educação", "comunidade": "comunidade", "familia_legado": "legado"}
ROLE_EN = {"estabilizador": "stabiliser", "motor de retorno": "return engine",
           "crescimento (assume mais risco do que o seu peso)": "growth (more risk than its weight)",
           "diversificador": "diversifier"}
STRATEGY_EN = {
    "max_sharpe": "maximum Sharpe ratio", "min_variance": "minimum variance", "risk_parity": "risk parity",
    "max_diversification": "maximum diversification", "hrp": "hierarchical risk parity",
    "min_cvar": "minimum CVaR (smallest average loss on the worst days)",
    "robust_mean_variance": "mean-variance robust to estimation error", "target_volatility": "the profile's target volatility",
    "equal_weight": "equal weights within the profile limits",
    "utility": "personal utility (expected return minus a risk penalty calibrated to the client's profile, objective "
               "and horizon, within the volatility ceiling)"}
DEFAULT_WINDOW_DAYS = 63          # three months when the rules give no end of trading
PCT_TOL = 1e-9


# ----------------------------------------------------------------------------- inputs
class ClientProfile(BaseModel):
    """The client as the case describes it (read from the text and confirmed or typed by the team)."""

    model_config = ConfigDict(extra="forbid")

    nome: str = Field("", max_length=80)
    idade: int | None = Field(None, ge=14, le=110)
    horizonte_anos: float = Field(..., gt=0, le=50)
    perfil_risco: Literal["conservador", "moderado", "arrojado"] = "moderado"
    objetivo_tipo: Literal["crescimento", "renda", "preservacao"] | None = None
    moeda_caso: Literal["USD", "BRL", "EUR"] = "USD"
    patrimonio_cliente: float | None = Field(None, gt=0, le=1e12)
    valor_alvo_cliente: float | None = Field(None, gt=0, le=1e13)
    aporte_anual_cliente: float | None = Field(None, ge=0, le=1e11)
    retirada_anual_cliente: float | None = Field(None, ge=0, le=1e11)
    retorno_exigido: float | None = Field(None, gt=-0.5, lt=1.0)
    tolerancia_perda: float | None = Field(None, gt=0, lt=1)
    necessidade_liquidez: float | None = Field(None, ge=0, le=1)
    exclusoes: list[ExclusionKey] = Field(default_factory=list, max_length=5)
    setores_excluidos: list[str] = Field(default_factory=list, max_length=20)
    ativos_excluidos: list[str] = Field(default_factory=list, max_length=100)
    temas: list[ThemeKey] = Field(default_factory=list, max_length=6)
    metas_intermediarias: list[str] = Field(default_factory=list, max_length=10)
    observacoes: str = Field("", max_length=1500)
    citacoes: dict[str, str] = Field(default_factory=dict, max_length=30)
    inflacao_hipotetica: float = Field(0.03, ge=0, le=0.5)

    @field_validator("citacoes")
    @classmethod
    def _quotes(cls, v: dict[str, str]) -> dict[str, str]:
        return {k[:40]: s[:700] for k, s in v.items()}

    @field_validator("metas_intermediarias")
    @classmethod
    def _goals(cls, v: list[str]) -> list[str]:
        return [s[:300] for s in v if s.strip()]


class CompetitionRules(BaseModel):
    """The rulebook. Every field is optional except the capital: a rule that is not given is not applied."""

    model_config = ConfigDict(extra="forbid")

    nome: str = Field("", max_length=120)
    capital: float = Field(..., gt=0, le=1e10)
    moeda: Literal["USD", "BRL"] = "USD"
    mercados: list[Literal["US", "BR"]] = Field(default_factory=lambda: ["US"], min_length=1, max_length=2)
    preco_minimo: float | None = Field(None, gt=0, le=1e4)
    peso_maximo: float | None = Field(None, gt=0, le=1)
    peso_minimo: float = Field(0.01, ge=0, le=0.5, description="menor posição, em fração do capital")
    min_posicoes: int | None = Field(None, ge=1, le=100)
    max_posicoes: int | None = Field(None, ge=2, le=200)
    min_etfs: int = Field(0, ge=0, le=20)
    etfs_aprovados: list[str] = Field(default_factory=list, max_length=600)
    min_setores: int | None = Field(None, ge=1, le=15)
    caixa_maximo: float | None = Field(None, ge=0, le=1)
    max_operacoes: int | None = Field(None, ge=1, le=10_000)
    limite_volume: float | None = Field(None, gt=0, le=100)
    permitir_fracoes: bool = False
    fim_negociacao: date | None = None

    @field_validator("etfs_aprovados")
    @classmethod
    def _etfs(cls, v: list[str]) -> list[str]:
        return sorted({norm_ticker(t) for t in v if t.strip()})

    @model_validator(mode="after")
    def _coherent(self) -> CompetitionRules:
        """Rules that cannot all hold are refused before any data is fetched, with the arithmetic."""
        lo, hi, cap = self.min_posicoes, self.max_posicoes, self.peso_maximo
        if lo and hi and lo > hi:
            raise ValueError(f"o mínimo de posições ({lo}) passa do máximo ({hi})")
        if cap and hi and cap * hi < 1 - PCT_TOL:
            raise ValueError(f"com no máximo {hi} posições de até {pct(cap, 0)} cada, só {pct(cap * hi, 0)} do capital "
                             "pode ser investido: aumente o máximo de posições ou o peso máximo")
        if cap is not None and self.peso_minimo > cap + PCT_TOL:
            raise ValueError(f"a menor posição ({pct(self.peso_minimo)}) passa do peso máximo ({pct(cap)})")
        if lo and self.peso_minimo * lo > 1 + PCT_TOL:
            raise ValueError(f"{lo} posições de pelo menos {pct(self.peso_minimo)} passam de 100% do capital")
        for what, n in (("ETFs", self.min_etfs), ("setores", self.min_setores or 0)):
            if hi and n > hi:
                raise ValueError(f"a regra pede {n} {what}, mais do que o máximo de {hi} posições")
        if self.etfs_aprovados and self.min_etfs > len(self.etfs_aprovados):
            raise ValueError("a regra pede mais ETFs do que a lista aprovada tem")
        return self


class CompetitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cliente: ClientProfile
    regras: CompetitionRules
    peso_alinhamento: float = Field(0.30, ge=0.0, le=1.0)
    n_simulacoes: int = Field(3_000, ge=500, le=10_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    universo: Literal["completo", "curado"] = "completo"
    max_ativos: int = Field(40, ge=10, le=60)

    def key(self) -> str:
        return "competicao|" + self.model_dump_json()


# ----------------------------------------------------------------------------- helpers
def _exclusions(cli: ClientProfile, universo: str, cache_root: Path, fetch: FetchFn | None
                ) -> tuple[dict[str, str], list[str], list[str]]:
    """Tickers the client excludes (with the reason), sectors excluded, and notes."""
    out: dict[str, str] = {}
    notes: list[str] = []
    sectors = [canonical_sector(x) or x for x in cli.setores_excluidos]
    for k in cli.exclusoes:
        sectors.extend(EXCLUSION_SECTORS.get(k, ()))
        for t in EXCLUSION_TICKERS.get(k, ()):
            out.setdefault(norm_ticker(t), f"pedido do cliente: sem {EXCLUSION_PT[k]}")
    pats = {k: re.compile(EXCLUSION_PATTERNS[k]) for k in cli.exclusoes if k in EXCLUSION_PATTERNS}
    if pats and universo == "completo":
        listings = load_market_listings(cache_root, fetch)
        if listings.available:
            for la in listings.all():
                text = _fold(f"{la.industry} {la.name}")
                for k, rx in pats.items():
                    if rx.search(text):
                        out.setdefault(la.ticker, f"pedido do cliente: sem {EXCLUSION_PT[k]} (classificação: "
                                                  f"{la.industry or la.name})")
        else:
            notes.append("listagens das bolsas indisponíveis agora: as exclusões do cliente valem para o catálogo e "
                         "para a lista de empresas do site; confira as ações que aparecerem")
    for t in cli.ativos_excluidos:
        if t.strip():
            out.setdefault(norm_ticker(t), "excluído pela equipe")
    return out, sorted(set(sectors)), notes


def _goal(cli: ClientProfile, capital: float) -> tuple[float, float, str, str]:
    """(goal on the competition's capital, yearly contribution on that scale, how — Portuguese, English)."""
    h = cli.horizonte_anos
    ccy = cli.moeda_caso
    if cli.valor_alvo_cliente and cli.patrimonio_cliente:
        k = capital / cli.patrimonio_cliente
        contrib = (cli.aporte_anual_cliente or 0.0) * k
        extra_pt = (f" e aportes de {money_pt(cli.aporte_anual_cliente, ccy)} por ano"
                    if cli.aporte_anual_cliente else "")
        extra_en = f" plus {cli.aporte_anual_cliente:,.0f} a year" if cli.aporte_anual_cliente else ""
        return (cli.valor_alvo_cliente * k, contrib,
                (f"meta do cliente ({money_pt(cli.valor_alvo_cliente, ccy)} em {num(h, 0)} anos, partindo de "
                f"{money_pt(cli.patrimonio_cliente, ccy)}{extra_pt}) na escala do capital da competição (×{num(k, 2)}); "
                "a chance de atingir não muda, porque tudo cresce na mesma proporção"),
                (f"the client's goal ({cli.valor_alvo_cliente:,.0f} in {h:g} years from {cli.patrimonio_cliente:,.0f}"
                f"{extra_en}, in {ccy}) scaled to the competition's capital (x{k:.2f}); scaling every amount by the "
                "same factor leaves the probability unchanged"))
    if cli.retorno_exigido is not None:
        r = cli.retorno_exigido
        return (capital * (1 + r) ** h, 0.0, f"retorno exigido de {pct(r)} ao ano por {num(h, 0)} anos",
                f"the required return of {r * 100:.1f}% a year for {h:g} years")
    infl = cli.inflacao_hipotetica
    why = ("; a meta citada não pôde ser usada sem saber quanto o cliente tem hoje"
           if cli.valor_alvo_cliente and not cli.patrimonio_cliente else "")
    return (capital * (1 + infl) ** h, 0.0,
            (f"sem meta em dinheiro utilizável: objetivo = manter o poder de compra (inflação hipotética de {pct(infl)} "
            f"ao ano por {num(h, 0)} anos){why}"),
            (f"no usable money goal in the case: keep purchasing power (assumed inflation of {infl * 100:.1f}% a year "
            f"for {h:g} years)"))


def _motive(cli: ClientProfile) -> str:
    """The client's values, in the words the alignment lexicon reads (the case quotes are shown apart)."""
    words = [THEME_WORDS[k] for k in cli.temas]
    if cli.objetivo_tipo == "renda":
        words.append("renda passiva")
    return ("Valores e interesses do cliente lidos no caso: " + "; ".join(words) + ".") if words else ""


def _asset(t: str, res: PurposeResult, alloc: dict[str, dict]) -> CatalogAsset:
    """The catalogue entry of a held asset, rebuilt from the dataset metadata and the engine's allocation row."""
    m = res.analysis.dataset.meta[t]
    a = alloc.get(t, {})
    return CatalogAsset(t, str(a.get("nome", t)), str(a.get("classe", m.market)), m.asset_class, m.sector, m.country,
                        m.currency, str(a.get("negocio", "")))


def _role(w: float, risk: float, ret: float | None) -> str:
    """Same reading as the engine's role of a position (core/purpose/engine.py), in English."""
    if risk < 0.5 * w:
        return "stabiliser"
    if ret is not None and ret > 1.2 * max(risk, w):
        return "return engine"
    if risk > 1.5 * w:
        return "growth (more risk than its weight)"
    return "diversifier"


def _class_label(a: CatalogAsset) -> tuple[str, str]:
    """(English, Portuguese) allocation bucket of an asset."""
    tr = traits(a)
    if tr.cash or a.ticker in CASH_TICKERS:
        return "Short-term Treasuries (cash-like)", "Tesouro americano curto (quase caixa)"
    if a.asset_class == "fixed_income":
        return "Bonds", "Renda fixa"
    if a.asset_class == "real_estate":
        return "Real estate (REITs)", "Imóveis (REITs/FIIs)"
    if a.asset_class == "commodity":
        return "Gold and commodities", "Ouro e commodities"
    if a.country == "BR" or a.ticker.endswith(".SA"):
        return "Brazilian equities", "Ações brasileiras"
    if a.country not in ("US",) or re.search(r"fora dos eua|emergentes|globais", a.business, re.IGNORECASE):
        return "International equities", "Ações internacionais"
    return "US equities", "Ações dos EUA"


def _orders(rows: list[dict], capital: float, cap: float | None, fractions: bool) -> tuple[list[dict], float, list[str]]:
    """Purchase rows with each position trimmed (share by share) to the per-asset cap at the purchase price."""
    out, notes = [], []
    cash = capital - sum(float(r["valor"]) for r in rows)
    for r in rows:
        q, p = float(r["quantidade"]), float(r["preco"])
        step = 1e-4 if fractions else 1.0
        if cap is not None and p > 0 and q * p > cap * capital + 1e-6:
            q_max = math.floor(cap * capital / p / step + 1e-9) * step
            cash += (q - q_max) * p
            notes.append(f"{r['ticker']}: {num(q - q_max, 0 if not fractions else 4)} ação(ões) a menos para não passar "
                         f"de {pct(cap, 0)} do capital")
            q = q_max
        if q > 0:
            out.append({**r, "quantidade": round(q, 4), "valor": round(q * p, 2)})
    return out, cash, notes


def _window(day: date, end: date | None) -> tuple[int, str, str, str]:
    """Trading days to the end of the competition (or a stated default), the end as text (Portuguese, English) and
    a note."""
    if end is not None and end > day:
        n = int(np.busday_count(day + timedelta(days=1), end + timedelta(days=1)))
        return max(n, 1), end.strftime("%d/%m/%Y"), end.isoformat(), ""
    note = ("a data de fim já passou: usada a janela padrão de 3 meses" if end is not None else
            "sem data de fim nas regras: usada a janela padrão de 3 meses (63 pregões)")
    return DEFAULT_WINDOW_DAYS, f"{DEFAULT_WINDOW_DAYS} pregões", f"{DEFAULT_WINDOW_DAYS} trading days", note


# ----------------------------------------------------------------------------- the build
def build(req: CompetitionRequest, *, provider: object | None = None, rf_fetch: FetchFn | None = None,
          cache_dir: Path | None = None, output_dir: Path | None = None,
          progress: Callable[[str], None] | None = None, budget_s: float | None = None,
          today: date | None = None) -> dict[str, object]:
    day = today or datetime.now(UTC).date()
    cli, rules = req.cliente, req.regras
    cache_root = cache_dir or Path(".cache") / "market_data"
    if progress:
        progress("lendo as regras e o perfil do cliente…")
    excluded, sectors_out, notes = _exclusions(cli, req.universo, cache_root, rf_fetch)
    target, contrib, goal_pt, goal_en = _goal(cli, rules.capital)
    classes = ["acoes_eua", "etf_eua", "ouro"] + (["acoes_br", "fii", "etf_br", "renda_fixa_br"]
                                                  if "BR" in rules.mercados else [])
    preq = PurposeRequest(
        capital_inicial=rules.capital, aporte_periodico=contrib, periodicidade="anual" if contrib > 0 else "nenhum",
        valor_alvo=target, moeda_objetivo=rules.moeda, horizonte_anos=cli.horizonte_anos,
        perfil_risco=cli.perfil_risco, objetivo_tipo=cli.objetivo_tipo, tolerancia_perda=cli.tolerancia_perda,
        necessidade_liquidez=cli.necessidade_liquidez, mercados=list(rules.mercados), permitir_cripto=False,
        ativos_excluidos=sorted(excluded), setores_excluidos=sectors_out, motivo_investidor=_motive(cli),
        peso_alinhamento=req.peso_alinhamento, n_simulacoes=req.n_simulacoes, seed=req.seed, universo=req.universo,
        max_ativos=min(60, max(req.max_ativos, (rules.min_posicoes or 0) + 10)), offline=False, classes=classes,
        fracionario_eua=rules.permitir_fracoes, valor_minimo_posicao=min(1e7, rules.capital * rules.peso_minimo),
        preco_minimo=rules.preco_minimo, peso_maximo_ativo=rules.peso_maximo, min_posicoes=rules.min_posicoes,
        max_posicoes=rules.max_posicoes, min_etfs=rules.min_etfs, etfs_permitidos=rules.etfs_aprovados or None,
        min_setores=rules.min_setores)
    try:
        res = run_purpose(preq, output_dir=output_dir, provider=provider, cache_dir=cache_root, rf_fetch=rf_fetch,
                          progress=progress, fast=True, download_budget_s=budget_s)
    except AmountTooSmall as e:
        raise ValueError(str(e)) from e
    return _payload(req, res, excluded, sectors_out, notes, goal_pt, goal_en, day)


def _payload(req: CompetitionRequest, res: PurposeResult, excluded: dict[str, str], sectors_out: list[str],
             notes: list[str], goal_pt: str, goal_en: str, day: date) -> dict[str, object]:
    cli, rules = req.cliente, req.regras
    cap_total, ccy = rules.capital, rules.moeda
    an = res.analysis
    ds = an.dataset
    alloc = {str(a["ticker"]): a for a in res.allocation}
    # the whole capital is bought now (the client's contributions count only for the goal): shares at the last close
    w_final = res.combined.final[res.combined.final > 0]
    cat = {t: _asset(t, res, alloc) for t in w_final.index}
    bought = plan(w_final, ds.prices.iloc[-1], cat, cap_total, rules.permitir_fracoes, ccy)
    orders, cash, trim_notes = _orders([r for r in bought.rows if float(r["quantidade"]) > 0], cap_total,
                                       rules.peso_maximo, rules.permitir_fracoes)
    trim_notes = [*(f"{t}: {why}" for t, why in bought.skipped.items()), *trim_notes]
    held = [str(o["ticker"]) for o in orders]
    assets = {t: cat[t] for t in held}
    invested = sum(float(o["valor"]) for o in orders)
    w = pd.Series({str(o["ticker"]): float(o["valor"]) / cap_total for o in orders}, dtype=float)

    # risk numbers of the purchased weights (uninvested cash: no return, no risk)
    cov = an.cov.cov.loc[held, held].to_numpy()
    mu = an.mu.loc[held].to_numpy()
    wv = w.loc[held].to_numpy()
    var_p = float(wv @ cov @ wv)
    vol = math.sqrt(max(var_p, 0.0))
    ret = float(wv @ mu)
    rc = wv * (cov @ wv) / var_p if var_p > 0 else np.zeros(len(wv))
    ret_share = wv * mu / ret if ret > 0 else np.full(len(wv), np.nan)
    daily = an.returns[held].fillna(0.0).to_numpy() @ wv
    growth = np.cumprod(1.0 + daily)
    dd = float((growth / np.maximum.accumulate(growth) - 1.0).min()) if len(growth) else None
    n_days, end_txt, end_en, window_note = _window(day, rules.fim_negociacao)
    worst = worst_dates = worst_dates_en = None
    if len(daily) > n_days:
        logs = np.log1p(daily)
        roll = pd.Series(logs, index=an.returns.index).rolling(n_days).sum().dropna()
        if len(roll):
            worst = float(np.expm1(roll.min()))
            i_end = roll.idxmin()
            i_start = an.returns.index[an.returns.index.get_loc(i_end) - n_days + 1]
            worst_dates = f"{i_start:%d/%m/%Y} a {i_end:%d/%m/%Y}"
            worst_dates_en = f"{i_start:%Y-%m-%d} to {i_end:%Y-%m-%d}"
    t_years = n_days / 252.0
    m_log = (ret - 0.5 * var_p) * t_years
    s_log = vol * math.sqrt(t_years)
    start_value = invested
    q = {k: cash + start_value * math.exp(m_log + s_log * norm.ppf(p)) for k, p in (("p10", 0.1), ("p50", 0.5),
                                                                                       ("p90", 0.9))}
    p_loss10 = float(norm.cdf((math.log(max(1e-9, (0.9 * cap_total - cash) / start_value)) - m_log) / s_log)
                     ) if s_log > 0 and start_value > 0 else None
    p_gain = float(1 - norm.cdf((math.log(max(1e-9, (cap_total - cash) / start_value)) - m_log) / s_log)
                   ) if s_log > 0 and start_value > 0 else None

    # orders, with what the competition and the client need to see
    vols = ds.volumes if ds.volumes is not None else None
    positions = []
    for i, o in enumerate(orders):
        t = str(o["ticker"])
        a = assets[t]
        row = alloc.get(t, {})
        last_vol = avg_vol = None
        if vols is not None and t in vols.columns:
            v = vols[t].dropna()
            v = v[v > 0]
            if len(v):
                last_vol, avg_vol = float(v.iloc[-1]), float(v.tail(20).mean())
        sec = business_sector(a)
        positions.append({
            "ticker": t, "nome": a.name, "logo": row.get("logo"), "classe": a.macro_class, "etf": is_etf(a),
            "setor": SECTOR_LABEL.get(sec, sec) if sec else None, "setor_chave": sec,
            "quantidade": float(o["quantidade"]), "preco": float(o["preco"]), "valor": float(o["valor"]),
            "peso": float(o["valor"]) / cap_total, "peso_alvo": float(o.get("peso_alvo", 0.0)),
            "participacao_risco": float(rc[i]), "papel": str(row.get("papel", "")),
            "papel_en": _role(float(wv[i]), float(rc[i]), None if np.isnan(ret_share[i]) else float(ret_share[i])),
            "proposito": str(row.get("proposito", "")), "negocio": a.business, "fatos_H": row.get("fatos_H"),
            "estimativas_E": row.get("estimativas_E"), "volume_ultimo": last_vol, "volume_medio_20d": avg_vol,
            "ordem_sobre_volume": (float(o["quantidade"]) / last_vol) if last_vol else None})

    # rule by rule, measured on the orders
    checks: list[dict[str, object]] = []

    def check(rule: str, required: str, got: str, ok: bool | None, note: str = "") -> None:
        checks.append({"regra": rule, "exigido": required, "obtido": got, "ok": ok, "nota": note})

    check("Capital inicial", money_pt(cap_total, ccy), f"{money_pt(invested, ccy)} investidos + {money_pt(cash, ccy)} "
          "em caixa", abs(invested + cash - cap_total) < 0.01 and cash >= -0.01)
    if rules.preco_minimo is not None:
        low = min(positions, key=lambda p: p["preco"]) if positions else None
        check("Preço mínimo por ação", f"{money_pt(rules.preco_minimo, ccy)} ou mais",
              f"menor preço: {low['ticker']} a {money_pt(low['preco'], ccy)}" if low else "sem posições",
              bool(low) and low["preco"] >= rules.preco_minimo - 1e-9,
              "medido no último fechamento; confira o preço no simulador na hora da compra")
    n_etf = sum(1 for p in positions if p["etf"])
    if rules.min_etfs:
        check("ETFs", f"pelo menos {rules.min_etfs}", f"{n_etf}: " + (", ".join(p["ticker"] for p in positions
                                                                            if p["etf"]) or "nenhum"),
              n_etf >= rules.min_etfs)
    if rules.etfs_aprovados:
        outside = [p["ticker"] for p in positions if p["etf"] and p["ticker"] not in rules.etfs_aprovados]
        check("ETFs da lista aprovada", "todos os ETFs da carteira na lista", "todos na lista" if not outside else
              "fora da lista: " + ", ".join(outside), not outside)
    elif rules.min_etfs:
        check("ETFs da lista aprovada", "ETF da lista aprovada", "lista não informada", None,
              "cole a lista de ETFs aprovados nas regras para o site conferir")
    top = max(positions, key=lambda p: p["peso"]) if positions else None
    if rules.peso_maximo is not None:
        check("Peso máximo por ativo", f"até {pct(rules.peso_maximo)}",
              f"maior: {top['ticker']} com {pct(top['peso'], 2)}" if top else "sem posições",
              bool(top) and top["peso"] <= rules.peso_maximo + 1e-9, "medido nos preços de compra; com as oscilações "
                                                                    "o peso muda, confira antes de cada nova compra")
    n_pos = len(positions)
    if rules.min_posicoes or rules.max_posicoes:
        req_txt = " e ".join(x for x in (f"pelo menos {rules.min_posicoes}" if rules.min_posicoes else "",
                                         f"no máximo {rules.max_posicoes}" if rules.max_posicoes else "") if x)
        check("Número de posições", req_txt, str(n_pos),
              (not rules.min_posicoes or n_pos >= rules.min_posicoes)
              and (not rules.max_posicoes or n_pos <= rules.max_posicoes))
    secs = sorted({str(p["setor"]) for p in positions if p["setor"]})
    if rules.min_setores:
        check("Setores diferentes", f"pelo menos {rules.min_setores}", f"{len(secs)}: {', '.join(secs) or 'nenhum'}",
              len(secs) >= rules.min_setores, "contam setores de negócio (ações e ETFs setoriais); ETFs amplos, renda "
                                              "fixa e ouro não contam como setor")
    if rules.caixa_maximo is not None:
        check("Caixa não investido", f"até {pct(rules.caixa_maximo)}", pct(cash / cap_total, 2),
              cash / cap_total <= rules.caixa_maximo + 1e-9)
    if rules.max_operacoes:
        check("Operações", f"até {rules.max_operacoes} no total", f"{n_pos} ordens de compra agora",
              n_pos <= rules.max_operacoes, f"sobram {max(0, rules.max_operacoes - n_pos)} para rebalancear e trocar "
                                            "posições")
    if rules.limite_volume:
        over = [p["ticker"] for p in positions if p["ordem_sobre_volume"] is not None
                and p["ordem_sobre_volume"] > rules.limite_volume]
        missing = [p["ticker"] for p in positions if p["ordem_sobre_volume"] is None]
        check("Tamanho da ordem", f"até {num(rules.limite_volume, 1)}× o volume do dia",
              ("acima: " + ", ".join(over)) if over else
              ("dentro do limite" + (f"; sem volume para {', '.join(missing)}" if missing else "")),
              False if over else (None if missing else True),
              "comparado com o volume do último pregão nos dados; o simulador usa o volume do dia da ordem")
    check("Sem margem e sem venda a descoberto", "só compras com o próprio capital", "só compras, sem empréstimo",
          cash >= -0.01)
    if not rules.permitir_fracoes:
        frac = [p["ticker"] for p in positions if abs(p["quantidade"] - round(p["quantidade"])) > 1e-9]
        check("Ações inteiras", "quantidades inteiras", "todas inteiras" if not frac else "frações em: " + ", ".join(frac),
              not frac)
    bad = [p["ticker"] for p in positions if p["ticker"] in excluded or (p["setor_chave"] in sectors_out)]
    if cli.exclusoes or cli.ativos_excluidos or sectors_out:
        check("Exclusões do cliente", "nada do que o cliente excluiu", "nenhuma posição excluída" if not bad else
              "em carteira: " + ", ".join(bad), not bad,
              f"{len(excluded)} empresas e {len(sectors_out)} setor(es) fora da análise")
    ok_all = all(c["ok"] is not False for c in checks)

    # the client, need by need
    quotes = cli.citacoes
    income = sum(p["peso"] for p in positions if traits(assets[p["ticker"]]).income)
    defensive = sum(p["peso"] for p in positions if assets[p["ticker"]].asset_class == "fixed_income")
    cashlike = sum(p["peso"] for p in positions if traits(assets[p["ticker"]]).cash or p["ticker"] in CASH_TICKERS)
    vol_cap = float(res.policy.get("teto_volatilidade") or 0.0)
    fit: list[dict[str, object]] = []

    def need(what: str, key: str, how: str, ok: bool | None = None) -> None:
        fit.append({"necessidade": what, "caso": quotes.get(key, ""), "como": how, "ok": ok})

    need(f"Perfil {cli.perfil_risco}", "perfil_risco",
         f"volatilidade estimada [E] de {pct(vol)} ao ano, para um teto de {pct(vol_cap)} do perfil "
         f"{res.policy.get('perfil_final')}", vol <= vol_cap + 1e-6 if vol_cap else None)
    need(f"Horizonte de {num(cli.horizonte_anos, 0)} anos", "horizonte_anos",
         "; ".join(str(r) for r in res.policy.get("regras", []) if "Horizonte" in str(r)) or
         "horizonte longo: limites do perfil, sem cortes adicionais por prazo")
    obj = cli.objetivo_tipo or "crescimento"
    if obj == "renda":
        need("Renda", "objetivo_tipo", f"{pct(income)} em ativos que pagam renda (REITs, ETFs de dividendos e de "
                                       "títulos)", income >= 0.5 - 1e-6)
    elif obj == "preservacao":
        need("Preservar o capital", "objetivo_tipo", f"{pct(defensive)} em renda fixa, {pct(cashlike)} quase caixa",
             None)
    else:
        need("Crescimento", "objetivo_tipo", f"retorno esperado [E] de {pct(ret)} ao ano com {pct(1 - defensive)} fora "
                                             "da renda fixa")
    goal = res.goal
    lo95, hi95 = goal.get("ic95", (None, None))  # type: ignore[misc]
    need("Meta", "valor_alvo_cliente", f"{goal_pt}: chance estimada [E] de {pct(goal.get('p_objetivo'), 0)} "  # type: ignore[arg-type]
                                       f"(intervalo de 95% da simulação: {pct(lo95, 0)} a {pct(hi95, 0)})")
    if cli.tolerancia_perda:
        need(f"Aceita perder até {pct(cli.tolerancia_perda, 0)}", "tolerancia_perda",
             f"maior queda destes pesos no histórico [H]: {pct(dd)}; teto de volatilidade derivado: {pct(vol_cap)}",
             None if dd is None else abs(dd) <= cli.tolerancia_perda + 1e-9)
    if cli.necessidade_liquidez:
        need(f"Liquidez de {pct(cli.necessidade_liquidez, 0)} em 12 meses", "necessidade_liquidez",
             f"{pct(cashlike)} em títulos curtos do Tesouro (quase caixa); todas as posições são negociadas em bolsa",
             cashlike >= cli.necessidade_liquidez - 1e-6)
    if cli.exclusoes:
        need("Exclusões", "exclusoes", "nenhuma posição em " + ", ".join(EXCLUSION_PT[k] for k in cli.exclusoes)
             + f" ({len(excluded)} empresas e {len(sectors_out)} setor(es) fora da análise)", not bad)
    if cli.temas:
        al = res.alignment
        tilted = sum(p["peso"] for p in positions if al.computed and p["ticker"] in al.scores
                     and al.scores[p["ticker"]].matched)
        need("Valores do cliente", "temas", f"{pct(tilted)} da carteira em ativos ligados a "
                                            + ", ".join(THEME_PT[k] for k in cli.temas)
             + f" (inclinação de {pct(req.peso_alinhamento, 0)} dentro dos limites de risco)")
    if cli.retirada_anual_cliente:
        need("Retiradas periódicas", "retirada_anual_cliente",
             "a simulação não desconta retiradas: com elas, a chance de atingir a meta é menor que a mostrada", None)

    # allocation by bucket
    buckets: dict[tuple[str, str], float] = {}
    for p in positions:
        k = _class_label(assets[p["ticker"]])
        buckets[k] = buckets.get(k, 0.0) + p["peso"]
    if cash / cap_total >= 0.001:
        buckets[("Uninvested cash", "Caixa não investido")] = cash / cap_total
    allocation = [{"classe_en": k[0], "classe_pt": k[1], "peso": v}
                  for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])]
    why_alloc = [(f"the {PROFILE_EN.get(str(res.policy.get('perfil_final')), res.policy.get('perfil_final'))} "
                 "profile's limits per asset class and per asset")]
    if res.policy.get("caixa_minimo"):
        why_alloc.append(f"at least {float(res.policy['caixa_minimo']) * 100:.0f}% in cash-like Treasuries")  # type: ignore[arg-type]
    if res.policy.get("renda_minima"):
        why_alloc.append(f"at least {float(res.policy['renda_minima']) * 100:.0f}% in income-paying assets")  # type: ignore[arg-type]
    if rules.peso_maximo:
        why_alloc.append(f"no position above {rules.peso_maximo * 100:.0f}%")
    if cli.exclusoes:
        why_alloc.append("no " + ", ".join(EXCLUSION_EN[k] for k in cli.exclusoes))

    # rules in English for the IPS
    rules_en = [f"starting capital {rules.capital:,.0f} {ccy}"]
    if rules.preco_minimo is not None:
        rules_en.append(f"securities priced at {rules.preco_minimo:,.2f} {ccy} or more")
    if rules.min_etfs:
        rules_en.append(f"at least {rules.min_etfs} ETF(s)" + (" from the approved list" if rules.etfs_aprovados else ""))
    if rules.peso_maximo is not None:
        rules_en.append(f"no position above {rules.peso_maximo * 100:.0f}% of the portfolio")
    if rules.min_posicoes:
        rules_en.append(f"at least {rules.min_posicoes} positions")
    if rules.max_posicoes:
        rules_en.append(f"at most {rules.max_posicoes} positions")
    if rules.min_setores:
        rules_en.append(f"at least {rules.min_setores} sectors")
    if rules.caixa_maximo is not None:
        rules_en.append(f"cash up to {rules.caixa_maximo * 100:.0f}%")
    if rules.max_operacoes:
        rules_en.append(f"at most {rules.max_operacoes} trades")
    if rules.limite_volume:
        rules_en.append(f"each order up to {rules.limite_volume:g}x the security's daily volume")
    rules_en.append("no margin and no short selling")
    if not rules.permitir_fracoes:
        rules_en.append("whole shares only")

    uni = res.universe
    funnel = uni.get("funil") or {}
    universe_en = (f"{'all US-listed stocks (Nasdaq screener: NYSE, Nasdaq, NYSE American) and the curated ETFs' if funnel.get('used_listings') else 'the curated catalogue of the site'}"  # type: ignore[union-attr]
                   f" -> {len(uni.get('candidatos', []))} candidates after screening on risk, quality, liquidity and "  # type: ignore[arg-type]
                   f"correlation; daily prices from {str(uni.get('periodo')).replace(' a ', ' to ')} (Yahoo Finance, "
                   "adjusted closes)")
    strat_en = (f"the {STRATEGY_EN.get(an.selected_name, an.selected_name)} portfolio chosen out of sample "
                f"(walk-forward validation), then tilted {req.peso_alinhamento * 100:.0f}% toward the client's values "
                "without leaving the risk limits; positions chosen to meet every counting rule")
    conc = res.concentration
    metrics = {"retorno_E": ret, "vol_E": vol, "erro_padrao_retorno": res.expected.get("erro_padrao_retorno_carteira"),
               "cagr_necessario": res.required.rate, "p_meta": goal.get("p_objetivo"), "ic95": [lo95, hi95],
               "p10_final": float(res.scenarios.loc["pessimista (P10)", "patrimonio_final"])
               if "pessimista (P10)" in res.scenarios.index else None,
               "drawdown_H": dd, "pior_janela_H": worst, "pior_janela_datas": worst_dates,
               "pior_janela_datas_en": worst_dates_en, "teto_vol": vol_cap,
               "n": n_pos, "enb": conc.enb if conc else None,
               "maior_ticker": top["ticker"] if top else None, "maior_peso": top["peso"] if top else None}
    window = {"fim": end_txt, "fim_en": end_en, "dias": n_days, "nota": window_note, "p10": q["p10"], "p50": q["p50"], "p90": q["p90"],
              "p_perda_10": p_loss10, "p_ganho": p_gain, "modelo": "lognormal com retorno [E] e covariância estimados "
                                                                  "pelo motor; caixa sem rendimento"}
    ctx = {"competicao": rules.nome, "data": day.isoformat(), "moeda": ccy, "capital": cap_total,
           "investido": invested, "caixa": cash, "data_precos": uni.get("ultima_cotacao"),
           "perfil_final": res.policy.get("perfil_final"), "meta_como_pt": goal_pt, "meta_como_en": goal_en,
           "cliente": {"nome": cli.nome, "idade": cli.idade, "horizonte": cli.horizonte_anos, "perfil": cli.perfil_risco,
                       "objetivo": cli.objetivo_tipo, "patrimonio": cli.patrimonio_cliente,
                       "meta": cli.valor_alvo_cliente, "aporte_anual": cli.aporte_anual_cliente,
                       "retirada_anual": cli.retirada_anual_cliente, "tolerancia_perda": cli.tolerancia_perda,
                       "liquidez": cli.necessidade_liquidez, "exclusoes": list(cli.exclusoes), "temas": list(cli.temas),
                       "metas_intermediarias": cli.metas_intermediarias, "observacoes": cli.observacoes,
                       "citacoes": quotes, "moeda_caso": cli.moeda_caso},
           "metricas": metrics, "janela": window, "alocacao": allocation, "porque_alocacao_en": why_alloc,
           "regras_en": rules_en, "universo_en": universe_en, "estrategia_en": strat_en, "posicoes": positions,
           "max_operacoes": rules.max_operacoes}
    ips = ips_english(ctx)
    resumo = summary_pt(ctx)
    language = []
    quoted = [*quotes.values(), cli.observacoes, *cli.metas_intermediarias]
    clean_ips = ips
    for s in quoted:
        if s:
            clean_ips = clean_ips.replace(s, "")          # the case's own words are quoted, not generated
    if hits := english_promises(clean_ips):
        language.append(f"linguagem de promessa no IPS: {hits}")
    if hits := guarantee_phrases(resumo):
        language.append(f"linguagem de garantia no resumo: {hits}")

    alerts = [a.__dict__ for a in res.alerts]
    if language:
        alerts.append({"category": "linguagem", "severity": "FAIL", "message": "; ".join(language)})
    csv = "ticker,acao,quantidade,preco_referencia,valor\n" + "\n".join(
        f"{p['ticker']},COMPRA,{p['quantidade']:g},{p['preco']:.2f},{p['valor']:.2f}" for p in positions)
    return {
        "aviso": ("Pesquisa, não recomendação: a carteira sai de um modelo com dados históricos e estimativas; nada é "
                  "executado aqui — a equipe digita as ordens no simulador da competição e responde pelas escolhas."),
        "cumpre_regras": ok_all, "conferencia": checks, "cliente_atendido": fit,
        "posicoes": positions, "caixa": cash, "investido": invested, "capital": cap_total, "moeda": ccy,
        "data_precos": uni.get("ultima_cotacao"), "alocacao": allocation,
        "metricas": metrics, "janela": window, "cenarios_horizonte": res.scenarios.reset_index().to_dict(orient="records"),
        "objetivo": {"como": goal_pt, "valor_alvo_na_escala": res.request.valor_alvo, "p": goal.get("p_objetivo"),
                     "ic95": [lo95, hi95], "cagr_necessario": res.required.rate,
                     "explicacao_cagr": res.required.explanation},
        "estrategia": {"nome": STRATEGY_PT.get(an.selected_name, an.selected_name), "texto": res.strategy_text},
        "ips_en": ips, "resumo_pt": resumo, "ordens_csv": csv,
        "exclusoes": {"empresas": len(excluded), "setores": [SECTOR_LABEL.get(s, s) for s in sectors_out],
                      "exemplos": sorted(excluded)[:30]},
        "fora_pelo_preco": {t: why for t, why in (res.execution.get("fora_por_tamanho_minimo") or {}).items()  # type: ignore[union-attr]
                            if "da regra" in str(why)},
        "notas": notes + trim_notes + ([window_note] if window_note else []),
        "regras_aplicadas": res.policy.get("regras", []), "hipoteses": res.policy.get("hipoteses", []),
        "universo": {"origem": uni.get("origem"), "periodo": uni.get("periodo"),
                     "ultima_cotacao": uni.get("ultima_cotacao"), "candidatos": len(uni.get("candidatos", [])),  # type: ignore[arg-type]
                     "funil": funnel},
        "rf": {k: {"taxa": v, "fonte": res.rf_source.get(k)} for k, v in res.rf_by_currency.items()},
        "fontes": res.sources, "alertas": alerts, "etapas": res.steps,
        "logos": {p["ticker"]: p["logo"] for p in positions if p.get("logo")},
    }
