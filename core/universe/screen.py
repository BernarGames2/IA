"""Pre-selection funnel: from every listed asset to the N candidates the optimiser can handle.

Optimising over thousands of assets per request is neither feasible (one 5-year download per
asset, a covariance matrix with more assets than observations) nor statistically meaningful.
The funnel is transparent and uses NO past performance (that would bias the backtest):

1. pool = curated catalogue + exchange listings (B3 shares/FIIs, US stocks), de-duplicated;
2. eligibility = liquidity/size floors (B3: R$ 5 mi traded on the reference day; US: market
   cap of US$ 2 bi) and supported instruments (ETFs only from the curated catalogue, whose
   asset class is known);
3. core diversification set (fixed income, broad indices, gold) — keeps the profile limits
   feasible;
4. theme slots: assets whose sector tags serve the themes the investor wrote, by size;
5. fill: round-robin over (market, sector) groups by size, so no sector dominates.

Size ranks are computed within each market (B3 by traded volume, US by market cap) and the two
markets are interleaved, because the two measures are not comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.purpose.catalog import CatalogAsset
from core.universe.listings import ListedAsset, MarketListings

CORE = ("IMAB11.SA", "B5P211.SA", "IRFM11.SA", "BOVA11.SA", "IVVB11.SA", "SPY", "AGG", "SHY", "TIP", "GLD")
B3_MIN_VOLUME = 5e6          # BRL traded on the reference day
US_MIN_MARKET_CAP = 2e9      # USD

_EQ_BR = "risco de ações: volatilidade elevada, risco de setor e de execução da empresa"
_EQ_US = "risco de ações e cambial (cotado em USD)"
_FII = "risco imobiliário: vacância, inadimplência e sensibilidade a juros"
_REIT = "risco imobiliário e de juros dos EUA e cambial (cotado em USD)"

# Nasdaq sector/industry -> theme tags (documented, deterministic; industry rules first)
_INDUSTRY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"solar|renewable|alternative energ|wind", ("energia_limpa", "sustentabilidade")),
    (r"water supply|water util|waste|environmental", ("saneamento", "infraestrutura")),
    (r"biotech|pharmac|medical|hospital|health|diagnos|dental|drug", ("saude",)),
    (r"educat|school|training", ("educacao",)),
    (r"real estate investment trust|reit", ("imoveis",)),
    (r"software|semiconductor|computer|edp services|internet|electronic comp|technology|data process",
     ("tecnologia",)),
    ((r"electric utilit|power generation|natural gas distribution|telecommunications equipment|"
      r"telecommunications services"), ("infraestrutura",)),
    (r"oil|gas production|petroleum|integrated oil|coal", ("petroleo_gas",)),
    (r"bank|insurance|investment|finance|credit|savings institution", ("financeiro",)),
)
_SECTOR_RULES = {"technology": ("tecnologia",), "health care": ("saude",), "finance": ("financeiro",),
                 "energy": ("petroleo_gas",), "utilities": ("infraestrutura",), "real estate": ("imoveis",),
                 "consumer discretionary": ("consumo",), "consumer staples": ("consumo",),
                 "industrials": ("industria",), "basic materials": ("industria",),
                 "telecommunications": ("infraestrutura",)}


def us_tags(sector: str, industry: str) -> tuple[str, ...]:
    ind = industry.lower()
    for rx, tags in _INDUSTRY_RULES:
        if re.search(rx, ind):
            return tags
    return _SECTOR_RULES.get(sector.lower(), ())


def listing_to_catalog(a: ListedAsset) -> CatalogAsset | None:
    """Convert a listed asset into a catalogue entry; None when its asset class is unknown."""
    if a.market == "US" and a.kind == "acao":
        tags = us_tags(a.sector, a.industry)
        reit = "imoveis" in tags
        business = (f"{a.industry} ({a.sector}) — classificação setorial da Nasdaq" if a.industry or a.sector
                    else "setor não informado pela fonte")
        return CatalogAsset(a.ticker, a.name, "acoes_eua", "real_estate" if reit else "equity",
                            (a.sector or "nao_classificado").lower(), "US", "USD", business,
                            (*tags, "diversificacao_global"), _REIT if reit else _EQ_US)
    if a.market == "B3" and a.kind == "acao":
        return CatalogAsset(a.ticker, a.name, "acoes_br", "equity", "nao_classificado", "BR", "BRL",
                            "ação listada na B3; setor não informado pela fonte (sem tema atribuído)", (), _EQ_BR)
    if a.market == "B3" and a.kind == "fii":
        return CatalogAsset(a.ticker, a.name, "fii", "real_estate", "imobiliario", "BR", "BRL",
                            "fundo imobiliário listado na B3", ("imoveis", "renda_recorrente"), _FII)
    return None                                   # ETFs outside the catalogue: underlying class unknown


@dataclass
class Funnel:
    listed_b3: int = 0
    listed_us: int = 0
    pool: int = 0
    eligible: int = 0
    selected: int = 0
    core: list[str] = field(default_factory=list)
    themed: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    used_listings: bool = False
    criteria: str = ""

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _market(a: CatalogAsset) -> str:
    return "B3" if a.ticker.endswith(".SA") else "US"


def preselect(curated: list[CatalogAsset], listings: MarketListings | None, theme_tags: set[str], cap: int,
              classes: list[str] | None = None) -> tuple[list[CatalogAsset], Funnel]:
    fun = Funnel(criteria=(f"liquidez/tamanho mínimos (B3: R$ {B3_MIN_VOLUME / 1e6:.0f} mi negociados no dia de "
                           f"referência; EUA: valor de mercado de US$ {US_MIN_MARKET_CAP / 1e9:.0f} bi), núcleo de "
                           "diversificação, ativos ligados aos seus temas e os maiores de cada setor; desempenho "
                           "passado NÃO é usado na pré-seleção"))
    size: dict[str, float] = {}
    pool: dict[str, CatalogAsset] = {}
    order: dict[str, int] = {}
    for i, a in enumerate(curated):                 # curated first: its metadata wins
        pool[a.ticker] = a
        order[a.ticker] = i
    eligible_dyn: set[str] = set()
    if listings is not None and listings.available:
        fun.used_listings = True
        fun.listed_b3, fun.listed_us = len(listings.b3), len(listings.us)
        fun.sources, fun.errors = dict(listings.sources), dict(listings.errors)
        for la in listings.all():
            if la.size is not None:
                size[la.ticker] = la.size
            if la.ticker in pool:
                continue
            ok = (la.size is not None and ((la.market == "B3" and la.size >= B3_MIN_VOLUME)
                                           or (la.market == "US" and la.size >= US_MIN_MARKET_CAP)))
            if not ok:
                continue
            c = listing_to_catalog(la)
            if c is None:
                continue
            pool[c.ticker] = c
            eligible_dyn.add(c.ticker)
    elif listings is not None:
        fun.errors = dict(listings.errors)
    if classes:
        pool = {t: a for t, a in pool.items() if a.macro_class in classes}
    fun.pool = len(pool) + (fun.listed_b3 + fun.listed_us - len(eligible_dyn) if fun.used_listings else 0)
    fun.eligible = len(pool)

    def rank_key(t: str) -> tuple[int, float, int]:
        # larger size first; curated names without size keep their editorial order
        return (0 if t in size else 1, -size.get(t, 0.0), order.get(t, 10**6))

    chosen: list[str] = []

    def take(t: str) -> None:
        if t in pool and t not in chosen and len(chosen) < cap:
            chosen.append(t)

    for t in CORE:
        take(t)
    fun.core = list(chosen)
    if theme_tags:
        themed = [t for t, a in pool.items() if set(a.tags) & theme_tags and t not in chosen]
        quota = max(1, int(0.4 * cap))
        by_mkt = {m: sorted([t for t in themed if _market(pool[t]) == m], key=rank_key) for m in ("B3", "US")}
        while quota > 0 and any(by_mkt.values()):
            for m in ("B3", "US"):
                if by_mkt[m] and quota > 0:
                    take(by_mkt[m].pop(0))
                    quota -= 1
        fun.themed = [t for t in chosen if t not in fun.core]
    # fill: alternate markets; within a market, cycle over sectors taking the largest remaining name
    groups: dict[str, dict[str, list[str]]] = {"B3": {}, "US": {}}
    for t, a in pool.items():
        if t not in chosen:
            groups[_market(a)].setdefault(a.sector, []).append(t)
    cycles = {}
    for m, gs in groups.items():
        for g in gs.values():
            g.sort(key=rank_key)
        cycles[m] = sorted(gs, key=lambda sec, gs=gs: rank_key(gs[sec][0]))
    pos = {"B3": 0, "US": 0}

    def next_from(m: str) -> str | None:
        secs = [sec for sec in cycles[m] if groups[m][sec]]
        if not secs:
            return None
        sec = secs[pos[m] % len(secs)]
        pos[m] += 1
        return groups[m][sec].pop(0)

    while len(chosen) < cap:
        picked = False
        for m in ("B3", "US"):
            t = next_from(m)
            if t is not None and len(chosen) < cap:
                take(t)
                picked = True
        if not picked:
            break
    fun.selected = len(chosen)
    return [pool[t] for t in chosen], fun
