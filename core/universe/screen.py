"""Stage A of the universe pipeline: from every listed asset to a broad pool worth downloading.

Downloading and estimating thousands of assets per request is neither feasible (one 5-year series
per asset) nor statistically meaningful (more assets than observations). Stage A uses only
information that does not depend on past returns, so it cannot chase performance:

1. discover: curated catalogue (instruments whose class is documented, e.g. US ETFs) + exchange
   listings (B3: every share, FII and ETF traded on the reference day; US: every common stock in the
   Nasdaq screener) + the investor's current holdings;
2. normalise and classify: B3 ETFs outside the catalogue get a class from their ticker/name
   (fixed income, gold, crypto, international or Brazilian equity index) or are left out when the
   class cannot be identified; US stocks get sector/industry from Nasdaq;
3. de-duplicate: one share class per company (PETR3/PETR4, GOOG/GOOGL — the most liquid stays);
4. filter: investor preferences (markets, excluded tickers/sectors, crypto permission, horizon-
   incompatible long bonds) and liquidity/size floors (B3: R$ 5 mi traded on the reference day; US:
   market cap of US$ 2 bi);
5. pool: stratified by class and sector, largest first within each stratum, so every class the
   profile may need is represented; no ticker is ever forced in. Assets linked to the investor's
   themes get up to 30% of the pool.

Stage B (``core/universe/scoring.py``) then downloads the pool, measures risk, quality, liquidity
and correlation, and picks the candidates for the optimiser.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from core.purpose.catalog import CatalogAsset
from core.purpose.themes import _fold
from core.universe.listings import ListedAsset, MarketListings

B3_MIN_VOLUME = 5e6          # BRL traded on the reference day
US_MIN_MARKET_CAP = 2e9      # USD
THEME_SHARE = 0.30           # share of the pool reserved for assets linked to the investor's themes
MIN_PER_CLASS = 3            # each allowed class keeps at least this many names in the pool (if available)

_EQ_BR = "risco de ações: volatilidade elevada, risco de setor e de execução da empresa"
_EQ_US = "risco de ações e cambial (cotado em USD)"
_FII = "risco imobiliário: vacância, inadimplência e sensibilidade a juros"
_REIT = "risco imobiliário e de juros dos EUA e cambial (cotado em USD)"
_RF_BR = "risco de taxa de juros (marcação a mercado) e de inflação implícita"
_GOLD = "commodity sem fluxo de caixa; risco cambial"
_CRYPTO = "volatilidade extrema, risco regulatório e de custódia; sem fluxo de caixa"
_ETF_BR = "risco de ações do índice replicado"
_ETF_INT = "risco de ações do índice replicado e cambial"

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

# B3 ETFs outside the catalogue: class inferred from ticker root + short name (documented; unknown -> out)
_B3_ETF_RULES: tuple[tuple[str, str, str, str, str, tuple[str, ...], str], ...] = (
    # (pattern, macro_class, asset_class, sector, description, tags, risk)
    (r"SELIC|LFTS|TESOURO SELIC|\bDI\b|CDI", "renda_fixa_br", "fixed_income", "titulos_publicos",
     "ETF de títulos pós-fixados (Tesouro Selic/DI): caixa, prazo curto", ("preservacao_capital",), _RF_BR),
    (r"B5P2|IB5M|IMA-?B ?5|IMAB5|\bB5\b", "renda_fixa_br", "fixed_income", "titulos_publicos",
     "ETF de títulos IPCA+ de até 5 anos: prazo curto", ("preservacao_capital", "protecao_inflacao"), _RF_BR),
    (r"IMAB|IMA-?B|IMBB|IPCA|INFLA", "renda_fixa_br", "fixed_income", "titulos_publicos",
     "ETF de títulos IPCA+: prazo longo", ("preservacao_capital", "protecao_inflacao"), _RF_BR),
    (r"IRFM|IRF-?M|\bPRE\b|FIXA|RENDA ?FIXA|DEBB|DEBENT|IDKA", "renda_fixa_br", "fixed_income", "titulos",
     "ETF de renda fixa: prazo médio", ("preservacao_capital",), _RF_BR),
    (r"OURO|GOLD", "ouro", "commodity", "ouro", "ETF que acompanha o ouro", ("reserva_valor",), _GOLD),
    (r"BTC|BITC|ETH|HASH|CRIPT|CRYPT|QBTC|QETH|SOLH|BITH|DEFI|XRP", "cripto", "crypto", "cripto",
     "ETF de criptoativos", ("especulativo",), _CRYPTO),
    (r"IVVB|S&P|SP ?500|SPXI|NASD|QQQ|WRLD|ACWI|EURP|EURO|ASIA|XINA|CHIN|JAPA|EUAA|USAL|GLOB|INTE?R|EMER|BDOM",
     "etf_br", "equity", "indice", "ETF listado na B3 de ações internacionais (exposição cambial)",
     ("diversificacao_global",), _ETF_INT),
    (r"BOVA|BOVV|BOVB|XBOV|IBOV|BRAX|IBRX|PIBB|SMAL|SMLL|DIVO|DIVD|IDIV|ECOO|ISUS|FIND|MATB|UTIL|GOVE|BBSD|SMAC",
     "etf_br", "equity", "indice", "ETF de índice de ações brasileiras", ("diversificacao_local",), _ETF_BR),
)


def us_tags(sector: str, industry: str) -> tuple[str, ...]:
    ind = industry.lower()
    for rx, tags in _INDUSTRY_RULES:
        if re.search(rx, ind):
            return tags
    return _SECTOR_RULES.get(sector.lower(), ())


def classify_b3_etf(a: ListedAsset) -> CatalogAsset | None:
    """Class of a B3 ETF from its ticker root and short name; None when it cannot be identified."""
    text = f"{a.ticker.removesuffix('.SA')} {_fold(a.name).upper()}"
    for rx, macro, cls, sector, desc, tags, risk in _B3_ETF_RULES:
        if re.search(rx, text):
            return CatalogAsset(a.ticker, a.name, macro, cls, sector, "BR", "BRL",
                                f"{desc} (classe inferida pelo código/nome '{a.name}' na listagem da B3)", tags, risk)
    return None


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
    if a.market == "B3" and a.kind == "etf":
        return classify_b3_etf(a)
    return None                                   # US ETFs outside the catalogue: no class/liquidity in the sources


@dataclass
class Funnel:
    listed_b3: int = 0
    listed_us: int = 0
    pool: int = 0                      # assets considered after classification (catalogue + listings)
    eligible: int = 0                  # after preferences, liquidity floors and de-duplication
    selected: int = 0                  # handed to stage B (downloaded)
    themed: list[str] = field(default_factory=list)
    excluded_by_preference: dict[str, str] = field(default_factory=dict)
    below_liquidity_floor: int = 0
    unknown_class: int = 0
    share_class_duplicates: dict[str, str] = field(default_factory=dict)      # dropped -> kept
    per_class: dict[str, int] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    used_listings: bool = False
    criteria: str = ""

    def as_dict(self) -> dict[str, object]:
        d = dict(self.__dict__)
        excl = d.pop("excluded_by_preference")
        d["excluidos_por_preferencia"] = len(excl)
        d["exemplos_excluidos_por_preferencia"] = dict(list(excl.items())[:12])
        return d


def _market(a: CatalogAsset) -> str:
    return "B3" if a.ticker.endswith(".SA") else ("CAIXA" if a.macro_class == "caixa" else "US")


_CLASS_WORDS = re.compile(r"\b(class [a-z]|common stock|capital stock|ordinary shares?|inc|corp(oration)?|co|ltd|plc|sa|"
                          r"holdings?|the|n\.?v\.?|ag|se|adr|ads|american depositary shares?)\b")


def company_key(a: CatalogAsset, listed_name: str | None = None) -> str | None:
    """Same company, different share classes: B3 equities share the 4-letter root; US names match
    once share-class words are removed (the exchange's own name is preferred, as catalogue names are
    editorial). FIIs, ETFs and funds are never merged this way."""
    if a.macro_class == "acoes_br":
        return "B3:" + a.ticker[:4]
    if a.macro_class == "acoes_eua":
        name = _CLASS_WORDS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", (listed_name or a.name).lower()))
        return "US:" + " ".join(name.split())
    return None


def preselect(curated: list[CatalogAsset], listings: MarketListings | None, theme_tags: set[str], cap: int,
              classes: list[str] | None = None, allows: Callable[[CatalogAsset], str | None] | None = None,
              always: list[CatalogAsset] | None = None) -> tuple[list[CatalogAsset], Funnel]:
    """Stage A. ``cap`` = size of the pool to download; ``allows`` = investor filter (None or the
    reason the asset is not allowed); ``always`` = assets the investor already holds (analysed so the
    current portfolio can be compared — they still have to pass the data checks)."""
    fun = Funnel(criteria=(f"classe identificada; preferências do investidor; liquidez/tamanho mínimos (B3: R$ "
                           f"{B3_MIN_VOLUME / 1e6:.0f} mi negociados no dia de referência; EUA: valor de mercado de US$ "
                           f"{US_MIN_MARKET_CAP / 1e9:.0f} bi); uma classe de ação por empresa; amostra estratificada "
                           "por classe e setor, maiores primeiro; desempenho passado NÃO é usado nesta etapa"))
    size: dict[str, float] = {}
    pool: dict[str, CatalogAsset] = {}
    order: dict[str, int] = {}
    for i, a in enumerate(curated):                 # curated first: its documented metadata wins
        pool[a.ticker] = a
        order[a.ticker] = i
    if listings is not None and listings.available:
        fun.used_listings = True
        fun.listed_b3, fun.listed_us = len(listings.b3), len(listings.us)
        fun.sources, fun.errors = dict(listings.sources), dict(listings.errors)
        for la in listings.all():
            if la.size is not None:
                size[la.ticker] = la.size
            if la.ticker in pool:
                continue
            c = listing_to_catalog(la)
            if c is None:
                fun.unknown_class += 1
                continue
            floor = B3_MIN_VOLUME if la.market == "B3" else US_MIN_MARKET_CAP
            if la.size is None or la.size < floor:
                fun.below_liquidity_floor += 1
                continue
            pool[c.ticker] = c
    elif listings is not None:
        fun.errors = dict(listings.errors)
    fun.pool = len(pool)
    if classes:
        pool = {t: a for t, a in pool.items() if a.macro_class in classes}
    if allows is not None:
        for t in list(pool):
            why = allows(pool[t])
            if why:
                fun.excluded_by_preference[t] = why
                del pool[t]

    def rank_key(t: str) -> tuple[int, float, int]:
        # larger size first; curated names without size keep their editorial order
        return (0 if t in size else 1, -size.get(t, 0.0), order.get(t, 10**6))

    listed_name = {la.ticker: la.name for la in listings.all()} if fun.used_listings and listings else {}
    best_of: dict[str, str] = {}
    for t in sorted(pool, key=rank_key):
        k = company_key(pool[t], listed_name.get(t))
        if k is None:
            continue
        if k in best_of:
            fun.share_class_duplicates[t] = best_of[k]
            del pool[t]
        else:
            best_of[k] = t
    fun.eligible = len(pool)

    chosen: list[str] = []

    def take(t: str) -> None:
        if t in pool and t not in chosen and len(chosen) < cap:
            chosen.append(t)

    for a in always or []:                          # the investor's own holdings are always analysed
        if a.ticker not in pool:
            pool[a.ticker] = a
        take(a.ticker)
    # every allowed class keeps a minimum presence (largest first)
    by_class: dict[str, list[str]] = {}
    for t, a in pool.items():
        by_class.setdefault(a.macro_class, []).append(t)
    for _cls, members in sorted(by_class.items()):
        for t in sorted(members, key=rank_key)[:MIN_PER_CLASS]:
            take(t)
    if theme_tags:
        themed = [t for t, a in pool.items() if set(a.tags) & theme_tags and t not in chosen]
        quota = max(1, int(THEME_SHARE * cap))
        by_mkt = {m: sorted([t for t in themed if _market(pool[t]) == m], key=rank_key) for m in ("B3", "US")}
        while quota > 0 and any(by_mkt.values()):
            for m in ("B3", "US"):
                if by_mkt[m] and quota > 0:
                    take(by_mkt[m].pop(0))
                    quota -= 1
        fun.themed = [t for t in chosen if set(pool[t].tags) & theme_tags]
    # fill: cycle over (class, sector) strata, largest remaining name of each stratum in turn
    strata: dict[tuple[str, str], list[str]] = {}
    for t, a in pool.items():
        if t not in chosen:
            strata.setdefault((a.macro_class, a.sector), []).append(t)
    for g in strata.values():
        g.sort(key=rank_key)
    cycle = sorted(strata, key=lambda k: rank_key(strata[k][0]))
    while len(chosen) < cap and any(strata[k] for k in cycle):
        for k in cycle:
            if strata[k] and len(chosen) < cap:
                take(strata[k].pop(0))
    fun.selected = len(chosen)
    for t in chosen:
        fun.per_class[pool[t].macro_class] = fun.per_class.get(pool[t].macro_class, 0) + 1
    return [pool[t] for t in chosen], fun
