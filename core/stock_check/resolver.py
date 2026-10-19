"""Resolve what the user typed ("PETR4", "petrobras", "AAPL", "Banco do Brasil") into a ticker.

Order: an explicit ticker (the whole input) > a B3 code found inside the text > a known company
name (accent-insensitive). Short or ambiguous names ("vale", "meta", "tim", "vivo", "ouro") only
match when they are the whole input, so a sentence such as "vale a pena?" never maps to VALE3.
Anything else is rejected with a request for the ticker — nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.purpose.catalog import REAL_CATALOG, SYNTHETIC_CATALOG, CatalogAsset
from core.purpose.themes import _fold
from core.universe.listings import ListedAsset, MarketListings

B3_CODE = re.compile(r"^[A-Z]{4}\d{1,2}$")
B3_IN_TEXT = re.compile(r"\b([A-Za-z]{4}\d{1,2})\b")
TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")
US_TICKER = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
CRYPTO_TICKER = re.compile(r"^[A-Z0-9]{2,6}-USD$")

# well-known names outside the analysis catalogue: (ticker, name, factual one-line business)
EXTRA: dict[str, tuple[str, str, str]] = {
    "bradesco": ("BBDC4.SA", "Bradesco PN", "banco de varejo, atacado e seguros"),
    "itausa": ("ITSA4.SA", "Itaúsa PN", "holding com participação relevante no Itaú Unibanco e outras empresas"),
    "eletrobras": ("ELET3.SA", "Eletrobras ON", "geração e transmissão de energia elétrica"),
    "sabesp": ("SBSP3.SA", "Sabesp ON", "saneamento básico (água e esgoto) em São Paulo"),
    "localiza": ("RENT3.SA", "Localiza ON", "aluguel de carros e gestão de frotas"),
    "b3": ("B3SA3.SA", "B3 ON", "bolsa de valores e infraestrutura de mercado do Brasil"),
    "gerdau": ("GGBR4.SA", "Gerdau PN", "siderurgia (produção de aço)"),
    "embraer": ("EMBR3.SA", "Embraer ON", "aviões comerciais, executivos e de defesa"),
    "magazine luiza": ("MGLU3.SA", "Magazine Luiza ON", "varejo físico e comércio eletrônico"),
    "magalu": ("MGLU3.SA", "Magazine Luiza ON", "varejo físico e comércio eletrônico"),
    "lojas renner": ("LREN3.SA", "Lojas Renner ON", "varejo de moda"),
    "renner": ("LREN3.SA", "Lojas Renner ON", "varejo de moda"),
    "klabin": ("KLBN11.SA", "Klabin units", "papel e embalagens"),
    "taesa": ("TAEE11.SA", "Taesa units", "transmissão de energia elétrica"),
    "cemig": ("CMIG4.SA", "Cemig PN", "geração, transmissão e distribuição de energia em Minas Gerais"),
    "rumo": ("RAIL3.SA", "Rumo ON", "logística ferroviária"),
    "prio": ("PRIO3.SA", "PRIO ON", "exploração e produção de petróleo"),
    "csn": ("CSNA3.SA", "CSN ON", "siderurgia e mineração"),
    "hypera": ("HYPE3.SA", "Hypera ON", "medicamentos"),
    "petrobras on": ("PETR3.SA", "Petrobras ON", "exploração, produção e refino de petróleo e gás"),
    "tesla": ("TSLA", "Tesla", "carros elétricos e armazenamento de energia"),
    "amazon": ("AMZN", "Amazon", "comércio eletrônico e computação em nuvem (AWS)"),
    "netflix": ("NFLX", "Netflix", "streaming de vídeo por assinatura"),
    "coca cola": ("KO", "Coca-Cola", "bebidas não alcoólicas"),
    "coca-cola": ("KO", "Coca-Cola", "bebidas não alcoólicas"),
    "nubank": ("NU", "Nu Holdings (Nubank)", "banco digital"),
    "mercado livre": ("MELI", "MercadoLibre", "comércio eletrônico e pagamentos na América Latina"),
    "meta": ("META", "Meta Platforms", "redes sociais e publicidade digital"),
    "facebook": ("META", "Meta Platforms", "redes sociais e publicidade digital"),
    "vivo": ("VIVT3.SA", "Telefônica Brasil (Vivo) ON", "telecomunicações"),
    "tim": ("TIMS3.SA", "TIM ON", "telefonia móvel e banda larga"),
}
# catalogue names / common short names
ALIASES: dict[str, str] = {
    "petrobras": "PETR4.SA", "vale": "VALE3.SA", "itau": "ITUB4.SA", "itau unibanco": "ITUB4.SA",
    "weg": "WEGE3.SA", "engie": "EGIE3.SA", "raia drogasil": "RADL3.SA", "drogasil": "RADL3.SA",
    "droga raia": "RADL3.SA", "raia": "RADL3.SA", "pague menos": "PGMN3.SA", "panvel": "PNVL3.SA",
    "fleury": "FLRY3.SA", "cogna": "COGN3.SA", "suzano": "SUZB3.SA", "banco do brasil": "BBAS3.SA",
    "ambev": "ABEV3.SA", "totvs": "TOTS3.SA", "rede d'or": "RDOR3.SA", "rede dor": "RDOR3.SA",
    "hapvida": "HAPV3.SA", "yduqs": "YDUQ3.SA", "ibovespa": "BOVA11.SA", "apple": "AAPL",
    "microsoft": "MSFT", "nvidia": "NVDA", "google": "GOOGL", "alphabet": "GOOGL", "johnson & johnson": "JNJ",
    "johnson": "JNJ", "s&p 500": "SPY", "sp500": "SPY", "s&p500": "SPY", "ouro": "GLD", "bitcoin": "BTC-USD",
}
EXACT_ONLY = {"vale", "meta", "tim", "vivo", "ouro", "b3", "prio", "rumo", "csn", "weg"}


@dataclass(frozen=True)
class Resolved:
    ticker: str
    name: str
    business: str | None          # None = not in the catalogue: no description is invented
    asset: CatalogAsset | None
    how: str                      # how the input was interpreted (shown to the user)
    listed: ListedAsset | None = None   # entry of the full-market listings, when resolved through them


def _from_ticker(t: str, how: str) -> Resolved:
    cat = {a.ticker: a for a in (*REAL_CATALOG, *SYNTHETIC_CATALOG)}
    if t in cat:
        a = cat[t]
        return Resolved(t, a.name, a.business.replace("[sintético] ", ""), a, how)
    for tk, name, business in EXTRA.values():
        if tk == t:
            return Resolved(t, name, business, None, how)
    return Resolved(t, t, None, None, how)


def _from_listing(a: ListedAsset, how: str) -> Resolved:
    r = _from_ticker(a.ticker, how)
    if r.asset is not None or r.business is not None:
        return Resolved(r.ticker, r.name, r.business, r.asset, how, a)   # curated metadata wins
    business = (f"{a.industry} ({a.sector}) — classificação setorial da Nasdaq" if a.industry or a.sector
                else None)
    return Resolved(a.ticker, a.name, business, None, how, a)


def is_plain_code(r: Resolved) -> bool:
    """True when the input was taken as a bare code that no catalogue/alias recognises."""
    return r.asset is None and r.business is None and r.how.startswith("código ") and \
        not r.how.startswith(("código B3", "código sintético"))


def resolve(text: str, listings: MarketListings | None = None) -> Resolved:
    """``listings`` (full B3/US listings, optional) enables name search over every listed
    company and checks bare codes against the listed tickers."""
    raw = (text or "").strip()
    if not raw or len(raw) > 60:
        raise ValueError("informe o código ou o nome de uma ação (até 60 caracteres)")
    up = raw.upper().replace(" ", "")
    if up.startswith("SINT_"):
        return _from_ticker(up, "código sintético")
    if B3_CODE.match(up):
        return _from_ticker(up + ".SA", f"código B3 {up}")
    if up.endswith(".SA") and B3_CODE.match(up[:-3]):
        return _from_ticker(up, f"código B3 {up[:-3]}")
    folded = _fold(raw).strip(" ?!.")
    if folded in ALIASES:
        return _from_ticker(ALIASES[folded], f"nome '{raw}'")
    if folded in EXTRA:
        return _from_ticker(EXTRA[folded][0], f"nome '{raw}'")
    has_listing = listings is not None and listings.available
    if " " not in raw and TICKER.match(up):
        plausible = bool(US_TICKER.match(up) or CRYPTO_TICKER.match(up)) or any(c in up for c in "^=.")
        if has_listing:
            known = listings.by_ticker()  # type: ignore[union-attr]
            if up in known:
                return _from_listing(known[up], f"código {up} (listado)")
            hit = listings.search(raw)  # type: ignore[union-attr]
            if hit is not None:
                return _from_listing(hit, f"nome '{raw}' encontrado na listagem ({hit.ticker})")
            if not plausible:
                raise ValueError(f"não encontrei {raw!r} nas listagens da B3 e dos EUA: confira o código ou o nome")
        return _from_ticker(up, f"código {up}")
    m = B3_IN_TEXT.search(raw)
    if m and B3_CODE.match(m.group(1).upper()):
        code = m.group(1).upper()
        return _from_ticker(code + ".SA", f"código B3 {code} encontrado no texto")
    names = sorted([*ALIASES.items(), *((k, v[0]) for k, v in EXTRA.items())], key=lambda kv: -len(kv[0]))
    for name, tk in names:
        if name in EXACT_ONLY:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", folded):
            return _from_ticker(tk, f"nome '{name}' encontrado no texto")
    if has_listing:
        hit = listings.search(raw)  # type: ignore[union-attr]
        if hit is not None:
            return _from_listing(hit, f"nome '{raw}' encontrado na listagem ({hit.ticker})")
    raise ValueError(f"não reconheci a ação em {raw!r}: digite o código (ex.: PETR4, VALE3, AAPL)")
