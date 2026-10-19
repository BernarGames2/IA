"""Complete exchange listings, downloaded at run time and cached (never invented).

* B3: official daily historical-quotes file **COTAHIST_D<ddmmyyyy>.ZIP** (fixed-width layout
  published by B3). Every security traded on the cash market that day, with its short name,
  specification (ON/PN/UNT/CI…) and the day's financial volume — used as the liquidity proxy.
* US (NYSE + Nasdaq + NYSE American): Nasdaq stock screener (JSON with sector, industry,
  market cap and volume for every listed common stock); if it is unavailable, the official
  Nasdaq Trader symbol directory (names and ETF flags only — no size, so it serves name search
  but not ranking).

Each source fails independently; failures are reported, and the caller falls back to the
curated catalogue. Listings are cached for 24 h.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from core.market_data.yahoo_chart import FetchFn, urllib_fetch
from core.purpose.themes import _fold
from utils.logging_config import get_logger

log = get_logger("universe.listings")

COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{d:%d%m%Y}.ZIP"
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
SYMDIR_URLS = ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
               "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
CACHE_TTL_S = 24 * 3600


@dataclass(frozen=True)
class ListedAsset:
    ticker: str                 # Yahoo form: PETR4.SA, AAPL, BRK-B
    name: str
    market: str                 # "B3" | "US"
    currency: str
    kind: str                   # acao | fii | etf
    sector: str = ""
    industry: str = ""
    size: float | None = None   # B3: financial volume of the day (BRL); US: market cap (USD)
    size_kind: str = ""         # "volume_dia_brl" | "market_cap_usd" | ""
    source: str = ""


@dataclass
class MarketListings:
    b3: list[ListedAsset] = field(default_factory=list)
    us: list[ListedAsset] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def all(self) -> list[ListedAsset]:
        return [*self.b3, *self.us]

    @property
    def available(self) -> bool:
        return bool(self.b3 or self.us)

    def by_ticker(self) -> dict[str, ListedAsset]:
        return {a.ticker: a for a in self.all()}

    def search(self, text: str) -> ListedAsset | None:
        """Best name match (accent-insensitive): exact > prefix > whole word; ties -> larger size."""
        q = _clean_name(text)
        if len(q) < 3:
            return None
        best: tuple[int, float, ListedAsset] | None = None
        for a in self.all():
            n = _clean_name(a.name)
            if not n:
                continue
            if n == q:
                score = 3
            elif n.startswith(q):
                score = 2
            elif re.search(r"(?<![a-z0-9])" + re.escape(q) + r"(?![a-z0-9])", n):
                score = 1
            else:
                continue
            key = (score, a.size or 0.0)
            if best is None or key > best[:2]:
                best = (score, a.size or 0.0, a)
        return best[2] if best else None


_NAME_NOISE = re.compile(r"\b(s\.?a\.?|sa|inc|corp|corporation|co|company|ltd|plc|holdings?|group|"
                         r"common stock|ordinary shares|class [abc]|on|pn|pna|pnb|unt|n[12]|nm|ej|ed|"
                         r"american depositary shares?|ads|adr|acao|acoes|da|de|do|the)\b")


def _clean_name(s: str) -> str:
    f = _fold(s).replace(".", " ").replace(",", " ")
    f = _NAME_NOISE.sub(" ", f)
    return re.sub(r"\s+", " ", f).strip()


# ----------------------------------------------------------------------------- B3 (COTAHIST)
def parse_cotahist(text: str) -> list[ListedAsset]:
    """Parse COTAHIST detail records (TIPREG 01) of the cash market (TPMERC 010).

    Layout (1-based positions): CODBDI 11-12, CODNEG 13-24, TPMERC 25-27, NOMRES 28-39,
    ESPECI 40-49, PREULT 109-121 (2 implied decimals), VOLTOT 171-188 (2 implied decimals).
    Kept: shares (ESPECI ON/PN/UNT, CODBDI 02), real-estate funds (CODBDI 12) and ETFs
    (CODBDI 14, ESPECI CI). BDRs, rights, receipts, odd lots and derivatives are dropped.
    """
    out: dict[str, ListedAsset] = {}
    for line in text.splitlines():
        if len(line) < 188 or line[0:2] != "01" or line[24:27] != "010":
            continue
        codbdi = line[10:12]
        ticker = line[12:24].strip()
        name = line[27:39].strip()
        especi = line[39:49].strip()
        try:
            volume = int(line[170:188]) / 100.0
        except ValueError:
            continue
        if especi.startswith(("DR", "BDR")):
            continue
        if codbdi == "02" and especi.split(" ")[0] in {"ON", "PN", "PNA", "PNB", "PNC", "UNT"}:
            kind = "acao"
        elif codbdi == "12":
            kind = "fii"
        elif codbdi == "14" and especi.startswith("CI"):
            kind = "etf"
        else:
            continue
        if not re.fullmatch(r"[A-Z0-9]{4}\d{1,2}", ticker):
            continue
        key = ticker + ".SA"
        prev = out.get(key)
        if prev is None or volume > (prev.size or 0):
            out[key] = ListedAsset(key, f"{name} {especi.split(' ')[0]}".strip(), "B3", "BRL", kind, size=volume,
                                   size_kind="volume_dia_brl", source="B3 COTAHIST (cotações históricas diárias)")
    return list(out.values())


def fetch_b3_listing(fetch: FetchFn, today: date | None = None, max_days_back: int = 8,
                     timeout_s: float = 20.0) -> tuple[list[ListedAsset], str]:
    d = today or datetime.now(ZoneInfo("America/Sao_Paulo")).date()     # B3 calendar day
    last_err = "sem tentativa"
    for k in range(1, max_days_back + 1):
        day = d - timedelta(days=k)
        if day.weekday() >= 5:
            continue
        url = COTAHIST_URL.format(d=day)
        try:
            status, body = fetch(url, timeout_s)
        except OSError as e:
            last_err = f"erro de rede: {e}"
            break                                  # unreachable host: do not try every day
        if status != 200 or body[:2] != b"PK":
            last_err = f"HTTP {status} para {day:%d/%m/%Y}"
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                text = z.read(z.namelist()[0]).decode("latin-1")
        except (zipfile.BadZipFile, IndexError, KeyError) as e:
            last_err = f"arquivo inválido: {e}"
            continue
        rows = parse_cotahist(text)
        if rows:
            return rows, f"B3 COTAHIST de {day:%d/%m/%Y}"
        last_err = f"arquivo de {day:%d/%m/%Y} sem registros do mercado à vista"
    raise RuntimeError(f"listagem B3 indisponível ({last_err})")


# ----------------------------------------------------------------------------- US (Nasdaq)
def _num(s: object) -> float | None:
    try:
        v = float(str(s).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    return v if v > 0 else None


_US_EXCLUDE = re.compile(r"\b(warrants?|rights?|units?|acquisition corp|notes due|preferred|depositary shares "
                         r"representing)\b", re.IGNORECASE)


def parse_nasdaq_screener(payload: object) -> list[ListedAsset]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return []
    rows = payload["data"].get("rows") or []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol", "")).strip().upper()
        name = str(r.get("name", "")).strip()
        if not sym or "^" in sym or " " in sym or _US_EXCLUDE.search(name):
            continue
        sym = sym.replace("/", "-")
        if not re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", sym):
            continue
        out.append(ListedAsset(sym, name, "US", "USD", "acao", str(r.get("sector") or ""),
                               str(r.get("industry") or ""), _num(r.get("marketCap")), "market_cap_usd",
                               "Nasdaq stock screener (NYSE, Nasdaq, NYSE American)"))
    return out


def parse_symdir(text: str, other: bool) -> list[ListedAsset]:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    if not lines:
        return []
    head = lines[0].split("|")
    out = []
    for ln in lines[1:]:
        f = dict(zip(head, ln.split("|")))
        sym = (f.get("ACT Symbol") if other else f.get("Symbol")) or ""
        if f.get("Test Issue") == "Y" or not sym or "$" in sym:
            continue
        name = f.get("Security Name", "")
        if _US_EXCLUDE.search(name):
            continue
        sym = sym.replace(".", "-")
        out.append(ListedAsset(sym, name, "US", "USD", "etf" if f.get("ETF") == "Y" else "acao",
                               source="Nasdaq Trader symbol directory"))
    return out


def fetch_us_listing(fetch: FetchFn, timeout_s: float = 25.0) -> tuple[list[ListedAsset], str]:
    try:
        status, body = fetch(NASDAQ_SCREENER_URL, timeout_s)
        if status == 200:
            rows = parse_nasdaq_screener(json.loads(body.decode("utf-8")))
            if rows:
                return rows, "Nasdaq stock screener"
        err = f"screener HTTP {status}"
    except (OSError, ValueError, UnicodeDecodeError) as e:
        err = f"screener: {e}"
    rows: list[ListedAsset] = []
    for url, other in zip(SYMDIR_URLS, (False, True)):
        try:
            status, body = fetch(url, timeout_s)
        except OSError as e:
            raise RuntimeError(f"listagem EUA indisponível ({err}; diretório: erro de rede: {e})") from e
        if status == 200:
            rows += parse_symdir(body.decode("utf-8", "replace"), other)
    if rows:
        return rows, f"Nasdaq Trader symbol directory (sem tamanho/setor; {err})"
    raise RuntimeError(f"listagem EUA indisponível ({err})")


# ----------------------------------------------------------------------------- loader + cache
def load_market_listings(cache_dir: Path | None = None, fetch: FetchFn | None = None,
                         today: date | None = None) -> MarketListings:
    f = fetch or urllib_fetch
    out = MarketListings()
    cache = Path(cache_dir) / "listings.json" if cache_dir else None
    if cache is not None and cache.exists():
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
            if time.time() - float(raw["stored_at"]) < CACHE_TTL_S:
                out.b3 = [ListedAsset(**x) for x in raw["b3"]]
                out.us = [ListedAsset(**x) for x in raw["us"]]
                out.sources = dict(raw["sources"])
                out.sources["cache"] = "listagens servidas do cache local (< 24 h)"
                return out
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        out.b3, out.sources["B3"] = fetch_b3_listing(f, today)
    except RuntimeError as e:
        out.errors["B3"] = str(e)
    try:
        out.us, out.sources["EUA"] = fetch_us_listing(f)
    except RuntimeError as e:
        out.errors["EUA"] = str(e)
    if cache is not None and out.available:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"stored_at": time.time(), "b3": [asdict(x) for x in out.b3],
                                         "us": [asdict(x) for x in out.us], "sources": out.sources}),
                             encoding="utf-8")
        except OSError as e:
            log.warning("listing cache write failed", extra={"error": str(e)})
    return out
