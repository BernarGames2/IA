"""Full-market universe: exchange listings (B3 COTAHIST, Nasdaq), name search, pre-selection
funnel and the engine with the complete universe. Network is never used (test doubles)."""

import io
import json
import zipfile
from datetime import date

import pytest

from core.purpose.catalog import REAL_CATALOG
from core.universe.listings import (
    ListedAsset,
    MarketListings,
    fetch_b3_listing,
    load_market_listings,
    parse_cotahist,
    parse_nasdaq_screener,
    parse_symdir,
)
from core.universe.screen import (
    B3_MIN_VOLUME,
    CORE,
    listing_to_catalog,
    preselect,
    us_tags,
)


def cota_line(ticker, nomres, especi, codbdi="02", tpmerc="010", voltot=1_000_000_000, preult=3050):
    """One COTAHIST detail record (TIPREG 01) with the official fixed-width positions."""
    line = [" "] * 245

    def put(a, b, value, right=False, fill=" "):
        w = b - a + 1
        v = str(value)
        v = v.rjust(w, fill) if right else v.ljust(w, fill)
        line[a - 1:b] = list(v[:w])

    put(1, 2, "01")
    put(3, 10, "20261016")
    put(11, 12, codbdi)
    put(13, 24, ticker)
    put(25, 27, tpmerc)
    put(28, 39, nomres)
    put(40, 49, especi)
    put(109, 121, preult, right=True, fill="0")
    put(171, 188, voltot, right=True, fill="0")
    return "".join(line)


COTAHIST = "\n".join([
    "00COTAHIST.2026BOVESPA 20261016" + " " * 214,
    cota_line("PETR4", "PETROBRAS", "PN  N2", voltot=250_000_000_000),     # R$ 2.5 bi
    cota_line("PETR3", "PETROBRAS", "ON  N2", voltot=90_000_000_000),
    cota_line("RAIZ4", "RAIZEN", "PN  N2", voltot=4_000_000_000),            # R$ 40 mi
    cota_line("TINY3", "MICRO CAP", "ON  NM", voltot=100_000_00),             # R$ 100 mil: below the floor
    cota_line("KNRI11", "FII KINEA", "CI", codbdi="12", voltot=1_500_000_000),
    cota_line("BOVA11", "ISHARES BOVA", "CI", codbdi="14", voltot=9_000_000_000),
    cota_line("XPTO11", "ETF NOVO", "CI", codbdi="14", voltot=900_000_000),   # ETF outside the catalogue
    cota_line("AAPL34", "APPLE", "DRN", voltot=500_000_000),                 # BDR: dropped
    cota_line("PETR4F", "PETROBRAS", "PN  N2", tpmerc="020"),                # odd lot: dropped
    "99COTAHIST.2026BOVESPA 20261016" + " " * 214,
])

SCREENER = {"data": {"rows": [
    {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "marketCap": "3,000,000,000,000.00",
     "sector": "Technology", "industry": "Computer Manufacturing"},
    {"symbol": "BRK/B", "name": "Berkshire Hathaway Inc. Class B", "marketCap": "900,000,000,000",
     "sector": "Finance", "industry": "Property-Casualty Insurers"},
    {"symbol": "LOPE", "name": "Grand Canyon Education, Inc. Common Stock", "marketCap": "5,000,000,000",
     "sector": "Real Estate", "industry": "Other Consumer Services Educational"},
    {"symbol": "MEDX", "name": "Medix Health Corp Common Stock", "marketCap": "8,000,000,000",
     "sector": "Health Care", "industry": "Medical Specialities"},
    {"symbol": "SMOL", "name": "Small Co Common Stock", "marketCap": "50,000,000",
     "sector": "Health Care", "industry": "Biotechnology: Pharmaceutical Preparations"},
    {"symbol": "SPAC", "name": "Blue Acquisition Corp Warrant", "marketCap": "", "sector": "", "industry": ""},
    {"symbol": "ABC^A", "name": "ABC Preferred", "marketCap": "1,000,000,000", "sector": "", "industry": ""},
]}}


def _zip(text, name="COTAHIST_D16102026.TXT"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text.encode("latin-1"))
    return buf.getvalue()


def _http(url, timeout):
    if "COTAHIST_D16102026" in url:
        return 200, _zip(COTAHIST)
    if "COTAHIST" in url:
        return 404, b""
    if "api.nasdaq.com" in url:
        return 200, json.dumps(SCREENER).encode()
    if "bcb.gov.br" in url:
        return 200, b'[{"data":"16/10/2026","valor":"15.00"}]'
    return 404, b""


# ----------------------------------------------------------------------------- parsing
def test_parse_cotahist_keeps_shares_fiis_etfs_and_drops_the_rest():
    rows = {a.ticker: a for a in parse_cotahist(COTAHIST)}
    assert set(rows) == {"PETR4.SA", "PETR3.SA", "RAIZ4.SA", "TINY3.SA", "KNRI11.SA", "BOVA11.SA", "XPTO11.SA"}
    assert rows["PETR4.SA"].kind == "acao" and rows["PETR4.SA"].size == pytest.approx(2.5e9)
    assert rows["KNRI11.SA"].kind == "fii" and rows["BOVA11.SA"].kind == "etf"
    assert rows["RAIZ4.SA"].name == "RAIZEN PN" and rows["RAIZ4.SA"].currency == "BRL"


def test_fetch_b3_listing_walks_back_to_the_last_business_day():
    rows, src = fetch_b3_listing(_http, today=date(2026, 10, 19))       # Monday -> Friday's file
    assert len(rows) == 7 and "16/10/2026" in src
    with pytest.raises(RuntimeError, match="erro de rede"):
        fetch_b3_listing(lambda u, t: (_ for _ in ()).throw(OSError("proxy 403")), today=date(2026, 10, 19))


def test_parse_nasdaq_screener_and_symbol_directory():
    rows = {a.ticker: a for a in parse_nasdaq_screener(SCREENER)}
    assert set(rows) == {"AAPL", "BRK-B", "LOPE", "MEDX", "SMOL"}          # warrant and preferred dropped
    assert rows["AAPL"].size == pytest.approx(3e12) and rows["AAPL"].sector == "Technology"
    assert parse_nasdaq_screener([{"not": "a dict"}]) == []
    text = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\nZZZT|Test Co|Q|Y|N|100|N|N\n"
            "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\nFile Creation Time: 1017202600:00|||||||")
    sd = {a.ticker: a for a in parse_symdir(text, other=False)}
    assert set(sd) == {"AAPL", "QQQ"} and sd["QQQ"].kind == "etf" and sd["AAPL"].size is None


def test_load_market_listings_and_cache(tmp_path):
    lst = load_market_listings(tmp_path, _http, today=date(2026, 10, 19))
    assert lst.available and len(lst.b3) == 7 and len(lst.us) == 5 and not lst.errors

    def boom(url, timeout):
        raise AssertionError("must be served from cache")

    again = load_market_listings(tmp_path, boom, today=date(2026, 10, 19))
    assert len(again.all()) == 12 and "cache" in again.sources
    down = load_market_listings(None, lambda u, t: (_ for _ in ()).throw(OSError("blocked")))
    assert not down.available and set(down.errors) == {"B3", "EUA"}


def test_name_search_prefers_exact_then_size():
    lst = load_market_listings(None, _http, today=date(2026, 10, 19))
    assert lst.search("Petrobras").ticker == "PETR4.SA"                    # exact; larger volume wins
    assert lst.search("raízen").ticker == "RAIZ4.SA"
    assert lst.search("Grand Canyon").ticker == "LOPE"
    assert lst.search("ab") is None and lst.search("inexistente xyz") is None


# ----------------------------------------------------------------------------- funnel
def test_us_tags_and_conversion():
    assert us_tags("Health Care", "Medical Specialities") == ("saude",)
    assert us_tags("Technology", "Computer Manufacturing") == ("tecnologia",)
    assert "educacao" in us_tags("Real Estate", "Other Consumer Services Educational")
    c = listing_to_catalog(ListedAsset("XPTO11.SA", "ETF NOVO CI", "B3", "BRL", "etf", size=9e6))
    assert c is None                                                        # unknown ETF class: excluded
    b = listing_to_catalog(ListedAsset("RAIZ4.SA", "RAIZEN PN", "B3", "BRL", "acao", size=4e7))
    assert b.tags == () and "setor não informado" in b.business


def test_preselect_funnel_respects_floors_core_themes_and_cap():
    lst = load_market_listings(None, _http, today=date(2026, 10, 19))
    sel, fun = preselect(list(REAL_CATALOG), lst, {"saude"}, 30)
    tick = [a.ticker for a in sel]
    assert len(sel) == 30 and fun.used_listings and fun.listed_b3 == 7 and fun.listed_us == 5
    assert tick[:len(fun.core)] == [t for t in CORE if t in tick]
    assert "MEDX" in tick                                                   # listed health company -> theme slot
    assert "SMOL" not in tick and "TINY3.SA" not in tick and "XPTO11.SA" not in tick   # below floors / unknown ETF
    assert all(a.ticker in tick for a in sel) and len(set(tick)) == len(tick)
    assert fun.eligible > len(REAL_CATALOG)                                 # listings added eligible names
    assert "NÃO é usado" in fun.criteria
    _, none = preselect(list(REAL_CATALOG), MarketListings(errors={"B3": "x"}), set(), 20)
    assert not none.used_listings and none.errors == {"B3": "x"} and none.selected == 20
    only_fii, _ = preselect(list(REAL_CATALOG), lst, set(), 10, classes=["fii"])
    assert {a.macro_class for a in only_fii} == {"fii"}
    assert B3_MIN_VOLUME > 1e5


def test_resolver_uses_listings_for_unknown_names():
    from core.stock_check.resolver import is_plain_code, resolve

    lst = load_market_listings(None, _http, today=date(2026, 10, 19))
    assert is_plain_code(resolve("Raizen"))                                  # without listings: bare code
    r = resolve("Raizen", lst)
    assert r.ticker == "RAIZ4.SA" and "listagem" in r.how
    g = resolve("Grand Canyon", lst)
    assert g.ticker == "LOPE" and "Nasdaq" in (g.business or "")
    with pytest.raises(ValueError, match="não (encontrei|reconheci)"):
        resolve("Inexistentissima", lst)
    with pytest.raises(ValueError, match="não encontrei"):
        resolve("Zebrafoo", lst)                                            # one word, not a plausible ticker
    assert resolve("AAPL", lst).asset is not None                           # curated metadata still wins


# ----------------------------------------------------------------------------- engine
def test_engine_complete_universe_reports_the_funnel(tmp_path):
    from core.purpose.engine import PurposeRequest, run_purpose
    from tests.test_real_data import _fake_market, _Prov, _real_symbols

    req = PurposeRequest(capital_inicial=100_000, aporte_periodico=1_000, valor_alvo=250_000, horizonte_anos=5,
                         motivo_investidor="Quero investir em saúde.", offline=False, n_simulacoes=500,
                         max_ativos=20, universo="completo")
    r = run_purpose(req, output_dir=tmp_path, provider=_Prov(_fake_market(_real_symbols())), cache_dir=tmp_path,
                    rf_fetch=_http, fast=True)
    fun = r.universe["funil"]
    assert fun["used_listings"] and fun["listed_b3"] == 7 and fun["selected"] == 20
    assert any("todas as ações listadas" in a.message for a in r.alerts)
    assert len(r.universe["candidatos"]) == 20
    # listing-only names without price history are excluded and reported, never proxied
    for why in r.universe["excluidos_por_dados"].values():
        assert "download failure" in why or "history" in why


def test_web_universe_limits():
    from fastapi.testclient import TestClient

    import webapp.server as srv

    c = TestClient(srv.app)
    assert c.post("/api/proposito", json={"valor_alvo": 1e5, "max_ativos": 61}).status_code == 422
    assert c.post("/api/proposito", json={"valor_alvo": 1e5, "universo": "tudo"}).status_code == 422
