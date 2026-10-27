"""Market of the day and latest prices: quotes read from the chart metadata (price, previous close, time of the last
trade, session open or closed, the delay the source reports), the market screen built from the exchanges' lists, and
the 'Ação do dia' at the latest price — test doubles only (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from core.market_data.base import DataUnavailableError, EmptyDataError
from core.market_data.quotes import parse_quote
from core.market_watch.overview import INDICES, MarketRequest, overview
from core.stock_check.daily_pick import apply_live_prices
from tests.market_doubles import (
    Prov,
    http,
    quote_json,
    quotes_double,
    quotes_from_market,
    structured_market,
)
from tests.test_universe import COTAHIST, _http, _zip
from webapp import server

NOW = datetime(2026, 10, 26, 17, 30, tzinfo=UTC)


def listings_any_day(url: str, timeout: float) -> tuple[int, bytes]:
    """The B3 and Nasdaq listing doubles, the B3 file served for whatever day is asked."""
    if "COTAHIST" in url:
        return 200, _zip(COTAHIST)
    return _http(url, timeout)


def no_listings(url: str, timeout: float) -> tuple[int, bytes]:
    return 503, b""


INDEX_BOOK = {"^BVSP": (120_000.0, 125_000.0, "BRL"), "USDBRL=X": (5.60, 5.40, "BRL"), "^GSPC": (6_000.0, 5_990.0, "USD"),
              "^IXIC": (20_000.0, 19_900.0, "USD")}                           # Brent left out: reported, not filled
STOCKS = {"PETR4.SA": (36.0, 40.0, "BRL"), "PETR3.SA": (39.0, 40.0, "BRL"), "RAIZ4.SA": (2.10, 2.00, "BRL"),
          "TINY3.SA": (10.0, 10.0, "BRL")}


def test_quote_from_the_chart_metadata():
    q = parse_quote("PETR4.SA", quote_json("PETR4.SA", 38.5, 40.0, "BRL", name="Petrobras"), NOW)
    assert q.price == 38.5 and q.previous_close == 40.0 and q.change_pct == pytest.approx(-0.0375)
    assert q.change == pytest.approx(-1.5) and q.currency == "BRL" and q.name == "Petrobras"
    assert q.time_local == "14:30" and q.date_local == "2026-10-26" and q.market_open is True and q.delay_min == 15
    assert len(q.intraday) == 6 and q.intraday[-1] == pytest.approx(38.5)
    closed = parse_quote("PETR4.SA", quote_json("PETR4.SA", 38.5, 40.0, "BRL"), datetime(2026, 10, 26, 22, 0, tzinfo=UTC))
    assert closed.market_open is False
    payload = quote_json("AAPL", 190.0, 188.0, "USD", delay=None)
    del payload["chart"]["result"][0]["meta"]["regularMarketPrice"]          # the last 5-minute close is used
    q2 = parse_quote("AAPL", payload, NOW)
    assert q2.price == pytest.approx(190.0) and q2.delay_min is None
    with pytest.raises(EmptyDataError):
        parse_quote("ZZZZ", {"chart": {"result": None, "error": {"description": "No data found"}}}, NOW)


def test_market_overview_from_the_b3_list(tmp_path):
    quotes = quotes_double({**INDEX_BOOK, **STOCKS})
    out = overview(MarketRequest(mercado="BR", quantidade=10), quotes, cache_dir=tmp_path, fetch=listings_any_day,
                   now=NOW)
    assert [r["ticker"] for r in out["cotacoes"]] == sorted(STOCKS) and "volume financeiro" in out["universo"]
    assert [r["ticker"] for r in out["baixas"]] == ["PETR4.SA", "PETR3.SA"]          # biggest drop first
    assert [r["ticker"] for r in out["altas"]] == ["RAIZ4.SA"] and out["altas"][0]["variacao_pct"] == pytest.approx(0.05)
    assert out["amplitude"] == {"sobem": 1, "caem": 2, "estaveis": 1, "variacao_media": pytest.approx(
        (-0.10 - 0.025 + 0.05 + 0.0) / 4)}
    idx = {r["ticker"]: r for r in out["indices"]}
    assert len(idx) == len(INDICES) and idx["^BVSP"]["variacao_pct"] == pytest.approx(-0.04)
    assert "erro" in idx["BZ=F"] and idx["USDBRL=X"]["preco"] == 5.60                # missing index said, not filled
    assert out["pregao_aberto"] is True and out["atraso_informado_min"] == [15] and "não é tempo real" in out["aviso"]
    assert out["mais_negociadas"][0]["volume_financeiro"] == pytest.approx(2_000_000 * 39.0)


def test_market_overview_without_the_lists_and_without_quotes(tmp_path):
    cat = {t: (50.0, 49.0, "BRL") for t in ("PETR4.SA", "VALE3.SA", "ITUB4.SA")}
    out = overview(MarketRequest(mercado="BR", quantidade=10), quotes_double({**INDEX_BOOK, **cat}),
                   cache_dir=tmp_path, fetch=no_listings, now=NOW)
    assert "catálogo do site" in out["universo"] and {r["ticker"] for r in out["cotacoes"]} == set(cat)
    assert len(out["sem_cotacao"]) == out["acompanhadas"] - 3                        # the rest: reported, not filled
    with pytest.raises(DataUnavailableError):
        overview(MarketRequest(mercado="US", quantidade=10), quotes_double({}), cache_dir=tmp_path, fetch=no_listings)


def test_market_endpoint(monkeypatch, tmp_path):
    quotes = quotes_double({**INDEX_BOOK, **STOCKS})
    monkeypatch.setattr(server, "_quote_source", lambda: (lambda progress: quotes))
    monkeypatch.setattr(server, "_rf_fetch", lambda: listings_any_day)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/mercado", json={"mercado": "BR", "quantidade": 10})
    assert r.status_code == 200, r.text
    assert r.json()["baixas"][0]["ticker"] == "PETR4.SA" and r.json()["do_cache"] is False
    assert c.post("/api/mercado", json={"mercado": "BR", "quantidade": 10}).json()["do_cache"] is True
    assert c.post("/api/mercado", json={"mercado": "XX"}).status_code == 422
    ev = [json.loads(x) for x in c.post("/api/mercado?stream=1", json={"mercado": "BR", "quantidade": 12}).text.splitlines()
          if x.strip()]
    assert ev[0]["tipo"] == "progresso" and ev[-1]["tipo"] == "resultado" and ev[-1]["dados"]["com_cotacao"] == 4
    monkeypatch.setattr(server, "_quote_source", lambda: (lambda progress: quotes_double({})))
    assert c.post("/api/mercado", json={"mercado": "US", "quantidade": 11}).status_code == 503
    ev = [json.loads(x) for x in c.post("/api/mercado?stream=1", json={"mercado": "US", "quantidade": 13}).text.splitlines()
          if x.strip()]
    assert ev[-1]["tipo"] == "erro" and ev[-1]["status"] == 503 and "cotações indisponíveis" in ev[-1]["detail"]


def test_page_has_the_market_screen_and_the_short_buy_lists():
    page = server._INDEX_HTML
    assert 'data-view="mercado"' in page and page.count('href="#/mercado"') >= 3 and '"/api/mercado"' in page
    assert '"mercado"' in page.split("const VIEW_NAMES")[1].split(";")[0]
    assert "não é tempo real" in page and "o motivo de uma alta ou queda está nas notícias" in page
    assert 'id="p-buy"' in page and 'id="p-full"' in page and "Preço agora" in page and "Quantas comprar" in page


PICK = {"origem": "real", "valor": 1_000.0, "avisos": ["aviso original"],
        "escolha": {"ticker": "WEGE3.SA", "nome": "WEG", "preco": 40.0, "quantidade": 25, "custo": 1_000.0, "sobra": 0.0},
        "alternativas": [{"ticker": "AAPL", "nome": "Apple", "preco": 1_000.0, "quantidade": 1, "custo": 1_000.0,
                          "sobra": 0.0},
                         {"ticker": "ITUB4.SA", "nome": "Itaú", "preco": 30.0, "quantidade": 33, "custo": 990.0,
                          "sobra": 10.0}]}


def test_ação_do_dia_at_the_latest_price():
    book = {"WEGE3.SA": (50.0, 40.0, "BRL"), "AAPL": (180.0, 182.0, "USD"), "USDBRL=X": (5.5, 5.5, "BRL")}
    d = apply_live_prices(PICK, quotes_double(book))
    p, a = d["escolha"], d["alternativas"]
    assert p["preco"] == 50.0 and p["quantidade"] == 20 and p["custo"] == 1_000.0 and p["preco_fechamento"] == 40.0
    assert p["ao_vivo"]["variacao_dia"] == pytest.approx(0.25) and p["ao_vivo"]["hora"] == "14:30"
    assert a[0]["preco"] == pytest.approx(990.0) and a[0]["quantidade"] == 1 and a[0]["ao_vivo"]["cambio"] == 5.5
    assert a[0]["ao_vivo"]["preco_original"] == 180.0 and a[0]["ao_vivo"]["moeda"] == "USD"
    assert a[1]["ao_vivo"] is None and "fechamento" in a[1]["ao_vivo_nota"]           # no quote: the close stays
    assert d["precos_ao_vivo"] is True and d["avisos"] == ["aviso original"]
    dear = apply_live_prices(PICK, quotes_double({**book, "WEGE3.SA": (1_200.0, 40.0, "BRL")}))
    assert dear["escolha"]["quantidade"] == 0 and "AAPL" in dear["avisos"][0]           # said, with what fits
    assert apply_live_prices({**PICK, "origem": "sintetico"}, quotes_double(book)) == {**PICK, "origem": "sintetico"}


def test_ação_do_dia_endpoint_with_latest_prices(monkeypatch, tmp_path):
    market = structured_market()
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(market))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "_quote_source", lambda: (lambda progress: quotes_double(quotes_from_market(market))))
    c = TestClient(server.app)
    d = c.post("/api/acao-do-dia", json={"valor": 5_000, "perfil": "moderado"}).json()
    p = d["escolha"]
    assert d["precos_ao_vivo"] is True and p["ao_vivo"]["variacao_dia"] == pytest.approx(0.01)
    assert p["custo"] <= 5_000 and p["preco"] == pytest.approx(p["preco_fechamento"] * 1.01, rel=0.05)
