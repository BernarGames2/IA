"""End-to-end runs shaped like production: full B3/US listings, Yahoo chart answers with the
quirks seen in the wild (the current day repeated as a separate "live" bar, missing volumes,
null prices, different holiday calendars, unknown tickers), the dollar converted to reais and
the web endpoints streaming. Network is never used: every HTTP answer is a test double.

Regression for: "cannot reindex on an axis with duplicate labels" on the "Completo" universe."""

import json
import zlib

import numpy as np
import pandas as pd
import pytest

from core.data_loader import convert_currency
from core.market_data.base import last_per_date
from core.market_data.cache import SeriesCache
from core.market_data.yahoo_chart import YahooChartProvider, parse_chart
from tests.test_universe import _http as listings_http

END = pd.Timestamp("2026-10-16")
BR_HOLIDAYS = {"2026-02-16", "2026-02-17", "2026-04-21", "2026-09-07"}
US_HOLIDAYS = {"2025-11-27", "2026-01-19", "2026-07-03", "2026-09-07"}
MISSING = {"LOPE", "MEDX", "RAIZ4.SA", "TINY3.SA"}          # listed, but the chart endpoint has nothing


def _chart(symbol: str) -> dict:
    """A Yahoo chart answer: daily bars, then today's bar AGAIN as a live bar (same local date)."""
    if symbol.endswith(".SA"):
        ccy, tz, hour, holidays = "BRL", "America/Sao_Paulo", "13:00", BR_HOLIDAYS
    elif symbol.endswith("=X"):
        ccy, tz, hour, holidays = "BRL", "Europe/London", "00:00", set()
    else:
        ccy, tz, hour, holidays = "USD", "America/New_York", "14:30", US_HOLIDAYS
    days = [d for d in pd.bdate_range(end=END, periods=1300) if d.strftime("%Y-%m-%d") not in holidays]
    ts = [int(pd.Timestamp(f"{d.date()}T{hour}Z").timestamp()) for d in days]
    ts.append(int(pd.Timestamp(f"{END.date()}T19:45Z").timestamp()))          # the live bar
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    level = 4.0 if symbol == "^IRX" else 5.2 if symbol.endswith("=X") else 40.0
    vol = 0.0 if symbol == "^IRX" else 0.006 if symbol.endswith("=X") else 0.017
    px = level * np.exp(np.cumsum(rng.normal(0.0003, vol, len(ts))))
    adj = [round(float(x), 4) for x in px]
    adj[len(adj) // 2] = None                                                # a null in the middle
    volume = [int(v) for v in rng.integers(1e5, 5e6, len(ts))]
    volume[-1] = None                                                        # live bar without volume
    return {"chart": {"result": [{
        "meta": {"currency": ccy, "exchangeName": "X", "instrumentType": "EQUITY", "exchangeTimezoneName": tz},
        "timestamp": ts, "indicators": {"quote": [{"close": adj, "volume": volume}], "adjclose": [{"adjclose": adj}]},
    }], "error": None}}


def yahoo_http(url: str, timeout: float) -> tuple[int, bytes]:
    sym = url.split("/chart/")[1].split("?")[0].replace("%5E", "^").replace("%3D", "=")
    if sym in MISSING:
        return 404, json.dumps({"chart": {"result": None, "error": {"code": "Not Found"}}}).encode()
    return 200, json.dumps(_chart(sym)).encode()


def test_parser_keeps_one_quote_per_day_the_latest():
    rs = parse_chart("PETR4.SA", _chart("PETR4.SA"))
    assert not rs.close.index.has_duplicates and not rs.volume.index.has_duplicates
    assert rs.close.index[-1] == END and any("repetida" in n for n in rs.notes)
    raw = _chart("PETR4.SA")["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
    assert rs.close.iloc[-1] == pytest.approx(raw[-1])                       # the live (latest) quote


def test_consumers_tolerate_repeated_dates_from_old_caches():
    idx = pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-16"])
    fx = pd.Series([5.0, 5.1, 5.2, 5.3], index=idx)
    out, missing = convert_currency(pd.Series([10.0, 11.0], index=idx[1:3]), fx)
    assert out.tolist() == pytest.approx([51.0, 58.3]) and missing == 0      # last quote of the 16th (5.3)
    assert last_per_date(fx).tolist() == [5.0, 5.1, 5.3]


@pytest.fixture()
def prod(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import webapp.server as srv

    monkeypatch.setattr(srv, "_market_provider", lambda: YahooChartProvider(
        cache=SeriesCache(tmp_path / "md", 12.0), fetch_fn=yahoo_http, sleep=lambda s: None, min_interval_s=0))
    monkeypatch.setattr(srv, "_rf_fetch", lambda: listings_http)
    monkeypatch.setattr(srv, "_OUTPUT_DIR", tmp_path)
    srv._purpose_cache.clear()
    return TestClient(srv.app)


def _stream(client, url, body):
    r = client.post(url + "?stream=1", json=body)
    assert r.status_code == 200, r.text
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    final = events[-1]
    assert final["tipo"] == "resultado", final
    return final["dados"]


def test_complete_universe_with_dollar_assets_and_live_bars(prod):
    body = {"valor_alvo": 900_000, "capital_inicial": 1, "aporte_periodico": 2_000, "periodicidade": "mensal",
            "horizonte_anos": 20, "perfil_risco": "moderado", "fonte_dados": "real", "universo": "completo",
            "max_ativos": 40, "n_simulacoes": 1_000}
    d = _stream(prod, "/api/proposito", body)
    assert d["dados_sinteticos"] is False and d["universo"]["funil"]["used_listings"]
    assert {a["moeda"] for a in d["onde_investir"]} >= {"BRL"} and d["cagr"]["necessario"] is not None
    assert d["universo"]["ultima_cotacao"] == "2026-10-16"
    excl = d["universo"]["excluidos_por_dados"]
    assert all(s in excl for s in ("LOPE", "MEDX") if s in d["universo"]["candidatos"])
    # starting from zero works on the same data
    d0 = _stream(prod, "/api/proposito", {**body, "capital_inicial": 0, "seed": 9})
    assert d0["entrada"]["capital_inicial"] == 0 and 0 <= d0["objetivo"]["p_objetivo"] <= 1


def test_single_stock_check_with_live_bars(prod):
    for q in ("PGMN3", "AAPL"):
        d = _stream(prod, "/api/acao", {"acao": q, "n_simulacoes": 1_000})
        assert d["veredito"]["rotulo"] in {"FAVORÁVEL", "NEUTRO", "DESFAVORÁVEL"} and d["periodo"][1] == "2026-10-16"
        assert d["por_que_pode_crescer"] or d["por_que_pode_nao_crescer_tanto"]


def test_library_failures_are_reported_as_internal_errors():
    from fastapi import HTTPException

    import webapp.server as srv

    try:
        pd.Series([1.0, 2.0], index=[0, 0]).reindex([0, 1])
    except ValueError as e:
        err = srv._analysis_error(e)
    assert isinstance(err, HTTPException) and err.status_code == 500 and "falha do sistema" in err.detail
    try:
        raise ValueError("horizonte deve estar entre 0,25 e 30 anos")
    except ValueError as e:
        err = srv._analysis_error(e)
    assert err.status_code == 400 and "horizonte" in err.detail
