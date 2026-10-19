"""Real-data path: Yahoo chart provider (stdlib), risk-free sources, and the purpose engine /
web endpoint wired to them. Network is never used: HTTP and providers are test doubles."""

import json

import numpy as np
import pandas as pd
import pytest

from core.market_data.base import DownloadError, RawSeries
from core.market_data.cache import SeriesCache
from core.market_data.risk_free import fetch_live_rates, fetch_selic, fetch_us_tbill
from core.market_data.yahoo_chart import YahooChartProvider, chart_url, parse_chart


def _payload(ts, adj, close=None, volume=None, currency="BRL", tz="America/Sao_Paulo"):
    ind = {"quote": [{"close": close if close is not None else adj, "volume": volume or [1000] * len(ts)}]}
    if adj is not None:
        ind["adjclose"] = [{"adjclose": adj}]
    return {"chart": {"result": [{"meta": {"currency": currency, "exchangeName": "SAO", "instrumentType": "EQUITY",
                                           "exchangeTimezoneName": tz},
                                  "timestamp": ts, "indicators": ind}], "error": None}}


def _ts(s):
    return int(pd.Timestamp(s).timestamp())


# ----------------------------------------------------------------------------- parsing
def test_parse_chart_uses_adjclose_local_dates_and_drops_nulls():
    ts = [_ts("2026-10-14T13:00Z"), _ts("2026-10-15T13:00Z"), _ts("2026-10-16T13:00Z")]
    rs = parse_chart("PETR4.SA", _payload(ts, [30.5, None, 31.2], close=[32.0, 32.1, 33.0]))
    assert list(rs.close.index.strftime("%Y-%m-%d")) == ["2026-10-14", "2026-10-16"]
    assert rs.close.tolist() == [30.5, 31.2]                       # adjusted, not raw close
    assert rs.price_field == "adjclose" and "dividendos" in rs.adjusted_for and rs.currency == "BRL"


def test_parse_chart_exchange_timezone_sets_the_calendar_date():
    # 02:30 UTC on the 16th is still the 15th in New York
    rs = parse_chart("AAPL", _payload([_ts("2026-10-16T02:30Z")], [100.0], currency="USD", tz="America/New_York"))
    assert str(rs.close.index[0].date()) == "2026-10-15" and rs.currency == "USD"


def test_parse_chart_without_adjclose_says_so_and_errors_are_explicit():
    rs = parse_chart("USDBRL=X", _payload([_ts("2026-10-16T12:00Z")], None, close=[5.1]))
    assert rs.price_field == "close" and "somente desdobramentos" in rs.adjusted_for
    with pytest.raises(Exception, match="delisted"):
        parse_chart("XXXX", {"chart": {"result": None, "error": {"code": "Not Found",
                                                                 "description": "symbol may be delisted"}}})


def test_chart_url_rejects_unsafe_symbols():
    assert "PETR4.SA" in chart_url("PETR4.SA", "5y", "1d") and "range=5y" in chart_url("PETR4.SA", "5y", "1d")
    with pytest.raises(Exception, match="inválido"):
        chart_url("../../etc/passwd", "5y", "1d")


# ----------------------------------------------------------------------------- provider
class _Http:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


OK_BODY = json.dumps(_payload([_ts("2026-10-15T13:00Z"), _ts("2026-10-16T13:00Z")], [10.0, 10.5])).encode()


def test_provider_retries_429_then_succeeds_alternating_hosts():
    http, sleeps = _Http([(429, b""), (200, OK_BODY)]), []
    p = YahooChartProvider(fetch_fn=http, sleep=sleeps.append, min_interval_s=0)
    got = p.fetch(["VALE3.SA"], "5y", "1d")
    assert got["VALE3.SA"].close.tolist() == [10.0, 10.5] and p.failures == {}
    assert "query1" in http.urls[0] and "query2" in http.urls[1] and 1.0 in sleeps


def test_provider_404_is_definitive_and_network_errors_exhaust_retries():
    p = YahooChartProvider(fetch_fn=_Http([(404, b"{}")]), sleep=lambda s: None, min_interval_s=0)
    assert p.fetch(["NAOEXISTE3.SA"], "5y", "1d") == {} and "404" in p.failures["NAOEXISTE3.SA"]
    assert p.n_requests == 1
    q = YahooChartProvider(fetch_fn=_Http([OSError("CONNECT tunnel failed, response 403")]), sleep=lambda s: None,
                           min_interval_s=0, max_retries=3)
    syms = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "EGIE3.SA"]
    assert q.fetch(syms, "5y", "1d") == {}
    assert "erro de rede" in q.failures["PETR4.SA"]
    assert q.n_requests == 3 * 2                     # 2 quick attempts x 3 tickers, then the circuit opens
    assert "não tentado" in q.failures["EGIE3.SA"]
    r = YahooChartProvider(fetch_fn=_Http([(200, b"<html>")]), sleep=lambda s: None, min_interval_s=0, max_retries=1)
    assert r.fetch(["PETR4.SA"], "5y", "1d") == {} and "resposta inválida" in r.failures["PETR4.SA"]


def test_provider_cache_avoids_second_download(tmp_path):
    http = _Http([(200, OK_BODY)])
    p = YahooChartProvider(cache=SeriesCache(tmp_path), fetch_fn=http, sleep=lambda s: None, min_interval_s=0)
    p.fetch(["WEGE3.SA"], "5y", "1d")
    again = YahooChartProvider(cache=SeriesCache(tmp_path), fetch_fn=_Http([OSError("offline")]),
                               sleep=lambda s: None, min_interval_s=0).fetch(["WEGE3.SA"], "5y", "1d")
    assert again["WEGE3.SA"].cache_hit and again["WEGE3.SA"].close.tolist() == [10.0, 10.5]
    assert len(http.urls) == 1


# ----------------------------------------------------------------------------- risk-free
def test_fetch_selic_parses_bcb_and_rejects_bad_answers():
    q = fetch_selic(lambda url, t: (200, b'[{"data":"16/10/2026","valor":"15.00"}]'))
    assert q.rate == pytest.approx(0.15) and q.as_of == "16/10/2026" and "SGS 432" in q.source
    assert fetch_selic(lambda url, t: (200, b'[{"data":"16/10/2026","valor":"14,25"}]')).rate == pytest.approx(0.1425)
    for bad in ((500, b""), (200, b"[]"), (200, b'[{"data":"x","valor":"250"}]')):
        with pytest.raises(DownloadError):
            fetch_selic(lambda url, t, b=bad: b)


class _Prov:
    """Minimal MarketDataProvider test double."""

    name = "fake"

    def __init__(self, series):
        self.series = series
        self.failures = {}

    def fetch(self, symbols, period, interval):
        return {s: self.series[s] for s in symbols if s in self.series}


def test_fetch_us_tbill_and_live_rates():
    irx = RawSeries("^IRX", pd.Series([4.10, 4.05], index=pd.to_datetime(["2026-10-15", "2026-10-16"])), None,
                    "fake", "adjclose", "n/a", "USD", "1d")
    assert fetch_us_tbill(_Prov({"^IRX": irx})).rate == pytest.approx(0.0405)
    quotes, errors = fetch_live_rates({"BRL", "USD", "EUR"}, _Prov({"^IRX": irx}),
                                      lambda url, t: (200, b'[{"data":"16/10/2026","valor":"15.00"}]'))
    assert set(quotes) == {"BRL", "USD"} and "sem fonte automática" in errors["EUR"]
    _, err2 = fetch_live_rates({"USD"}, _Prov({}))
    assert "USD" in err2


# ----------------------------------------------------------------------------- engine + web (test doubles)
def _fake_market(symbols_ccy, n=1300, end="2026-10-16"):
    idx = pd.bdate_range(end=end, periods=n)
    out = {}
    for i, (sym, ccy) in enumerate(sorted(symbols_ccy.items())):
        rng = np.random.default_rng(1000 + i)
        if sym == "^IRX":
            x = pd.Series(np.full(n, 4.0), index=idx)
        elif sym.endswith("=X"):
            x = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.008, n))), index=idx)
        else:
            vol = 0.004 if sym in ("IMAB11.SA", "B5P211.SA", "AGG") else 0.017
            x = pd.Series(50 * np.exp(np.cumsum(rng.normal(0.0004, vol, n))), index=idx)
        out[sym] = RawSeries(sym, x.rename(sym), None, "fake_test_double", "adjclose", "teste", ccy, "1d",
                             ingested_at_utc="2026-10-16T21:00:00+00:00")
    return out


def _real_symbols():
    from core.purpose.catalog import REAL_CATALOG

    m = {a.ticker: a.currency for a in REAL_CATALOG}
    m.update({"USDBRL=X": "BRL", "^IRX": "USD"})
    return m


def test_engine_real_mode_reports_where_to_invest_and_why(tmp_path):
    from core.purpose.engine import PurposeRequest, run_purpose
    from core.purpose.report import render_markdown

    req = PurposeRequest(capital_inicial=100_000, aporte_periodico=1_000, valor_alvo=250_000, horizonte_anos=5,
                         motivo_investidor="Quero investir em saúde e tecnologia.", offline=False,
                         classes=["acoes_br", "acoes_eua", "renda_fixa_br"], n_simulacoes=500)
    r = run_purpose(req, output_dir=tmp_path, provider=_Prov(_fake_market(_real_symbols())), cache_dir=tmp_path,
                    rf_fetch=lambda url, t: (200, b'[{"data":"16/10/2026","valor":"15.00"}]'))
    assert not r.is_synthetic and r.is_final_portfolio and r.universe["origem"].startswith("real")
    assert "Banco Central" in r.rf_source["BRL"] and "^IRX" in r.rf_source["USD"]
    assert r.rf_by_currency == {"BRL": pytest.approx(0.15), "USD": pytest.approx(0.04)}
    assert not any(a.category == "dados" and a.severity == "FAIL" for a in r.alerts)
    assert any("DADOS REAIS" in a.message for a in r.alerts)
    alloc = r.allocation
    assert alloc and sum(a["valor_inicial"] for a in alloc) == pytest.approx(100_000)
    assert sum(a["aporte_por_periodo"] for a in alloc) == pytest.approx(1_000)
    assert all(a["por_que"] and a["negocio"] and a["ticker"] in r.table.index for a in alloc)
    assert [a["peso"] for a in alloc] == sorted((a["peso"] for a in alloc), reverse=True)
    assert "Carteira-base quantitativa" in r.strategy_text
    md = render_markdown(r)
    assert "Onde investir e por quê" in md and "Dados reais" in md and "SINTÉTICOS" not in md


def test_engine_real_mode_without_data_fails_loudly(tmp_path):
    from core.market_data.base import DataUnavailableError
    from core.purpose.engine import PurposeRequest, run_purpose

    req = PurposeRequest(capital_inicial=1_000, valor_alvo=2_000, horizonte_anos=5, offline=False, n_simulacoes=500)
    with pytest.raises(DataUnavailableError):
        run_purpose(req, output_dir=tmp_path, provider=_Prov({}), cache_dir=tmp_path,
                    rf_fetch=lambda url, t: (500, b""))


def test_web_real_mode_and_unavailable_data(monkeypatch):
    from fastapi.testclient import TestClient

    import webapp.server as srv

    client = TestClient(srv.app)
    body = {"valor_alvo": 300_000, "capital_inicial": 100_000, "horizonte_anos": 5, "n_simulacoes": 500,
            "motivo_investidor": "saúde", "fonte_dados": "real"}
    monkeypatch.setattr(srv, "_market_provider", lambda: _Prov({}))
    monkeypatch.setattr(srv, "_rf_fetch", lambda: (lambda url, t: (500, b"")))
    r = client.post("/api/proposito", json=body)
    assert r.status_code == 503 and "dados reais indisponíveis" in r.json()["detail"]
    assert "Nenhum dado sintético" in r.json()["detail"]
    monkeypatch.setattr(srv, "_market_provider", lambda: _Prov(_fake_market(_real_symbols())))
    monkeypatch.setattr(srv, "_rf_fetch", lambda: (lambda url, t: (200, b'[{"data":"16/10/2026","valor":"15.00"}]')))
    d = client.post("/api/proposito", json={**body, "seed": 7}).json()
    assert d["dados_sinteticos"] is False and d["carteira_final"] is True
    assert d["onde_investir"] and {"ticker", "nome", "peso", "valor_inicial", "por_que"} <= set(d["onde_investir"][0])
    assert d["universo"]["ultima_cotacao"] == "2026-10-16"


# ----------------------------------------------------------------------------- robustness for web use
def test_provider_parallel_after_probe_and_throttling_breaker():
    seen = []

    def ok(url, timeout):
        seen.append(url)
        return 200, OK_BODY

    syms = [f"T{i:02d}" for i in range(12)]
    p = YahooChartProvider(fetch_fn=ok, sleep=lambda s: None, min_interval_s=0, max_workers=4)
    got = p.fetch(syms, "5y", "1d")
    assert set(got) == set(syms) and p.failures == {} and len(seen) == 12

    q = YahooChartProvider(fetch_fn=_Http([(429, b"")]), sleep=lambda s: None, min_interval_s=0, max_retries=2)
    assert q.fetch(syms, "5y", "1d") == {}
    assert q.n_requests == 3 * 3                       # 3 probes x 3 attempts, then the breaker opens
    assert "HTTP 429" in q.failures["T00"] and "não tentado" in q.failures["T11"]


def test_provider_time_budget_marks_unstarted_symbols():
    now = {"t": 0.0}

    def slow(url, timeout):
        now["t"] += 4.0                                 # every request "takes" 4 s on the fake clock
        return 200, OK_BODY

    p = YahooChartProvider(fetch_fn=slow, sleep=lambda s: None, min_interval_s=0, max_workers=1,
                           time_budget_s=10.0, clock=lambda: now["t"])
    got = p.fetch(["A1", "A2", "A3", "A4", "A5"], "5y", "1d")
    assert len(got) == 3 and "tempo de download esgotado" in p.failures["A5"]


def test_provider_reports_progress():
    msgs = []
    p = YahooChartProvider(fetch_fn=_Http([(200, OK_BODY)]), sleep=lambda s: None, min_interval_s=0,
                           progress=msgs.append)
    p.fetch(["X1", "X2"], "5y", "1d")
    assert len(msgs) == 2 and all(m.startswith("cotações ") for m in msgs) and "2/2" in msgs[-1]


def _events(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_streamed_endpoints_end_with_result_or_error(monkeypatch):
    from fastapi.testclient import TestClient

    import webapp.server as srv

    c = TestClient(srv.app)
    r = c.post("/api/acao?stream=1", json={"acao": "SINT_SAUDE3", "fonte_dados": "sintetico", "n_simulacoes": 500})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    ev = _events(r.text)
    assert ev[0]["tipo"] == "progresso" and ev[-1]["tipo"] == "resultado"
    assert ev[-1]["dados"]["ticker"] == "SINT_SAUDE3" and any("simulando" in e.get("msg", "") for e in ev)
    monkeypatch.setattr(srv, "_market_provider", lambda: _Prov({}))
    monkeypatch.setattr(srv, "_rf_fetch", lambda: (lambda url, t: (500, b"")))
    ev = _events(c.post("/api/proposito?stream=1", json={"valor_alvo": 300_000, "n_simulacoes": 500,
                                                         "seed": 11}).text)
    assert ev[-1]["tipo"] == "erro" and ev[-1]["status"] == 503 and "indisponíveis" in ev[-1]["detail"]
    ev = _events(c.post("/api/analisar?stream=1", json={"profile": "conservador", "simulations": 500,
                                                        "horizon_years": 1}).text)
    assert ev[-1]["tipo"] == "resultado" and "meta" in ev[-1]["dados"]
    assert c.post("/api/acao?stream=1", json={"acao": "SINT_SAUDE3", "valor": -1}).status_code == 422
