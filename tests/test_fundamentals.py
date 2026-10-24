"""Fundamentals over time, value-trap detector, structured thesis and opportunity radar, on real-data paths with
test doubles of every source (Yahoo statements, estimates and calendar; B3; BCB; prices) — no network."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, date, datetime
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.fundamentals import radar
from core.fundamentals.analysis import fcf, growth, piotroski, valuation, year_ratios
from core.fundamentals.sources import (
    FIELDS,
    Analysts,
    fetch_analysts,
    fetch_annual,
    parse_analysts,
    parse_timeseries,
)
from core.fundamentals.thesis import build_thesis
from core.fundamentals.value_trap import detect, price_stats
from core.purpose.language import guarantee_phrases
from core.stock_check.company import CompanyProfile, YahooFundamentals
from core.stock_check.sectors import SEGMENTS
from tests.market_doubles import END, Prov, cdi_json, structured_market
from webapp import server

YEARS = ("2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31")
B = 1e9
DETERIORATING = {  # cheap and getting worse
    "receita": [500 * B, 480 * B, 450 * B, 420 * B], "lucro_bruto": [250 * B, 230 * B, 200 * B, 180 * B],
    "lucro_operacional": [150 * B, 130 * B, 110 * B, 90 * B], "lucro_liquido": [100 * B, 80 * B, 60 * B, 40 * B],
    "ebitda": [200 * B, 180 * B, 150 * B, 100 * B], "fco": [140 * B, 110 * B, 70 * B, 30 * B],
    "capex": [-60 * B, -60 * B, -65 * B, -70 * B], "divida_total": [200 * B, 250 * B, 300 * B, 380 * B],
    "caixa": [50 * B, 40 * B, 30 * B, 20 * B], "patrimonio": [400 * B, 390 * B, 370 * B, 340 * B],
    "ativos": [900 * B, 920 * B, 950 * B, 1000 * B], "ativo_circulante": [200 * B, 190 * B, 170 * B, 150 * B],
    "passivo_circulante": [150 * B, 155 * B, 160 * B, 170 * B], "acoes": [13.0 * B, 13.2 * B, 13.5 * B, 13.9 * B],
    "dividendos_pagos": [-50 * B, -45 * B, -40 * B, -35 * B], "lpa": [7.7, 6.1, 4.4, 2.9]}
GROWER = {  # expensive, growing, improving
    "receita": [30 * B, 34.5 * B, 39.7 * B, 45.6 * B], "lucro_bruto": [9 * B, 10.7 * B, 12.7 * B, 15.0 * B],
    "lucro_operacional": [5 * B, 6.2 * B, 7.5 * B, 9.1 * B], "lucro_liquido": [4 * B, 5 * B, 6.1 * B, 7.4 * B],
    "ebitda": [5.8 * B, 7.1 * B, 8.5 * B, 10.3 * B], "fco": [4.5 * B, 5.8 * B, 7 * B, 8.5 * B],
    "fcl": [3 * B, 4 * B, 5 * B, 6.2 * B], "divida_total": [3 * B, 2.8 * B, 2.5 * B, 2.2 * B],
    "caixa": [5 * B, 6 * B, 7 * B, 8 * B], "patrimonio": [15 * B, 18 * B, 21 * B, 25 * B],
    "ativos": [35 * B, 38 * B, 42 * B, 47 * B], "ativo_circulante": [20 * B, 22 * B, 25 * B, 28 * B],
    "passivo_circulante": [10 * B, 10.5 * B, 11 * B, 11.5 * B], "acoes": [4.19 * B] * 4,
    "dividendos_pagos": [-1.5 * B, -1.8 * B, -2.2 * B, -2.6 * B], "lpa": [0.95, 1.19, 1.46, 1.77]}
SNAP = {"PETR4.SA": (20.0, 13.9 * B, 40 * B, 35 * B), "WEGE3.SA": (50.0, 4.19 * B, 7.4 * B, 2.6 * B)}


def _ts(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def ts_json(sym: str, annual: dict[str, list[float | None]], ccy: str = "BRL") -> dict:
    result = []
    for key, yahoo in FIELDS.items():
        vals = annual.get(key)
        if vals is None:
            continue
        t = "annual" + yahoo
        rows = [None if v is None else {"asOfDate": y, "periodType": "12M", "currencyCode": ccy,
                                        "reportedValue": {"raw": v, "fmt": "x"}} for y, v in zip(YEARS, vals, strict=True)]
        result.append({"meta": {"symbol": [sym], "type": [t]}, "timestamp": [_ts(y) for y in YEARS], t: rows})
    return {"timeseries": {"result": result, "error": None}}


def snap_json(price: float, shares: float, ni: float, div: float, ccy: str = "BRL") -> dict:
    mcap = price * shares
    return {"quoteSummary": {"result": [{
        "assetProfile": {"industry": "x", "sector": "y"},
        "financialData": {"currentPrice": {"raw": price}, "financialCurrency": ccy, "revenueGrowth": {"raw": 0.05},
                          "earningsGrowth": {"raw": 0.02}, "returnOnEquity": {"raw": 0.12}},
        "defaultKeyStatistics": {},
        "summaryDetail": {"marketCap": {"raw": mcap}, "trailingPE": {"raw": mcap / ni}, "currency": ccy,
                          "dividendYield": {"raw": div / mcap}}}], "error": None}}


def analysts_json(cur: float, d90: float, up: int, down: int, growth_: float, n: int = 9) -> dict:
    return {"quoteSummary": {"result": [{
        "earningsTrend": {"trend": [
            {"period": "0y", "growth": {"raw": growth_}, "earningsEstimate": {"numberOfAnalysts": {"raw": n}},
             "epsTrend": {"current": {"raw": cur}, "7daysAgo": {"raw": cur}, "30daysAgo": {"raw": cur},
                          "60daysAgo": {"raw": d90}, "90daysAgo": {"raw": d90}},
             "epsRevisions": {"upLast30days": {"raw": up}, "downLast30days": {"raw": down}}},
            {"period": "+1y", "epsTrend": {"current": {"raw": cur * 1.1}}}]},
        "recommendationTrend": {"trend": [{"period": "0m", "strongBuy": 1, "buy": 3, "hold": 4, "sell": 1, "strongSell": 0}]},
        "calendarEvents": {"earnings": {"earningsDate": [{"raw": _ts("2026-11-06")}]},
                           "exDividendDate": {"raw": _ts("2026-08-20")}, "dividendDate": {"raw": _ts("2026-12-15")}}}],
        "error": None}}


def _generic(sym: str) -> tuple[dict, tuple]:
    """Deterministic statements for any other ticker (for the radar)."""
    rng = np.random.default_rng(int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16))
    g, m = rng.uniform(-0.08, 0.2), rng.uniform(0.04, 0.3)
    rev = [10 * B * (1 + g) ** i for i in range(4)]
    ni = [r * m * rng.uniform(0.4, 0.7) for r in rev]
    ann = {"receita": rev, "lucro_bruto": [r * (m + 0.2) for r in rev], "lucro_operacional": [r * m for r in rev],
           "lucro_liquido": ni, "ebitda": [r * (m + 0.05) for r in rev], "fco": [x * rng.uniform(0.7, 1.4) for x in ni],
           "capex": [-r * 0.04 for r in rev], "divida_total": [rev[0] * rng.uniform(0.1, 1.5)] * 4, "caixa": [rev[0] * 0.1] * 4,
           "patrimonio": [rev[0] * 0.8 * (1.05 ** i) for i in range(4)], "ativos": [rev[0] * 2 * (1.04 ** i) for i in range(4)],
           "ativo_circulante": [rev[0] * 0.6] * 4, "passivo_circulante": [rev[0] * 0.4] * 4, "acoes": [1 * B] * 4,
           "dividendos_pagos": [-x * 0.4 for x in ni]}
    price = float(rng.uniform(5, 60))
    return ann, (price, 1 * B, ni[-1], ni[-1] * 0.4)


class Router:
    """Fake HTTP: BCB (Selic, CDI), B3 (down), Yahoo crumb, statements, snapshot and estimates."""

    def __init__(self, yahoo: bool = True, slow: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.yahoo, self.slow = yahoo, set(slow or ())

    def __call__(self, url: str, timeout: float) -> tuple[int, bytes]:
        self.calls.append(url)
        if "bcdata.sgs.12" in url:
            return 200, cdi_json(n=2600)
        if "bcb.gov.br" in url:
            return 200, b'[{"data":"16/10/2026","valor":"15.00"}]'
        if "sistemaswebb3" in url:
            return 503, b""
        if "fc.yahoo.com" in url:
            return 404, b""
        if "getcrumb" in url:
            return (200, b"c1") if self.yahoo else (429, b"")
        sym = url.split("?")[0].rsplit("/", 1)[1].replace("%2E", ".").replace("%5E", "^")
        if sym in self.slow:
            time.sleep(2.5)
        if "fundamentals-timeseries" in url:
            ann = {"PETR4.SA": DETERIORATING, "WEGE3.SA": GROWER}.get(sym) or _generic(sym)[0]
            return 200, json.dumps(ts_json(sym, ann)).encode()
        if "quoteSummary" in url and "earningsTrend" in url:
            if sym == "PETR4.SA":
                return 200, json.dumps(analysts_json(4.5, 5.2, 1, 5, -0.12)).encode()
            return 200, json.dumps(analysts_json(1.9, 1.75, 4, 0, 0.15)).encode()
        if "quoteSummary" in url:
            p = SNAP.get(sym) or _generic(sym)[1]
            return 200, json.dumps(snap_json(*p)).encode()
        return 404, b""


def _annual(d: dict) -> object:
    return parse_timeseries(ts_json("X", d), "2026-10-16")


# ----------------------------------------------------------------------------- parsing and ratios
def test_statements_parsing_and_ratios_by_hand():
    raw = ts_json("X", DETERIORATING)
    raw["timeseries"]["result"][0]["annualTotalRevenue"][1] = None                      # a missing year stays missing
    raw["timeseries"]["result"][0]["annualTotalRevenue"].append(
        {"asOfDate": "2026-01-02", "currencyCode": "BRL", "reportedValue": {"raw": 421 * B}})  # restated 2 days later
    a = parse_timeseries(raw, "2026-10-16")
    assert len(a.anos) == 4 and a.anos[-1] == "2026-01-02" and a.moeda == "BRL"
    assert a.get("receita", 1) is None and a.get("receita", -1) == 421 * B
    d = _annual(DETERIORATING)
    r = year_ratios(d, -1)
    assert r["margem_operacional"] == pytest.approx(90 / 420)
    assert r["roe"] == pytest.approx(40 / ((340 + 370) / 2))
    assert r["fcl"] == pytest.approx(-40 * B) == fcf(d, -1)                          # 30 - 70, no published FCF
    assert r["divida_liquida_ebitda"] == pytest.approx((380 - 20) / 100)
    assert r["cobertura_dividendos"] == pytest.approx(-40 / 35)
    g = growth(d)
    assert g["receita_cagr_3a"] == pytest.approx((420 / 500) ** (1 / 3) - 1)
    assert g["acoes_variacao"] == pytest.approx(13.9 / 13.0 - 1)
    v = valuation(d, {"valor_mercado": 278 * B, "pl": 6.95, "moeda_cotacao": "BRL"})
    assert v["p_vp"] == pytest.approx(278 / 340) and v["ev_ebitda"] == pytest.approx((278 + 380 - 20) / 100)
    assert v["fcl_yield"] == pytest.approx(-40 / 278) and v["lucro_yield"] == pytest.approx(1 / 6.95)
    assert valuation(d, {"valor_mercado": 1e9, "moeda_cotacao": "USD"})["p_vp"] is None   # never mixes currencies
    with pytest.raises(Exception, match="formato"):
        parse_timeseries({"x": 1}, "2026-10-16")


def test_piotroski_by_hand():
    good = piotroski(_annual(GROWER))
    assert good["pontos"] == 9 and good["medidos"] == 9
    bad = piotroski(_annual(DETERIORATING))
    pts = {t["teste"]: t["ponto"] for t in bad["testes"]}
    assert bad["pontos"] == 2 and pts["lucro sobre ativos positivo"] == 1 and pts["geração de caixa operacional positiva"] == 1
    assert pts["caixa operacional maior que o lucro"] == 0 and pts["sem emissão de ações"] == 0
    thin = piotroski(_annual({"receita": [1, 2, 3, 4]}))
    assert thin["medidos"] < 6


def test_analysts_parsing():
    an = parse_analysts(analysts_json(4.5, 5.2, 1, 5, -0.12)["quoteSummary"]["result"][0], "2026-10-16")
    assert an.eps_ano["current"] == 4.5 and an.eps_ano["90daysAgo"] == 5.2 and an.revisoes_30d == {"alta": 1, "baixa": 5}
    assert an.n_analistas == 9 and an.proximo_resultado == "2026-11-06" and an.data_com == "2026-08-20"
    assert an.recomendacoes["hold"] == 4 and "terceiros" in an.fonte


# ----------------------------------------------------------------------------- value trap
def _px(trend: float, n: int = 1300) -> pd.Series:
    return pd.Series(50 * np.exp(np.linspace(0, trend, n)), index=pd.bdate_range(end=END, periods=n))


def test_value_trap_levels():
    d = _annual(DETERIORATING)
    snap = {"valor_mercado": 278 * B, "pl": 6.95, "dividend_yield": 35 / 278, "moeda_cotacao": "BRL"}
    an = parse_analysts(analysts_json(4.5, 5.2, 1, 5, -0.12)["quoteSummary"]["result"][0], "x")
    t = detect(d, snap, an, price_stats(_px(-2.0)), financial=False)
    assert t["parece_barata"] and t["nivel"] == "alto" and t["acionados"] >= 8
    hit = {s["sinal"]: s["acionado"] for s in t["sinais"]}
    assert hit["receita em queda"] and hit["dívida alta"] and hit["dividendos acima do caixa livre"]
    assert hit["analistas cortando estimativas (terceiros)"] and hit["preço em tendência de queda"]
    assert hit["dividendos altos demais"] is False and any("P/L de 7" in c for c in t["criterios_barata"])
    g = _annual(GROWER)
    healthy_cheap = detect(g, {"valor_mercado": 45 * B, "pl": 6.1, "moeda_cotacao": "BRL"}, None, price_stats(_px(0.5)))
    assert healthy_cheap["parece_barata"] and healthy_cheap["nivel"] == "baixo"
    pricey = detect(g, {"valor_mercado": 209.5 * B, "pl": 28.3, "moeda_cotacao": "BRL"}, None, price_stats(_px(0.5)))
    assert not pricey["parece_barata"] and pricey["nivel"].startswith("não se aplica")
    bank = detect(_annual({"receita": [1e9] * 4, "lucro_liquido": [2e8] * 4}), {"pl": 5.0}, None, {}, financial=True)
    assert bank["nivel"] == "indeterminado" and bank["piotroski"] is None
    assert not guarantee_phrases(json.dumps(t, ensure_ascii=False))


# ----------------------------------------------------------------------------- thesis
def test_thesis_structure_and_invalidation():
    d = _annual(DETERIORATING)
    snap = {"valor_mercado": 278 * B, "pl": 6.95, "dividend_yield": 35 / 278, "moeda_cotacao": "BRL"}
    an = parse_analysts(analysts_json(4.5, 5.2, 1, 5, -0.12)["quoteSummary"]["result"][0], "2026-10-16")
    ps = price_stats(_px(-2.0))
    trap = detect(d, snap, an, ps)
    p = CompanyProfile("PETR4.SA", "Petrobras PN", "petróleo", "catálogo", SEGMENTS["petroleo"], "catálogo",
                       fundamentos=snap, fundamentos_fonte="Yahoo (teste)", pais="BR")
    th = build_thesis(p, d, an, None, trap, ps, date(2026, 10, 16))
    assert th["contra_tese"].startswith("O preço baixo pode ser merecido")
    assert [c["data"] for c in th["catalisadores"]] == ["2026-11-06", "2026-12-15"]      # the past ex-date is left out
    inval = {i["condicao"]: i for i in th["invalidacao"]}
    assert inval["receita do próximo ano fiscal mais de 5% abaixo da última"]["limite"] == "BRL 399,0 bi"
    assert any(x["base"] == "setor" for x in th["a_favor"]) and any(x["base"] == "medido" for x in th["contra"])
    assert th["qualidade_dos_dados"]["nivel"] in ("fraca", "média", "boa")
    topics = [x["texto"].split(":")[0].lower() for x in th["contra"]]
    assert sum("receita" in t for t in topics) == 1                                   # no duplicate of the same fact
    assert not guarantee_phrases(json.dumps(th, ensure_ascii=False))


# ----------------------------------------------------------------------------- end to end
MARKET_10Y = structured_market(n=2600)


def test_thesis_endpoint_and_sources_down(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y))
    monkeypatch.setattr(server, "_rf_fetch", lambda: Router())
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/tese", json={"ativo": "WEGE3", "horizonte_anos": 3})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ticker"] == "WEGE3.SA" and len(d["fundamentos"]["anos"]) == 4
    assert d["fundamentos"]["piotroski"]["pontos"] == 9 and not d["armadilha"]["parece_barata"]
    assert d["analistas"]["n_analistas"] == 9 and d["tese"]["a_favor"] and d["previsao"]["probabilidades"]["p_superar_rf"] > 0
    assert any(x["texto"].startswith("Receita crescendo") for x in d["tese"]["a_favor"])
    assert any("B3" in m for m in d["indisponivel"])                                  # B3 down: reported
    monkeypatch.setattr(server, "_rf_fetch", lambda: Router(yahoo=False))
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path / "frio")
    r2 = c.post("/api/tese", json={"ativo": "VALE3"})
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["fundamentos"]["anos"] == [] and d2["analistas"] is None                 # nothing invented
    assert any("demonstrações" in m for m in d2["indisponivel"]) and d2["tese"]["precisa_ser_verdade"]
    assert c.post("/api/tese", json={"ativo": "BTC-USD"}).status_code == 400
    assert c.post("/api/tese", json={"ativo": "SINT_ENERGIA3"}).status_code == 400


def test_radar_ranking_trap_penalty_and_budget(tmp_path):
    prov = Prov(structured_market())
    client = YahooFundamentals(Router(slow={"VALE3.SA"}))
    out = radar.scan(["BR"], "valor", prov, client, tmp_path, budget_s=1.5, top=10)
    cand = out["candidatos"]
    assert len(cand) == 10 and [x["posicao"] for x in cand] == list(range(1, 11))
    assert all(a["pontuacao"] >= b["pontuacao"] for a, b in pairwise(cand))
    assert sum(out["pesos"].values()) == pytest.approx(1.0)
    assert "VALE3.SA" in out["tempo_esgotado"] and out["analisados"] > 50
    for x in cand:
        assert x["evidencias"] and all(any(ch.isdigit() for ch in e) for e in x["evidencias"])
        assert x["desfaz_se"] and 0.5 <= x["cobertura"] <= 1
    # the cheap-and-deteriorating stock pays the trap penalty
    full = radar.scan(["BR"], "valor", prov, YahooFundamentals(Router()), tmp_path, budget_s=20, top=20)
    allc = {x["ticker"]: x for x in full["candidatos"]}
    again = radar.scan(["BR"], "valor", prov, YahooFundamentals(Router()), tmp_path, budget_s=20, top=20)
    assert [x["ticker"] for x in again["candidatos"]] == list(allc)                     # same data, same ranking
    sc = radar.score([radar.Item(a, MARKET_10Y[a.ticker].close, None, None) for a in radar.REAL_CATALOG
                      if a.macro_class == "acoes_br"][:20], "equilibrado")
    assert set(radar.FACTORS) <= set(sc.columns)
    pet = radar.scan(["BR"], "valor", prov, YahooFundamentals(Router()), tmp_path, budget_s=20, top=86)
    p4 = next(x for x in pet["candidatos"] if x["ticker"] == "PETR4.SA")
    assert p4["armadilha"]["parece_barata"] and p4["penalidade_armadilha"] >= 0.5
    with pytest.raises(ValueError, match="mercado"):
        radar.scan([], "valor", prov, client, tmp_path)


def test_radar_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(structured_market()))
    monkeypatch.setattr(server, "_rf_fetch", lambda: Router())
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/radar", json={"mercados": ["BR", "US"], "estilo": "qualidade", "quantidade": 5})
    assert r.status_code == 200, r.text
    assert len(r.json()["candidatos"]) == 5 and r.json()["estilo"] == "qualidade"
    assert c.post("/api/radar", json={"mercados": ["XX"]}).status_code == 422
    assert c.post("/api/radar", json={"estilo": "sorte"}).status_code == 422


def test_sources_cache(tmp_path):
    r = Router()
    cl = YahooFundamentals(r)
    a1 = fetch_annual(cl, "WEGE3.SA", tmp_path, now=1e9)
    n = len(r.calls)
    a2 = fetch_annual(cl, "WEGE3.SA", tmp_path, now=1e9 + 3600)
    assert len(r.calls) == n and a1.valores == a2.valores                              # served from the cache
    fetch_annual(cl, "WEGE3.SA", tmp_path, now=1e9 + 2 * 86400)
    assert len(r.calls) > n                                                            # expired: fetched again
    an = fetch_analysts(cl, "WEGE3.SA", tmp_path, now=1e9)
    assert isinstance(an, Analysts) and an.eps_ano["current"] == 1.9
