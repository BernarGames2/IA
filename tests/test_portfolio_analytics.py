"""The investor's own portfolio: dashboard (value, evolution, returns, benchmarks, allocation, alerts) and the
advanced risk report, on real-data paths with test doubles (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.market_data.base import RawSeries
from core.portfolio_analytics.dashboard import build_dashboard, twr_index, xirr
from core.portfolio_analytics.data import PortfolioRequest, build_book, load_market
from core.portfolio_analytics.risk import build_risk
from core.purpose.language import guarantee_phrases
from tests.market_doubles import END, Prov, http, structured_market
from webapp import server

REF = date.fromisoformat(END)
MARKET = structured_market()
MARKET_10Y = structured_market(n=2600)


def _req(**kw) -> PortfolioRequest:
    return PortfolioRequest(data_referencia=REF, **kw)


def _close(sym: str, d: str | None = None, market=MARKET) -> float:
    s = market[sym].close
    return float(s.loc[:pd.Timestamp(d)].iloc[-1] if d else s.iloc[-1])


def test_twr_and_xirr_by_hand():
    v = pd.Series([100.0, 110.0, 220.0, 231.0], index=pd.bdate_range("2025-01-01", periods=4))
    f = pd.Series([0.0, 0.0, 100.0, 0.0], index=v.index)
    idx = twr_index(v, f)
    assert idx.iloc[-1] == pytest.approx(1.10 * (220 / 210) * 1.05)
    d0 = pd.Timestamp("2025-01-01")
    assert xirr([(d0, -100.0), (d0 + pd.Timedelta(days=365), 110.0)]) == pytest.approx(0.10, abs=1e-8)
    assert xirr([(d0, 100.0)]) is None                          # no investment, no rate


def test_book_from_movements_and_oversell_refused():
    m = load_market(["PETR4.SA"], Prov(MARKET), http, "5y", REF)
    moves = [{"data": "2025-01-06", "ticker": "petr4", "tipo": "compra", "quantidade": 100, "preco": 40},
             {"data": "2025-09-01", "ticker": "PETR4", "tipo": "venda", "quantidade": 40},
             {"data": "2025-10-01", "ticker": "PETR4", "tipo": "dividendo", "valor": 120}]
    b = build_book(_req(movimentos=moves), m)
    assert b.quantity == {"PETR4.SA": 60.0} and b.from_movements
    assert b.flows.loc[pd.Timestamp("2025-01-06"), "PETR4.SA"] == pytest.approx(4000.0)
    sale = -b.flows.loc[pd.Timestamp("2025-09-01"), "PETR4.SA"]
    assert sale == pytest.approx(40 * _close("PETR4.SA", "2025-09-01"))      # no price typed: that day's close
    assert float(b.dividends.sum()) == 120.0
    with pytest.raises(ValueError, match="maior que a quantidade"):
        build_book(_req(movimentos=[moves[0], {**moves[1], "quantidade": 101}]), m)


def test_dashboard_values_positions_in_reais_and_compares_with_benchmarks():
    req = _req(posicoes=[{"ticker": "PETR4", "quantidade": 100}, {"ticker": "AAPL", "quantidade": 10},
                         {"ticker": "CAIXA_CDI", "valor": 5000}], periodo="1a", perfil="moderado")
    d = build_dashboard(req, Prov(MARKET), http)
    expect = 100 * _close("PETR4.SA") + 10 * _close("AAPL") * _close("USDBRL=X") + 5000
    assert d["patrimonio"] == pytest.approx(expect, rel=1e-9)
    assert sum(p["peso"] for p in d["posicoes"]) == pytest.approx(1.0)
    for dim in ("ativo", "classe", "setor", "moeda"):
        assert sum(d["alocacao"][dim].values()) == pytest.approx(1.0)
    assert d["modo"] == "hipotetico" and any("hipótese" in a for a in d["avisos"])
    assert {b["nome"] for b in d["periodo"]["benchmarks"]} == {"CDI", "Ibovespa (BOVA11)", "S&P 500 em reais (IVVB11)"}
    ev = d["evolucao"]
    assert ev["carteira"][0] == pytest.approx(100.0) and all(v[0] == pytest.approx(100.0) for v in ev["benchmarks"].values())
    start = pd.Timestamp(d["periodo"]["inicio"])
    bova = MARKET["BOVA11.SA"].close
    want = float(bova.iloc[-1] / bova.loc[:start].iloc[-1] - 1)
    got = next(b["retorno"] for b in d["periodo"]["benchmarks"] if b["nome"].startswith("Ibovespa"))
    assert got == pytest.approx(want)                               # same window as the portfolio
    assert any("acima do limite de 30% por ativo do perfil moderado" in a["texto"] for a in d["alertas"])
    assert guarantee_phrases(" ".join(d["avisos"] + [a["texto"] for a in d["alertas"]])) == []


def test_dashboard_with_movements_measures_returns_and_result():
    moves = [{"data": "2025-01-06", "ticker": "PETR4", "tipo": "compra", "quantidade": 100}]
    d = build_dashboard(_req(movimentos=moves, periodo="inicio"), Prov(MARKET), http)
    buy = _close("PETR4.SA", "2025-01-06")
    assert d["modo"] == "movimentacoes"
    assert d["periodo"]["retorno_twr"] == pytest.approx(_close("PETR4.SA") / buy - 1.0)
    assert d["resultado"]["total_investido_liquido"] == pytest.approx(100 * buy)
    assert d["resultado"]["resultado"] == pytest.approx(100 * (_close("PETR4.SA") - buy))
    days = (pd.Timestamp(d["data_referencia"]) - pd.Timestamp("2025-01-06")).days
    assert d["periodo"]["retorno_mwr_anual"] == pytest.approx((_close("PETR4.SA") / buy) ** (365 / days) - 1, rel=1e-6)


def test_missing_ticker_is_reported_never_filled_and_no_price_after_the_reference_date():
    d = build_dashboard(_req(posicoes=[{"ticker": "PETR4", "quantidade": 10}, {"ticker": "ZZZZ3", "quantidade": 5}]),
                        Prov(MARKET), http)
    assert "ZZZZ3.SA" in d["sem_dados"] and all(p["ticker"] != "ZZZZ3.SA" for p in d["posicoes"])
    assert d["patrimonio"] == pytest.approx(10 * _close("PETR4.SA"))
    early = date(2026, 6, 30)
    d2 = build_dashboard(PortfolioRequest(posicoes=[{"ticker": "PETR4", "quantidade": 10}], data_referencia=early),
                         Prov(MARKET), http)
    assert d2["data_referencia"] <= "2026-06-30" and d2["patrimonio"] == pytest.approx(10 * _close("PETR4.SA", "2026-06-30"))


def test_risk_report_adds_up_and_states_its_limits():
    req = _req(posicoes=[{"ticker": "PETR4", "quantidade": 100}, {"ticker": "VALE3", "quantidade": 80},
                         {"ticker": "AAPL", "quantidade": 10}, {"ticker": "CAIXA_CDI", "valor": 5000}], confianca=0.95)
    d = build_risk(req, Prov(MARKET_10Y), http)
    assert sum(c["contribuicao"] for c in d["contribuicao_risco"]) == pytest.approx(1.0)
    assert sum(d["pesos"].values()) == pytest.approx(1.0)
    hist = next(r for r in d["var"] if r["metodo"] == "histórico" and r["confianca"] == 0.95)
    assert 0 < hist["var"] <= hist["cvar"] and hist["var_reais"] == pytest.approx(hist["var"] * d["patrimonio"])
    covid = next(s for s in d["estresse_historico"] if s["nome"].startswith("Crise da COVID"))
    assert covid["retorno"] is not None and covid["cobertura"] >= 0.8         # 10 years of prices cover 2020
    for h in d["estresse_hipotetico"]:
        assert h["impacto"] == pytest.approx(h["beta"] * h["choque"]) and "hipótese" in h["base"]
    assert d["sensibilidades"] and all(s["forca"].startswith("relação estatisticamente") for s in d["sensibilidades"])
    assert d["correlacao"]["dias_estresse"] >= 30 and -1 <= d["correlacao"]["media_estresse"] <= 1
    text = " ".join([d["o_que_explica"]["texto"], *d["avisos"]])
    assert guarantee_phrases(text) == [] and "perda futura pode ser maior" in text
    short = build_risk(req, Prov(MARKET), http)                            # 5 years of prices: no 2020 data
    covid5 = next(s for s in short["estresse_historico"] if s["nome"].startswith("Crise da COVID"))
    assert covid5["retorno"] is None and "sem dados suficientes" in covid5["nota"]


def test_liquidity_uses_traded_value_and_says_when_volume_is_missing():
    m = dict(MARKET_10Y)
    s = m["VALE3.SA"]
    m["VALE3.SA"] = RawSeries(s.symbol, s.close, None, s.provider, s.price_field, s.adjusted_for, s.currency, s.frequency)
    d = build_risk(_req(posicoes=[{"ticker": "PETR4", "quantidade": 1000}, {"ticker": "VALE3", "quantidade": 10}]), Prov(m), http)
    petr = next(x for x in d["liquidez"] if x["ticker"] == "PETR4.SA")
    adtv = float((MARKET_10Y["PETR4.SA"].close * MARKET_10Y["PETR4.SA"].volume).iloc[-60:].mean())
    assert petr["dias_para_vender"] == pytest.approx(1000 * _close("PETR4.SA", market=MARKET_10Y) / (0.2 * adtv))
    vale = next(x for x in d["liquidez"] if x["ticker"] == "VALE3.SA")
    assert vale["dias_para_vender"] is None and "não medido" in vale["nota"]


def test_endpoints(monkeypatch):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http)
    c = TestClient(server.app)
    body = {"posicoes": [{"ticker": "PETR4", "quantidade": 10}], "data_referencia": END}
    r = c.post("/api/carteira/painel", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["patrimonio"] == pytest.approx(10 * _close("PETR4.SA")) and r.json()["do_cache"] is False
    assert c.post("/api/carteira/painel", json=body).json()["do_cache"] is True
    assert c.post("/api/carteira/painel", json={"posicoes": []}).status_code == 422
    bad = {"movimentos": [{"data": "2025-01-06", "ticker": "PETR4", "tipo": "venda", "quantidade": 5}], "data_referencia": END}
    r2 = c.post("/api/carteira/painel", json=bad)
    assert r2.status_code == 400 and "maior que a quantidade" in r2.json()["detail"]
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y))
    r3 = c.post("/api/carteira/risco", json={**body, "posicoes": [{"ticker": "PETR4", "quantidade": 10},
                                                                   {"ticker": "VALE3", "quantidade": 10}]})
    assert r3.status_code == 200, r3.text
    assert np.isfinite(r3.json()["volatilidade"]["historica"])
