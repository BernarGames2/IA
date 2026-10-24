"""Business futures (discounted cash flow under explicit hypotheses) and economic chain effects, checked by hand and
with planted relations in test data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.fundamentals.sources import parse_timeseries
from core.market_data.base import RawSeries
from core.purpose.language import guarantee_phrases
from core.scenarios.api import defaults
from core.scenarios.chain import chain_effects
from core.scenarios.dcf import Base, implied_growth, simulate, value_per_share
from tests.market_doubles import END, Prov, structured_market
from tests.test_fundamentals import GROWER, Router, ts_json
from webapp import server

P = {"crescimento": (0.0, 0.05, 0.10), "margem_fcl": (0.08, 0.10, 0.12), "custo_capital": (0.10, 0.12, 0.14),
     "crescimento_perpetuo": (0.01, 0.02, 0.03)}


# ----------------------------------------------------------------------------- discounted cash flow
def test_value_by_hand():
    b = Base(receita=100.0, margem_atual=0.10, acoes=10.0, preco=8.0)
    assert float(value_per_share(b, 0.0, 0.10, 0.10, 0.0)) == pytest.approx(10.0)        # 10 a year forever at 10%
    # a growing perpetuity: 5 explicit years + terminal value equal CF1 / (ke - g) exactly
    g, ke = 0.03, 0.10
    assert float(value_per_share(b, g, 0.10, ke, g)) == pytest.approx(100 * 1.03 * 0.10 / (ke - g) / 10)
    # the margin moves in a straight line from today's to the target in year 5
    b2 = Base(receita=100.0, margem_atual=0.0, acoes=1.0, preco=1.0)
    flows = [100 * 0.2 * t / 5 for t in range(1, 6)]
    expected = sum(f / 1.1 ** t for t, f in enumerate(flows, 1)) + 100 * 0.2 / 0.1 / 1.1 ** 5
    assert float(value_per_share(b2, 0.0, 0.2, 0.1, 0.0)) == pytest.approx(expected)
    # banks and insurers: net income x payout
    fb = Base(receita=100.0, margem_atual=None, acoes=10.0, preco=5.0, financeira=True)
    assert float(value_per_share(fb, 0.0, 0.0, 0.10, 0.0, payout=0.5)) == pytest.approx(100 * 0.5 / 0.10 / 10)


def test_reverse_dcf_scenarios_and_simulation():
    b = Base(receita=100.0, margem_atual=0.10, acoes=10.0, preco=8.0)
    g = implied_growth(b, 0.10, 0.12, 0.02)
    assert g is not None and float(value_per_share(b, g, 0.10, 0.12, 0.02)) == pytest.approx(8.0, rel=1e-6)
    r = simulate(b, P)
    v = r["valor_por_acao"]
    assert v["pessimista"] < v["base"] < v["otimista"]
    assert v["base"] == pytest.approx(float(value_per_share(b, 0.05, 0.10, 0.12, 0.02)))
    assert r["potencial"]["base"] == pytest.approx(v["base"] / 8.0 - 1)
    pc = list(r["simulacao"]["percentis"].values())
    assert pc == sorted(pc) and 0 <= r["simulacao"]["p_acima_do_preco"] <= 1
    amp = [t["amplitude"] for t in r["sensibilidade"]]
    assert amp == sorted(amp, reverse=True)
    ke = next(t for t in r["sensibilidade"] if t["chave"] == "custo_capital")
    assert ke["com_valor_baixo"] > ke["com_valor_alto"]                                  # higher cost of capital, lower value
    assert simulate(b, P)["simulacao"] == r["simulacao"]                                  # same seed, same draws
    assert 0 < r["parcela_perpetuidade"] < 1


def test_hypotheses_are_validated():
    b = Base(receita=100.0, margem_atual=0.10, acoes=10.0, preco=8.0)
    with pytest.raises(ValueError, match="1 ponto percentual"):
        simulate(b, {**P, "custo_capital": (0.08, 0.10, 0.12), "crescimento_perpetuo": (0.01, 0.02, 0.075)})
    with pytest.raises(ValueError, match="entre"):
        simulate(b, {**P, "custo_capital": (0.10, 0.12, 0.90)})
    with pytest.raises(ValueError, match="pessimista"):
        simulate(b, {**P, "crescimento": (0.10, 0.05, 0.0)})
    with pytest.raises(ValueError, match="payout"):
        simulate(Base(100.0, None, 10.0, 5.0, True), P)


def test_defaults_come_from_the_statements():
    a = parse_timeseries(ts_json("X", GROWER), "2026-10-16")
    p, origin, pay, _ = defaults(a, False, 0.15, 1.2, "BRL")
    assert p["crescimento"][1] == pytest.approx((45.6 / 30) ** (1 / 3) - 1)
    margins = sorted(GROWER["fcl"][i] / GROWER["receita"][i] for i in range(4))
    assert p["margem_fcl"][1] == pytest.approx((margins[1] + margins[2]) / 2)
    assert p["custo_capital"][1] == pytest.approx(0.15 + 1.2 * 0.04)
    assert p["custo_capital"][0] - p["crescimento_perpetuo"][2] >= 0.01 - 1e-12 and pay is None
    assert "balanços" in origin["crescimento"] and "hipótese" in origin["crescimento_perpetuo"]


MARKET_10Y = structured_market(n=2600)


def test_scenarios_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y))
    monkeypatch.setattr(server, "_rf_fetch", lambda: Router())
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/cenarios", json={"ativo": "WEGE3"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ticker"] == "WEGE3.SA" and d["base"]["acoes"] == pytest.approx(4.19e9)
    assert set(d["premissas"]) == {"crescimento", "margem_fcl", "custo_capital", "crescimento_perpetuo"}
    assert d["resultado"]["valor_por_acao"]["base"] > 0 and "Selic" in d["fonte_taxa"]
    mine = {k: v["valores"] for k, v in d["premissas"].items()}
    mine["crescimento"] = [0.0, 0.02, 0.04]
    r2 = c.post("/api/cenarios", json={"ativo": "WEGE3", "premissas": mine})
    assert r2.status_code == 200 and r2.json()["premissas"]["crescimento"]["origem"] == "informada por você"
    assert r2.json()["resultado"]["valor_por_acao"]["base"] < d["resultado"]["valor_por_acao"]["base"]
    bad = {**mine, "crescimento_perpetuo": [0.0, 0.02, mine["custo_capital"][0]]}
    assert c.post("/api/cenarios", json={"ativo": "WEGE3", "premissas": bad}).status_code == 400
    assert c.post("/api/cenarios", json={"ativo": "WEGE3", "premissas": {"sorte": [0, 0, 0]}}).status_code == 422
    assert c.post("/api/cenarios", json={"ativo": "BTC-USD"}).status_code == 400
    assert not guarantee_phrases(r.text)


# ----------------------------------------------------------------------------- chain effects
def _planted(betas: dict[str, float], proxy: str = "USDBRL=X", n: int = 1300, seed: int = 3) -> dict[str, RawSeries]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=END, periods=n)
    rp = rng.normal(0, 0.01, n)
    out = {proxy: RawSeries(proxy, pd.Series(5 * np.exp(np.cumsum(rp)), index=idx), None, "teste", "close", "t", "BRL", "1d")}
    for t, b in betas.items():
        r = b * rp + rng.normal(0, 0.01, n)
        out[t] = RawSeries(t, pd.Series(30 * np.exp(np.cumsum(r)), index=idx), None, "teste", "close", "t", "BRL", "1d")
    return out


def test_chain_recovers_planted_betas_and_reads_agreement():
    data = _planted({"VALE3.SA": 0.8, "MGLU3.SA": -0.8, "EGIE3.SA": 0.0})
    out = chain_effects("dolar", 10, "BR", Prov(data), ["VALE3", "MGLU3", "EGIE3"])
    eff = {r["ticker"]: r for r in out["acoes"]}
    assert eff["VALE3.SA"]["beta"] == pytest.approx(0.8, abs=0.2) and eff["MGLU3.SA"]["beta"] == pytest.approx(-0.8, abs=0.2)
    assert eff["VALE3.SA"]["efeito"] == pytest.approx(eff["VALE3.SA"]["beta"] * 0.10)
    lo, hi = eff["VALE3.SA"]["faixa_95"]
    assert lo < eff["VALE3.SA"]["efeito"] < hi
    sec = {s["setor"]: s for s in out["setores"]}
    assert sec["mineracao"]["concordancia"] == "dados confirmam" and sec["mineracao"]["esperado"] == "+"
    assert sec["consumo"]["concordancia"] == "dados confirmam" and sec["consumo"]["esperado"] == "−"
    assert sec["utilidades"]["concordancia"].startswith("teoria ambígua")
    flipped = chain_effects("dolar", 10, "BR", Prov(_planted({"MGLU3.SA": 0.8})), ["MGLU3"])
    assert flipped["setores"][0]["concordancia"] == "dados contrariam"
    rates = chain_effects("juros_br", 1, "BR", Prov(_planted({"VALE3.SA": 0.5}, proxy="IRFM11.SA")), ["VALE3"])
    assert rates["choque"]["movimento_indicador"] == pytest.approx(-0.02)
    assert rates["acoes"][0]["efeito"] == pytest.approx(rates["acoes"][0]["beta"] * -0.02)
    assert out["leitura"] and out["cadeia"] and not guarantee_phrases(str(out))
    with pytest.raises(ValueError, match="tamanho"):
        chain_effects("dolar", 0, "BR", Prov(data), ["VALE3"])


def test_chain_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(structured_market(extra={"BZ=F": "USD"})))
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/cadeia", json={"choque": "petroleo", "tamanho": 20, "mercado": "BR"})
    assert r.status_code == 200, r.text
    assert r.json()["setores"] and r.json()["choque"]["indicador"] == "BZ=F"
    assert c.post("/api/cadeia", json={"choque": "chuva"}).status_code == 422
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(structured_market(), fail={"IEF"}))
    assert c.post("/api/cadeia", json={"choque": "juros_eua", "mercado": "US"}).status_code == 503
