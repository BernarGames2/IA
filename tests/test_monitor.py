"""On-demand follow-up of saved theses: statement conditions wait for a new fiscal year, estimates and prices are
checked at any time, and nothing is invented when a source fails (test doubles only)."""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from core.fundamentals.monitor import Condition, MonitorRequest, WatchedThesis, check
from tests.market_doubles import Prov, structured_market
from tests.test_fundamentals import DETERIORATING, Router, ts_json
from webapp import server

MARKET = structured_market()


class NewYearRouter(Router):
    """Same doubles, but WEGE3's statements gain a 2026 fiscal year with revenue down 10%."""

    def __call__(self, url: str, timeout: float) -> tuple[int, bytes]:
        if "fundamentals-timeseries" in url and "WEGE3" in url:
            ann = {k: v[1:] + [v[-1] * (0.9 if k == "receita" else 1.0)] for k, v in DETERIORATING.items()}
            years = ("2023-12-31", "2024-12-31", "2025-12-31", "2026-12-31")
            raw = ts_json("WEGE3.SA", ann)
            for item in raw["timeseries"]["result"]:
                t = item["meta"]["type"][0]
                for row, y in zip(item[t], years, strict=True):
                    if row:
                        row["asOfDate"] = y
            return 200, json.dumps(raw).encode()
        return super().__call__(url, timeout)


def _thesis(ticker: str, conds: list[Condition], created: str = "2026-03-02") -> WatchedThesis:
    return WatchedThesis(ticker=ticker, criada_em=date.fromisoformat(created), condicoes=conds)


def test_statement_conditions_wait_for_a_new_fiscal_year(tmp_path):
    revenue = Condition(chave="receita", valor_limite=400e9, direcao="abaixo", condicao="receita −5%", ano_fiscal="2025-12-31")
    margin = Condition(chave="margem_operacional", valor_limite=0.25, direcao="abaixo", condicao="margem −3 p.p.",
                       ano_fiscal="2025-12-31")
    req = MonitorRequest(teses=[_thesis("PETR4", [revenue, margin])])
    out = check(req, provider=Prov(MARKET), fetch=Router(), cache_dir=tmp_path, today=date(2026, 10, 16))
    states = {c["chave"]: c["estado"] for c in out["teses"][0]["condicoes"]}
    assert states == {"receita": "aguardando o próximo balanço", "margem_operacional": "aguardando o próximo balanço"}
    # a later fiscal year is published: the conditions are measured (revenue 420 x 0.9 = 378 bi < 400 bi)
    req2 = MonitorRequest(teses=[_thesis("WEGE3", [revenue])])
    out2 = check(req2, provider=Prov(MARKET), fetch=NewYearRouter(), cache_dir=tmp_path / "b", today=date(2026, 10, 16))
    c = out2["teses"][0]["condicoes"][0]
    assert c["estado"] == "acionada" and c["valor_atual"] == pytest.approx(378e9) and out2["acionadas"] == 1


def test_estimate_and_price_conditions(tmp_path):
    px = MARKET["VALE3.SA"].close
    low = float(px.loc["2026-03-02":].min())
    conds = [Condition(chave="lpa_estimado", valor_limite=2.0, direcao="abaixo", condicao="LPA −10%"),
             Condition(chave="preco", valor_limite=low + 0.01, direcao="abaixo", condicao="cenário dos 5% piores",
                       ate=date(2027, 3, 2)),
             Condition(chave="preco", valor_limite=low - 0.01, direcao="abaixo", condicao="limite mais baixo")]
    out = check(MonitorRequest(teses=[_thesis("VALE3", conds)]), provider=Prov(MARKET), fetch=Router(), cache_dir=tmp_path,
                today=date(2026, 10, 16))
    got = out["teses"][0]
    st = [c["estado"] for c in got["condicoes"]]
    assert st == ["acionada", "acionada", "dentro do limite"]                 # Router: current LPA 1.9 < 2.0
    assert got["condicoes"][1]["valor_atual"] == pytest.approx(low)
    assert got["preco_atual"] == pytest.approx(float(px.iloc[-1])) and "conferido_em" in out
    fresh = check(MonitorRequest(teses=[_thesis("VALE3", conds[1:2], created="2026-10-17")]), provider=Prov(MARKET),
                  fetch=Router(), cache_dir=tmp_path, today=date(2026, 10, 17))
    assert fresh["teses"][0]["condicoes"][0]["estado"] == "aguardando novas cotações"          # no close after the thesis yet
    assert "segundo plano" in out["aviso"]


def test_sources_down_are_reported_not_filled(tmp_path):
    conds = [Condition(chave="receita", valor_limite=1.0, direcao="abaixo", ano_fiscal="2025-12-31"),
             Condition(chave="lpa_estimado", valor_limite=1.0, direcao="abaixo")]
    out = check(MonitorRequest(teses=[_thesis("PETR4", conds)]), provider=Prov(MARKET), fetch=Router(yahoo=False),
                cache_dir=tmp_path, today=date(2026, 10, 16))
    t = out["teses"][0]
    assert [c["estado"] for c in t["condicoes"]] == ["sem dados", "sem dados"]
    assert any("demonstrações" in w for w in t["avisos"]) and any("estimativas" in w for w in t["avisos"])


def test_monitor_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET))
    monkeypatch.setattr(server, "_rf_fetch", lambda: Router())
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    body = {"teses": [{"ticker": "PETR4", "criada_em": "2026-03-02",
                       "condicoes": [{"chave": "preco", "valor_limite": 1.0, "direcao": "abaixo", "condicao": "x"}]}]}
    r = c.post("/api/monitorar", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["teses"][0]["condicoes"][0]["estado"] == "dentro do limite"
    assert c.post("/api/monitorar", json={"teses": []}).status_code == 422
    bad = {"teses": [{"ticker": "PETR4", "criada_em": "2026-03-02", "condicoes": [{"chave": "sorte", "valor_limite": 1,
                                                                                 "direcao": "abaixo"}]}]}
    assert c.post("/api/monitorar", json=bad).status_code == 422
