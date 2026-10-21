"""Ação do dia: with the money the investor has today, the stock the fixed rules point to — affordable, within the
profile's risk ceiling for a single stock, aware of what the investor already holds, reproducible, honestly worded.
Real-data path with test doubles (no network)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from core.purpose.language import guarantee_phrases
from core.stock_check.daily_pick import CEILING, daily_pick
from tests.market_doubles import Prov, http, structured_market
from webapp import server

MARKET = structured_market()


def _pick(amount=1_000.0, profile="moderado", **kw):
    return daily_pick(amount, profile, provider=Prov(MARKET), rf_fetch=http, **kw)


def _text(r) -> str:
    return json.dumps(r.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize("profile", ["conservador", "moderado", "arrojado"])
def test_the_pick_fits_the_money_and_the_profile(profile):
    r = _pick(1_000.0, profile)
    p = r.pick
    assert p is not None and 0 < p["custo"] <= 1_000.0 and p["quantidade"] >= 1
    assert p["sobra"] == pytest.approx(1_000.0 - p["custo"], abs=0.01) and p["quantidade"] == int(p["quantidade"])
    vmax, dmin = CEILING[profile]
    for e in [p, *r.alternatives]:
        assert e["custo"] <= 1_000.0
    f = p["fatos_H"]
    assert vmax is None or f["volatilidade"] <= vmax
    assert dmin is None or f["queda_maxima"] >= dmin
    assert len({p["ticker"], *(a["ticker"] for a in r.alternatives)}) == 1 + len(r.alternatives) <= 6
    assert p["por_que"] and p["negocio"] and "concentra o risco" in p["riscos"]


def test_same_data_and_inputs_give_the_same_pick_and_the_wording_is_probabilistic():
    a, b = _pick(), _pick()
    assert a.to_dict() == b.to_dict()
    text = _text(a)
    assert guarantee_phrases(text) == [] and "não prevê o preço de amanhã" in text and "não é recomendação" in text


def test_markets_limit_the_universe():
    assert _pick(markets=["BR"]).pick["ticker"].endswith(".SA")
    us = _pick(5_000.0, markets=["US"])
    assert us.pick is not None and not us.pick["ticker"].endswith(".SA")


def test_what_the_investor_holds_is_not_suggested_again_and_counts_for_diversification():
    first = _pick().pick["ticker"]
    r = _pick(holdings={first: 5_000.0})
    assert r.pick["ticker"] != first and r.left_out["já na sua carteira"] >= 1
    assert r.holdings_used == [first] and r.pick["correlacao_com_sua_carteira"] is not None
    assert not any("não informou o que já tem" in w for w in r.warnings)
    assert any("não informou o que já tem" in w for w in _pick().warnings)


def test_too_little_money_gives_no_pick_and_says_why():
    r = _pick(0.5)
    assert r.pick is None and r.alternatives == []
    assert "não dá para comprar 1 ação" in r.warnings[0] and "frações de ações dos EUA" in r.warnings[0]


def test_us_fractions_when_the_broker_allows_them():
    whole = _pick(100.0, markets=["US"])
    frac = _pick(100.0, markets=["US"], us_fractional=True)
    assert whole.pick is None
    assert frac.pick is not None and 0 < frac.pick["custo"] <= 100.0 and frac.pick["quantidade"] < 1
    assert "frações" in frac.pick["regra_de_lote"]


def test_synthetic_demonstration_is_labelled():
    r = daily_pick(1_000.0, "moderado", offline=True)
    assert r.origin == "sintetico" and r.pick["ticker"].startswith("SINT_") and r.pick["logo"] is None
    assert "sintético" in r.sources[0]["fonte"]


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        daily_pick(0.0, "moderado", offline=True)
    with pytest.raises(ValueError):
        daily_pick(100.0, "agressivo", offline=True)


def test_endpoint(monkeypatch):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http)
    c = TestClient(server.app)
    body = {"valor": 1_000, "perfil": "moderado", "carteira_atual": {"PETR4": 2_000}}
    r1 = c.post("/api/acao-do-dia", json=body)
    assert r1.status_code == 200, r1.text
    d = r1.json()
    assert d["escolha"]["ticker"] != "PETR4.SA" and d["carteira_considerada"] == ["PETR4.SA"]
    assert d["do_cache"] is False and d["calculado_em_utc"] and d["regras"] and d["aviso"]
    assert c.post("/api/acao-do-dia", json=body).json()["do_cache"] is True
    syn = c.post("/api/acao-do-dia", json={"valor": 1_000, "fonte_dados": "sintetico"}).json()
    assert syn["origem"] == "sintetico" and syn["escolha"]["ticker"].startswith("SINT_")
    assert c.post("/api/acao-do-dia", json={"valor": 0}).status_code == 422
    assert c.post("/api/acao-do-dia", json={"valor": 100, "carteira_atual": {"PETR4": -5}}).status_code == 400


def test_page_has_the_section_and_a_link_to_it():
    page = server._INDEX_HTML
    assert 'id="d-form"' in page and "/api/acao-do-dia" in page and 'href="#d-title"' in page
    assert "não prevê o preço de amanhã e não é recomendação" in page
