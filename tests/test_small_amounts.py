"""Small amounts: the money must buy a portfolio within the investor's limits — every position at least its minimum
purchase (one lot, and the 'Compra mínima por ativo') and every weight within the profile's limits. Below that the
answer is a plain refusal that names the least amount, never a target made of positions the money cannot buy (the
site showed R$ 1 + R$ 1/month spread over seven assets, R$ 0 in every row). No network: test doubles."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.optimization.constraints import semicontinuous_weights
from core.purpose.engine import AmountTooSmall, PurposeRequest, run_purpose
from core.purpose.language import guarantee_phrases
from models.portfolio import GroupConstraint, PortfolioConstraints
from tests.market_doubles import Prov, http, structured_market
from webapp import server

MARKET = structured_market()


class NoData:
    """A provider that fails the test if the engine asks it for anything."""

    def __getattr__(self, name):
        raise AssertionError(f"dados de mercado pedidos antes da recusa ({name})")


def _no_rf(url, timeout):
    raise AssertionError("taxa livre de risco pedida antes da recusa")


def _req(capital: float, aporte: float = 0.0, **kw) -> PurposeRequest:
    return PurposeRequest(capital_inicial=capital, aporte_periodico=aporte,
                          periodicidade="mensal" if aporte else "nenhum", valor_alvo=50_000, horizonte_anos=8,
                          n_simulacoes=500, **kw)


def _run(req: PurposeRequest, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return run_purpose(req, output_dir=tmp_path, cache_dir=tmp_path, fast=True,
                       **({} if req.offline else {"provider": Prov(MARKET), "rf_fetch": http}))


def _least(msg: str) -> float:
    found = re.search(r"pede pelo menos R\$ ([\d.]+,\d{2})", msg)
    assert found, msg
    return float(found.group(1).replace(".", "").replace(",", "."))


def _assert_buyable(r) -> None:
    """Every position reaches the minimum purchase with the reference amount, within the per-asset limit, and the
    first purchase buys all of them."""
    ref, m = r.execution["valor_de_referencia"], r.request.valor_minimo_posicao
    cap = r.analysis.profile.research_limits.max_weight_per_asset
    w = r.combined.final[r.combined.final > 1e-9]
    assert len(w) >= 2 and w.sum() == pytest.approx(1.0)
    assert all(x * ref >= m - 0.01 for x in w), w.to_dict()
    assert all(x <= cap + 1e-6 for x in w), w.to_dict()
    first = r.execution["compra_inicial"]
    assert first["fora_desta_compra"] == {} and {p["ticker"] for p in first["posicoes"]} == set(w.index)
    assert not [a.message for a in r.alerts if a.category == "execucao"
                or (a.severity == "FAIL" and ("INVIÁVEIS" in a.message or "violou" in a.message))]


def test_one_real_plus_one_real_a_month_is_refused_at_once(tmp_path):
    with pytest.raises(AmountTooSmall) as e:
        run_purpose(_req(1, 1), output_dir=tmp_path, cache_dir=tmp_path, provider=NoData(), rf_fetch=_no_rf,
                    fast=True)
    msg = str(e.value)
    assert msg.startswith("Valor pequeno demais") and "com R$ 13,00 entre o que você tem hoje" in msg
    assert "não deixa nenhum ativo passar de 30%" in msg and "pelo menos 4 posições de R$ 100,00" in msg
    assert _least(msg) == 400 and "Ação do dia" in msg and "'Compra mínima por ativo'" in msg
    assert guarantee_phrases(msg) == []


@pytest.mark.parametrize(("kw", "least"), [({"perfil_risco": "conservador"}, 300), ({"perfil_risco": "moderado"}, 400),
                                           ({"perfil_risco": "arrojado"}, 300), ({"valor_minimo_posicao": 50}, 200),
                                           ({"necessidade_liquidez": 0.2}, 200)])
def test_the_first_minimum_follows_the_cap_and_the_minimum_purchase(kw, least, tmp_path):
    """No asset above the per-asset cap means at least ceil(1 / cap) positions of the minimum purchase each; under a
    cash minimum, cash may hold the rest, so two positions."""
    with pytest.raises(AmountTooSmall) as e:
        run_purpose(_req(150, **kw), output_dir=tmp_path, cache_dir=tmp_path, provider=NoData(), rf_fetch=_no_rf,
                    fast=True)
    assert _least(str(e.value)) == least
    assert ("não deixa nenhum ativo passar" in str(e.value)) is ("necessidade_liquidez" not in kw)


def test_below_todays_least_amount_the_refusal_names_it_and_that_amount_is_enough(tmp_path):
    with pytest.raises(AmountTooSmall) as e:
        _run(_req(600, offline=True), tmp_path / "a")
    msg = str(e.value)
    assert "não há combinação de ativos em que cada posição compre pelo menos 1 ação ou cota" in msg
    assert "Com os preços de hoje, a menor carteira possível" in msg and "frações de ações dos EUA" in msg
    least = _least(msg)
    assert least > 600 and least % 10 == 0
    _assert_buyable(_run(_req(least, offline=True), tmp_path / "b"))


def test_a_small_portfolio_holds_only_positions_the_money_buys(tmp_path):
    """R$ 1.000 with real-data doubles: buying the largest theoretical positions first spent the money on
    expensive lots before the fixed-income minimum; the held set is now chosen exactly."""
    _assert_buyable(_run(_req(1_000, offline=False), tmp_path))


def test_contributions_alone_build_the_portfolio_over_the_year(tmp_path):
    r = _run(_req(0, 50, offline=False), tmp_path)
    assert r.execution["valor_de_referencia"] == pytest.approx(600.0)
    nxt = r.execution["proximo_aporte"]
    assert nxt["posicoes"] and all("compra mínima de R$ 100,00" in why for why in nxt["fora_desta_compra"].values())


def test_exact_held_set_where_buying_the_largest_first_fails():
    """Two stocks lead the target but their lots (45% each) leave no room for the 20% fixed-income minimum."""
    pc = PortfolioConstraints(["A", "B", "C", "F"], np.zeros(4), np.full(4, 0.5),
                              [GroupConstraint("renda fixa (mínimo)", ("F",), 0.2, 1.0)])
    floor, target = np.array([0.45, 0.45, 0.1, 0.2]), np.array([0.4, 0.4, 0.1, 0.1])
    w = semicontinuous_weights(pc, floor, target)
    held = w > 0
    assert w.sum() == pytest.approx(1.0) and held[3] and w[3] >= 0.2 - 1e-9 and not (held[0] and held[1])
    assert np.all(w[held] >= floor[held] - 1e-9) and np.all(w <= 0.5 + 1e-9)
    assert np.abs(w - target).sum() == pytest.approx(0.8)
    again = semicontinuous_weights(pc, floor, target, avoid=[list(np.flatnonzero(held))])
    assert again is not None and set(np.flatnonzero(again > 0)) != set(np.flatnonzero(held))
    assert semicontinuous_weights(pc, floor, target, pool=[0, 1, 2]) is None       # no fixed income to hold
    assert semicontinuous_weights(pc, floor, min_held=4) is None                    # four lots add up to 120%


def test_the_site_says_it_plainly_with_status_400(monkeypatch):
    monkeypatch.setattr(server, "_market_provider", lambda: NoData())
    monkeypatch.setattr(server, "_rf_fetch", lambda: _no_rf)
    c = TestClient(server.app)
    body = {"capital_inicial": 1, "aporte_periodico": 1, "periodicidade": "mensal", "valor_alvo": 10_000,
            "horizonte_anos": 5}
    r = c.post("/api/proposito", json=body)
    assert r.status_code == 400 and r.json()["detail"].startswith("Valor pequeno demais")
    assert "não pôde ser executada" not in r.json()["detail"]
    events = [json.loads(x) for x in c.post("/api/proposito?stream=1", json=body).text.splitlines() if x.strip()]
    assert events[-1]["tipo"] == "erro" and events[-1]["status"] == 400
    assert events[-1]["detail"].startswith("Valor pequeno demais")


def test_page_shows_the_refusal_as_a_notice_and_small_values_with_cents():
    page = server._INDEX_HTML
    assert 'e.status === 400 ? "Atenção: " + e.message' in page and "errStatus = ev.status" in page
    assert "fmtMoneyAuto(a.valor_alvo)" in page and "fmtMoneyAuto(a.aporte_por_periodo)" in page
