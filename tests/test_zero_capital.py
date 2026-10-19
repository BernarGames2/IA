"""Starting from zero: no initial capital, only periodic contributions (e.g. R$ 2.000 per month).
Required CAGR, request validation, the simulation, the purpose engine and the web endpoint."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core.purpose.engine import PurposeRequest, run_purpose
from core.purpose.language import guarantee_phrases
from core.purpose.report import render_markdown, to_dict
from core.purpose.required_cagr import future_value, required_cagr
from core.simulation.engine import CashflowPlan, SimulationSettings, simulate_portfolio
from core.simulation.models import GBMModel


def test_required_cagr_with_contributions_only():
    r = required_cagr(0.0, 2_000.0, "mensal", 400_000.0, 10)
    assert r.status == "ok" and r.total_contributed == pytest.approx(240_000.0)
    assert future_value(r.rate, 0.0, 2_000.0, 12, 10) == pytest.approx(400_000.0, rel=1e-9)
    assert "sem capital inicial" in r.explanation
    # below the money put in: the goal needs no return at all
    assert required_cagr(0.0, 2_000.0, "mensal", 200_000.0, 10).status == "atingivel_sem_retorno"
    for args in ((0.0, 0.0, "mensal"), (0.0, 2_000.0, "nenhum")):
        with pytest.raises(ValueError):
            required_cagr(*args, 400_000.0, 10)
    with pytest.raises(ValueError):                              # no yearly contribution falls inside 6 months
        required_cagr(0.0, 2_000.0, "anual", 10_000.0, 0.5)


def test_request_accepts_zero_capital_only_with_contributions():
    base = {"valor_alvo": 400_000, "horizonte_anos": 10}
    ok = PurposeRequest(**base, aporte_periodico=2_000, periodicidade="mensal")
    assert ok.capital_inicial == 0.0                             # optional: defaults to starting from zero
    for bad in ({"aporte_periodico": 0}, {"aporte_periodico": 0, "periodicidade": "nenhum"},
                {"aporte_periodico": 2_000, "periodicidade": "anual", "horizonte_anos": 0.5}):
        with pytest.raises(ValidationError, match="sem capital inicial"):
            PurposeRequest(**{**base, "capital_inicial": 0, **bad})
    with pytest.raises(ValidationError):
        PurposeRequest(**base, capital_inicial=-1, aporte_periodico=2_000)


def test_simulation_starts_from_contributions():
    m = GBMModel(np.array([0.08, 0.03]), np.array([[0.04, 0.0], [0.0, 0.0004]]), 1 / 12)
    ss = SimulationSettings(n_paths=3_000, n_steps=120, seed=1, initial_capital=0.0, rebalance_every=1,
                            periods_per_year=12, record_every=1, goal=300_000)
    r = simulate_portfolio(m, np.array([0.6, 0.4]), ss, CashflowPlan(monthly_contribution=2_000.0, steps_per_month=1))
    s = r.summary
    assert s["net_invested"] == pytest.approx(240_000.0) and r.percentiles["p50"].iloc[0] == 0.0
    assert 0 < s["p_goal"] < 1 and np.isfinite(r.terminal_wealth).all() and (r.terminal_wealth > 0).all()
    # the first-step risk is measured on the first month with money invested, not on an empty portfolio
    assert r.first_step_returns.std() > 0.01 and s["var_es_1step"]["levels"]["0.95"]["var"] > 0
    assert np.isfinite(r.max_drawdown_unit).all()


def test_purpose_engine_from_zero(tmp_path):
    req = PurposeRequest(capital_inicial=0, aporte_periodico=2_000, periodicidade="mensal", valor_alvo=400_000,
                         horizonte_anos=10, n_simulacoes=1_000, taxa_livre_risco={"BRL": 0.105},
                         motivo_investidor="Quero pagar a faculdade dos meus filhos")
    res = run_purpose(req, output_dir=tmp_path, fast=True)
    assert res.required.rate is not None and np.isfinite(res.required.rate)
    assert all(a["valor_inicial"] == 0 for a in res.allocation)
    assert sum(a["aporte_por_periodo"] for a in res.allocation) == pytest.approx(2_000.0, rel=1e-6)
    d = to_dict(res)
    json.dumps(d, allow_nan=False)                                # no NaN or infinity anywhere
    assert 0 <= d["objetivo"]["p_objetivo"] <= 1
    md = render_markdown(res)
    assert "começa só com os aportes" in md and "sem capital inicial" in md
    assert guarantee_phrases(md.replace(req.motivo_investidor, "")) == []


def test_web_endpoint_from_zero():
    from fastapi.testclient import TestClient

    import webapp.server as srv

    c = TestClient(srv.app)
    body = {"valor_alvo": 60_000, "capital_inicial": 0, "aporte_periodico": 2_000, "horizonte_anos": 2,
            "n_simulacoes": 500, "taxa_livre_risco": {"BRL": 0.105}, "fonte_dados": "sintetico"}
    r = c.post("/api/proposito", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["entrada"]["capital_inicial"] == 0 and d["cagr"]["necessario"] is not None
    bad = c.post("/api/proposito", json={**body, "aporte_periodico": 0})
    assert bad.status_code == 400 and "sem capital inicial" in bad.json()["detail"]
    assert "pydantic" not in bad.json()["detail"] and "validation error" not in bad.json()["detail"]
    html = (Path(__file__).resolve().parents[1] / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'name="capital_inicial" type="number" min="0"' in html and "Vai começar do zero? Deixe 0" in html
