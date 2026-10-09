"""Web API (Vercel entrypoint) tests — offline, synthetic data only."""

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    from webapp.server import app

    return TestClient(app)


FAST = {"profile": "conservador", "simulations": 500, "horizon_years": 1}


def test_pyproject_entrypoint_resolves_to_asgi_app():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module, attr = cfg["tool"]["vercel"]["entrypoint"].split(":")
    app = getattr(importlib.import_module(module), attr)
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)
    deps = " ".join(cfg["project"]["dependencies"])
    for heavy in ("matplotlib", "yfinance", "pytest"):
        assert heavy not in deps  # keeps the serverless bundle small


def test_web_import_does_not_pull_plotting_or_provider_libs():
    code = ("import sys, webapp.server; "
            "print(','.join(m for m in ('matplotlib', 'yfinance', 'requests') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "Quant Portfolio Intelligence" in r.text and "DADOS SINTÉTICOS" in r.text
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["data"] == "synthetic-only"


def test_analyze_returns_validated_summary(client):
    r = client.post("/api/analisar", json=FAST)
    assert r.status_code == 200
    d = r.json()
    assert d["meta"]["synthetic"] is True and d["meta"]["mode"] == "rapido"
    w = d["selected"]["weights"]
    assert abs(sum(x["weight"] for x in w) - 1) < 1e-6
    assert abs(sum(x["rc_pct"] for x in w) - 1) < 1e-6
    assert d["validation"]["overall"] in ("PASS", "WARNING") and d["validation"]["validated"]
    mc = d["monte_carlo"]["profile"]
    assert len(mc["fan"]["years"]) == len(mc["fan"]["p50"]) and sum(mc["histogram"]["counts"]) <= 500
    assert all(r["es"] >= r["var"] - 1e-12 for r in d["var_es"] if r["var"] is not None)
    assert d["backtest"] is None  # fast mode: no walk-forward


def test_analyze_is_deterministic_and_cached(client):
    a = client.post("/api/analisar", json=FAST).json()
    b = client.post("/api/analisar", json=FAST).json()
    assert a["selected"] == b["selected"] and a["meta"]["run_id"] == b["meta"]["run_id"]


@pytest.mark.parametrize("payload", [
    {"simulations": 10_000_000},
    {"profile": "agressivo"},
    {"capital": -5},
    {"unexpected": 1},
    {"horizon_years": 100},
])
def test_invalid_inputs_rejected(client, payload):
    assert client.post("/api/analisar", json=payload).status_code == 422


def test_ask_grounded_answer_and_refusals(client):
    r = client.post("/api/perguntar", json={"params": FAST, "question": "Qual o risco da carteira?"}).json()
    assert r["verified"] and r["claims"] and all(c["call_id"] for c in r["claims"])
    r2 = client.post("/api/perguntar", json={"params": FAST,
                                             "question": "O SINT_CRIPTO vai subir amanhã? Devo comprar?"}).json()
    assert any("ordens" in x for x in r2["refusals"]) and any("Não é possível concluir" in x for x in r2["refusals"])
    assert client.post("/api/perguntar", json={"params": FAST, "question": "x" * 501}).status_code == 422


def test_page_never_injects_server_text_as_html():
    html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "eval(" not in html
