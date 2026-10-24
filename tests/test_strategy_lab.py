"""Strategy lab (published hypotheses, walk-forward with costs and overfitting controls) and the optimisation
laboratory on real prices — test doubles only, with planted signals so the verdicts can be checked."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.strategy_lab.lab import CASH, INDEX, build_returns, run_lab
from tests.market_doubles import END, Prov, cdi_json, http, structured_market
from webapp import server

N = 2600


def _market(drifts: list[float], vol: float = 0.25, seed: int = 11) -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=END, periods=N)
    d = vol / np.sqrt(252)
    rets = np.column_stack([rng.normal(mu / 252 - d * d / 2, d, N) for mu in drifts])
    prices = {f"S{i:02d}.SA": pd.Series(20 * np.exp(np.cumsum(rets[:, i])), index=idx) for i in range(len(drifts))}
    index = pd.Series(100 * np.exp(np.cumsum(rets.mean(axis=1))), index=idx)
    cash = pd.Series(100 * (1.10 ** (1 / 252)) ** np.arange(1, N + 1), index=idx)
    return prices, index, cash


def test_planted_momentum_is_found_and_no_signal_is_not():
    prices, index, cash = _market([0.30] * 10 + [-0.10] * 10)        # persistent winners: momentum must work
    frame, used, left = build_returns(prices, index, cash)
    assert len(frame) == 2520 and len(used) == 20 and left == [] and {CASH, INDEX} <= set(frame.columns)
    r = run_lab("momento", frame, used, 15)
    assert r["veredito"] == "há evidência" and all(c["ok"] for c in r["criterios"])
    best = max(r["variantes"], key=lambda v: v["sharpe_validacao"])
    assert r["variante_escolhida"] == best["variante"]                 # chosen on validation only
    assert r["sem_olhar_o_futuro"]["passou"] and r["sem_olhar_o_futuro"]["decisoes_conferidas"] > 10
    assert r["n_variantes"] == 4 and r["dsr"] >= 0.95
    flat_prices, flat_index, flat_cash = _market([0.10] * 20, seed=5)  # no signal at all
    f2, u2, _ = build_returns(flat_prices, flat_index, flat_cash)
    for hyp in ("momento", "reversao", "baixa_volatilidade"):
        assert run_lab(hyp, f2, u2, 15)["veredito"] != "há evidência", hyp


def test_costs_rules_and_cash():
    prices, index, cash = _market([0.30] * 10 + [-0.10] * 10)
    frame, used, _ = build_returns(prices, index, cash)
    cheap = run_lab("reversao", frame, used, 0)
    dear = run_lab("reversao", frame, used, 100)
    for a, b in zip(cheap["variantes"], dear["variantes"], strict=True):
        assert b["custo_anual"] > a["custo_anual"] > 0 and a["giro_anual"] > 1          # slippage (5 bps) always applies
    trend = run_lab("tendencia", frame, used, 15)
    assert {v["variante"] for v in trend["variantes"]} == {"média de 100 dias", "média de 200 dias"}
    no_cash, used2, _ = build_returns(prices, index, None)
    with pytest.raises(ValueError, match="caixa"):
        run_lab("tendencia", no_cash, used2, 15)
    assert any("zero" in w for w in run_lab("momento", no_cash, used2, 15)["avisos"])  # stated, never silent
    with pytest.raises(ValueError, match="hipótese"):
        run_lab("astrologia", frame, used, 15)


def test_window_and_gaps():
    prices, index, cash = _market([0.1] * 12)
    prices["S00.SA"] = prices["S00.SA"].iloc[-1500:]                   # 6 years only
    frame, used, left = build_returns(prices, index, cash)
    assert len(frame) == 2520 and "S00.SA" in left and len(used) == 11
    short = {k: v.iloc[-1500:] for k, v in prices.items()}             # nobody has 10 years: 5 are used
    f5, u5, _ = build_returns(short, index, cash)
    assert len(f5) == 1260 and len(u5) == 12


MARKET_10Y = structured_market(n=2600)


def http10(url: str, timeout: float) -> tuple[int, bytes]:
    if "bcdata.sgs.12" in url:
        return 200, cdi_json(n=2600)
    return http(url, timeout)


def test_strategy_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http10)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/estrategias", json={"hipotese": "baixa_volatilidade", "mercado": "BR"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["veredito"] in ("há evidência", "evidência fraca", "sem evidência") and d["caixa"] is True
    assert d["indice"] == "BOVA11.SA" and len(d["curvas"]["series"]) == 3 and d["sem_olhar_o_futuro"]["passou"]
    assert c.post("/api/estrategias", json={"hipotese": "sorte"}).status_code == 422
    mixed = {"hipotese": "momento", "ativos": ["PETR4", "VALE3", "ITUB4", "WEGE3", "AAPL", "MSFT", "NVDA", "GOOGL"]}
    assert c.post("/api/estrategias", json=mixed).status_code == 400
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y, fail={"BOVA11.SA"}))
    assert c.post("/api/estrategias", json={"hipotese": "momento", "mercado": "BR"}).status_code == 503


def test_optimisation_lab_on_real_prices(monkeypatch, tmp_path):
    mk = structured_market()
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(mk))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    body = {"fonte_dados": "real", "ativos": ["PETR4", "VALE3", "ITUB4", "WEGE3", "IVVB11"], "simulations": 2000, "seed": 3}
    r = c.post("/api/analisar", json=body)
    assert r.status_code == 200, r.text
    m = r.json()["meta"]
    assert m["synthetic"] is False and m["risk_free_rate"] == pytest.approx(0.15) and not m["risk_free_is_fallback"]
    assert set(m["symbols"]) == {"PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "IVVB11.SA", "BOVA11.SA"}
    ask = c.post("/api/perguntar", json={"params": body, "question": "Qual a volatilidade?"})
    assert ask.status_code == 200 and ask.json()["claims"]
    assert c.post("/api/analisar", json={"fonte_dados": "real", "ativos": ["PETR4"]}).status_code == 400
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(mk, fail=set(m["symbols"])))
    assert c.post("/api/analisar", json={**body, "seed": 4}).status_code == 503
    syn = c.post("/api/analisar", json={"simulations": 2000})
    assert syn.status_code == 200 and syn.json()["meta"]["synthetic"] is True        # the demonstration still works
