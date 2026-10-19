"""'Vale a pena?' single-asset check: resolution, rules, shrinkage, projection, real path (test
doubles), web endpoint and CLI. No network is used."""

import json

import numpy as np
import pandas as pd
import pytest

from core.market_data.base import DataUnavailableError, RawSeries
from core.purpose.language import guarantee_phrases
from core.stock_check.analyzer import (
    _expected,
    _verdict,
    analyze_stock,
    render_markdown,
    to_dict,
)
from core.stock_check.resolver import resolve


@pytest.mark.parametrize("text,ticker", [
    ("PETR4", "PETR4.SA"), ("petr4", "PETR4.SA"), ("Petrobras", "PETR4.SA"), ("Itaú", "ITUB4.SA"),
    ("vale", "VALE3.SA"), ("Vale a pena PETR4?", "PETR4.SA"), ("AAPL", "AAPL"), ("Tesla", "TSLA"),
    ("Banco do Brasil é bom?", "BBAS3.SA"), ("SINT_ENERGIA3", "SINT_ENERGIA3"), ("bova11.sa", "BOVA11.SA"),
])
def test_resolver_maps_codes_and_names(text, ticker):
    assert resolve(text).ticker == ticker


def test_resolver_never_guesses():
    for bad in ("vale a pena?", "", "quero investir em alguma coisa boa", "x" * 61):
        with pytest.raises(ValueError):
            resolve(bad)
    r = resolve("ZZZZ")                                      # unknown but well-formed: kept as typed
    assert r.ticker == "ZZZZ" and r.business is None


@pytest.mark.parametrize("p,t,label", [(0.65, 1.2, "FAVORÁVEL"), (0.65, 0.5, "NEUTRO"), (0.35, 0.5, "DESFAVORÁVEL"),
                                       (0.5, -1.5, "DESFAVORÁVEL"), (0.5, None, "NEUTRO"), (0.60, 1.0, "FAVORÁVEL"),
                                       (0.40, 0.0, "DESFAVORÁVEL")])
def test_verdict_rules(p, t, label):
    assert _verdict(p, t)["rotulo"] == label


def test_shrinkage_weights_history_by_its_precision():
    noisy = _expected({"beta": 1.0, "retorno_medio_anual": 0.40, "volatilidade": 0.40, "anos_de_dados": 2.0},
                      0.10, 0.04, 0.10)
    precise = _expected({"beta": 0.0, "retorno_medio_anual": 0.105, "volatilidade": 0.01, "anos_de_dados": 5.0},
                        0.10, 0.04, 0.10)
    assert noisy["peso_historico"] < 0.2 and precise["peso_historico"] > 0.99
    for e in (noisy, precise):
        assert e["retorno_esperado"] == pytest.approx(e["prior"] + e["peso_historico"]
                                                      * (e["retorno_historico"] - e["prior"]))
    assert noisy["prior"] == pytest.approx(0.14) and precise["prior"] == pytest.approx(0.10)


@pytest.fixture(scope="module")
def syn():
    return analyze_stock("SINT_ENERGIA3", horizon_years=3, amount=10_000, profile="moderado", offline=True,
                         n_paths=3000)


def test_synthetic_check_projection_is_consistent(syn):
    pj = syn.projection
    f = pj["fan"]
    for i in range(len(f["years"])):
        assert f["p5"][i] <= f["p25"][i] <= f["p50"][i] <= f["p75"][i] <= f["p95"][i]
    assert f["rf_line"][0] == pytest.approx(10_000) and f["rf_line"][-1] == pytest.approx(pj["valor_rf_no_horizonte"])
    vals = [c["valor_final"] for c in pj["cenarios"]]
    assert vals[0] >= vals[1] >= vals[2]
    p, n = pj["p_superar_rf"], pj["n_trajetorias"]
    assert pj["p_superar_rf_ep"] == pytest.approx(np.sqrt(p * (1 - p) / n))
    assert len(pj["sensibilidade"]) == 4 and pj["sensibilidade"][0]["p_superar_rf"] == pytest.approx(p)
    assert syn.verdict["rotulo"] in {"FAVORÁVEL", "NEUTRO", "DESFAVORÁVEL"}
    assert syn.stats["beta"] is not None and syn.rf_source.startswith("FALLBACK")


def test_synthetic_check_language_and_labels(syn):
    md = render_markdown(syn)
    assert "DADOS SINTÉTICOS" in md and "Segundo a simulação" in md and "não é preço-alvo" in md
    assert guarantee_phrases(md) == []
    json.dumps(to_dict(syn))


def test_mode_guards():
    with pytest.raises(ValueError, match="SINT_"):
        analyze_stock("PETR4", offline=True)
    with pytest.raises(ValueError, match="sintéticos"):
        analyze_stock("SINT_ENERGIA3", offline=False)
    with pytest.raises(ValueError):
        analyze_stock("SINT_ENERGIA3", horizon_years=0.1, offline=True)


class _Prov:
    name = "fake_test_double"

    def __init__(self, series):
        self.series = series
        self.failures = {}

    def fetch(self, symbols, period, interval):
        out = {s: self.series[s] for s in symbols if s in self.series}
        for s in symbols:
            if s not in out:
                self.failures[s] = f"{s}: erro de rede: simulado"
        return out


def _series(sym, n, ccy, seed, vol=0.018, drift=0.0005, with_volume=True):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    rng = np.random.default_rng(seed)
    px = pd.Series(30 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx, name=sym)
    volume = pd.Series(rng.integers(5_000_000, 9_000_000, n).astype(float), index=idx) if with_volume else None
    return RawSeries(sym, px, volume, "fake_test_double", "adjclose", "teste", ccy, "1d")


def test_real_path_with_test_doubles():
    common = _series("BOVA11.SA", 1250, "BRL", 1, vol=0.012)
    own = _series("PETR4.SA", 1250, "BRL", 2)
    own = RawSeries("PETR4.SA", (own.close * 0.5 + common.close * 0.5).rename("PETR4.SA"), own.volume,
                    own.provider, own.price_field, own.adjusted_for, "BRL", "1d")
    prov = _Prov({"PETR4.SA": own, "BOVA11.SA": common})
    r = analyze_stock("Petrobras", horizon_years=3, amount=5_000, profile="arrojado", provider=prov,
                      rf_fetch=lambda url, t: (200, b'[{"data":"16/10/2026","valor":"15.00"}]'), n_paths=2000)
    assert r.origin == "real" and r.currency == "BRL" and r.rf == pytest.approx(0.15)
    assert "Banco Central" in r.rf_source and r.last_date == "2026-10-16"
    crit = {c["criterio"]: c for c in r.verdict["criterios"]}
    assert any("Ibovespa" in k for k in crit) and any("Giro" in k for k in crit)
    assert 0.3 < r.stats["correlacao"] <= 1.0 and r.stats["beta"] is not None
    assert "não descrito" not in render_markdown(r)                       # catalogue asset: business known


def test_real_path_refuses_short_history_and_missing_data():
    prov = _Prov({"VALE3.SA": _series("VALE3.SA", 120, "BRL", 3)})
    with pytest.raises(DataUnavailableError, match="252"):
        analyze_stock("VALE3", provider=prov, rf_fetch=lambda url, t: (500, b""))
    with pytest.raises(DataUnavailableError, match="ITUB4"):
        analyze_stock("ITUB4", provider=_Prov({}), rf_fetch=lambda url, t: (500, b""))


def test_web_endpoint_stock(monkeypatch):
    from fastapi.testclient import TestClient

    import webapp.server as srv

    c = TestClient(srv.app)
    d = c.post("/api/acao", json={"acao": "SINT_TECH_US", "fonte_dados": "sintetico", "n_simulacoes": 1000}).json()
    assert d["ticker"] == "SINT_TECH_US" and d["moeda"] == "USD" and d["veredito"]["rotulo"]
    assert d["projecao"]["fan"]["p50"] and d["markdown"].startswith("# Vale a pena?")
    assert c.post("/api/acao", json={"acao": "vale a pena?", "fonte_dados": "sintetico"}).status_code == 400
    assert c.post("/api/acao", json={"acao": "PETR4", "valor": 0}).status_code == 422
    monkeypatch.setattr(srv, "_market_provider", lambda: _Prov({}))
    monkeypatch.setattr(srv, "_rf_fetch", lambda: (lambda url, t: (500, b"")))
    r = c.post("/api/acao", json={"acao": "PETR4", "seed": 3})
    assert r.status_code == 503 and "Nada foi estimado" in r.json()["detail"]


def test_cli_stock(tmp_path):
    from main import main

    assert main(["--offline", "--acao", "SINT_SAUDE3", "--horizon-years", "1", "--simulations", "500",
                 "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "acao" / "SINT_SAUDE3.md").exists() and (tmp_path / "acao" / "SINT_SAUDE3.json").exists()
    assert main(["--offline", "--acao", "PETR4", "--output-dir", str(tmp_path)]) == 2
