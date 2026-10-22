"""Probabilistic forecasts: the same model as the 'Vale a pena?' check in closed form, its historical calibration
without look-ahead, and the auditable registry (sealed records, hash chain, retroactive entries, pending until the
horizon ends), on real-data paths with test doubles (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm

from core.forecasting import audit, calibration
from core.forecasting.data import load, rf_growth, rf_index_from_annual_pct, rf_rate_at
from core.forecasting.forecast import build
from core.forecasting.model import fit, pit, prob_above, quantiles
from core.purpose.language import guarantee_phrases
from core.stock_check.analyzer import _expected, _gbm, _sim, _stats
from tests.market_doubles import END, Prov, cdi_json, http, structured_market
from webapp import server

MARKET_10Y = structured_market(n=2600)
KEY = b"chave-de-teste"


def http10(url: str, timeout: float) -> tuple[int, bytes]:
    """Listings + Selic as in ``http``, with 10 years of CDI."""
    if "bcdata.sgs.12" in url:
        return 200, cdi_json(n=2600)
    return http(url, timeout)


def _gbm_px(seed: int, n: int = 2600, mu: float = 0.12, vol: float = 0.28) -> pd.Series:
    rng = np.random.default_rng(seed)
    d = vol / np.sqrt(252)
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(mu / 252 - d * d / 2, d, n))),
                     index=pd.bdate_range(end=END, periods=n))


def _browser(obj):
    """What a browser hands back after JSON.parse/JSON.stringify: integral floats come back as integers."""
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x
    return json.loads(json.dumps(conv(obj)))


# ----------------------------------------------------------------------------- the model
def test_model_is_the_stock_check_model():
    px, bench = _gbm_px(1, 1300), _gbm_px(2, 1300, 0.10, 0.20)
    rf = 0.105
    st = _stats(px, bench, rf, None)
    exp = _expected(st, rf, 0.04, 0.10)
    r = px.pct_change(fill_method=None).dropna().to_numpy()
    br = bench.pct_change(fill_method=None).dropna().to_numpy()
    f = fit(r, rf, br, 0.04, 0.10)
    assert f.mu_e == pytest.approx(exp["retorno_esperado"], rel=1e-12)
    assert f.lam == pytest.approx(exp["peso_historico"], rel=1e-12) and f.prior == pytest.approx(exp["prior"])
    assert f.vol_log == pytest.approx(st["vol_log"], rel=1e-12) and f.beta == pytest.approx(st["beta"], rel=1e-12)
    # the closed form reproduces the stock check's Monte Carlo (same drift and volatility)
    sim = _sim(_gbm(f.mu_e, f.vol_log), 1.0, 1.0, 40_000, 7)
    tw = sim.terminal_wealth
    m, s = f.log_params(1.0)
    assert np.mean(tw >= 1 + rf) == pytest.approx(prob_above(m, s, rf), abs=0.01)
    assert np.quantile(tw - 1, 0.10) == pytest.approx(quantiles(m, s, (0.10,))[0.10], abs=0.01)


def test_scores_by_hand():
    p, o = np.array([0.9, 0.2, 0.6]), np.array([1.0, 0.0, 0.0])
    sc = calibration.score(p, o, np.array([0.5, 0.5, 0.5]))
    assert sc["brier"] == pytest.approx((0.01 + 0.04 + 0.36) / 3)
    assert sc["log_score"] == pytest.approx(-(np.log(0.9) + np.log(0.8) + np.log(0.4)) / 3)
    assert sc["brier_climatologia"] == pytest.approx(0.25)
    assert sc["habilidade_brier"] == pytest.approx(1 - sc["brier"] / 0.25)
    rel = {(b["de"], b["ate"]): b for b in sc["confiabilidade"]}
    assert rel[(0.8, 1.0)]["n"] == 1 and rel[(0.8, 1.0)]["observado"] == 1.0
    assert rel[(0.0, 0.2)]["n"] == 0 and rel[(0.2, 0.4)]["n"] == 1 and rel[(0.4, 0.6)]["n"] == 0
    assert pit(0.0, 1.0, np.e - 1) == pytest.approx(norm.cdf(1.0))
    assert prob_above(0.0, 1.0, -1.5) == 1.0 and prob_above(0.0, 1.0, 0.0) == pytest.approx(0.5)
    idx = rf_index_from_annual_pct(pd.Series([4.0, 4.0, 5.0], index=pd.bdate_range("2026-01-05", periods=3)))
    assert rf_rate_at(idx, idx.index[-1]) == pytest.approx(0.05) and rf_rate_at(idx, idx.index[0]) is None
    assert rf_growth(idx, idx.index[0], idx.index[-1]) == pytest.approx(1.04 ** (1 / 252) * 1.05 ** (1 / 252) - 1)


def test_calibration_uses_nothing_after_each_origin():
    px = _gbm_px(3)
    rf_idx = rf_index_from_annual_pct(pd.Series(10.0, index=px.index))
    k = 1800
    px2 = px.copy()
    px2.iloc[k:] *= 3.0                                   # the future changes...
    rf2 = rf_idx.copy()
    rf2.iloc[k:] *= 1.5
    a = calibration.run(px, 63, None, rf_idx, step=42)
    b = calibration.run(px2, 63, None, rf2, step=42)
    assert len(a) == len(b) > 20
    pos = {d: px.index.get_loc(pd.Timestamp(d)) for d in (r["data"] for r in a)}
    checked = 0
    for ra, rb in zip(a, b, strict=True):
        if pos[ra["data"]] < k:                           # ...forecasts made before it do not
            for f in ("p_alta", "p_superar_rf", "q10", "q50", "q90", "clim_alta", "clim_rf", "regime_vol",
                      "retorno_esperado"):
                assert ra[f] == rb[f], (ra["data"], f)
            checked += 1
            if pos[ra["data"]] + 63 < k:
                assert ra["realizado"] == rb["realizado"]
    assert checked > 10
    assert any(ra["realizado"] != rb["realizado"] for ra, rb in zip(a, b, strict=True))
    # an origin without the rate known on that day is skipped, never filled with zero
    assert calibration.run(px, 63, None, None, step=42) == []


def test_calibration_is_unbiased_when_the_model_is_true():
    rf_idx = rf_index_from_annual_pct(pd.Series(10.0, index=pd.bdate_range(end=END, periods=2600)))
    cov, above, below, pits = [], [], [], []
    for seed in range(12):
        df = pd.DataFrame(calibration.run(_gbm_px(100 + seed), 252, None, rf_idx, step=63))
        cov.append(df["dentro_10_90"].mean())
        above.append((df["realizado"] > df["q90"]).mean())
        below.append((df["realizado"] < df["q10"]).mean())
        pits.append(df["pit"].mean())
    assert 0.68 <= np.mean(cov) <= 0.90
    assert 0.03 <= np.mean(above) <= 0.18 and 0.03 <= np.mean(below) <= 0.18
    assert 0.42 <= np.mean(pits) <= 0.58
    s = calibration.summarise(calibration.run(_gbm_px(5), 252, None, rf_idx, step=21), 252)
    assert s["n_efetivo"] == pytest.approx(s["n"] * 21 / 252, abs=0.1) and "sobrepostos" in s["nota"]
    assert sum(s["pit_histograma"]) == s["n"] and s["evento_superar_rf"] is not None


# ----------------------------------------------------------------------------- forecast
def test_forecast_distribution_and_probabilities():
    s3 = load("PETR4.SA", Prov(MARKET_10Y), http10, end=date.fromisoformat(END))
    assert s3.currency == "BRL" and s3.bench_ticker == "BOVA11.SA" and s3.rf_index is not None
    out = build(s3, 1.0, 0.15, "Selic (teste)")
    d = out["registro"]["distribuicao"]
    m, s = d["m_log"], d["s_log"]
    q = out["quantis"]
    assert q["p05"] < q["p10"] < q["p25"] < q["p50"] < q["p75"] < q["p90"] < q["p95"]
    assert q["p50"] == pytest.approx(np.expm1(m))
    pr = out["probabilidades"]
    assert pr["p_alta"] == pytest.approx(norm.sf(-m / s))
    assert pr["p_superar_rf"] == pytest.approx(norm.sf((np.log(1.15) - m) / s))
    assert pr["p_queda_20"] == pytest.approx(norm.cdf((np.log(0.8) - m) / s))
    assert [c["retorno"] for c in out["cenarios"]] == [q["p10"], q["p50"], q["p90"]]
    assert sum(out["distribuicao"]["probabilidades"]) == pytest.approx(0.98, abs=1e-9)
    assert out["sensibilidade"][0]["p_superar_rf"] == pytest.approx(pr["p_superar_rf"])
    assert out["fonte_taxa"] == "Selic (teste)" and out["taxa_livre_de_risco"] == 0.15
    c = out["calibracao"]
    assert c["horizonte_pregoes"] == 252 and c["n"] >= 10 and c["evento_alta"]["brier"] >= 0
    assert out["registro"]["invalidacao"] and out["registro"]["premissas"]["modelo"].startswith("qpi-")
    assert not guarantee_phrases(json.dumps(out, ensure_ascii=False))
    # probability of beating the benchmark: exact under the model (bivariate normal of log returns)
    ref = out["referencia"]
    mb, sb, rho = ref["m_log"], ref["s_log"], ref["correlacao"]
    br = s3.bench.reindex(s3.px.index).pct_change(fill_method=None).iloc[-1260:].to_numpy()
    assert ref["ticker"] == "BOVA11.SA" and -1 <= rho <= 1
    assert sb == pytest.approx(np.log1p(br).std(ddof=1) * np.sqrt(252), rel=1e-9)
    assert pr["p_superar_referencia"] == pytest.approx(norm.cdf((m - mb) / np.sqrt(s * s + sb * sb - 2 * rho * s * sb)))
    z = np.random.default_rng(0).multivariate_normal([m, mb], [[s * s, rho * s * sb], [rho * s * sb, sb * sb]], 400_000)
    assert pr["p_superar_referencia"] == pytest.approx(np.mean(z[:, 0] > z[:, 1]), abs=0.005)


def test_forecast_without_rate_history_is_flagged_and_long_horizon_explained():
    def no_cdi(url, timeout):
        return (503, b"") if "bcdata.sgs.12" in url else http(url, timeout)

    s3 = load("PETR4.SA", Prov(MARKET_10Y), no_cdi, end=date.fromisoformat(END))
    assert s3.rf_index is None and any("taxa livre de risco histórica" in n for n in s3.notes)
    out = build(s3, 3.0)                      # no live rate either: demonstration value, flagged
    assert "DEMONSTRAÇÃO" in out["fonte_taxa"] and any("demonstração" in w for w in out["avisos"])
    assert out["calibracao"]["n"] == 0 and "nota_horizonte" in out["calibracao"]
    with pytest.raises(ValueError, match="horizonte"):
        build(s3, 0.1)


# ----------------------------------------------------------------------------- registry
def _record(as_of: str, horizon: float = 0.25):
    s3 = load("PETR4.SA", Prov(MARKET_10Y), http10, end=date.fromisoformat(as_of))
    return build(s3, horizon, 0.15, "Selic (teste)", with_calibration=False)["registro"]


def test_seal_register_and_chain():
    t = datetime(2025, 6, 3, 15, tzinfo=UTC)
    r1 = audit.seal(_record("2025-06-02"), KEY, now=t)
    r2 = audit.seal(_record("2025-06-02", 1.0), KEY, now=t)
    e1 = audit.register(_browser(r1), audit.GENESIS, KEY, now=t)       # survives a trip through the browser
    e2 = audit.register(_browser(r2), e1["hash"], KEY, now=t)
    chain = [_browser(e1), _browser(e2)]
    assert all(c["ok"] for c in audit.verify(chain, KEY))
    assert not audit.is_retroactive(e1["registro"], e1["registrado_em"])
    # a changed probability is caught by the signature at registration...
    bad = _browser(r1)
    bad["probabilidades"]["p_alta"] = 0.99
    with pytest.raises(ValueError, match="assinatura"):
        audit.register(bad, audit.GENESIS, KEY, now=t)
    # ...and by the chain afterwards, even when the hash is recomputed
    tampered = json.loads(json.dumps(chain))
    tampered[0]["registro"]["probabilidades"]["p_alta"] = 0.99
    v = audit.verify(tampered, KEY)
    assert not v[0]["ok"] and any("hash" in p for p in v[0]["problemas"])
    tampered[0]["hash"] = audit._entry_hash(tampered[0])
    v = audit.verify(tampered, KEY)
    assert not v[0]["ok"] and not v[1]["ok"] and any("cadeia" in p for p in v[1]["problemas"])
    # removing or reordering entries breaks the chain
    assert not audit.verify([chain[1]], KEY)[0]["ok"]
    assert not audit.verify([chain[1], chain[0]], KEY)[0]["ok"]
    # without the server key a full rewrite is not detectable — the limitation stated in the result
    loose = [audit.register(_browser(audit.seal(_record("2025-06-02"), None, now=t)), audit.GENESIS, None, now=t)]
    loose[0]["registro"]["probabilidades"]["p_alta"] = 0.99
    loose[0]["hash"] = audit._entry_hash(loose[0])
    assert audit.verify(loose, None)[0]["ok"]
    with pytest.raises(ValueError, match="incompleto"):
        audit.register({"ativo": {"ticker": "X"}}, audit.GENESIS, None, now=t)


def test_pending_evaluated_and_retroactive():
    t = datetime(2025, 6, 3, 15, tzinfo=UTC)
    rec = audit.seal(_record("2025-06-02"), KEY, now=t)                 # 3-month horizon from 2025-06-02
    e1 = audit.register(rec, audit.GENESIS, KEY, now=t)
    late = datetime(2025, 6, 20, 15, tzinfo=UTC)                       # 14 business days after the reference
    e2 = audit.register(audit.seal(_record("2025-06-02"), KEY, now=late), e1["hash"], KEY, now=late)
    prov = Prov(MARKET_10Y)
    mid = audit.evaluate([e1, e2], prov, http10, as_of=date(2025, 7, 15), key=KEY)
    st = [x["estado"] for x in mid["entradas"]]
    assert st == ["pendente", "retroativa"] and "metricas" not in mid
    px = MARKET_10Y["PETR4.SA"].close
    p0 = float(px.loc[:"2025-06-02"].iloc[-1])
    assert mid["entradas"][0]["retorno_ate_agora"] == pytest.approx(float(px.loc[:"2025-07-15"].iloc[-1]) / p0 - 1)
    done = audit.evaluate([e1, e2], prov, http10, as_of=date.fromisoformat(END), cost_bps=25, key=KEY)
    row = done["entradas"][0]
    assert [x["estado"] for x in done["entradas"]] == ["avaliada", "retroativa"]
    end = pd.Timestamp(rec["data_fim"])
    R = float(px.loc[:end].iloc[-1]) / p0 - 1
    assert row["realizado"] == pytest.approx(R)
    assert row["realizado_liquido"] == pytest.approx((1 + R) * (1 - 0.0025) ** 2 - 1)
    n_days = len(pd.bdate_range("2025-06-03", px.loc[:end].index[-1]))   # CDI double: 0.04% a business day
    assert row["rf_realizado"] == pytest.approx(1.0004 ** n_days - 1, rel=1e-9)
    m = done["metricas"]
    p_up = rec["probabilidades"]["p_alta"]
    assert m["n"] == 1 and m["alta"]["brier"] == pytest.approx((p_up - (R > 0)) ** 2)
    d = rec["distribuicao"]
    assert row["pit"] == pytest.approx(norm.cdf((np.log1p(R) - d["m_log"]) / d["s_log"]))
    assert any("menos de 30" in n for n in done["notas"])


def test_endpoints(monkeypatch):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(MARKET_10Y))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http10)
    monkeypatch.setenv(audit.KEY_ENV, "segredo-do-servidor")
    c = TestClient(server.app)
    r = c.post("/api/previsoes/prever", json={"ativo": "PETR4", "horizonte_anos": 1})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["ticker"] == "PETR4.SA" and out["registro"]["assinatura"] and out["do_cache"] is False
    assert "Selic" in out["fonte_taxa"] and out["calibracao"]["n"] >= 10
    reg = c.post("/api/previsoes/registrar", json={"registro": out["registro"]})
    assert reg.status_code == 200, reg.text
    entry = reg.json()["entrada"]
    assert reg.json()["assinado_pelo_servidor"] is True and entry["hash_anterior"] == audit.GENESIS
    forged = json.loads(json.dumps(out["registro"]))
    forged["probabilidades"]["p_superar_rf"] = 0.95
    assert c.post("/api/previsoes/registrar", json={"registro": forged}).status_code == 400
    ev = c.post("/api/previsoes/avaliar", json={"entradas": [entry]})
    assert ev.status_code == 200, ev.text
    lag = int(np.busday_count(date.fromisoformat(out["data_referencia"]), datetime.now(UTC).date()))
    assert ev.json()["entradas"][0]["estado"] == ("retroativa" if lag > audit.RETRO_BDAYS else "pendente")
    assert c.post("/api/previsoes/prever", json={"ativo": "SINT_ENERGIA3"}).status_code == 400
    assert c.post("/api/previsoes/prever", json={"ativo": "PETR4", "horizonte_anos": 20}).status_code == 422
    assert c.post("/api/previsoes/prever", json={"ativo": "ZZZZ9"}).status_code == 503
