"""Mission tests for the dynamic portfolio engine: the portfolio comes from data, the investor's own
limits and an executable plan — never from a fixed list — and it changes only when something real
changes. Real-data path with test doubles (no network): fake exchange listings, a structured market in
which asset families differ in drift and volatility, and the Banco Central series (Selic and CDI)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from core.market_data.base import DataUnavailableError
from core.market_data.cash import CASH_ASSET, CASH_TICKER
from core.optimization import optimizers as opt
from core.optimization.constraints import verify_weights
from core.purpose.catalog import REAL_CATALOG
from core.purpose.changes import MATERIAL, compare, fingerprint
from core.purpose.concentration import analyse, effective_bets
from core.purpose.engine import PurposeRequest, run_purpose
from core.purpose.feasibility import (
    contribution_plan,
    executable_universe,
    lot_size,
    plan,
)
from core.purpose.language import guarantee_phrases
from core.purpose.personalization import CASH_TICKERS, build_policy, traits
from core.purpose.report import render_markdown, to_dict
from core.universe.scoring import screen
from models.investor import RiskProfile
from models.portfolio import PortfolioConstraints
from tests.market_doubles import (
    Prov,
    http,
    http_without_cdi,
    real_symbols,
    structured_market,
)

MARKET = structured_market()
BASE = {"capital_inicial": 100_000, "aporte_periodico": 1_000, "valor_alvo": 500_000, "horizonte_anos": 10,
            "perfil_risco": "moderado", "offline": False, "universo": "completo", "max_ativos": 20, "n_simulacoes": 500}
CAT = {a.ticker: a for a in (*REAL_CATALOG, CASH_ASSET)}


def _trunc(market, k):
    """The same market seen k business days earlier."""
    return {s: dataclasses.replace(rs, close=rs.close.iloc[:-k], volume=None if rs.volume is None else rs.volume.iloc[:-k])
            for s, rs in market.items()}


def _http_no_listings(url, timeout):
    return http(url, timeout) if "bcb.gov.br" in url else (404, b"")


CONFIGS = {
    "base": {},
    "cons": {"perfil_risco": "conservador"},
    "aggr": {"perfil_risco": "arrojado"},
    "small": {"capital_inicial": 1_000, "aporte_periodico": 0, "periodicidade": "nenhum"},
    "big": {"capital_inicial": 2_000_000, "aporte_periodico": 0, "periodicidade": "nenhum"},
    "h15": {"horizonte_anos": 15},
    "h2": {"horizonte_anos": 2},
    "br": {"mercados": ["BR"]},
    "curado": {"_http": _http_no_listings},
    "market2": {"_market": structured_market(seed_offset=1)},
    "prev5": {"_market": _trunc(MARKET, 5)},
}


def _run(tmp, market=None, http_fn=http, prov=None, **kw):
    req = PurposeRequest(**{**BASE, **kw})
    prov = prov or Prov(MARKET if market is None else market)
    return run_purpose(req, output_dir=tmp, provider=prov, cache_dir=tmp, rf_fetch=http_fn, fast=True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    done = {}

    def get(name):
        if name not in done:
            kw = dict(CONFIGS[name])
            done[name] = _run(tmp_path_factory.mktemp(name), kw.pop("_market", None), kw.pop("_http", http), **kw)
        return done[name]
    return get


def _weights(r):
    w = r.combined.final
    return w[w > 1e-9]


def _share(r, pred):
    return float(sum(w for t, w in _weights(r).items() if pred(CAT[t])))


# ----------------------------------------------------------------------------- 1. profile
def test_1_profile_changes_the_constraints_and_can_change_the_portfolio(runs):
    cons, base, aggr = runs("cons"), runs("base"), runs("aggr")
    caps = [r.policy["teto_volatilidade"] for r in (cons, base, aggr)]
    lams = [r.policy["aversao_ao_risco"] for r in (cons, base, aggr)]
    assert caps == sorted(caps) and len(set(caps)) == 3                 # tighter risk cap for lower risk
    assert lams == sorted(lams, reverse=True) and len(set(lams)) == 3   # higher risk aversion for lower risk
    assert cons.policy["minimos_por_classe"]["fixed_income"] > base.policy["minimos_por_classe"]["fixed_income"]
    assert len({r.reproducibility["impressao_digital_carteira"] for r in (cons, base, aggr)}) == 3
    assert _share(cons, lambda a: a.asset_class == "fixed_income") >= 0.40 - 1e-6
    assert _share(cons, lambda a: a.asset_class == "equity") <= 0.30 + 1e-6
    for r in (cons, base, aggr):
        cap = r.analysis.profile.research_limits.target_volatility_band[1]
        assert r.combined.vol_final <= max(r.combined.vol_quant, cap) + 1e-6


# ----------------------------------------------------------------------------- 2. capital
def test_2_capital_change_recalculates_feasibility(runs):
    small, big = runs("small"), runs("big")
    assert small.execution["valor_de_referencia"] == 1_000 and big.execution["valor_de_referencia"] == 2_000_000
    out_small = len(small.execution["fora_por_tamanho_minimo"]) + len(small.execution["retirados_apos_otimizar"])
    out_big = len(big.execution["fora_por_tamanho_minimo"]) + len(big.execution["retirados_apos_otimizar"])
    assert out_small > out_big                       # small money: fewer assets fit a sensible purchase
    assert len(_weights(small)) < len(_weights(big))
    for r, amount in ((small, 1_000), (big, 2_000_000)):
        ci = r.execution["compra_inicial"]
        spent = sum(p["valor"] for p in ci["posicoes"])
        assert spent + ci["caixa_nao_investido"] == pytest.approx(amount, abs=0.05)
        assert ci["viavel"]                          # every target position buys at least one lot
        for p in ci["posicoes"]:
            step, _ = lot_size(CAT[p["ticker"]])
            assert p["quantidade"] / step == pytest.approx(round(p["quantidade"] / step), abs=1e-6)
            if not traits(CAT[p["ticker"]]).cash:
                assert p["valor"] >= min(100.0, p["preco"]) - 1e-6 or p["valor"] >= p["preco"] - 1e-6


# ----------------------------------------------------------------------------- 3. horizon
def test_3_horizon_changes_the_evaluation_and_the_portfolio_only_when_justified(runs):
    base, longer, short = runs("base"), runs("h15"), runs("h2")
    assert longer.required.rate != base.required.rate                    # the evaluation is redone
    assert longer.goal["p_objetivo"] != base.goal["p_objetivo"] or longer.goal["valor_alvo"] == base.goal["valor_alvo"]
    # 10 -> 15 years changes nothing in the investor's risk limits: same portfolio, no manufactured novelty
    assert longer.policy == {**base.policy}
    assert longer.reproducibility["impressao_digital_carteira"] == base.reproducibility["impressao_digital_carteira"]
    # 2 years: little time to recover from a fall -> cash floor, equity cap, no long bonds, other portfolio
    assert short.policy["perfil_final"] == "moderado" and short.policy["caixa_minimo"] >= 0.20
    assert any("Horizonte de 1 a 3 anos" in x for x in short.policy["regras"])
    assert _share(short, lambda a: traits(a).cash) >= 0.20 - 1e-6
    assert _share(short, lambda a: a.asset_class == "equity") <= 0.30 + 1e-6
    assert "IMAB11.SA" in short.universe["excluidos_por_preferencia"]      # long-duration bonds out
    assert short.reproducibility["impressao_digital_carteira"] != base.reproducibility["impressao_digital_carteira"]


# ----------------------------------------------------------------------------- 4. universe
def test_4_universe_change_changes_the_candidates(runs, tmp_path):
    base, br, cur = runs("base"), runs("br"), runs("curado")
    assert set(br.universe["candidatos"]) != set(base.universe["candidatos"])
    assert all(t.endswith(".SA") or t == CASH_TICKER for t in br.universe["baixados"])
    assert any("EUA fora" in why for why in br.universe["excluidos_por_preferencia"].values())
    assert not cur.universe["funil"]["used_listings"] and set(cur.universe["baixados"]) != set(base.universe["baixados"])
    assert any(a.severity == "WARNING" and "Listagens completas" in a.message for a in cur.alerts)
    top = _weights(base).idxmax()
    excl = _run(tmp_path, ativos_excluidos=[top.removesuffix(".SA")])
    assert top not in excl.universe["baixados"] and top not in _weights(excl).index
    assert excl.universe["excluidos_por_preferencia"][top] == "excluído por você"


def test_4b_assets_are_not_dropped_for_being_new_nor_added_for_variety(runs):
    base = runs("base")
    t = base.screening.table
    # every selected candidate has a reason grounded in data or in the investor's rules; no "variety" slot
    for tk in base.universe["candidatos"]:
        why = t.at[tk, "motivo"]
        assert any(k in why for k in ("pontuação", "necessário para cumprir", "tema", "recomendação", "já tem")), why
    assert not any("variedade" in str(m) for m in t["motivo"])


# ----------------------------------------------------------------------------- 5. market data
def test_5_new_market_data_triggers_a_new_analysis(runs, tmp_path):
    base, other = runs("base"), runs("market2")
    assert other.reproducibility["parametros"] == base.reproducibility["parametros"]
    assert other.reproducibility["dados"] != base.reproducibility["dados"]
    assert other.reproducibility["impressao_digital_carteira"] != base.reproducibility["impressao_digital_carteira"]
    # one downloaded asset whose prospects improve sharply in the data is picked up
    w0 = _weights(base)
    pool = [s for s in base.universe["baixados"] if s not in w0.index and CAT.get(s) is not None
            and CAT[s].asset_class == "equity" and s in MARKET]
    boosted = pool[0]
    r = _run(tmp_path, market=structured_market(drift_shift={boosted: 0.8}))
    assert boosted in r.universe["candidatos"] and r.combined.final.get(boosted, 0.0) > 0.005


# ----------------------------------------------------------------------------- 6. bad assets
def test_6_unavailable_duplicate_and_short_history_assets_are_handled(tmp_path):
    m = dict(MARKET)
    m["WEGE3.SA"] = dataclasses.replace(m["WEGE3.SA"], close=m["WEGE3.SA"].close.iloc[-100:],
                                        volume=m["WEGE3.SA"].volume.iloc[-100:])
    prov = Prov(m, fail={"ITUB4.SA"})
    r = _run(tmp_path, prov=prov, carteira_atual={"WEGE3": 5_000, "ITUB4": 5_000, "ZZZZ3": 1_000})
    ex = r.universe["excluidos_por_dados"]
    assert "history" in ex["WEGE3.SA"] and "download failure" in ex["ITUB4.SA"]
    assert "ZZZZ3.SA" in r.universe["fora_da_analise"]
    for t in ("WEGE3.SA", "ITUB4.SA", "ZZZZ3.SA"):
        assert t not in _weights(r).index
    assert any("ZZZZ3.SA (da sua carteira atual) não entra na análise" in a.message for a in r.alerts)
    assert {c.ticker: c.kind for c in r.changes.changes}["ITUB4.SA"] == "saiu"
    assert "dados insuficientes" in {c.ticker: c.reason for c in r.changes.changes}["ITUB4.SA"]


def test_6b_screen_drops_duplicate_exposures_and_keeps_required_groups():
    idx = pd.bdate_range(end="2026-10-16", periods=600)
    rng = np.random.default_rng(3)
    base = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 600))), index=idx)
    series = {"SPY": base, "VOO": base * 1.1,                               # same exposure, different price
              "QQQ": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 600))), index=idx),
              CASH_TICKER: pd.Series(100 * np.cumprod(np.full(600, 1.0004)), index=idx)}
    assets = {s: CAT[s] for s in series}
    sr = screen(series, None, assets, RiskProfile.MODERADO, "crescimento", 0.10, 3,
                required=[("o caixa mínimo", [CASH_TICKER], 1)], keep={"QQQ": "você já tem este ativo"})
    assert set(sr.duplicates) & {"SPY", "VOO"} and len(set(sr.duplicates) & {"SPY", "VOO"}) == 1
    assert CASH_TICKER in sr.selected and "QQQ" in sr.selected
    dropped = next(iter(set(sr.duplicates) & {"SPY", "VOO"}))
    assert "mesma exposição" in sr.table.at[dropped, "motivo"]
    assert sr.table.at["QQQ", "motivo"] == "você já tem este ativo"


# ----------------------------------------------------------------------------- 7. failures
def test_7_a_data_failure_never_returns_a_recommendation(tmp_path):
    every = set(real_symbols())
    with pytest.raises(DataUnavailableError):
        _run(tmp_path / "a", prov=Prov(MARKET, fail=every))
    most = {s for i, s in enumerate(sorted(every)) if i % 10 < 8}
    with pytest.raises(DataUnavailableError, match="não seria confiável"):
        _run(tmp_path / "b", prov=Prov(MARKET, fail=most))
    r = _run(tmp_path / "c", http_fn=http_without_cdi)              # the CDI source is down: said, not invented
    assert CASH_TICKER not in r.universe["baixados"]
    assert any(a.severity == "WARNING" and "CDI" in a.message for a in r.alerts)


def test_7b_web_never_serves_an_expired_real_result_and_the_page_hides_old_ones(monkeypatch):
    import webapp.server as srv

    srv._store("k-test", {"x": 1})
    assert srv._cached("k-test", True)["do_cache"] is True
    later = time.time() + srv._RESULT_TTL_S + 1
    monkeypatch.setattr(srv.time, "time", lambda: later)
    assert srv._cached("k-test", True) is None                          # real data: recomputed after 15 min
    srv._store("k-syn", {"x": 2})
    assert srv._cached("k-syn", False) is not None                      # synthetic: same seed, same numbers
    html = srv._INDEX_HTML
    i_hide, i_post = html.index('$("p-results").classList.add("hidden")'), html.index('postStream("/api/proposito"')
    assert i_hide < i_post and "um resultado antigo não substitui" in html


# ----------------------------------------------------------------------------- 8. limits
@pytest.mark.parametrize("name", ["base", "cons", "aggr", "small", "h2"])
def test_8_weights_limits_and_sums_are_validated(runs, name):
    r = runs(name)
    pc = r.analysis.constraints
    w = r.combined.final.reindex(pc.symbols).fillna(0.0).to_numpy()
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all()
    assert verify_weights(w, pc) == []
    lim = r.analysis.profile.research_limits
    for t, x in _weights(r).items():
        assert x <= (1.0 if traits(CAT[t]).cash else lim.max_weight_per_asset) + 1e-6
    ci = r.execution["compra_inicial"]
    assert sum(p["valor"] for p in ci["posicoes"]) <= r.request.capital_inicial + 0.01
    assert sum(p.get("peso_executado", p.get("peso_apos_compra")) for p in ci["posicoes"]) <= 1.0 + 1e-9
    assert r.execution["valor_de_referencia"] == pytest.approx(
        r.request.capital_inicial + (12 * r.request.aporte_periodico if r.request.periodicidade == "mensal" else 0))


# ----------------------------------------------------------------------------- 9. reproducible
def test_9_same_data_parameters_and_version_give_the_same_result(runs, tmp_path):
    base = runs("base")
    again = _run(tmp_path)
    assert again.reproducibility == base.reproducibility
    pd.testing.assert_series_equal(again.combined.final, base.combined.final)
    assert again.execution["compra_inicial"] == base.execution["compra_inicial"]
    assert again.analysis.selected_name == base.analysis.selected_name


# ----------------------------------------------------------------------------- 10. real changes only
def test_10_a_portfolio_is_shown_as_different_only_when_it_really_changes(runs, tmp_path):
    prev = runs("prev5")                                     # recommendation made 5 trading days ago
    saved = to_dict(prev)["para_guardar"]
    fresh = runs("base")                                     # what today's data alone would give
    assert fresh.reproducibility["impressao_digital_carteira"] != prev.reproducibility["impressao_digital_carteira"]
    same = _run(tmp_path / "same", carteira_anterior=saved["pesos"], carteira_anterior_parametros=saved["parametros"])
    st = same.changes.stability
    assert st["manter_anterior"] and st["ganho_utilidade_E"] <= st["erro_padrao_aprox"] + st["custo_da_troca"]
    assert same.changes.same and all(c.kind == "mantido" for c in same.changes.changes)
    assert same.changes.fingerprint_before == same.changes.fingerprint_after
    changed = _run(tmp_path / "changed", perfil_risco="conservador", carteira_anterior=saved["pesos"],
                   carteira_anterior_parametros=saved["parametros"])
    assert not changed.changes.same and "entradas mudaram" in changed.changes.stability["explicacao"]
    moved = [c for c in changed.changes.changes if c.kind != "mantido"]
    assert moved and all(c.reason for c in moved)


def test_10b_compare_counts_only_material_changes():
    before = {"A": 0.50, "B": 0.30, "C": 0.20}
    after = pd.Series({"A": 0.505, "B": 0.295, "C": 0.20})
    rep = compare(before, after, {}, "recomendação anterior")
    assert rep.same and all(c.kind == "mantido" for c in rep.changes)
    rep2 = compare(before, pd.Series({"A": 0.50, "B": 0.30 - MATERIAL * 2, "D": 0.20 + MATERIAL * 2}),
                   {"C": "triagem: x", "D": "entrou: y", "B": "reduziu: z"}, "recomendação anterior")
    kinds = {c.ticker: c.kind for c in rep2.changes}
    assert kinds == {"A": "mantido", "B": "reduziu", "C": "saiu", "D": "entrou"} and not rep2.same
    assert fingerprint({"A": 0.5004, "B": 0.4996}) == fingerprint({"A": 0.5001, "B": 0.4999})


# ----------------------------------------------------------------------------- selection
def test_equal_weight_is_a_baseline_never_the_choice(runs):
    import core.universe.screen as stage_a

    assert not hasattr(stage_a, "CORE")                      # no fixed list of tickers anywhere in stage A
    for name in ("base", "cons", "aggr", "small"):
        r = runs(name)
        assert r.analysis.selected_name != "equal_weight"
        sel = r.analysis.selection
        assert sel["ranking_metric"] == "utilidade_realizada" and sel["preferred"] == "utility"
        assert sel.get("test_metrics_equal_weight") is not None   # the 1/N benchmark is still reported
    base = runs("base")
    assert any("linha de base 1/N" in a.message for a in base.alerts)


def test_select_strategy_keeps_the_personal_portfolio_unless_clearly_beaten():
    from core.backtesting.walk_forward import BacktestSettings, walk_forward
    from core.model_validation.model_selection import select_strategy

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=900)
    r = pd.DataFrame(rng.normal(0.0004, 0.01, (900, 3)), index=idx, columns=["a", "b", "c"])
    near = {"equal_weight": lambda t, p: np.full(3, 1 / 3), "utility": lambda t, p: np.array([0.4, 0.3, 0.3]),
            "min_variance": lambda t, p: np.array([0.34, 0.33, 0.33])}
    bt = walk_forward(r, near, BacktestSettings(252, 21, 10, 5, 0.6, 0.0, 252), synthetic=True)
    sel = select_strategy(bt, "equal_weight", list(near), "utility", ("equal_weight",), 4.0)
    assert sel["selected"] == "utility" and sel["selected"] != "equal_weight"
    r2 = r.copy()
    r2["c"] = r2["c"] + 0.004                                # one asset clearly better out of sample
    far = {**near, "min_variance": lambda t, p: np.array([0.0, 0.0, 1.0])}
    bt2 = walk_forward(r2, far, BacktestSettings(252, 21, 10, 5, 0.6, 0.0, 252), synthetic=True)
    sel2 = select_strategy(bt2, "equal_weight", list(far), "utility", ("equal_weight",), 4.0)
    assert sel2["selected"] == "min_variance" == sel2["best_by_metric"]


def test_max_utility_reacts_to_risk_aversion_cap_and_turnover():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 4)) * np.array([0.01, 0.015, 0.02, 0.002])
    cov = pd.DataFrame(np.cov(x.T) * 252, index=list("abcd"), columns=list("abcd"))
    mu = pd.Series([0.08, 0.11, 0.14, 0.05], index=list("abcd"))
    pc = PortfolioConstraints.long_only(list("abcd"), max_weight=0.8)
    lo, hi = opt.max_utility(mu, cov, pc, 8.0), opt.max_utility(mu, cov, pc, 1.0)
    assert lo.success and hi.success and lo.volatility < hi.volatility
    capped = opt.max_utility(mu, cov, pc, 1.0, vol_cap=0.12)
    assert capped.volatility <= 0.12 + 1e-6
    prev = np.array([0.25, 0.25, 0.25, 0.25])
    sticky = opt.max_utility(mu, cov, dataclasses.replace(pc, previous_weights=prev), 4.0, turnover_penalty=1.0)
    assert np.abs(sticky.weights.to_numpy() - prev).sum() < np.abs(opt.max_utility(mu, cov, pc, 4.0).weights.to_numpy()
                                                                    - prev).sum()


# ----------------------------------------------------------------------------- personal rules
def test_short_horizon_income_liquidity_and_loss_rules(tmp_path):
    p = build_policy(profile=RiskProfile.MODERADO, horizon_years=0.5, base_currency="BRL", capital=10_000)
    assert p.min_cash >= 0.5 and p.limits.max_class_weight["equity"] <= 0.10 and p.long_duration_cap == 0.0
    assert p.profile == RiskProfile.CONSERVADOR                         # profiler: horizon below 2 years
    p2 = build_policy(profile=RiskProfile.MODERADO, horizon_years=10, base_currency="BRL", objective="renda",
                      liquidity_need=0.3, loss_tolerance=0.10, capital=10_000)
    assert p2.min_income == 0.5 and p2.min_cash == 0.3 and p2.vol_cap == pytest.approx(0.05)
    assert any("renda" in x for x in p2.rules) and not p2.assumptions[1:]
    r = _run(tmp_path, objetivo_tipo="renda", necessidade_liquidez=0.3, tolerancia_perda=0.10)
    assert _share(r, lambda a: traits(a).income) >= 0.5 - 1e-6
    assert _share(r, lambda a: traits(a).cash) >= 0.3 - 1e-6
    assert r.combined.vol_final <= max(r.combined.vol_quant, 0.05) + 1e-6


def test_missing_information_is_stated_not_presumed(runs):
    base = runs("base")
    hyp = " ".join(base.policy["hipoteses"])
    assert "Objetivo não informado" in hyp and "Tolerância a perdas não informada" in hyp
    assert "Mercados não informados" in hyp
    assert any(a.message.startswith("Hipótese") for a in base.alerts)


def test_crypto_only_when_allowed(runs, tmp_path):
    base = runs("base")
    assert not any(CAT[t].asset_class == "crypto" for t in base.universe["baixados"] if t in CAT)
    assert any("cripto não permitida" in w for w in base.universe["excluidos_por_preferencia"].values())
    r = _run(tmp_path, permitir_cripto=True, perfil_risco="arrojado")
    assert r.policy["cripto_permitida"] and _share(r, lambda a: a.asset_class == "crypto") <= 0.10 + 1e-6


# ----------------------------------------------------------------------------- contributions, holdings
def test_zero_capital_is_sized_for_a_year_of_contributions(tmp_path):
    r = _run(tmp_path, capital_inicial=0, aporte_periodico=2_000)
    assert r.execution["valor_de_referencia"] == pytest.approx(24_000)
    assert "12 meses" in r.execution["nota_referencia"] and r.execution["compra_inicial"] is None
    pa = r.execution["proximo_aporte"]
    assert sum(p["valor"] for p in pa["posicoes"]) + pa["caixa_nao_investido"] == pytest.approx(2_000, abs=0.05)
    assert all(a["valor_inicial"] == 0 for a in r.allocation)


def test_current_holdings_count_as_wealth_and_new_money_fills_the_gaps(tmp_path):
    held = {"PETR4": 30_000, "CAIXA_CDI": 5_000}
    r = _run(tmp_path, capital_inicial=10_000, carteira_atual=held)
    assert "inclui" in r.steps[0]["resultado"] and r.execution["valor_de_referencia"] == 45_000 + 12 * 1_000
    ci = r.execution["compra_inicial"]
    assert sum(p["valor"] for p in ci["posicoes"]) + ci["caixa_nao_investido"] == pytest.approx(10_000, abs=0.05)
    assert "peso_executado" not in (ci["posicoes"] or [{}])[0]          # a contribution plan, not a full purchase
    assert r.changes.reference == "carteira atual"
    if r.combined.final.get("PETR4.SA", 0.0) * 45_000 < 30_000 - max(100.0, 450.0):
        assert any(v["ticker"] == "PETR4.SA" for v in r.execution["vendas_sugeridas"])
        assert "imposto" in r.execution["nota_vendas"]


def test_feasibility_units():
    a = {t: CAT[t] for t in ("PETR4.SA", "BOVA11.SA", "AAPL", CASH_TICKER)}
    px = pd.Series({"PETR4.SA": 38.5, "BOVA11.SA": 128.0, "AAPL": 1250.0, CASH_TICKER: 103.7})
    keep, out = executable_universe(a, px, {"PETR4.SA": 0.3, "BOVA11.SA": 0.3, "AAPL": 0.3, CASH_TICKER: 1.0},
                                    1_000, 100.0, False)
    assert "AAPL" in out and "1.250,00" in out["AAPL"] and set(keep) == {"PETR4.SA", "BOVA11.SA", CASH_TICKER}
    keep2, _ = executable_universe(a, px, {"PETR4.SA": 0.3, "BOVA11.SA": 0.3, "AAPL": 0.3, CASH_TICKER: 1.0},
                                   1_000, 100.0, True)
    assert "AAPL" in keep2                                              # fractions of US shares allowed
    w = pd.Series({"PETR4.SA": 0.3, "BOVA11.SA": 0.3, CASH_TICKER: 0.4})
    p = plan(w, px, a, 1_000)
    assert sum(r["valor"] for r in p.rows) + p.cash_left == pytest.approx(1_000)
    assert p.cash_left < 0.01                                           # the rest goes to the divisible cash
    assert {r["ticker"]: r["quantidade"] for r in p.rows}["PETR4.SA"] == int({r["ticker"]: r["quantidade"]
                                                                               for r in p.rows}["PETR4.SA"])
    c = contribution_plan(w, {"PETR4.SA": 400.0}, px, a, 500.0)
    assert "PETR4.SA" not in {r["ticker"] for r in c.rows}                # already above its share: not bought
    assert sum(r["valor"] for r in c.rows) + c.cash_left == pytest.approx(500.0)


def test_concentration_counts_bets_not_tickers():
    rng = np.random.default_rng(2)
    f = rng.normal(0, 0.01, 600)
    same = pd.DataFrame({k: f + rng.normal(0, 0.0005, 600) for k in ("PETR4.SA", "VALE3.SA", "ITUB4.SA")})
    cov = same.cov() * 252
    assert effective_bets(np.full(3, 1 / 3), cov.to_numpy()) < 1.2
    w = pd.Series(np.full(3, 1 / 3), index=same.columns)
    rep = analyse(w, cov, same, CAT)
    assert rep.groups and set(rep.groups[0]["ativos"]) == set(same.columns) and rep.overlaps
    indep = pd.DataFrame(rng.normal(0, 0.01, (600, 3)), columns=["PETR4.SA", "VALE3.SA", "ITUB4.SA"])
    assert effective_bets(np.full(3, 1 / 3), (indep.cov() * 252).to_numpy()) > 2.5


# ----------------------------------------------------------------------------- outputs
def test_explanations_report_and_json_carry_the_new_evidence(runs):
    r = runs("base")
    for a in r.allocation:
        for k in ("mercado", "riscos", "motivo_selecao", "fatos_H", "estimativas_E", "hipoteses", "diversificacao",
                  "alternativas", "dados"):
            assert k in a and a[k] is not None
    md = render_markdown(r)
    heads = ["## 1. CAGR", "## 6. Alertas", "## 7. Plano de compra", "## 8. O que mudou", "## 9. Diversificação real",
             "## 10. Como os ativos foram escolhidos", "## 11. Suas regras", "## 12. Fontes dos dados"]
    pos = [md.index(h) for h in heads]
    assert pos == sorted(pos) and guarantee_phrases(md) == []
    assert not any(a.category == "linguagem" for a in r.alerts)
    d = to_dict(r)
    for k in ("regras_pessoais", "triagem", "execucao", "concentracao", "mudancas", "fontes", "reprodutibilidade",
              "para_guardar"):
        assert k in d
    assert d["para_guardar"]["parametros"] == r.reproducibility["parametros"]
    assert {s["dado"] for s in d["fontes"]} >= {"preços diários ajustados", "listagens de ativos", "taxa livre de risco",
                                               "caixa em reais"}
    assert "Não cobre outros países" in r.universe["cobertura"]
    assert sum(1 for t in CASH_TICKERS if t in _weights(r).index) <= 1


def test_web_endpoint_accepts_the_personal_fields(monkeypatch):
    from fastapi.testclient import TestClient

    import webapp.server as srv

    c = TestClient(srv.app)
    body = {"fonte_dados": "sintetico", "valor_alvo": 300_000, "capital_inicial": 10_000, "aporte_periodico": 1_000,
            "horizonte_anos": 8, "n_simulacoes": 500, "objetivo_tipo": "renda", "tolerancia_perda": 0.2,
            "necessidade_liquidez": 0.1, "mercados": ["BR"], "valor_minimo_posicao": 50,
            "carteira_atual": {"SINT_SELIC": 2_000}}
    r1 = c.post("/api/proposito", json=body)
    assert r1.status_code == 200, r1.text
    d = r1.json()
    assert d["do_cache"] is False and d["calculado_em_utc"] and d["para_guardar"]["pesos"]
    assert d["regras_pessoais"]["objetivo"] == "renda" and d["mudancas"]["referencia"] == "carteira atual"
    assert not any(t.startswith("SINT_") and t.endswith("_US") for t in d["para_guardar"]["pesos"])   # only BR
    again = c.post("/api/proposito", json=body).json()
    assert again["do_cache"] is True and again["para_guardar"] == d["para_guardar"]
    assert c.post("/api/proposito", json={**body, "tolerancia_perda": 1.5}).status_code == 422


def test_cli_accepts_the_personal_fields(tmp_path):
    import json

    from main import main

    rc = main(["--offline", "--proposito", "--objetivo", "300000", "--aporte", "1000", "--horizon-years", "8",
               "--simulations", "500", "--objetivo-tipo", "renda", "--queda-maxima", "0.2", "--mercados", "BR",
               "--carteira-atual", "SINT_SELIC=2000", "--compra-minima", "50", "--no-charts",
               "--output-dir", str(tmp_path)])
    assert rc == 0
    out = json.loads(next((tmp_path / "proposito").glob("*_proposito.json")).read_text(encoding="utf-8"))
    assert out["regras_pessoais"]["objetivo"] == "renda" and out["entrada"]["carteira_atual"] == {"SINT_SELIC": 2000.0}
    assert out["execucao"]["valor_minimo_posicao"] == 50 and out["mudancas"]["referencia"] == "carteira atual"
    assert main(["--offline", "--proposito", "--objetivo", "1", "--carteira-atual", "PETR4", "--output-dir",
                 str(tmp_path)]) == 2
