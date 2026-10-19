"""Purpose-aligned selection engine: required CAGR, themes, governance, alignment, sleeves,
language guard and the end-to-end pipeline (offline, synthetic data)."""

import json

import numpy as np
import pandas as pd
import pytest

from core.optimization.constraints import verify_weights
from core.purpose.alignment import combine, score_assets
from core.purpose.catalog import REAL_CATALOG, SYNTHETIC_CATALOG, by_ticker, expand
from core.purpose.governance import apply_governance
from core.purpose.language import guarantee_phrases
from core.purpose.required_cagr import future_value, required_cagr
from core.purpose.themes import extract_themes
from models.portfolio import GroupConstraint, PortfolioConstraints


# ----------------------------------------------------------------------------- step 1
def test_required_cagr_without_contributions_is_closed_form():
    r = required_cagr(100_000, 0, "nenhum", 200_000, 8)
    assert r.status == "ok"
    assert r.rate == pytest.approx(2 ** (1 / 8) - 1, rel=1e-10)


@pytest.mark.parametrize("per", ["mensal", "trimestral", "semestral", "anual"])
def test_required_cagr_solves_future_value(per):
    r = required_cagr(50_000, 1_500, per, 600_000, 12)
    m = {"mensal": 12, "trimestral": 4, "semestral": 2, "anual": 1}[per]
    assert future_value(r.rate, 50_000, 1_500, m, 12) == pytest.approx(600_000, rel=1e-9)
    assert r.total_contributed == pytest.approx(50_000 + 1_500 * m * 12)


def test_required_cagr_edge_cases():
    easy = required_cagr(100_000, 1_000, "mensal", 150_000, 5)    # contributions alone exceed the target
    assert easy.status == "atingivel_sem_retorno" and easy.rate <= 0
    assert required_cagr(1, 0, "nenhum", 1e12, 1).status == "inatingivel"
    with pytest.raises(ValueError):
        required_cagr(0, 0, "nenhum", 10, 1)
    with pytest.raises(ValueError):
        required_cagr(10, 0, "semanal", 20, 1)


# ----------------------------------------------------------------------------- step 6
def test_themes_quote_the_text_verbatim():
    txt = "Quero investir em SAÚDE e energia limpa; o objetivo é pagar a faculdade dos meus filhos."
    t = extract_themes(txt)
    assert {"saude", "sustentabilidade", "educacao", "familia_legado"} <= set(t.names())
    for th in t.themes:
        for ev in th.evidence:
            assert ev in txt                                   # nothing that was not written
    assert t.why == "objetivo é pagar a faculdade dos meus filhos"
    assert t.why in txt and t.confidence == "alta"


def test_negated_themes_are_not_counted():
    t = extract_themes("Não me importo com sustentabilidade, quero tecnologia e inovação para crescer.")
    assert "sustentabilidade" not in t.names() and "sustentabilidade" in t.negated
    assert "tecnologia" in t.names()
    t2 = extract_themes("Não tenho filhos mas quero educação de qualidade")
    assert "educacao" in t2.names() and "familia_legado" in t2.negated


def test_empty_and_vague_motives():
    e = extract_themes("   ")
    assert e.empty and e.themes == [] and e.confidence == "nenhuma"
    assert any("falta de input" in n for n in e.notes)
    v = extract_themes("quero ganhar dinheiro")
    assert v.themes == [] and v.confidence == "baixa"


# ----------------------------------------------------------------------------- step 5
def _flags(tmp_path, flags):
    p = tmp_path / "flags.json"
    p.write_text(json.dumps({"flags": flags}), encoding="utf-8")
    return p


def test_governance_default_excludes_and_unlock_includes(tmp_path):
    p = _flags(tmp_path, {
        "A": {"status": "sob_revisao", "reason": "r", "source": "fonte X"},
        "B": {"status": "excluido", "reason": "r", "source": "fonte Y"},
        "C": {"status": "sob_revisao", "reason": "sem fonte"},          # rejected: no source
    })
    d = apply_governance(["A", "B", "C", "D"], False, p)
    assert d.kept == ["C", "D"] and set(d.removed) == {"A", "B"} and "C" in d.rejected_entries
    u = apply_governance(["A", "B", "C", "D"], True, p)
    assert u.kept == ["A", "C", "D"] and "B" in u.removed and "A" in u.included_under_review


def test_request_flags_merge_with_file_and_need_a_source(tmp_path):
    p = _flags(tmp_path, {"A": {"status": "sob_revisao", "reason": "r", "source": "fonte X"}})
    extra = {"A": {"status": "excluido", "reason": "r2", "source": "fonte Z"},       # request wins for A
             "D": {"status": "sob_revisao", "reason": "sem fonte"}}                     # rejected
    g = apply_governance(["A", "D"], True, p, extra)
    assert g.kept == ["D"] and "fonte Z" in g.removed["A"] and "D" in g.rejected_entries


def test_shipped_flags_contain_no_real_company_allegation():
    d = apply_governance([a.ticker for a in REAL_CATALOG])
    assert d.removed == {} and d.included_under_review == {}


# ----------------------------------------------------------------------------- steps 7-8
def _toy_problem():
    syms = ["RF", "SAUDE", "ENERGIA", "VAREJO"]
    vol = np.array([0.02, 0.25, 0.28, 0.35])
    corr = np.full((4, 4), 0.4)
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=syms, columns=syms)
    mu = pd.Series([0.10, 0.14, 0.15, 0.16], index=syms)
    pc = PortfolioConstraints(syms, np.zeros(4), np.full(4, 0.5),
                              [GroupConstraint("class:equity", ("SAUDE", "ENERGIA", "VAREJO"), 0.0, 0.6)])
    w_q = pd.Series([0.5, 0.1, 0.1, 0.3], index=syms)
    cat = by_ticker(SYNTHETIC_CATALOG)
    assets = [cat["SINT_SELIC"], cat["SINT_SAUDE3"], cat["SINT_ENERGIA3"], cat["SINT_VAREJO3"]]
    assets = [type(a)(**{**a.__dict__, "ticker": s}) for a, s in zip(assets, syms)]
    return syms, cov, mu, pc, w_q, assets


def test_alignment_scores_and_combination_invariants():
    syms, cov, mu, pc, w_q, assets = _toy_problem()
    al = score_assets(extract_themes("Quero apoiar saúde e energia limpa."), assets)
    assert al.computed
    s = {k: v.score for k, v in al.scores.items()}
    assert s["SAUDE"] == pytest.approx(0.5) and s["ENERGIA"] == pytest.approx(0.5)
    assert s["RF"] == 0.0 and s["VAREJO"] == 0.0
    assert all(al.scores[k].reason for k in syms)
    c = combine(w_q, al, mu, cov, pc, alpha=0.3, te_budget=0.05, vol_cap=0.12)
    assert c.violations == []
    assert not verify_weights(c.final.to_numpy(), pc)
    assert c.tracking_error <= 0.3 * 0.05 + 1e-6
    assert c.vol_final <= max(c.vol_quant, 0.12) + 1e-6
    assert c.alignment_final > c.alignment_quant                  # alignment moved weights toward the themes
    np.testing.assert_allclose(c.final.to_numpy(), 0.7 * w_q.to_numpy() + 0.3 * c.alignment_target.to_numpy(),
                               atol=1e-9)


def test_alignment_never_replaces_the_optimizer():
    syms, cov, mu, pc, w_q, assets = _toy_problem()
    empty = score_assets(extract_themes(""), assets)
    assert not empty.computed and all(v.score is None for v in empty.scores.values())
    c = combine(w_q, empty, mu, cov, pc, alpha=0.3)
    pd.testing.assert_series_equal(c.final, w_q.loc[syms])
    values_only = score_assets(extract_themes("Quero deixar um legado para a minha família e ter disciplina."), assets)
    assert not values_only.computed and values_only.unmapped_themes      # values are not mapped to sectors
    al = score_assets(extract_themes("saúde"), assets)
    c0 = combine(w_q, al, mu, cov, pc, alpha=0.0)
    pd.testing.assert_series_equal(c0.final, w_q.loc[syms])
    with pytest.raises(ValueError):
        combine(w_q, al, mu, cov, pc, alpha=1.5)


# ----------------------------------------------------------------------------- language guard
def test_guarantee_language_detected():
    assert guarantee_phrases("Esta carteira vai gerar 20% ao ano")
    assert guarantee_phrases("Certamente você atingirá o objetivo")
    assert guarantee_phrases("retorno GARANTIDO")
    assert guarantee_phrases("não oferece garantia de resultado; no cenário base, segundo a simulação") == []


def test_catalog_expansion_lists_individual_tickers():
    assert all("_" not in a.ticker or a.ticker.startswith("SINT_") for a in SYNTHETIC_CATALOG)
    fii = expand(REAL_CATALOG, ["fii"])
    assert {"KNRI11.SA", "HGLG11.SA"} <= {a.ticker for a in fii} and all(a.macro_class == "fii" for a in fii)
    assert len(REAL_CATALOG) >= 200 and len({a.ticker for a in REAL_CATALOG}) == len(REAL_CATALOG)
    with pytest.raises(ValueError):
        expand(REAL_CATALOG, ["classe_inexistente"])


# ----------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def purpose_run(tmp_path_factory):
    from core.purpose.engine import PurposeRequest, run_purpose

    req = PurposeRequest(capital_inicial=100_000, aporte_periodico=3_000, periodicidade="trimestral",
                         valor_alvo=250_000, horizonte_anos=3, perfil_risco="moderado",
                         taxa_livre_risco={"BRL": 0.105, "USD": 0.043},
                         motivo_investidor="Quero apoiar saúde e energia limpa. O objetivo é a faculdade dos meus filhos.",
                         n_simulacoes=1000)
    return run_purpose(req, output_dir=tmp_path_factory.mktemp("purpose"))


def test_engine_follows_the_nine_steps_and_output_format(purpose_run):
    from core.purpose.report import render_markdown

    r = purpose_run
    assert [s["etapa"] for s in r.steps] == list(range(1, 10))
    md = render_markdown(r)
    heads = ["## 1. CAGR necessário vs. CAGR esperado", "## 2. Carteira", "## 3. Justificativa por ativo",
             "## 4. Cenários", "## 5. Probabilidade de atingir o objetivo", "## 6. Alertas de validação"]
    pos = [md.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "| ticker | classe | peso | score_alinhamento | moeda |" in md
    assert guarantee_phrases(md) == []
    assert not any(a.category == "linguagem" for a in r.alerts)


def test_engine_flags_synthetic_and_respects_rf_per_currency(purpose_run):
    r = purpose_run
    assert r.is_synthetic and not r.is_final_portfolio
    assert any(a.severity == "FAIL" and "NÃO é uma carteira final" in a.message for a in r.alerts)
    assert r.rf_by_currency == {"BRL": 0.105, "USD": 0.043}
    assert set(r.rf_source.values()) == {"informada pelo usuário"}
    ns = r.native_stats
    assert set(ns.loc[ns["moeda"] == "USD", "rf_moeda"]) == {0.043}
    assert set(ns.loc[ns["moeda"] == "BRL", "rf_moeda"]) == {0.105}
    for s in r.sleeves_final:
        assert s.rf == r.rf_by_currency[s.currency]


def test_engine_final_portfolio_respects_all_limits(purpose_run):
    r = purpose_run
    res = r.analysis
    w = r.table["peso"].reindex(res.constraints.symbols).to_numpy()
    assert w.sum() == pytest.approx(1.0) and not verify_weights(w, res.constraints)
    assert "SINT_BANCO4" not in r.table.index                    # governance: under review -> out by default
    assert "SINT_BANCO4" in r.governance.removed
    assert r.combined.tracking_error <= r.request.peso_alinhamento * r.request.orcamento_tracking_error + 1e-6
    gap = r.expected["distancia_fronteira_final_E"]
    assert gap is not None and 0.0 <= gap < 0.05            # inside the feasible set, close to the frontier
    cap = res.profile.research_limits.target_volatility_band[1]
    assert r.combined.vol_final <= max(r.combined.vol_quant, cap) + 1e-6
    sc = r.table["score_alinhamento"]
    assert sc.between(0, 1).all()


def test_engine_dual_theses_are_separate(purpose_run):
    r = purpose_run
    held = set(r.table.index[r.table["peso"] > 0])
    assert {t["ticker"] for t in r.theses} == held
    for t in r.theses:
        assert t["tese_financeira"] and t["tese_proposito"] and t["tese_financeira"] != t["tese_proposito"]
        assert "Moat" in t["tese_financeira"] and "Conexão" not in t["tese_financeira"]


def test_engine_scenarios_goal_probability_and_json(purpose_run):
    from core.purpose.report import to_dict

    r = purpose_run
    sc = r.scenarios["patrimonio_final"].to_numpy()
    assert sc[0] >= sc[1] >= sc[2]
    p, se, n = r.goal["p_objetivo"], r.goal["erro_padrao"], r.goal["n_trajetorias"]
    assert 0 <= p <= 1 and se == pytest.approx(np.sqrt(p * (1 - p) / n))
    assert r.required.rate is not None and r.expected["cagr_esperado_mediano_simulado"] is not None
    # the median scenario's equivalent CAGR is the "expected CAGR" shown side by side with the required one
    assert r.scenarios.iloc[1]["cagr_equivalente"] == pytest.approx(r.expected["cagr_esperado_mediano_simulado"])
    json.dumps(to_dict(r))


def test_engine_empty_motive_runs_quant_only(tmp_path):
    from core.purpose.engine import PurposeRequest, run_purpose
    from core.purpose.report import render_markdown

    req = PurposeRequest(capital_inicial=50_000, aporte_periodico=0, periodicidade="nenhum", valor_alvo=60_000,
                         horizonte_anos=2, perfil_risco="conservador", motivo_investidor="", n_simulacoes=500,
                         incluir_sob_revisao=True)
    r = run_purpose(req, output_dir=tmp_path)
    assert not r.alignment.computed and r.table["score_alinhamento"].isna().all()
    assert r.combined.alignment_target is None                           # no alignment portfolio was built
    np.testing.assert_allclose(r.combined.final.to_numpy(), r.combined.quant.to_numpy(), atol=0.01)  # dust only
    md = render_markdown(r)
    assert "não calculado por falta de input" in md
    assert "SINT_BANCO4" in r.governance.included_under_review           # explicit unlock
    assert any(a.category == "governanca" and a.severity == "WARNING" for a in r.alerts)


def test_request_validation():
    from core.purpose.engine import PurposeRequest

    with pytest.raises(ValueError):
        PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, aporte_periodico=10, periodicidade="nenhum")
    with pytest.raises(ValueError):
        PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, moeda_objetivo="USD", offline=True)
    with pytest.raises(ValueError):
        PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, taxa_livre_risco={"BRL": 3.0})
    with pytest.raises(ValueError):
        PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, extra_field=1)
    with pytest.raises(ValueError, match="moeda do objetivo"):
        PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, moeda="USD")
    assert PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, moeda="brl").moeda == "brl"


def test_missing_rf_for_unknown_currency_is_never_zero():
    from core.purpose.engine import PurposeRequest, _resolve_rf

    req = PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, moeda_objetivo="GBP", offline=False)
    with pytest.raises(ValueError, match="nunca é assumida igual a zero"):
        _resolve_rf(req, {"GBP"})


def test_real_universe_excludes_short_histories_without_proxy():
    """Online path with a fake provider: a ticker with < 252 observations is excluded and listed."""
    from config import AppConfig
    from core.market_data.base import RawSeries
    from core.purpose.engine import PurposeRequest, _load_universe

    idx = pd.bdate_range("2021-01-04", periods=400)

    def rs(sym, n, ccy="BRL"):
        x = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(len(sym)).normal(0, 0.01, n))), index=idx[-n:])
        return RawSeries(sym, x, None, "fake", "adj_close", "splits+dividends", ccy, "1d")

    class Fake:
        name = "fake"

        def __init__(self):
            self.failures = {"ITUB4.SA": "HTTP 404"}

        def fetch(self, symbols, period, interval):
            data = {"PETR4.SA": rs("PETR4.SA", 400), "VALE3.SA": rs("VALE3.SA", 400), "WEGE3.SA": rs("WEGE3.SA", 120)}
            return {s: data[s] for s in symbols if s in data}

    req = PurposeRequest(capital_inicial=1, valor_alvo=2, horizonte_anos=1, offline=False, classes=["acoes_br"])
    assets = expand(REAL_CATALOG, ["acoes_br"])
    ds, truth = _load_universe(req, assets, AppConfig(offline=False, risk_free_rate=0.1), Fake())
    assert truth is None and set(ds.symbols) == {"PETR4.SA", "VALE3.SA"}
    excl = ds.quality.excluded_symbols
    assert "WEGE3.SA" in excl and "120" in excl["WEGE3.SA"]           # short history: excluded, not proxied
    assert "ITUB4.SA" in excl                                         # download failure documented


def test_cli_purpose_mode(tmp_path):
    from main import main

    code = main(["--offline", "--proposito", "--objetivo", "150000", "--capital", "100000", "--aporte", "1000",
                 "--horizon-years", "2", "--simulations", "500", "--rf", "BRL=0.105", "--rf", "USD=0.043",
                 "--motivo", "Quero investir em saúde", "--output-dir", str(tmp_path)])
    assert code == 0
    out = tmp_path / "proposito"
    md = next(out.glob("*_proposito.md")).read_text(encoding="utf-8")
    data = json.loads(next(out.glob("*_proposito.json")).read_text(encoding="utf-8"))
    assert "NÃO É UMA CARTEIRA FINAL" in md and data["rf_por_moeda"] == {"BRL": 0.105, "USD": 0.043}
    assert main(["--offline", "--proposito", "--output-dir", str(tmp_path)]) == 2          # --objetivo missing
    assert main(["--offline", "--proposito", "--objetivo", "1", "--rf", "BRL", "--output-dir", str(tmp_path)]) == 2


def test_engine_infeasible_profile_is_flagged_not_hidden(tmp_path):
    """Conservative profile (>= 40% fixed income) on an equity-only universe: reported, never hidden."""
    from core.purpose.engine import PurposeRequest, run_purpose

    req = PurposeRequest(capital_inicial=100_000, valor_alvo=120_000, horizonte_anos=2, perfil_risco="conservador",
                         classes=["acoes_br"], motivo_investidor="saúde", n_simulacoes=500)
    r = run_purpose(req, output_dir=tmp_path)
    assert any(a.severity == "FAIL" and a.category == "validacao" and "INVIÁVEIS" in a.message for a in r.alerts)
    step4 = next(s for s in r.steps if s["etapa"] == 4)["resultado"]
    assert "backtest NÃO executado" in step4
    lo, hi = r.goal["ic95"]
    assert 0.0 <= lo <= r.goal["p_objetivo"] <= hi <= 1.0


def test_monthly_goal_simulation_matches_daily():
    """The goal simulation runs on monthly steps (monthly rebalancing and contributions): for a GBM
    this is exact in distribution, so it must agree with the daily simulation within MC error."""
    from types import SimpleNamespace

    from config import AppConfig
    from core.purpose.engine import PurposeRequest, _simulate
    from core.simulation.engine import (
        CashflowPlan,
        SimulationSettings,
        simulate_portfolio,
    )
    from core.simulation.models import GBMModel

    cfg = AppConfig(offline=True, risk_free_rate=0.1)
    mu, cov, w = np.array([0.10, 0.06]), np.array([[0.04, 0.006], [0.006, 0.01]]), np.array([0.6, 0.4])
    req = PurposeRequest(capital_inicial=10_000, aporte_periodico=200, valor_alvo=19_500, horizonte_anos=3)
    m = _simulate(req, SimpleNamespace(config=cfg), GBMModel(mu, cov, 1 / 12), w, 20_000, 1)
    d = simulate_portfolio(GBMModel(mu, cov, 1 / 252), w, SimulationSettings(
        n_paths=20_000, n_steps=756, seed=2, initial_capital=10_000, rebalance_every=21,
        rebalance_cost_bps=cfg.transaction_cost_bps + cfg.spread_bps, inflation_annual=cfg.inflation_annual,
        periods_per_year=252, goal=19_500, record_every=21), CashflowPlan(200, 0, 21, cfg.transaction_cost_bps))
    se = np.hypot(m.summary["p_goal_mc_se"], d.summary["p_goal_mc_se"])
    assert 0.05 < d.summary["p_goal"] < 0.95
    assert abs(m.summary["p_goal"] - d.summary["p_goal"]) < 4 * se
    assert m.summary["net_invested"] == d.summary["net_invested"] == 10_000 + 200 * 36

