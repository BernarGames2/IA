"""Regression tests for the defects found by the review of the dynamic portfolio engine: purchase plans,
cash, lots and fingerprints; the universe (history, cash, currencies, sectors, tickers, governance); the
selection (volatility band and cap); and the web/CLI input parsing. Real-data path with test doubles."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.market_data.cash import CASH_ASSET, CASH_TICKER
from core.optimization.constraints import verify_weights
from core.portfolio_engine import REFERENCE_ONLY, _within_vol_cap, band_filter
from core.purpose.catalog import REAL_CATALOG, SYNTHETIC_CATALOG, canonical_sector
from core.purpose.changes import stability_test
from core.purpose.engine import (
    PurposeRequest,
    _param_fingerprint,
    norm_ticker,
    run_purpose,
)
from core.purpose.feasibility import (
    contribution_plan,
    executable_universe,
    lot_size,
    min_weight,
    plan,
    unit_price,
)
from core.purpose.personalization import build_policy, traits
from core.purpose.report import _plan_table, _qty, render_markdown, to_dict
from core.universe.listings import ListedAsset
from core.universe.scoring import MIN_CANDIDATE_OBS, screen
from core.universe.screen import listing_to_catalog
from models.investor import RiskProfile
from models.portfolio import GroupConstraint, OptimizationResult, PortfolioConstraints
from tests.market_doubles import Prov, http, structured_market
from utils.validation import ValidationError, parse_amount

MARKET = structured_market()
BASE = {"capital_inicial": 100_000, "aporte_periodico": 1_000, "valor_alvo": 500_000, "horizonte_anos": 10,
        "perfil_risco": "moderado", "offline": False, "universo": "completo", "max_ativos": 20, "n_simulacoes": 500}
CAT = {a.ticker: a for a in (*REAL_CATALOG, CASH_ASSET)}
SYN = {a.ticker: a for a in SYNTHETIC_CATALOG}
OFFLINE = {"capital_inicial": 10_000, "aporte_periodico": 1_000, "valor_alvo": 300_000, "horizonte_anos": 8,
           "n_simulacoes": 500, "offline": True}


def _run(tmp, market=None, **kw):
    req = PurposeRequest(**{**BASE, **kw})
    return run_purpose(req, output_dir=tmp, provider=Prov(MARKET if market is None else market), cache_dir=tmp,
                       rf_fetch=http, fast=True)


def _policy(**kw):
    args = {"profile": RiskProfile.MODERADO, "horizon_years": 10, "base_currency": "BRL", "capital": 100_000.0, **kw}
    return build_policy(**args)


# ----------------------------------------------------------------------------- lots and money
def test_only_the_cdi_model_is_money_exchange_cash_like_etfs_trade_in_units():
    shy, lfts = CAT["SHY"], CAT.get("LFTS11.SA")
    assert lot_size(shy) == (1.0, "ação inteira") and unit_price(shy, 450.0) == 450.0
    assert min_weight(shy, 450.0, 1_000.0, 100.0, False) == pytest.approx(0.45)        # one share, not R$ 100
    assert lot_size(CAT[CASH_TICKER])[0] == 0.01 and unit_price(CAT[CASH_TICKER], 103.7) == 1.0
    if lfts is not None:
        assert lot_size(lfts) == (1.0, "lote de 1 cota")
    p = plan(pd.Series({"SHY": 0.4, "PETR4.SA": 0.3, "BOVA11.SA": 0.3}),
             pd.Series({"SHY": 452.10, "PETR4.SA": 38.5, "BOVA11.SA": 128.0}), CAT, 10_000.0)
    shy_row = next(r for r in p.rows if r["ticker"] == "SHY")
    assert shy_row["quantidade"] == int(shy_row["quantidade"]) and shy_row["preco"] == 452.10
    assert not shy_row["em_dinheiro"] and p.cash_left > 0                  # the leftover is not parked in SHY


def test_b3_crypto_etf_trades_in_whole_units_direct_crypto_in_fractions():
    assert lot_size(CAT["HASH11.SA"]) == (1.0, "lote de 1 cota")
    assert lot_size(CAT["BTC-USD"])[0] == 1e-8


def test_minimum_purchase_message_names_what_actually_excludes():
    _, out = executable_universe({"PETR4.SA": CAT["PETR4.SA"]}, pd.Series({"PETR4.SA": 38.5}), {"PETR4.SA": 0.10},
                                 1_000.0, 150.0, False, "BRL")
    assert out["PETR4.SA"].startswith("a compra mínima de R$ 150,00 passa do peso máximo de 10%")
    _, out = executable_universe({"BOVA11.SA": CAT["BOVA11.SA"]}, pd.Series({"BOVA11.SA": 128.0}),
                                 {"BOVA11.SA": 0.10}, 1_000.0, 50.0, False, "BRL")
    assert out["BOVA11.SA"].startswith("1 cota custa R$ 128,00: acima do peso máximo")


def test_a_lot_dearer_than_one_contribution_is_bought_with_cash_set_aside():
    w = pd.Series({"SINT_SAUDE_US": 0.30, "SINT_IPCA": 0.30, "SINT_ENERGIA3": 0.30, "SINT_SELIC": 0.10})
    px = pd.Series({"SINT_SAUDE_US": 1524.06, "SINT_IPCA": 235.36, "SINT_ENERGIA3": 469.09, "SINT_SELIC": 103.0})
    hold: dict[str, float] = {}
    redeemed = 0.0
    for _ in range(24):
        p = contribution_plan(w, hold, px, SYN, 1_000.0)
        hold = p.holdings_after(hold)
        redeemed += p.redeemed
        assert sum(r["valor"] for r in p.rows) - p.redeemed + p.cash_left == pytest.approx(1_000.0, abs=0.02)
    tot = sum(hold.values())
    assert hold.get("SINT_SAUDE_US", 0.0) / tot > 0.20 and redeemed > 0          # bought with money set aside
    assert all(v >= -1e-6 for v in hold.values())


def test_contribution_plan_counts_holdings_outside_the_target_and_says_when_nothing_more_comes():
    w = pd.Series({"SINT_IPCA": 0.5, "SINT_SELIC": 0.5})
    px = pd.Series({"SINT_IPCA": 235.36, "SINT_SELIC": 103.0})
    p = contribution_plan(w, {"ABCD3.SA": 30_000.0}, px, SYN, 10_000.0, more_coming=False)
    assert sum(r["peso_apos_compra"] for r in p.rows) == pytest.approx(0.25, abs=0.01)   # 30k stays outside
    q = contribution_plan(pd.Series({"SINT_SAUDE_US": 0.5, "SINT_ENERGIA3": 0.5}), {},
                          pd.Series({"SINT_SAUDE_US": 1524.06, "SINT_ENERGIA3": 469.09}), SYN, 300.0,
                          more_coming=False)
    assert q.rows == [] and all("não coube" in v for v in q.skipped.values())
    assert not any("próximos aportes" in n for n in q.notes)
    lines: list[str] = []
    _plan_table(lines, q.as_dict(), "BRL")
    text = "\n".join(lines)
    assert "Nenhuma compra nesta etapa." in text and "300,00" in text and "SINT_SAUDE_US" in text


def test_quantities_keep_the_decimals_they_need():
    assert _qty(0.00016393, False) == "0,00016393" and _qty(7.0, False) == "7" and _qty(669.13, True) == "669,13"


# ----------------------------------------------------------------------------- fingerprint and stability
def test_parameter_fingerprint_ignores_typing_order_and_simulation_count():
    base = {k: v for k, v in OFFLINE.items() if k != "n_simulacoes"}
    a = _param_fingerprint(PurposeRequest(**base, carteira_atual={"SINT_SELIC": 1000, "SINT_IPCA": 2000},
                                          mercados=["BR", "US"], n_simulacoes=1000))
    b = _param_fingerprint(PurposeRequest(**base, carteira_atual={"SINT_IPCA": 2000, "SINT_SELIC": 1000},
                                          mercados=["US", "BR"], n_simulacoes=2000))
    c = _param_fingerprint(PurposeRequest(**base, carteira_atual={"SINT_IPCA": 2000, "SINT_SELIC": 1500},
                                          mercados=["US", "BR"]))
    assert a == b != c


def test_stability_wording_never_says_a_negative_improvement():
    idx = ["A", "B"]
    mu, cov = pd.Series([0.08, 0.08], index=idx), pd.DataFrame(np.diag([0.04, 0.04]), index=idx, columns=idx)
    out = stability_test(pd.Series([0.5, 0.5], index=idx), pd.Series([0.6, 0.4], index=idx), mu, cov, 4.0, 5.0, 0.002)
    assert out["manter_anterior"] and "não melhora" in out["explicacao"] and "melhora a utilidade esperada [E] em -" \
        not in out["explicacao"]


def test_kept_previous_recommendation_is_not_listed_as_removed(tmp_path):
    r0 = run_purpose(PurposeRequest(**OFFLINE), output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    saved = to_dict(r0)["para_guardar"]
    prev = dict(saved["pesos"])
    small = min(prev, key=prev.get)
    prev["SINT_LOGI11"] = prev.get("SINT_LOGI11", 0.0) + 0.015
    prev[small] -= 0.015 if prev[small] > 0.02 else 0.0
    r1 = run_purpose(PurposeRequest(**OFFLINE, carteira_anterior=prev, carteira_anterior_parametros=saved["parametros"]),
                     output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    held = {t for t, x in r1.combined.final.items() if x > 0}
    assert not held & set(r1.execution["retirados_apos_otimizar"])


# ----------------------------------------------------------------------------- holdings and plans (engine)
def test_holdings_only_without_new_money_get_a_rebalance_plan(tmp_path):
    r = run_purpose(PurposeRequest(**{**OFFLINE, "capital_inicial": 0, "aporte_periodico": 0, "periodicidade": "nenhum",
                                      "carteira_atual": {"SINT_ENERGIA3": 40_000}}),
                    output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    ex = r.execution
    assert ex["compra_inicial"] is None and ex["proximo_aporte"] is None
    assert ex["vendas_sugeridas"] and ex["reequilibrio"] is not None
    sold = sum(v["excesso"] for v in ex["vendas_sugeridas"])
    rb = ex["reequilibrio"]
    assert rb["valor"] == pytest.approx(sold) and rb["posicoes"]
    step12 = next(st["resultado"] for st in r.steps if st["etapa"] == 12)
    assert "sem compra inicial (sem capital)" not in step12 and "se vender o excesso" in step12
    md = render_markdown(r)
    assert "Se vender o excesso" in md


def test_cash_above_the_target_is_redeemed_by_the_plan_not_listed_as_a_sale(tmp_path):
    r = run_purpose(PurposeRequest(**{**OFFLINE, "capital_inicial": 0, "aporte_periodico": 0, "periodicidade": "nenhum",
                                      "carteira_atual": {"SINT_SELIC": 50_000}}),
                    output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    ex = r.execution
    assert not [v for v in ex["vendas_sugeridas"] if v["ticker"] == "SINT_SELIC"]
    rb = ex["reequilibrio"]
    assert rb is not None and rb["posicoes"] and rb["resgate_do_caixa"] > 0 and rb["resgate_de"] == "SINT_SELIC"
    assert sum(p["valor"] for p in rb["posicoes"]) == pytest.approx(rb["resgate_do_caixa"] + rb["valor"]
                                                                    - rb["caixa_nao_investido"], abs=0.05)


def test_allocation_reports_target_value_and_what_is_held(tmp_path):
    r = run_purpose(PurposeRequest(**{**OFFLINE, "carteira_atual": {"SINT_ENERGIA3": 30_000}}),
                    output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    start = 40_000
    for a in r.allocation:
        assert a["valor_alvo"] == pytest.approx(start * a["peso"])
        assert a["valor_atual"] == (30_000 if a["ticker"] == "SINT_ENERGIA3" else 0.0)
    assert "você tem" in render_markdown(r)


def test_reference_note_counts_the_months_inside_a_short_horizon(tmp_path):
    r = run_purpose(PurposeRequest(**{**OFFLINE, "horizonte_anos": 0.5, "valor_alvo": 20_000}),
                    output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    assert "próximos 6 meses" in r.execution["nota_referencia"] and "12 meses" not in r.execution["nota_referencia"]


def test_sources_list_the_cdi_apart_from_the_price_provider(tmp_path):
    r = _run(tmp_path)
    prices = r.sources[0]
    assert "bcb" not in prices["fonte"].lower()
    cash = next(s for s in r.sources if s["dado"] == "caixa em reais")
    assert "CDI" in cash["fonte"] and cash.get("obtido_em_utc") and "atraso" in cash["limite"]
    price_times = {p.ingested_at_utc for s, p in r.analysis.dataset.provenance.items() if s != CASH_TICKER}
    assert r.universe["obtido_em_utc"] == max(price_times)


# ----------------------------------------------------------------------------- universe
def test_short_history_holding_is_not_analysed_and_does_not_break_the_request(tmp_path):
    m = dict(MARKET)
    w = m["WEGE3.SA"]
    m["WEGE3.SA"] = dataclasses.replace(w, close=w.close.iloc[-300:], volume=w.volume.iloc[-300:])
    r = _run(tmp_path, market=m, carteira_atual={"WEGE3": 5_000})
    assert "histórico curto" in r.universe["fora_da_analise"]["WEGE3.SA"]
    assert "WEGE3.SA" not in r.universe["candidatos"] and r.universe["n_datas"] >= MIN_CANDIDATE_OBS
    assert any("WEGE3.SA (da sua carteira atual) não entra na análise: histórico curto" in a.message for a in r.alerts)


class _BudgetProv(Prov):
    """The download time runs out after the first third of the series (the rest is never tried)."""

    def fetch(self, symbols, period, interval):
        got = super().fetch(symbols, period, interval)
        arrived = sorted(got)[: max(12, len(got) // 3)]
        for t in got:
            if t not in arrived:
                self.failures[t] = "não tentado: tempo de download esgotado"
        return {t: got[t] for t in arrived}


def test_series_left_untried_by_the_time_budget_do_not_count_as_source_failures(tmp_path):
    req = PurposeRequest(**BASE)
    r = run_purpose(req, output_dir=tmp_path, provider=_BudgetProv(MARKET), cache_dir=tmp_path, rf_fetch=http,
                    fast=True)
    assert any("tempo de download acabou" in a.message for a in r.alerts)
    assert len(r.universe["candidatos"]) >= 2


def _series(n, seed, vol=0.012, drift=0.0004):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


def test_screen_forced_picks_pass_the_cap_and_cash_stays_out_of_the_class_scores():
    n = 900
    s = {"SPY": _series(n, 1), "QQQ": _series(n, 2, 0.016), "AGG": _series(n, 3, 0.004, 0.0002),
         "BND": _series(n, 4, 0.004, 0.0001), "IEF": _series(n, 5, 0.006, 0.0002)}
    a = {k: CAT[k] for k in s}
    sr = screen(s, None, a, RiskProfile.MODERADO, "crescimento", 0.10, 2,
                required=[("o mínimo de renda fixa", ["AGG", "BND", "IEF"], 1)],
                keep={"SPY": "você já tem este ativo", "QQQ": "você já tem este ativo"})
    assert {"SPY", "QQQ"} <= set(sr.selected) and set(sr.selected) & {"AGG", "BND", "IEF"}       # 3 > cap of 2
    cdi = pd.Series(100 * np.cumprod(np.full(n, 1.0004)), index=s["SPY"].index)       # near-zero volatility
    with_cash = screen({**s, CASH_TICKER: cdi}, None, {**a, CASH_TICKER: CAT[CASH_TICKER]}, RiskProfile.MODERADO,
                       "crescimento", 0.15, 2)
    alone = screen(s, None, a, RiskProfile.MODERADO, "crescimento", 0.15, 2)
    for t in ("AGG", "BND", "IEF"):
        assert with_cash.table.at[t, "pontuacao"] == pytest.approx(alone.table.at[t, "pontuacao"])
    assert CASH_TICKER in with_cash.selected and "caixa de liquidez diária" in with_cash.table.at[CASH_TICKER, "motivo"]


def test_screen_leaves_short_series_out_and_never_offers_them_as_alternatives():
    s = {"SPY": _series(900, 1), "VOO": _series(900, 6), "QQQ": _series(500, 2, 0.016)}
    sr = screen(s, None, {k: CAT[k] for k in s}, RiskProfile.MODERADO, "crescimento", 0.10, 1)
    assert "QQQ" in sr.short_history and "QQQ" not in sr.selected
    assert sr.table.at["QQQ", "decisao"] == "descartado" and "histórico curto" in sr.table.at["QQQ", "motivo"]
    assert all("QQQ" not in sr.alternatives(t) for t in sr.selected)


def test_cash_and_foreign_exposure_follow_the_investors_currency():
    pol = _policy()
    assert not pol.is_cash(CAT["SHY"]) and pol.is_cash(CAT[CASH_TICKER])                  # SHY carries the dollar
    groups = {g.name: set(g.members) for g in pol.groups([CAT[t] for t in ("IVVB11.SA", "GOLD11.SA", "BOVA11.SA",
                                                                           "PETR4.SA", "AAPL", CASH_TICKER)])}
    foreign = groups["moeda estrangeira (máximo)"]
    assert {"IVVB11.SA", "GOLD11.SA", "AAPL"} <= foreign and not foreign & {"BOVA11.SA", "PETR4.SA", CASH_TICKER}
    assert traits(listing_to_catalog(ListedAsset("XINA11.SA", "TREND CHINA CI", "B3", "BRL", "etf",
                                                 size=9e6))).exposure == "USD"


def test_only_us_markets_lift_the_foreign_currency_limit_and_say_so():
    pol = _policy(markets=["US"])
    assert pol.foreign_cap == 1.0 and any("só EUA" in r for r in pol.rules)
    assert not [g for g in pol.groups([CAT["AAPL"], CAT["MSFT"]]) if g.name.startswith("moeda")]


def test_sector_names_are_normalised_and_unknown_ones_are_reported():
    pol = _policy(excluded_sectors=["Petróleo e Gás", "Technology", "xyz"])
    assert pol.excluded_sectors == {"petroleo_gas", "tecnologia"}
    assert pol.allows(CAT["PETR4.SA"]) and pol.allows(CAT["AAPL"]) and pol.allows(CAT["XLE"])
    assert pol.allows(CAT["ITUB4.SA"]) is None
    assert any("'xyz' não reconhecido" in w for w in pol.warnings)
    assert canonical_sector("bancos") == "financeiro"


def test_listed_us_sectors_share_the_catalogue_keys_and_caps():
    c = listing_to_catalog(ListedAsset("ZZTC", "Zeta Tech Inc", "US", "USD", "acao", "Technology", "Computer Software",
                                       size=5e9))
    assert c is not None and c.sector == "tecnologia"
    pol = _policy()
    tech = next(g for g in pol.groups([c, CAT["AAPL"], CAT["TOTS3.SA"]]) if g.name.startswith("setor tecnologia"))
    assert set(tech.members) == {"ZZTC", "AAPL", "TOTS3.SA"}


def test_ima_b_5_plus_is_long_duration():
    c = listing_to_catalog(ListedAsset("IB5M11.SA", "IT NOW IMA-B 5+ CI", "B3", "BRL", "etf", size=9e6))
    assert c is not None and "prazo longo" in c.business and traits(c).duration == "longa"


@pytest.mark.parametrize(("typed", "code"), [("petr4", "PETR4.SA"), ("b3sa3", "B3SA3.SA"), ("B5P211", "B5P211.SA"),
                                             ("PETR4F", "PETR4.SA"), ("brk.b", "BRK-B"), ("BRK/B", "BRK-B"),
                                             ("AAPL", "AAPL"), ("BTC-USD", "BTC-USD"), ("PETR4.SA", "PETR4.SA")])
def test_tickers_are_normalised(typed, code):
    assert norm_ticker(typed) == code


def test_classes_may_include_cash(tmp_path):
    r = run_purpose(PurposeRequest(**{**OFFLINE, "classes": ["acoes_br", "caixa"]}),
                    output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    assert "SINT_SELIC" in r.universe["baixados"]


def test_governance_is_applied_before_the_screening(tmp_path):
    flag = {"VALE3.SA": {"status": "excluido", "reason": "teste", "source": "teste unitário"}}
    r = _run(tmp_path, restricoes_governanca=flag)
    assert "VALE3.SA" in r.governance.removed and "VALE3.SA" not in r.screening.table.index
    assert all("VALE3.SA" not in [x["ticker"] for x in a["alternativas"]] for a in r.allocation)


def test_empty_ticker_in_holdings_is_an_error_not_silently_dropped():
    with pytest.raises(ValueError, match="sem código"):
        PurposeRequest(**{**OFFLINE, "carteira_atual": {" ": 1000}})


# ----------------------------------------------------------------------------- selection
def test_band_filter_never_hands_the_choice_to_a_baseline():
    vols = pd.Series({"equal_weight": 0.08, "utility": 0.15, "max_sharpe": 0.20})
    elig, excluded, warn = band_filter(["equal_weight", "utility", "max_sharpe"], vols, 0.10, 0.10, REFERENCE_ONLY)
    assert elig == ["equal_weight", "utility", "max_sharpe"] and excluded == [] and warn
    elig, excluded, warn = band_filter(["equal_weight", "utility", "max_sharpe"], vols.replace(0.15, 0.105), 0.10,
                                       0.10, REFERENCE_ONLY)
    assert elig == ["equal_weight", "utility"] and excluded == ["max_sharpe"] and not warn


def test_chosen_weights_are_brought_within_the_volatility_cap():
    syms = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.30, 0.10, 0.02]) ** 2, index=syms, columns=syms)
    mu = pd.Series([0.15, 0.09, 0.05], index=syms)
    pc = PortfolioConstraints(syms, np.zeros(3), np.full(3, 0.7), [GroupConstraint("g", ("A", "B"), 0.2, 1.0)])
    w = pd.Series([0.7, 0.3, 0.0], index=syms)
    vol = float(np.sqrt(w @ cov @ w))
    res = OptimizationResult("max_sharpe", w, True, "ok", float(w @ mu), vol, None, True)
    out = _within_vol_cap(res, cov, pc, mu, 0.0, 0.10)
    assert out is not None and out.volatility <= 0.10 * (1 + 1e-4) and not verify_weights(out.weights.to_numpy(), pc)
    assert _within_vol_cap(res, cov, pc, mu, 0.0, 0.001) is None           # no portfolio fits: said by the caller


# ----------------------------------------------------------------------------- input parsing
@pytest.mark.parametrize(("typed", "value"), [("1500", 1500.0), ("1.500", 1500.0), ("1.500,00", 1500.0),
                                              ("1,500.00", 1500.0), ("1,500", 1500.0), ("1500,5", 1500.5),
                                              ("1500.5", 1500.5), ("R$ 1.500.000,00", 1_500_000.0), ("0,50", 0.5)])
def test_amounts_in_brazilian_and_us_formats(typed, value):
    assert parse_amount(typed) == value


@pytest.mark.parametrize("typed", ["", "abc", "1,500,00", "1.2.3", "-5"])
def test_invalid_amounts_are_rejected(typed):
    with pytest.raises(ValidationError):
        parse_amount(typed)


def test_cli_reads_holdings_in_brazilian_format(tmp_path):
    import main

    base = ["--offline", "--proposito", "--objetivo", "300000", "--horizon-years", "8", "--capital", "10000",
            "--output-dir", str(tmp_path)]
    args = main.build_parser().parse_args([*base, "--carteira-atual", "SINT_SELIC=1.500,00",
                                           "--carteira-atual", "SINT_IPCA=2,000.50"])
    req = main.purpose_request_from_args(args, main.config_from_args(args))
    assert req.carteira_atual == {"SINT_SELIC": 1500.0, "SINT_IPCA": 2000.5}
    bad = main.build_parser().parse_args([*base, "--carteira-atual", "=100"])
    with pytest.raises(ValueError):
        main.purpose_request_from_args(bad, main.config_from_args(bad))


def test_page_amount_parser_and_headings_are_fixed():
    page = (Path(__file__).resolve().parents[1] / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    assert "function parseAmount(v)" in page and "parseAmount(m[2])" in page
    assert 'el("h3", {}, lab)' not in page                 # the heading was the label map ("[object Object]")
    assert "os dados sintéticos não mudam" in page
    assert json.dumps("Se vender o excesso")[1:-1] in page
