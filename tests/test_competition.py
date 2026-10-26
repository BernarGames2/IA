"""Investment-competition page: reading the client case and the rulebook (English and Portuguese, every value with the
sentence it came from, nothing guessed), rules that cannot hold, and a portfolio that meets every rule it can measure —
built on test doubles of the market (no network)."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.competition.build import (
    ClientProfile,
    CompetitionRequest,
    CompetitionRules,
    _goal,
    _orders,
    _window,
)
from core.competition.ips import english_promises
from core.competition.reader import parse_tickers, read_case, read_rules
from core.purpose.language import guarantee_phrases
from tests.market_doubles import Prov, http, structured_market
from webapp import server

TODAY = date(2026, 10, 16)

CASE_EN = """Meet the Rivera family. Maria Rivera is a 45-year-old nurse and her husband Tom is a teacher. Together they \
have $250,000 in savings that they would like to invest. Their goal is to accumulate $1,000,000 by the time Maria \
retires in 20 years. They can contribute $1,000 per month to the portfolio. They also want to pay for their \
daughter's college tuition of $120,000 in 8 years.

The Riveras describe themselves as moderate investors: they are not comfortable with large swings in their savings \
and could not stomach a loss of more than 20% in a single year. They also keep an emergency fund of $25,000 for the \
next year.

They care deeply about the environment and do not want to invest in fossil fuels, tobacco or weapons manufacturers. \
Maria is passionate about healthcare innovation. Tom earns $60,000 per year."""

CASE_PT = """A cliente, Ana, tem 38 anos e é médica. Ela tem R$ 400 mil guardados e quer ter R$ 2 milhões daqui a 15 anos \
para se aposentar. Ana se considera arrojada, mas não aceita perder mais de 25% em um ano. Ela pode aportar R$ 3 mil \
por mês. Ana não quer investir em empresas de petróleo nem de cigarros e valoriza a educação. Precisa de uma reserva \
de R$ 50 mil no próximo ano."""

RULES = """Trading Details 2026-2027. Each team receives $500,000 in virtual cash to build its portfolio on the \
simulator. Any stock from any exchange priced at $5 or higher may be traded. Portfolios must hold at least one ETF \
from the approved ETF list. No single stock may exceed 25% of the portfolio value. Teams must hold at least 10 \
stocks. Margin trading is prohibited and short selling is not permitted. Each team may make a maximum of 200 \
trades. A team cannot trade more than twice a security's current daily volume. Registration closes September 11, \
2026. Trading begins September 28 and ends December 4, 2026. The final report deadline is December 12, 2026."""


def _quoted(text: str, reading: dict) -> None:
    """Every value carries verbatim sentences of the pasted text."""
    for f in reading["achados"]:
        for part in f["trecho"].split(" … "):
            assert part and part in text, (f["campo"], part)


def test_reads_the_case_in_english_and_portuguese():
    en = read_case(CASE_EN, TODAY).as_dict()
    _quoted(CASE_EN, en)
    s = en["sugestao"]
    assert en["idioma"] == "en" and s["idade"] == 45 and s["moeda_caso"] == "USD"
    assert s["patrimonio_cliente"] == 250_000 and s["valor_alvo_cliente"] == 1_000_000
    assert s["horizonte_anos"] == 20 and s["aporte_anual_cliente"] == 12_000       # $1,000 a month; Tom's salary ignored
    assert "retirada_anual_cliente" not in s
    assert s["perfil_risco"] == "moderado"            # the client's own label wins over the described behaviour
    assert s["objetivo_tipo"] == "crescimento" and s["tolerancia_perda"] == pytest.approx(0.20)
    assert s["necessidade_liquidez"] == pytest.approx(0.10)                         # $25,000 of $250,000
    assert s["exclusoes"] == ["armas", "combustiveis_fosseis", "tabaco"]
    assert {"saude", "sustentabilidade"} <= set(s["temas"])
    assert any(f["campo"] == "meta_intermediaria" and f["valor"] == 120_000 for f in en["achados"])
    pt = read_case(CASE_PT, TODAY).as_dict()
    _quoted(CASE_PT, pt)
    s = pt["sugestao"]
    assert pt["idioma"] == "pt" and s["idade"] == 38 and s["moeda_caso"] == "BRL"
    assert s["patrimonio_cliente"] == 400_000 and s["valor_alvo_cliente"] == 2_000_000 and s["horizonte_anos"] == 15
    assert s["aporte_anual_cliente"] == 36_000 and s["perfil_risco"] == "arrojado"
    assert s["tolerancia_perda"] == pytest.approx(0.25) and s["necessidade_liquidez"] == pytest.approx(0.125)
    assert s["exclusoes"] == ["combustiveis_fosseis", "tabaco"] and s["temas"] == ["educacao"]
    assert "objetivo principal (crescer, renda ou proteger)" in pt["faltando"]         # not written: not guessed


def test_reads_the_rules():
    r = read_rules(RULES, TODAY).as_dict()
    _quoted(RULES, r)
    s = r["sugestao"]
    assert s["capital"] == 500_000 and s["moeda"] == "USD" and s["preco_minimo"] == 5
    assert s["min_etfs"] == 1 and s["lista_etfs_aprovados"] is True and s["peso_maximo"] == pytest.approx(0.25)
    assert s["min_posicoes"] == 10 and "max_posicoes" not in s
    assert s["sem_margem"] and s["sem_venda_descoberto"] and s["max_operacoes"] == 200 and s["limite_volume"] == 2
    assert s["inicio_negociacao"] == "2026-09-28" and s["fim_negociacao"] == "2026-12-04"   # not the registration/report
    assert r["faltando"] == []
    pt = read_rules("Cada equipe recebe US$ 100.000 em dinheiro virtual. Nenhuma ação pode ter mais de 20% da carteira: "
                    "no máximo 20% por ação. Pelo menos 8 ações e no máximo 15 ativos. Pelo menos 2 ETFs.", TODAY).as_dict()
    s = pt["sugestao"]
    assert s["capital"] == 100_000 and s["peso_maximo"] == pytest.approx(0.20)
    assert s["min_posicoes"] == 8 and s["max_posicoes"] == 15 and s["min_etfs"] == 2


def test_the_reader_does_not_guess():
    text = ("John is not an aggressive investor and is not interested in technology. He earns $90,000 a year and "
            "works in the oil industry. He has $80,000 to invest.")
    s = read_case(text, TODAY).as_dict()["sugestao"]
    assert "perfil_risco" not in s and "temas" not in s and "exclusoes" not in s
    assert "aporte_anual_cliente" not in s and s["patrimonio_cliente"] == 80_000
    empty = read_case("", TODAY).as_dict()
    assert empty["achados"] == [] and empty["faltando"]
    assert parse_tickers("Approved ETF list: SPY, QQQ; VTI\nAGG - iShares Core US Aggregate Bond ETF\nBRK.B") == \
        ["SPY", "QQQ", "VTI", "AGG", "BRK.B"]


def test_rules_that_cannot_hold_are_refused():
    with pytest.raises(ValidationError, match="só 60%"):
        CompetitionRules(capital=500_000, max_posicoes=3, peso_maximo=0.20)
    with pytest.raises(ValidationError, match="passa do máximo"):
        CompetitionRules(capital=500_000, min_posicoes=12, max_posicoes=10)
    with pytest.raises(ValidationError, match="lista aprovada"):
        CompetitionRules(capital=500_000, min_etfs=3, etfs_aprovados=["SPY", "QQQ"])
    with pytest.raises(ValidationError, match="menor posição"):
        CompetitionRules(capital=500_000, peso_maximo=0.05, peso_minimo=0.10)


def test_goal_scale_window_and_trimming():
    cli = ClientProfile(horizonte_anos=20, patrimonio_cliente=250_000, valor_alvo_cliente=1_000_000,
                        aporte_anual_cliente=12_000)
    target, contrib, how, _ = _goal(cli, 500_000)
    assert target == pytest.approx(2_000_000) and contrib == pytest.approx(24_000) and "×2,00" in how
    target, contrib, how, _ = _goal(ClientProfile(horizonte_anos=10, retorno_exigido=0.06), 100_000)
    assert target == pytest.approx(100_000 * 1.06 ** 10) and contrib == 0 and "retorno exigido" in how
    target, _, how, _ = _goal(ClientProfile(horizonte_anos=10, valor_alvo_cliente=5e5), 100_000)
    assert target == pytest.approx(100_000 * 1.03 ** 10) and "poder de compra" in how and "não pôde" in how
    n, txt, iso, note = _window(TODAY, date(2026, 12, 4))
    assert n == 35 and txt == "04/12/2026" and iso == "2026-12-04" and note == ""
    assert _window(TODAY, None)[0] == 63 and "sem data" in _window(TODAY, None)[3]
    rows = [{"ticker": "AAA", "quantidade": 300.0, "preco": 100.0, "valor": 30_000.0},
            {"ticker": "BBB", "quantidade": 700.0, "preco": 100.0, "valor": 70_000.0}]
    out, cash, notes = _orders(rows, 100_000, 0.65, False)
    assert out[1]["quantidade"] == 650 and cash == pytest.approx(5_000) and "BBB" in notes[0]


def _market(cheap: tuple[str, ...] = ()):
    mk = structured_market(drift_shift={t: 0.25 for t in cheap})
    for t in cheap:          # same returns, a price below the rule's minimum
        mk[t] = dataclasses.replace(mk[t], close=mk[t].close / 1000)
        assert float(mk[t].close.iloc[-1]) < 5
    return mk


APPROVED = ["AGG", "BND", "SHY", "SPY", "VOO", "SCHD", "VYM", "XLV"]


def _request(**rules) -> CompetitionRequest:
    return CompetitionRequest(
        cliente={"horizonte_anos": 20, "perfil_risco": "moderado", "objetivo_tipo": "crescimento", "idade": 45,
                 "patrimonio_cliente": 250_000, "valor_alvo_cliente": 1_000_000, "aporte_anual_cliente": 12_000,
                 "tolerancia_perda": 0.25, "exclusoes": ["combustiveis_fosseis", "tabaco", "armas"], "temas": ["saude"],
                 "citacoes": {"perfil_risco": "The Riveras describe themselves as moderate investors"}},
        regras={"nome": "teste", "capital": 500_000, "preco_minimo": 5, "peso_maximo": 0.20, "min_posicoes": 12,
                "max_posicoes": 20, "min_etfs": 2, "etfs_aprovados": APPROVED, "min_setores": 5, "caixa_maximo": 0.02,
                "max_operacoes": 200, "limite_volume": 2, "fim_negociacao": "2026-12-04", **rules},
        n_simulacoes=1000, max_ativos=30)


def test_build_meets_every_rule(tmp_path):
    from core.competition.build import build

    cheap = ("MSFT", "JNJ")
    out = build(_request(), provider=Prov(_market(cheap)), rf_fetch=http, cache_dir=tmp_path, output_dir=tmp_path,
                today=TODAY)
    checks = {c["regra"]: c for c in out["conferencia"]}
    assert out["cumpre_regras"] and all(c["ok"] is True for c in out["conferencia"]), out["conferencia"]
    pos = out["posicoes"]
    tickers = {p["ticker"] for p in pos}
    assert 12 <= len(pos) <= 20 and max(p["peso"] for p in pos) <= 0.20 + 1e-9
    assert all(float(p["quantidade"]).is_integer() and p["preco"] >= 5 for p in pos)
    etfs = [p["ticker"] for p in pos if p["etf"]]
    assert len(etfs) >= 2 and set(etfs) <= set(APPROVED)                          # only approved ETFs
    assert len({p["setor"] for p in pos if p["setor"]}) >= 5
    assert not tickers & {"XOM", "CVX", "XLE", "LMT", "BA", "RTX"}               # the client's exclusions
    assert not tickers & set(cheap) and set(out["fora_pelo_preco"]) & set(cheap)   # left out by the price rule
    assert all("abaixo do mínimo" in why for why in out["fora_pelo_preco"].values())
    assert out["caixa"] >= 0 and out["caixa"] / 500_000 <= 0.02
    assert abs(out["investido"] + out["caixa"] - 500_000) < 0.01
    assert "Exclusões do cliente" in checks and checks["Operações"]["obtido"].startswith(str(len(pos)))
    assert out["ordens_csv"].count("\n") == len(pos)
    j = out["janela"]
    assert j["dias"] == 35 and j["p10"] < j["p50"] < j["p90"] and 0 <= j["p_perda_10"] <= 1
    assert english_promises(out["ips_en"]) == [] and guarantee_phrases(out["resumo_pt"]) == []
    assert "INVESTMENT POLICY STATEMENT" in out["ips_en"] and "moderate investors" in out["ips_en"]
    assert "×2,00" in out["objetivo"]["como"] and not [a for a in out["alertas"] if a["category"] == "linguagem"]
    fit = {f["necessidade"]: f for f in out["cliente_atendido"]}
    assert fit["Exclusões"]["ok"] is True and fit["Perfil moderado"]["caso"]


def test_build_reports_a_rule_it_cannot_meet(tmp_path):
    from core.competition.build import build

    out = build(_request(etfs_aprovados=["ZZZZ", "YYYY"], min_etfs=1), provider=Prov(_market()), rf_fetch=http,
                cache_dir=tmp_path, output_dir=tmp_path, today=TODAY)
    checks = {c["regra"]: c for c in out["conferencia"]}
    assert out["cumpre_regras"] is False and checks["ETFs"]["ok"] is False             # said, never hidden


def test_competition_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(_market()))
    monkeypatch.setattr(server, "_rf_fetch", lambda: http)
    monkeypatch.setattr(server, "_OUTPUT_DIR", tmp_path)
    c = TestClient(server.app)
    r = c.post("/api/competicao/ler", json={"desafio": CASE_EN, "regras": RULES, "etfs": "SPY, QQQ"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["desafio"]["sugestao"]["horizonte_anos"] == 20 and d["regras"]["sugestao"]["capital"] == 500_000
    assert d["etfs"] == ["SPY", "QQQ"]
    assert c.post("/api/competicao/ler", json={}).status_code == 400
    body = _request().model_dump(mode="json")
    body["universo"] = "curado"
    r = c.post("/api/competicao/montar", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["cumpre_regras"] is True and r.json()["posicoes"]
    bad = _request().model_dump(mode="json")
    bad["regras"]["max_posicoes"] = 3
    assert c.post("/api/competicao/montar", json=bad).status_code == 422
    monkeypatch.setattr(server, "_market_provider", lambda: Prov(_market(), fail=set(structured_market())))
    body["seed"] = 7
    assert c.post("/api/competicao/montar", json=body).status_code == 503
