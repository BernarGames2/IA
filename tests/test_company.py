"""Company profile and growth reasons for the stock check: segment knowledge base, B3 and Yahoo
parsing (test doubles, no network), rules with their sources, and the end-to-end check."""

import base64
import json

import numpy as np
import pandas as pd
import pytest

from core.market_data.base import RawSeries
from core.purpose.catalog import REAL_CATALOG, SYNTHETIC_CATALOG
from core.purpose.language import guarantee_phrases
from core.stock_check.analyzer import analyze_stock, render_markdown, to_dict
from core.stock_check.company import (
    CompanyProfile,
    YahooFundamentals,
    fetch_b3_company,
    fundamentals_table,
    growth_case,
    load_company_profile,
    parse_summary,
)
from core.stock_check.resolver import resolve
from core.stock_check.sectors import SEGMENTS, classify_segment, segment_for_class


def _segment_of(a):
    biz = a.business.replace("[sintético] ", "")
    seg = segment_for_class(a.macro_class, a.asset_class, biz)
    if seg is None:
        found = classify_segment([("catálogo", biz)])
        seg = found[0] if found else None
    return seg


def test_knowledge_base_is_probabilistic_and_complete():
    for s in SEGMENTS.values():
        assert s.como_ganha and s.cresce and s.limita, s.key
        for text in (s.como_ganha, *(t for t, _ in s.cresce), *(t for t, _ in s.limita)):
            assert guarantee_phrases(text) == [], (s.key, text)
        assert {scope for _, scope in (*s.cresce, *s.limita)} <= {"*", "BR", "US"}


def test_every_catalogue_asset_has_a_segment():
    missing = [a.ticker for a in (*REAL_CATALOG, *SYNTHETIC_CATALOG) if _segment_of(a) is None]
    assert missing == []


@pytest.mark.parametrize("ticker,key", [
    ("RADL3.SA", "farmacias"), ("PGMN3.SA", "farmacias"), ("PNVL3.SA", "farmacias"), ("CVS", "farmacias"),
    ("HYPE3.SA", "farmaceutica"), ("HAPV3.SA", "planos_saude"), ("RDOR3.SA", "hospitais"), ("FLRY3.SA", "diagnosticos"),
    ("BBDC4.SA", "bancos"), ("ORCL", "software"), ("BBSE3.SA", "seguros"), ("BPAC11.SA", "gestao_bolsa"),
    ("ITSA4.SA", "holding"), ("SBSP3.SA", "saneamento"), ("EQTL3.SA", "energia_distribuicao"),
    ("TAEE11.SA", "energia_transmissao"), ("EGIE3.SA", "energia_geracao"), ("UGPA3.SA", "combustiveis"),
    ("ASML", "semicondutores"), ("GE", "aeroespacial"), ("WEGE3.SA", "bens_capital"), ("MULT3.SA", "shoppings"),
    ("HGLG11.SA", "fii_tijolo"), ("KNCR11.SA", "fii_papel"), ("DUOL", "educacao"), ("PG", "beleza_higiene"),
    ("BOVA11.SA", "etf_acoes"), ("IMAB11.SA", "etf_renda_fixa"), ("GLD", "ouro"), ("BTC-USD", "cripto"),
])
def test_segment_examples(ticker, key):
    a = {x.ticker: x for x in REAL_CATALOG}[ticker]
    assert _segment_of(a).key == key


@pytest.mark.parametrize("text,key", [
    ("Pharmaceutical Retailers", "farmacias"), ("Retail-Drug Stores and Proprietary Stores", "farmacias"),
    ("Saúde / Comércio e Distribuição / Medicamentos e Outros Produtos", "farmacias"),
    ("Comércio varejista de produtos farmacêuticos, perfumaria e cosméticos", "farmacias"),
    ("Drug Manufacturers - General", "farmaceutica"), ("Financeiro / Intermediários Financeiros / Bancos", "bancos"),
    ("Banks - Regional", "bancos"), ("Utilities - Regulated Water", "saneamento"),
    ("Oil & Gas Refining & Marketing", "combustiveis"), ("Oil & Gas Integrated", "petroleo"),
    ("Semiconductors", "semicondutores"), ("Software - Application", "software"), ("Airlines", "aviacao"),
    ("Household & Personal Products", "beleza_higiene"), ("Residential Construction", "construcao"),
])
def test_classification_from_source_texts(text, key):
    seg, src = classify_segment([("fonte", text)])
    assert seg.key == key and src == "fonte"


def test_generic_sector_is_only_a_fallback():
    seg, src = classify_segment([("Yahoo — setor", "Utilities"), ("Yahoo — indústria", "Utilities - Renewable")])
    assert seg.key == "energia_geracao" and src == "Yahoo — indústria"
    seg, _ = classify_segment([("setor", "Healthcare")])
    assert seg.key == "saude_geral"
    assert classify_segment([("x", "algo sem relação")]) is None


# ------------------------------------------------------------------ sources (test doubles)
B3_LIST = {"results": [{"codeCVM": "24910", "issuingCompany": "PGMN", "companyName": "EMPREENDIMENTOS PAGUE MENOS S.A.",
                        "tradingName": "PAGUE MENOS"}]}
B3_DETAIL = {"companyName": "EMPREENDIMENTOS PAGUE MENOS S.A.", "tradingName": "PAGUE MENOS",
             "industryClassification": "Saúde / Comércio e Distribuição / Medicamentos e Outros Produtos",
             "activity": "Comércio Varejista de Produtos Farmacêuticos. Perfumaria e Cosméticos",
             "website": "www.paguemenos.com.br"}
YH = {"quoteSummary": {"result": [{
    "assetProfile": {"city": "Fortaleza", "country": "Brazil", "industry": "Pharmaceutical Retailers",
                     "sector": "Healthcare", "fullTimeEmployees": 25000,
                     "longBusinessSummary": "Operates a chain of drugstores in Brazil. It sells medicines."},
    "financialData": {"currentPrice": {"raw": 4.0}, "targetMeanPrice": {"raw": 5.2},
                      "numberOfAnalystOpinions": {"raw": 6},
                      "recommendationKey": "buy", "revenueGrowth": {"raw": 0.12, "fmt": "12.00%"},
                      "earningsGrowth": {"raw": -0.3}, "profitMargins": {"raw": 0.012}, "returnOnEquity": {"raw": 0.04},
                      "debtToEquity": {"raw": 180.0, "fmt": "180.00"}, "freeCashflow": {"raw": -1.2e8},
                      "totalRevenue": {"raw": 1.2e10}, "financialCurrency": "BRL"},
    "defaultKeyStatistics": {"mostRecentQuarter": {"raw": 1782777600}},
    "summaryDetail": {"trailingPE": {"raw": 40.0}, "forwardPE": {"raw": 20.0},
                      "dividendYield": {"raw": 0.01, "fmt": "1.00%"}, "marketCap": {"raw": 2.4e9},
                      "currency": "BRL"}}], "error": None}}


class Router:
    """Fake HTTP: BCB Selic, B3 listed companies and the Yahoo cookie/crumb/quoteSummary flow."""

    def __init__(self, b3=True, yahoo=True, stale_crumb=False):
        self.calls, self.b3, self.yahoo, self.stale = [], b3, yahoo, stale_crumb
        self.crumbs = iter(["c1", "c2", "c3"])

    def __call__(self, url, timeout):
        self.calls.append(url)
        if "bcb.gov.br" in url:
            return 200, b'[{"data":"16/10/2026","valor":"15.00"}]'
        if "sistemaswebb3-listados" in url:
            if not self.b3:
                return 503, b""
            payload = json.loads(base64.b64decode(url.rsplit("/", 1)[1]))
            return 200, json.dumps(B3_DETAIL if "codeCVM" in payload else B3_LIST).encode()
        if "fc.yahoo.com" in url:
            return 404, b"<html>"
        if "getcrumb" in url:
            return (200, next(self.crumbs).encode()) if self.yahoo else (429, b"Too Many Requests")
        if "quoteSummary" in url:
            if self.stale and "crumb=c1" in url:
                return 401, b'{"finance":{"error":{"code":"Unauthorized","description":"Invalid Crumb"}}}'
            return 200, json.dumps(YH).encode()
        return 500, b""


def test_b3_parsing_and_payload():
    r = Router()
    d = fetch_b3_company("PGMN", r)
    first = json.loads(base64.b64decode(r.calls[0].rsplit("/", 1)[1]))
    assert first["company"] == "PGMN" and first["language"] == "pt-br"
    assert d["atividade"] == "Comércio Varejista de Produtos Farmacêuticos, Perfumaria e Cosméticos"
    assert d["classificacao"] == "Saúde › Comércio e Distribuição › Medicamentos e Outros Produtos"


def test_yahoo_crumb_refresh_and_parsing():
    r = Router(stale_crumb=True)
    res = YahooFundamentals(r).summary("PGMN3.SA")
    summaries = [u for u in r.calls if "quoteSummary" in u]
    assert len(summaries) == 2 and "crumb=c1" in summaries[0] and "crumb=c2" in summaries[1]
    prof, f = parse_summary(res)
    assert prof["industria_en"] == "Pharmaceutical Retailers" and prof["funcionarios"] == 25000
    assert f["crescimento_receita"] == pytest.approx(0.12) and f["divida_patrimonio"] == pytest.approx(1.8)
    assert f["dividend_yield"] == pytest.approx(0.01) and f["recomendacao"] == "buy"
    assert f["ultimo_trimestre"] == "2026-06-30" and "pl_projetado" in f and "margem_bruta" not in f


def test_profile_combines_sources_and_reports_failures(tmp_path):
    res = resolve("pague menos")
    p = load_company_profile(res, fetch=Router(), cache_dir=tmp_path)
    assert p.segmento.key == "farmacias" and p.segmento_fonte == "catálogo"
    assert "farmácias" in p.descricao and p.descricao_fonte.startswith("catálogo")
    assert p.atividade_b3.startswith("Comércio Varejista") and p.classificacao_fonte == "B3"
    assert p.sede == "Fortaleza, Brasil" and p.razao_social == "Empreendimentos Pague Menos S.A."
    assert p.fundamentos["roe"] == pytest.approx(0.04)
    assert "último trimestre reportado: 2026-06-30" in p.fundamentos_fonte
    # cached: a second call does not touch the network
    r2 = Router()
    p2 = load_company_profile(res, fetch=r2, cache_dir=tmp_path)
    assert r2.calls == [] and p2.atividade_b3 == p.atividade_b3
    # sources down: reported, nothing invented
    down = load_company_profile(resolve("PNVL3"), fetch=Router(b3=False, yahoo=False), cache_dir=tmp_path)
    assert down.fundamentos == {} and down.atividade_b3 is None and "farmácias" in down.descricao
    assert any(u.startswith("B3:") for u in down.indisponivel) and any("Yahoo" in u for u in down.indisponivel)
    assert not (tmp_path / "perfis" / "PNVL3.SA.json").exists()        # failures are not cached


def test_unlisted_company_is_described_from_b3_only(tmp_path):
    from core.stock_check.resolver import Resolved

    res = Resolved("PGMN3.SA", "PGMN3", None, None, "código B3 PGMN3")       # no catalogue entry
    p = load_company_profile(res, fetch=Router(), cache_dir=tmp_path)
    assert p.descricao_fonte == "B3 — atividade registrada" and p.segmento.key == "farmacias"
    assert p.segmento_fonte == "B3 — atividade registrada"


def _profile(seg="farmacias", pais="BR", **f):
    return CompanyProfile("PGMN3.SA", "Pague Menos ON", "rede de farmácias", "catálogo", SEGMENTS[seg], "catálogo",
                          fundamentos=f, fundamentos_fonte="Yahoo Finance (teste)", pais=pais)


ST = {"retorno_12m": 0.25, "acima_media_200d": True, "t_excesso": 1.4, "excesso_anual": 0.08, "anos_de_dados": 5.0,
      "volatilidade": 0.45, "queda_maxima": -0.6, "beta": 1.4, "giro_diario_medio": 3e6}
PJ = {"p_superar_rf": 0.62, "horizonte_anos": 3.0}


def test_growth_rules_with_bases_and_sources():
    g = growth_case(_profile(crescimento_receita=0.12, crescimento_lucro=-0.3, margem_liquida=0.012, roe=0.04,
                             divida_patrimonio=1.8, pl=40.0, pl_projetado=20.0, dividend_yield=0.07,
                             fluxo_caixa_livre=-1.2e8, receita_12m=1.2e10, preco_atual=4.0, preco_alvo_medio=5.2,
                             n_analistas=6, recomendacao="buy", moeda_balanco="BRL"),
                    ST, PJ, 0.15, "CDI/Selic", "BRL", "teste")
    up = {x["texto"]: x["base"] for x in g["pode_crescer"]}
    down = {x["texto"]: x["base"] for x in g["pode_nao_crescer_tanto"]}
    has = lambda d, frag, base: any(frag in t and b == base for t, b in d.items())
    assert has(up, "Receita cresceu 12%", "fundamentos") and has(up, "6 analistas", "analistas")
    assert has(up, "lucro maior nos próximos 12 meses", "analistas") and has(up, "Tendência de alta", "historico")
    assert has(up, "62% de chance", "simulacao") and has(up, "Envelhecimento da população", "setor")
    assert has(down, "Lucro caiu 30%", "fundamentos") and has(down, "Margem líquida baixa", "fundamentos")
    assert has(down, "abaixo do CDI/Selic", "fundamentos") and has(down, "Dívida alta: 1.8", "fundamentos")
    assert has(down, "P/L de 40", "fundamentos") and has(down, "dividend yield de 7.0%", "fundamentos")
    assert has(down, "caixa livre negativo", "fundamentos") and has(down, "Juros altos", "juros")
    assert has(down, "volatilidade de 45%", "historico") and has(down, "Liquidez baixa", "historico")
    assert has(down, "CMED", "setor")                                       # Brazilian rule for a Brazilian company
    for x in g["pode_crescer"] + g["pode_nao_crescer_tanto"]:
        assert x["fonte"] and x["rotulo_base"] and guarantee_phrases(x["texto"]) == []


def test_growth_rules_respect_scope_and_financials():
    g = growth_case(_profile(pais="US", margem_liquida=0.01, divida_patrimonio=3.0), ST, PJ, 0.04, "T-bill", "USD", "t")
    texts = " ".join(x["texto"] for x in g["pode_nao_crescer_tanto"])
    bases = {x["base"] for x in g["pode_nao_crescer_tanto"]}
    assert "CMED" not in texts and "Cotada em dólar" in texts and "juros" not in bases and "cambio" in bases
    bank = growth_case(_profile(seg="bancos", margem_liquida=0.01, divida_patrimonio=9.0, roe=0.2), ST, PJ, 0.15,
                       "CDI/Selic", "BRL", "t")
    bt = " ".join(x["texto"] for x in bank["pode_nao_crescer_tanto"])
    assert "Dívida alta" not in bt and "Margem líquida" not in bt              # not meaningful for banks
    assert any("ROE) de 20%" in x["texto"] for x in bank["pode_crescer"])


def test_fundamentals_table_marks_missing_values():
    rows = {r["indicador"]: r["valor"] for r in fundamentals_table(_profile(roe=0.04, preco_alvo_medio=5.2,
                                                                            n_analistas=6, recomendacao="buy"), "BRL")}
    assert rows["Retorno sobre o patrimônio (ROE)"] == "4.0%" and rows["Margem líquida"] == "indisponível"
    assert rows["Preço-alvo médio dos analistas"] == "R$ 5.20 (6 analistas; recomendação média: compra)"
    assert fundamentals_table(_profile(), "BRL") == []


def _series(sym, n, seed, vol=0.02):
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    rng = np.random.default_rng(seed)
    px = pd.Series(10 * np.exp(np.cumsum(rng.normal(0.0004, vol, n))), index=idx, name=sym)
    vol_s = pd.Series(rng.integers(500_000, 900_000, n).astype(float), index=idx)
    return RawSeries(sym, px, vol_s, "fake_test_double", "adjclose", "teste", "BRL", "1d")


class _Prov:
    name = "fake_test_double"

    def __init__(self, series):
        self.series, self.failures = series, {}

    def fetch(self, symbols, period, interval):
        return {s: self.series[s] for s in symbols if s in self.series}


def test_stock_check_shows_what_the_company_does_and_why(tmp_path):
    prov = _Prov({"PGMN3.SA": _series("PGMN3.SA", 1250, 7), "BOVA11.SA": _series("BOVA11.SA", 1250, 8, 0.012)})
    r = analyze_stock("pague menos", horizon_years=3, provider=prov, rf_fetch=Router(), cache_dir=tmp_path,
                      n_paths=1500)
    assert r.company.segmento.key == "farmacias" and r.company.atividade_b3
    assert r.growth["pode_crescer"] and r.growth["pode_nao_crescer_tanto"]
    md = render_markdown(r)
    for section in ("## O que a empresa faz", "## Por que pode crescer", "## Por que pode não crescer tanto",
                    "## Indicadores da empresa", "Como o segmento costuma ganhar dinheiro", "CMED"):
        assert section in md
    assert guarantee_phrases(md.replace("pague menos", "")) == []
    d = json.loads(json.dumps(to_dict(r)))
    assert d["empresa"]["segmento"]["chave"] == "farmacias" and d["empresa"]["indicadores"]
    assert d["por_que_pode_crescer"][0]["base"] and d["negocio"].startswith("rede de farmácias")
