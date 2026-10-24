"""Structured investment thesis: what is measured [H], what is estimated [E], what the sector suggests, what has to
be true, the dated catalysts, the conditions that would invalidate it and the strongest counter-argument.

Rule-based and reproducible: every bullet comes from a published number (with source and date), from the price
model of the forecasts (labelled as an estimate) or from the segment knowledge base (labelled "setor"). Nothing
here says what to do: the thesis is a hypothesis to be tested, and the invalidation conditions say how.
"""

from __future__ import annotations

from datetime import date

from core.fundamentals.analysis import dividends_paid, fcf, growth, year_ratios
from core.fundamentals.sources import Analysts, Annual
from core.purpose.language import guarantee_phrases
from core.stock_check.company import CompanyProfile
from utils.br_format import millions, num, pct

DISCLAIMER = ("Tese em teste, montada por regras a partir de dados publicados e de um modelo estatístico: não é "
              "recomendação de compra ou venda e não oferece garantia de resultado. Cada condição de invalidação diz "
              "como conferir se ela deixou de valer.")


def _item(text: str, base: str, src: str) -> dict[str, str]:
    return {"texto": text, "base": base, "fonte": src}


def build_thesis(p: CompanyProfile, a: Annual | None, an: Analysts | None, fc: dict[str, object] | None,
                 trap: dict[str, object], px_stats: dict[str, float | None], as_of: date) -> dict[str, object]:
    seg = p.segmento
    fin = bool(seg and seg.financeiro)
    src_fin = f"{a.fonte} ({a.consultado_em})" if a else ""
    src_an = f"{an.fonte} ({an.consultado_em})" if an else ""
    ccy = (a.moeda if a and a.moeda else None) or str(p.fundamentos.get("moeda_cotacao") or "")
    money = lambda x: f"{ccy} {millions(x)}".strip()
    facts, favor, against, premises, catalysts, invalid = [], [], [], [], [], []
    r_now = year_ratios(a, -1) if a is not None else {}
    gr = growth(a) if a is not None else {}
    last_year = a.anos[-1][:4] if a is not None else None

    # ---- facts [H]
    if a is not None:
        rev, ni = a.get("receita", -1), a.get("lucro_liquido", -1)
        if rev is not None:
            facts.append(_item(f"Receita de {money(rev)} no ano fiscal de {last_year}"
                               + ("" if gr.get("receita_cagr_3a") is None else
                                  f" ({pct(gr['receita_cagr_3a'], sign=True)} ao ano em 3 anos)"), "medido", src_fin))
        if ni is not None:
            facts.append(_item(f"Lucro líquido de {money(ni)} em {last_year}", "medido", src_fin))
        if r_now.get("margem_operacional") is not None and not fin:
            facts.append(_item(f"Margem operacional de {pct(r_now['margem_operacional'])}", "medido", src_fin))
        if r_now.get("roe") is not None:
            facts.append(_item(f"Retorno sobre o patrimônio (ROE) de {pct(r_now['roe'])}", "medido", src_fin))
        if r_now.get("fcl") is not None and not fin:
            facts.append(_item(f"Fluxo de caixa livre de {money(r_now['fcl'])} em {last_year}", "medido", src_fin))
        if r_now.get("divida_liquida_ebitda") is not None and not fin:
            facts.append(_item(f"Dívida líquida de {num(r_now['divida_liquida_ebitda'])}× o EBITDA", "medido", src_fin))
    fs = trap.get("piotroski")
    if isinstance(fs, dict) and fs.get("pontos") is not None and fs.get("medidos", 0) >= 6:
        facts.append(_item(f"Piotroski F-score de {fs['pontos']} em {fs['medidos']} testes (ano de {str(fs['ano'])[:4]})",
                           "medido", src_fin))
    mult = trap.get("multiplos") or {}
    if isinstance(mult, dict):
        if mult.get("pl") is not None:
            facts.append(_item(f"P/L de {num(mult['pl'])}", "medido", p.fundamentos_fonte or "Yahoo Finance"))
        if mult.get("dividend_yield") is not None:
            facts.append(_item(f"Dividendos de {pct(mult['dividend_yield'])} ao ano (últimos 12 meses)", "medido",
                               p.fundamentos_fonte or "Yahoo Finance"))
    # ---- estimates [E]
    estimates = []
    if fc is not None:
        pr = fc["probabilidades"]
        h = fc["horizonte_anos"]
        estimates.append(_item(f"Modelo de preços: chance de {pct(pr['p_superar_rf'], 0)} de superar o {fc['nome_taxa']} em "  # type: ignore[index]
                               f"{num(h, 0 if float(h) % 1 == 0 else 1)} ano(s); mediana de {pct(fc['quantis']['p50'], sign=True)} "  # type: ignore[index, arg-type]
                               f"e faixa de 80% de {pct(fc['quantis']['p10'], 0, sign=True)} a {pct(fc['quantis']['p90'], 0, sign=True)}",  # type: ignore[index]
                               "modelo", "previsão probabilística (só histórico de preços)"))
    if an is not None and an.crescimento_esperado is not None:
        estimates.append(_item(f"Consenso de analistas: lucro por ação {pct(an.crescimento_esperado, sign=True)} no ano"
                               + (f" ({an.n_analistas} analistas)" if an.n_analistas else ""), "terceiros", src_an))
    target, n_t = p.fundamentos.get("preco_alvo_medio"), p.fundamentos.get("n_analistas")
    price = px_stats.get("ultimo")
    if isinstance(target, float) and price:
        estimates.append(_item(f"Preço-alvo médio de analistas: {num(target, 2)} ({pct(target / price - 1, 0, sign=True)} "
                               f"sobre o último fechamento; {int(n_t) if n_t else '?'} analistas) — opinião de terceiros, "
                               "não usada nas contas", "terceiros", p.fundamentos_fonte or "Yahoo Finance"))
    # ---- for / against (measured first, then model, then sector)
    rc = gr.get("receita_cagr_3a")
    if rc is not None:
        (favor if rc >= 0.05 else against if rc < 0 else []).append(
            _item(f"Receita {'crescendo' if rc >= 0 else 'caindo'} {pct(abs(rc))} ao ano nos últimos 3 anos", "medido", src_fin))
    mo = r_now.get("margem_operacional")
    if mo is not None and not fin:
        if mo >= 0.15:
            favor.append(_item(f"Margem operacional alta, de {pct(mo)}: o negócio converte bem receita em lucro", "medido", src_fin))
        elif mo < 0.05:
            against.append(_item(f"Margem operacional baixa, de {pct(mo)}: pouca folga se os custos subirem", "medido", src_fin))
    roe = r_now.get("roe")
    if roe is not None:
        if roe >= 0.15:
            favor.append(_item(f"ROE de {pct(roe)}", "medido", src_fin))
        elif roe < 0.05:
            against.append(_item(f"ROE baixo, de {pct(roe)}", "medido", src_fin))
    fy = mult.get("fcl_yield") if isinstance(mult, dict) else None
    if fy is not None and not fin:
        if fy >= 0.05:
            favor.append(_item(f"Fluxo de caixa livre de {pct(fy)} do valor de mercado", "medido", src_fin))
        elif fy < 0:
            against.append(_item("Fluxo de caixa livre negativo no último ano", "medido", src_fin))
    nde = r_now.get("divida_liquida_ebitda")
    if nde is not None and not fin:
        if nde <= 1.5:
            favor.append(_item(f"Endividamento baixo: dívida líquida de {num(nde)}× o EBITDA", "medido", src_fin))
        elif nde > 3:
            against.append(_item(f"Endividamento alto: dívida líquida de {num(nde)}× o EBITDA", "medido", src_fin))
    if isinstance(fs, dict) and fs.get("pontos") is not None and fs.get("medidos", 0) >= 6 and fs["pontos"] >= 7:
        favor.append(_item(f"Fundamentos melhorando no ano: F-score de {fs['pontos']}", "medido", src_fin))
    covered = {"receita em queda": rc is not None and rc < 0, "fluxo de caixa livre negativo": fy is not None and fy < 0,
               "dívida alta": nde is not None and nde > 3 and not fin}
    for s in trap.get("sinais", []):  # type: ignore[union-attr]
        if s.get("acionado") and not covered.get(str(s.get("sinal"))):
            against.append(_item(f"{s['sinal'][0].upper()}{s['sinal'][1:]}: {s['valor']}", "medido", str(s.get("fonte"))))
    if trap.get("parece_barata") and trap.get("nivel") == "baixo":
        favor.append(_item("Parece barata (" + "; ".join(trap["criterios_barata"]) + ") com poucos sinais de deterioração",  # type: ignore[arg-type]
                           "medido", "regras do detector de armadilhas"))
    mom, above = px_stats.get("momento_12_1"), px_stats.get("acima_media_200")
    if mom is not None and above is not None and mom >= 0.10 and above:
        favor.append(_item(f"Preço em tendência de alta: {pct(mom, sign=True)} em 12 meses (sem o último) e acima da média de "
                           "200 dias", "medido", "cotações ajustadas"))
    if fc is not None:
        pb = fc["probabilidades"]["p_superar_rf"]  # type: ignore[index]
        if pb >= 0.6:
            favor.append(_item(f"Modelo de preços: {pct(pb, 0)} de chance de superar o {fc['nome_taxa']}", "modelo", "previsão"))
        elif pb <= 0.4:
            against.append(_item(f"Modelo de preços: só {pct(pb, 0)} de chance de superar o {fc['nome_taxa']}", "modelo", "previsão"))
        vol = fc["fatos_H"]["volatilidade"]  # type: ignore[index]
        if vol >= 0.40:
            against.append(_item(f"Oscila muito: volatilidade de {pct(vol, 0)} ao ano", "medido", "cotações ajustadas"))
        cal = fc.get("calibracao") or {}
        if isinstance(cal, dict) and cal.get("cobertura_10_90") is not None and cal["cobertura_10_90"] < 0.7:
            against.append(_item(f"Neste ativo, o modelo de preços errou a faixa de 80% com frequência (acertou "
                                 f"{pct(cal['cobertura_10_90'], 0)} das vezes): trate as chances com cautela", "modelo",
                                 "calibração histórica"))
    if an is not None:
        cur, old = an.eps_ano.get("current"), an.eps_ano.get("90daysAgo")
        if cur is not None and old is not None and old > 0 and cur / old - 1 >= 0.05:
            favor.append(_item(f"Analistas elevando a estimativa de lucro do ano ({pct(cur / old - 1, sign=True)} em 90 dias)",
                               "terceiros", src_an))
    if seg is not None:
        country = p.pais
        favor += [_item(t, "setor", "base de conhecimento por segmento") for t, sc in seg.cresce if sc in ("*", country)][:2]
        against += [_item(t, "setor", "base de conhecimento por segmento") for t, sc in seg.limita if sc in ("*", country)][:2]
    # ---- what has to be true
    if rc is not None:
        premises.append(f"A receita não encolher: nos últimos 3 anos ela variou {pct(rc, sign=True)} ao ano.")
    if mo is not None and not fin:
        premises.append(f"A margem operacional ficar perto de {pct(mo)} (custos sob controle).")
    if fc is not None:
        est = fc["estimativas_E"]
        premises.append(f"O retorno do passado ({pct(fc['fatos_H']['retorno_medio_anual'])} ao ano) não ser só sorte: a média "  # type: ignore[index]
                        f"histórica tem erro padrão de {pct(est['erro_padrao_media'])}, por isso o modelo dá a ela peso de "  # type: ignore[index]
                        f"{pct(est['peso_do_historico'], 0)}.")  # type: ignore[index]
        premises.append(f"A renda fixa render perto de {pct(fc['taxa_livre_de_risco'])} ao ano no período (taxa de hoje).")
    dp = dividends_paid(a, -1) if a is not None else None
    f_last = fcf(a, -1) if a is not None else None
    if dp and f_last is not None and not fin:
        premises.append(f"Os dividendos se sustentarem: o caixa livre do último ano cobriu {num(f_last / dp)}× o que foi pago.")
    # ---- dated catalysts (only upcoming dates published by the source)
    if an is not None:
        for d, ev in ((an.proximo_resultado, "divulgação de resultados (data informada pela fonte; pode mudar)"),
                      (an.data_com, "data com de dividendos"), (an.pagamento_dividendo, "pagamento de dividendos")):
            if d and d >= as_of.isoformat():
                catalysts.append({"data": d, "evento": ev, "fonte": src_an})
    catalysts.sort(key=lambda c: c["data"])
    # ---- invalidation: measurable, checkable later
    if a is not None and a.get("receita", -1) is not None:
        lim = a.get("receita", -1) * 0.95  # type: ignore[operator]
        invalid.append({"condicao": "receita do próximo ano fiscal mais de 5% abaixo da última",
                        "atual": money(a.get("receita", -1)), "limite": money(lim), "como_conferir": "próximo balanço anual",
                        "chave": "receita", "valor_limite": lim, "direcao": "abaixo", "ano_fiscal": a.anos[-1]})
    if mo is not None and not fin:
        invalid.append({"condicao": "margem operacional 3 pontos percentuais abaixo da atual", "atual": pct(mo),
                        "limite": pct(mo - 0.03), "como_conferir": "próximo balanço anual",
                        "chave": "margem_operacional", "valor_limite": mo - 0.03, "direcao": "abaixo",
                        "ano_fiscal": a.anos[-1] if a is not None else None})
    if nde is not None and not fin and nde <= 3.5:
        invalid.append({"condicao": "dívida líquida acima de 3,5× o EBITDA", "atual": f"{num(nde)}×", "limite": "3,5×",
                        "como_conferir": "próximo balanço anual", "chave": "divida_ebitda", "valor_limite": 3.5,
                        "direcao": "acima", "ano_fiscal": a.anos[-1] if a is not None else None})
    if an is not None and an.eps_ano.get("current") is not None:
        invalid.append({"condicao": "estimativa de lucro por ação do ano cair 10% ou mais (analistas)",
                        "atual": num(an.eps_ano["current"], 2), "limite": num(an.eps_ano["current"] * 0.9, 2),  # type: ignore[operator]
                        "como_conferir": "nova consulta da tese", "chave": "lpa_estimado",
                        "valor_limite": an.eps_ano["current"] * 0.9, "direcao": "abaixo", "ano_fiscal": None})  # type: ignore[operator]
    if fc is not None:
        q05 = fc["quantis"]["p05"]  # type: ignore[index]
        invalid.append({"condicao": "fechamento abaixo do cenário dos 5% piores do modelo antes do fim do prazo",
                        "atual": num(fc["preco_referencia"], 2), "limite": num(fc["preco_referencia"] * (1 + q05), 2),  # type: ignore[operator]
                        "como_conferir": f"cotação até {fc['registro']['data_fim']}", "chave": "preco",  # type: ignore[index]
                        "valor_limite": fc["preco_referencia"] * (1 + q05), "direcao": "abaixo",  # type: ignore[operator]
                        "ate": fc["registro"]["data_fim"]})  # type: ignore[index]
    if dp and not fin:
        invalid.append({"condicao": "dividendos pagos no próximo ano abaixo da metade dos do último", "atual": money(dp),
                        "limite": money(dp / 2), "como_conferir": "próximo balanço anual", "chave": "dividendos",
                        "valor_limite": dp / 2, "direcao": "abaixo", "ano_fiscal": a.anos[-1] if a is not None else None})
    # ---- counter-thesis and quality of the data behind it
    hits = [s for s in trap.get("sinais", []) if s.get("acionado")]  # type: ignore[union-attr]
    if trap.get("parece_barata") and trap.get("nivel") == "alto":
        counter = "O preço baixo pode ser merecido: " + "; ".join(f"{s['sinal']} ({s['valor']})" for s in hits[:3]) + "."
    elif against:
        measured_against = [x for x in against if x["base"] == "medido"] or against
        counter = "O ponto mais forte contra: " + measured_against[0]["texto"] + "."
    else:
        counter = "Nenhum ponto contra foi medido — o que também pode ser falta de dados, não ausência de risco."
    score, why = 0, []
    if a is not None and len(a.anos) >= 4:
        score += 1
        why.append(f"{len(a.anos)} anos de demonstrações")
    if isinstance(fs, dict) and fs.get("medidos", 0) >= 6:
        score += 1
        why.append("F-score medido")
    if an is not None and (an.n_analistas or 0) >= 5:
        score += 1
        why.append(f"{an.n_analistas} analistas com estimativas")
    cal = (fc or {}).get("calibracao") or {}
    if isinstance(cal, dict) and cal.get("n_efetivo") is not None and cal["n_efetivo"] >= 5 and 0.7 <= cal.get("cobertura_10_90", 0) <= 0.9:
        score += 1
        why.append("modelo de preços razoavelmente calibrado neste ativo")
    if fc is not None and fc["fatos_H"]["anos_de_dados"] >= 4.5:  # type: ignore[index]
        score += 1
        why.append("5 anos de cotações")
    level = "boa" if score >= 4 else "média" if score == 3 else "fraca"
    statement = (f"Hipótese em teste: {p.nome} pode render mais que a renda fixa no prazo — "
                 + (f"o modelo de preços estima {pct(fc['probabilidades']['p_superar_rf'], 0)} de chance [E]; " if fc else "")  # type: ignore[index]
                 + f"{len(favor)} ponto(s) a favor e {len(against)} contra (qualidade dos dados: {level}).")
    out = {"afirmacao": statement, "fatos": facts, "estimativas": estimates, "a_favor": favor, "contra": against,
           "precisa_ser_verdade": premises, "catalisadores": catalysts, "invalidacao": invalid, "contra_tese": counter,
           "qualidade_dos_dados": {"nivel": level, "pontos": score, "de": 5, "motivos": why}, "aviso": DISCLAIMER}
    texts = [statement, counter, DISCLAIMER, *premises, *(x["texto"] for x in (*facts, *estimates, *favor, *against)),
             *(x["condicao"] for x in invalid)]
    bad = guarantee_phrases(" ".join(texts))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out
