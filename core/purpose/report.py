"""Rendering of the purpose-engine result (Markdown + JSON-ready dict), in the required order:
1 required vs expected CAGR, 2 table, 3 theses, 4 scenarios, 5 P(goal) with SE, 6 alerts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from utils.serialization import to_jsonable

if TYPE_CHECKING:
    from core.purpose.engine import PurposeResult

DISCLAIMER = ("Ferramenta de pesquisa e simulação. Não é recomendação de investimento, não executa ordens e não "
              "oferece garantia de resultado: todos os números são estimativas [E], históricos [H] ou simulações [S] sujeitas a "
              "erro de modelo e de estimação.")


def _p(x: object, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "indisponível"
    return f"{float(x):.{d}%}"  # type: ignore[arg-type]


def _money(x: float, ccy: str) -> str:
    s = f"{x:,.0f}".replace(",", ".")
    return f"{ccy} {s}"


def _cell(x: object) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _num(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _plan_table(L: list[str], plan: dict[str, object], ccy: str) -> None:
    rows = plan.get("posicoes") or []
    if not rows:
        L.append("Nenhuma compra nesta etapa.")
        return
    key = "peso_executado" if "peso_executado" in rows[0] else "peso_apos_compra"  # type: ignore[index]
    L.append("| ativo | quantidade | preço (último da série diária) | valor | peso-alvo | "
             + ("peso comprado" if key == "peso_executado" else "peso depois da compra") + " | regra de lote |")
    L.append("|---|---|---|---|---|---|---|")
    for r_ in rows:  # type: ignore[union-attr]
        q = float(r_["quantidade"])
        qty = _num(q, 2) if r_["ticker"] == "CAIXA_CDI" or q != int(q) else f"{int(q)}"
        L.append(f"| {_cell(r_['ticker'])} | {qty} | {ccy} {_num(float(r_['preco']))} | {ccy} {_num(float(r_['valor']))} | "
                 f"{float(r_['peso_alvo']):.1%} | {float(r_[key]):.1%} | {_cell(r_['regra_de_lote'])} |")
    L.append(f"\nSobra sem investir: {ccy} {_num(float(plan.get('caixa_nao_investido', 0.0)))}."
             + (" " + " ".join(map(str, plan.get("notas") or [])) if plan.get("notas") else ""))
    for t, why in (plan.get("fora_desta_compra") or {}).items():  # type: ignore[union-attr]
        L.append(f"- {t}: {why}")


def render_markdown(r: PurposeResult) -> str:
    q = r.request
    ccy = q.moeda_objetivo
    L: list[str] = []
    L.append("# Carteira alinhada ao propósito — resultado")
    if r.is_synthetic:
        L.append("\n> **DEMONSTRAÇÃO COM DADOS SINTÉTICOS — NÃO É UMA CARTEIRA FINAL.** Os tickers `SINT_*` não "
                 "existem no mercado; os números ilustram o método.")
    else:
        L.append(f"\n> **Dados reais** — {r.universe['origem']}; período {r.universe['periodo']}; última cotação "
                 f"{r.universe.get('ultima_cotacao', '?')}. Resultado de pesquisa quantitativa, não recomendação.")
    L.append(f"\n_{DISCLAIMER}_\n")
    cap = (_money(q.capital_inicial, ccy) if q.capital_inicial > 0
           else f"{_money(0.0, ccy)} (começa só com os aportes)")
    L.append(f"Entradas: capital inicial {cap}; aporte {_money(q.aporte_periodico, ccy)} "
             f"({q.periodicidade}); objetivo {_money(q.valor_alvo, ccy)} em {q.horizonte_anos:g} anos; perfil "
             f"declarado {q.perfil_risco.value} (final após limites: {r.analysis.profile.final_profile.value}); "
             f"slider {1 - q.peso_alinhamento:.0%} quant / {q.peso_alinhamento:.0%} alinhamento; ativos sob revisão "
             f"{'INCLUÍDOS por opção explícita' if q.incluir_sob_revisao else 'excluídos (padrão)'}; objetivo "
             f"{q.objetivo_tipo or 'não informado'}; queda máxima aceita "
             f"{f'{q.tolerancia_perda:.0%}' if q.tolerancia_perda else 'não informada'}; liquidez em 12 meses "
             f"{f'{q.necessidade_liquidez:.0%}' if q.necessidade_liquidez is not None else 'não informada'}; mercados "
             f"{', '.join(q.mercados) if q.mercados else 'não informados'}; cripto "
             f"{'permitida' if q.permitir_cripto else 'não permitida'}"
             + (f"; excluídos por você: {', '.join(q.ativos_excluidos)}" if q.ativos_excluidos else "")
             + (f"; carteira atual: {len(q.carteira_atual)} posição(ões), {_money(sum(q.carteira_atual.values()), ccy)}"
                if q.carteira_atual else "") + ".")

    # 1
    e = r.expected
    L.append("\n## 1. CAGR necessário vs. CAGR esperado da carteira\n")
    L.append("| CAGR necessário | CAGR esperado da carteira (mediana simulada [S]) | Diferença | "
             "Retorno esperado médio [E] (taxa efetiva a.a., ± 1 EP) |")
    L.append("|---|---|---|---|")
    gap = e.get("gap")
    L.append(f"| {_p(r.required.rate)} | {_p(e['cagr_esperado_mediano_simulado'])} | "
             f"{'indisponível' if gap is None else f'{gap:+.2%}'} | "
             f"{_p(e['retorno_esperado_efetivo_E'])} ± {_p(e['erro_padrao_retorno_carteira'])} |")
    L.append(f"\n{r.required.explanation}. Ambas as taxas da comparação usam a mesma convenção (taxa efetiva anual "
             "aplicada ao mesmo capital e aos mesmos aportes): o CAGR esperado é a taxa que leva esse fluxo ao "
             "patrimônio MEDIANO simulado, já líquido de custos e do arrasto de volatilidade. O retorno esperado "
             "médio [E] é a média estimada pelo otimizador convertida para taxa efetiva; fica acima da mediana "
             "porque a média de um processo multiplicativo supera a sua mediana.")

    # 2
    L.append("\n## 2. Carteira\n")
    if r.strategy_text:
        L.append(r.strategy_text + "\n")
    if r.allocation:
        L.append("**Onde investir e por quê** (alocação do modelo; valor inicial = capital inicial × peso, aporte "
                 "por período = aporte × peso" + ("; sem capital inicial, cada aporte é dividido pelos pesos" if
                                                  q.capital_inicial == 0 else "") + "):\n")
        L.append("| ativo | peso | valor inicial | aporte por período | por quê |")
        L.append("|---|---|---|---|---|")
        for a in r.allocation:
            L.append(f"| {_cell(a['ticker'])} — {_cell(a['nome'])} | {a['peso']:.1%} | "
                     f"{_money(float(a['valor_inicial']), ccy)} | {_money(float(a['aporte_por_periodo']), ccy)} | "
                     f"{_cell(a['por_que'])} |")
        L.append("")
    L.append("| ticker | classe | peso | score_alinhamento | moeda | peso só-quant |")
    L.append("|---|---|---|---|---|---|")
    for t, row in r.table.iterrows():
        if row["peso"] <= 0 and row["peso_quant"] < 0.005:
            continue
        sc = "não calculado" if row["score_alinhamento"] is None or (isinstance(row["score_alinhamento"], float)
                                                                     and np.isnan(row["score_alinhamento"])) \
            else f"{row['score_alinhamento']:.2f}"
        L.append(f"| {_cell(t)} | {_cell(row['classe'])} | {row['peso']:.1%} | {sc} | {row['moeda']} | "
                 f"{row['peso_quant']:.1%} |")
    L.append("\nMotivo do score (uma frase por ativo elegível):\n")
    for t, row in r.table.iterrows():
        L.append(f"- **{_cell(t)}**: {_cell(row['motivo_score'])}")

    # 3
    L.append("\n## 3. Justificativa por ativo\n")
    if r.themes.why:
        L.append(f"Porquê declarado (citação literal): \"{r.themes.why}\"\n")
    alloc = {str(a["ticker"]): a for a in r.allocation}
    for th in r.theses:
        L.append(f"### {th['ticker']} — {th['nome']}\n")
        L.append(f"**(a) Tese financeira.** {th['tese_financeira']}\n")
        L.append(f"**(b) Tese de propósito.** {th['tese_proposito']}\n")
        a = alloc.get(str(th["ticker"]))
        if a is not None:
            L.append(f"- Mercado: {a['mercado']}; classe {a['classe']}; moeda {a['moeda']}.")
            L.append(f"- Por que entrou na lista de candidatos: {_cell(a['motivo_selecao'])}.")
            L.append(f"- Fatos [H]: {_cell(a['fatos_H'])}. Dados: {_cell(a['dados'])}.")
            L.append(f"- Estimativas [E]: {_cell(a['estimativas_E'])}.")
            L.append(f"- Hipóteses: {_cell(a['hipoteses'])}.")
            L.append(f"- Diversificação: {_cell(a['diversificacao'])}.")
            alts = a.get("alternativas") or []
            L.append("- Alternativas consideradas e não escolhidas: " + ("; ".join(
                f"{x['ticker']} (pontuação {float(x['pontuacao']):+.2f}: {_cell(x['motivo'])})" for x in alts)  # type: ignore[union-attr]
                if alts else "nenhuma do mesmo grupo") + "\n")

    # 4
    L.append(f"\n## 4. Cenários no horizonte de {q.horizonte_anos:g} anos (segundo a simulação)\n")
    L.append("| cenário | patrimônio final nominal | real (inflação hipotética) | CAGR equivalente | atinge o objetivo? |")
    L.append("|---|---|---|---|---|")
    for name, row in r.scenarios.iterrows():
        L.append(f"| {name} | {_money(row['patrimonio_final'], ccy)} | {_money(row['patrimonio_final_real'], ccy)} | "
                 f"{_p(row['cagr_equivalente'])} | {'sim' if row['atinge_objetivo'] else 'não'} |")
    L.append(f"\nNo cenário base (mediana), o patrimônio simulado é {_money(r.scenarios.iloc[1]['patrimonio_final'], ccy)}; "
             f"em 10% das trajetórias fica abaixo de {_money(r.scenarios.iloc[2]['patrimonio_final'], ccy)}. "
             f"Inflação hipotética usada para valores reais: {r.goal['inflacao_hipotetica']:.1%} a.a.")

    # 5
    g = r.goal
    L.append("\n## 5. Probabilidade de atingir o objetivo (Monte Carlo)\n")
    L.append(f"P(patrimônio final ≥ {_money(g['valor_alvo'], ccy)}) = **{g['p_objetivo']:.1%}** "
             f"(erro padrão Monte Carlo {g['erro_padrao']:.2%}; IC95% de Wilson {g['ic95'][0]:.1%}–{g['ic95'][1]:.1%}; "
             f"{g['n_trajetorias']} trajetórias; {g['modelo']}).")
    L.append(f"Carteira só-quantitativa nas mesmas trajetórias: {g['p_objetivo_carteira_quant']:.1%} "
             f"(EP {g['erro_padrao_quant']:.2%}). P(terminar abaixo do total aportado "
             f"{_money(g['total_aportado'], ccy)}) = {g['p_abaixo_do_aportado']:.1%}.")
    L.append("\nSensibilidade a modelo e a erro de estimativa:\n")
    L.append("| modelo | P(objetivo) | EP | mediana final | CAGR equivalente mediano |")
    L.append("|---|---|---|---|---|")
    for name, row in r.sensitivity.iterrows():
        L.append(f"| {name} | {row['p_objetivo']:.1%} | {row['erro_padrao']:.2%} | "
                 f"{_money(row['mediana_final'], ccy)} | {_p(row['cagr_equivalente_mediano'])} |")
    L.append("\nO erro padrão acima mede apenas o ruído do Monte Carlo; a incerteza de modelo e de estimação "
             "(tabela) costuma ser bem maior.")

    # 6
    L.append("\n## 6. Alertas de validação\n")
    order = {"FAIL": 0, "WARNING": 1, "INFO": 2}
    for a in sorted(r.alerts, key=lambda x: order.get(x.severity, 3)):
        L.append(f"- **{a.severity}** [{a.category}] {_cell(a.message)}")

    # 7
    ex = r.execution
    L.append("\n## 7. Plano de compra (o que dá para comprar de fato)\n")
    if ex:
        L.append(f"{ex['nota_referencia']}. Preços: {ex['precos']}. Compra mínima por posição: "
                 f"{_money(float(ex['valor_minimo_posicao']), ccy)}; frações de ações dos EUA: "
                 f"{'sim' if ex['fracionario_eua'] else 'não (ação inteira)'}.\n")
        for t, why in (ex.get("fora_por_tamanho_minimo") or {}).items():  # type: ignore[union-attr]
            L.append(f"- fora antes de otimizar — {t}: {why}")
        for t, why in (ex.get("retirados_apos_otimizar") or {}).items():  # type: ignore[union-attr]
            L.append(f"- retirado depois de otimizar — {t}: {why}")
        if ex.get("compra_inicial"):
            L.append(f"\n**Compra inicial com {_money(q.capital_inicial, ccy)}:**\n")
            _plan_table(L, ex["compra_inicial"], ccy)  # type: ignore[arg-type]
        if ex.get("proximo_aporte"):
            L.append(f"\n**Próximo aporte de {_money(q.aporte_periodico, ccy)} ({q.periodicidade}):**\n")
            _plan_table(L, ex["proximo_aporte"], ccy)  # type: ignore[arg-type]
        for v in ex.get("vendas_sugeridas") or []:  # type: ignore[union-attr]
            L.append(f"- acima do alvo — {v['ticker']}: tem {_money(float(v['valor_atual']), ccy)}, alvo "
                     f"{_money(float(v['valor_alvo']), ccy)} ({_cell(v['motivo'])})")
        if ex.get("nota_vendas"):
            L.append(f"\n{ex['nota_vendas']}")

    # 8
    L.append("\n## 8. O que mudou e por quê\n")
    ch = r.changes
    if ch is None:
        L.append("Sem carteira anterior ou atual informada: nada a comparar. Impressão digital desta carteira: "
                 f"`{r.reproducibility.get('impressao_digital_carteira', '')}`.")
    else:
        L.append(f"Comparação com a {ch.reference}: " + "; ".join(ch.notes) + ". Impressão digital "
                 f"`{ch.fingerprint_before}` → `{ch.fingerprint_after}`.")
        if ch.stability:
            L.append(f"\nRegra de estabilidade: {ch.stability.get('explicacao', '')}")
        L.append("\n| ativo | antes | depois | o que aconteceu | por quê |")
        L.append("|---|---|---|---|---|")
        for c in ch.changes:
            L.append(f"| {_cell(c.ticker)} | {c.before:.1%} | {c.after:.1%} | {c.kind} | {_cell(c.reason)} |")

    # 9
    L.append("\n## 9. Diversificação real (pelo risco, não pelo número de ativos)\n")
    co = r.concentration
    if co is not None:
        L.append(f"{co.n_positions} posições; N efetivo pelos pesos {co.effective_n:.1f}; apostas independentes "
                 f"{co.enb:.1f} (razão de diversificação ao quadrado: 1 quando tudo anda junto, N para N posições "
                 f"iguais e independentes). Maior posição: {co.largest[0]} ({co.largest[1]:.1%}).")
        for g in co.groups:
            L.append(f"- andam juntos: {', '.join(g['ativos'])} — {float(g['peso']):.0%} do dinheiro, "  # type: ignore[arg-type]
                     f"{float(g['participacao_no_risco']):.0%} do risco (correlação média {g['correlacao_media']})")  # type: ignore[arg-type]
        for dim in ("moeda", "classe", "mercado", "setor"):
            ex_w, ex_r = co.exposure.get(dim, {}), co.risk_exposure.get(dim, {})
            L.append(f"- por {dim}: " + ", ".join(f"{k} {v:.0%} do dinheiro / {ex_r.get(k, 0.0):.0%} do risco"
                                                 for k, v in sorted(ex_w.items(), key=lambda kv: -kv[1])))
        for w_ in co.warnings:
            L.append(f"- **atenção:** {_cell(w_)}")

    # 10
    L.append("\n## 10. Como os ativos foram escolhidos\n")
    u = r.universe
    L.append(f"Cobertura consultada: {u.get('cobertura', '')}")
    fun = u.get("funil")
    if isinstance(fun, dict):
        L.append(f"\nFunil: {fun.get('listed_b3', 0)} listados na B3 e {fun.get('listed_us', 0)} nos EUA → "
                 f"{fun.get('eligible', 0)} elegíveis → {fun.get('selected', 0)} baixados → "
                 f"{len(u.get('candidatos', []))} candidatos para o otimizador. Critérios: {fun.get('criteria', '')}.")
    if r.screening is not None:
        t = r.screening.table
        L.append("\n| ativo | classe | pontuação | decisão | motivo |")
        L.append("|---|---|---|---|---|")
        for tk in list(t.index)[:60]:
            L.append(f"| {_cell(tk)} | {t.at[tk, 'classe']} | {float(t.at[tk, 'pontuacao']):+.2f} | "
                     f"{t.at[tk, 'decisao']} | {_cell(t.at[tk, 'motivo'])} |")
        if len(t) > 60:
            L.append(f"\n(+{len(t) - 60} ativos medidos com pontuação menor; lista completa no JSON.)")

    # 11
    L.append("\n## 11. Suas regras e as hipóteses usadas\n")
    pol = r.policy
    for rule in pol.get("regras", []) or ["limites padrão do perfil"]:  # type: ignore[union-attr]
        L.append(f"- regra: {_cell(rule)}")
    for h in pol.get("hipoteses", []) or []:  # type: ignore[union-attr]
        L.append(f"- hipótese (dado não informado): {_cell(h)}")

    # 12
    L.append("\n## 12. Fontes dos dados\n")
    for src in r.sources:
        L.append(f"- {src['dado']}: {_cell(src['fonte'])}" + (f" (data {src['data']})" if src.get("data") else "")
                 + (f" — limite: {_cell(src['limite'])}" if src.get("limite") else ""))
    rp = r.reproducibility
    if rp:
        L.append(f"\nReprodutibilidade: carteira `{rp.get('impressao_digital_carteira')}`, parâmetros "
                 f"`{rp.get('parametros')}`, dados `{rp.get('dados')}`, versão {rp.get('versao')}. {rp.get('nota', '')}.")

    # appendix
    L.append(f"\n## Apêndice — processo (etapas 1 a {len(r.steps)}, nesta ordem)\n")
    for s in r.steps:
        L.append(f"{s['etapa']}. **{s['titulo']}**: {_cell(s['resultado'])}")
    L.append("\nNota: o filtro de governança é aplicado como exclusão dura do universo ANTES do otimizador rodar "
             "(equivale a impor peso zero); remover depois e renormalizar quebraria os limites do perfil.")
    L.append("\n### Sleeves por moeda (cada um na própria moeda, com a própria rf) [H]\n")
    L.append("| moeda | ativos | peso na carteira | rf | fonte da rf | retorno médio nativo | vol nativa | Sharpe nativo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in r.sleeves_final:
        L.append(f"| {s.currency} | {', '.join(s.assets)} | {s.weight_in_portfolio:.1%} | {s.rf:.2%} | {s.rf_source} | "
                 f"{_p(s.expected_return_native)} | {_p(s.volatility_native)} | "
                 f"{'indisponível' if s.sharpe_native is None else f'{s.sharpe_native:.2f}'} |")
    L.append("\nA carteira agregada é avaliada na moeda do objetivo (com conversão cambial) e com a rf dessa moeda.")
    L.append("\n### Estatísticas por ativo na moeda nativa (antes da otimização) [H]\n")
    L.append("| ticker | moeda | rf da moeda | retorno médio | volatilidade | Sharpe | observações |")
    L.append("|---|---|---|---|---|---|---|")
    for t, row in r.native_stats.iterrows():
        sh = row["sharpe_nativo"]
        L.append(f"| {_cell(t)} | {row['moeda']} | {row['rf_moeda']:.2%} | {_p(row['retorno_medio_anual_nativo'])} | "
                 f"{_p(row['volatilidade_nativa'])} | {'indisponível' if sh is None or not np.isfinite(sh) else f'{sh:.2f}'} | "
                 f"{int(row['n_obs'])} |")
    th = r.themes
    L.append("\n### Extração de temas (determinística, sem modelo de linguagem)\n")
    if th.empty:
        L.append("motivo_investidor vazio: score de alinhamento não calculado por falta de input.")
    else:
        L.append(f"Confiança: **{th.confidence}**.")
        for t in th.themes:
            L.append(f"- {t.name} ({t.kind}): evidências literais {', '.join(repr(x) for x in t.evidence)}")
        if th.negated:
            L.append(f"- temas negados no texto (ignorados): {', '.join(th.negated)}")
        for n in th.notes:
            L.append(f"- nota: {n}")
    L.append(f"\nUniverso: origem {r.universe['origem']}; período {r.universe['periodo']}; "
             f"{len(r.universe['validados'])} validados de {len(r.universe['candidatos'])} candidatos.")
    return "\n".join(L) + "\n"


def to_dict(r: PurposeResult) -> dict[str, object]:
    return to_jsonable({
        "aviso": DISCLAIMER, "dados_sinteticos": r.is_synthetic, "carteira_final": r.is_final_portfolio,
        "entrada": r.request.model_dump(mode="json"),
        "cagr": {"necessario": r.required.rate, "status_necessario": r.required.status,
                 "explicacao": r.required.explanation, **r.expected},
        "carteira": r.table.reset_index().to_dict(orient="records"),
        "teses": r.theses, "cenarios": r.scenarios.reset_index().to_dict(orient="records"),
        "onde_investir": r.allocation, "estrategia_texto": r.strategy_text,
        "objetivo": r.goal, "sensibilidade": r.sensitivity.reset_index().to_dict(orient="records"),
        "alertas": [a.__dict__ for a in r.alerts], "etapas": r.steps, "universo": r.universe,
        "rf_por_moeda": r.rf_by_currency, "fonte_rf": r.rf_source,
        "sleeves": [s.__dict__ for s in r.sleeves_final],
        "temas": {"confianca": r.themes.confidence, "porque": r.themes.why,
                  "temas": [t.__dict__ for t in r.themes.themes], "negados": r.themes.negated,
                  "notas": r.themes.notes},
        "governanca": {"removidos": r.governance.removed, "incluidos_sob_revisao": r.governance.included_under_review},
        "estrategia_quant": r.analysis.selected_name,
        "selecao_estrategia": {k: r.analysis.selection.get(k) for k in ("rule", "ranking_metric", "scores",
                                                                        "best_by_metric", "within_one_se")},
        "validacao_quant": r.analysis.validation.overall.value if r.analysis.validation else None,
        "regras_pessoais": r.policy,
        "triagem": ([{"ticker": t, **{k: row[k] for k in ("classe", "mercado", "setor", "pontuacao", "decisao", "motivo",
                                                            "volatilidade_H", "queda_maxima_H", "sharpe_encolhido",
                                                            "liquidez_diaria", "grupo")}}
                     for t, row in r.screening.table.iterrows()] if r.screening is not None else []),
        "execucao": r.execution,
        "concentracao": r.concentration.as_dict() if r.concentration is not None else None,
        "mudancas": r.changes.as_dict() if r.changes is not None else None,
        "fontes": r.sources, "reprodutibilidade": r.reproducibility,
        # what the page stores to compare the next run with this one (same inputs + new prices -> stability rule)
        "para_guardar": {"pesos": {t: float(w) for t, w in r.combined.final.items() if float(w) >= 0.005},
                         "parametros": r.reproducibility.get("parametros")},
    })  # type: ignore[return-value]


def write_purpose_outputs(r: PurposeResult, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = r.analysis.run_id
    md = out_dir / f"{rid}_proposito.md"
    js = out_dir / f"{rid}_proposito.json"
    md.write_text(render_markdown(r), encoding="utf-8")
    js.write_text(json.dumps(to_dict(r), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": str(md), "json": str(js)}
