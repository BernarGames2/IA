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


def render_markdown(r: PurposeResult) -> str:
    q = r.request
    ccy = q.moeda_objetivo
    L: list[str] = []
    L.append("# Carteira alinhada ao propósito — resultado")
    if r.is_synthetic:
        L.append("\n> **DEMONSTRAÇÃO COM DADOS SINTÉTICOS — NÃO É UMA CARTEIRA FINAL.** Os tickers `SINT_*` não "
                 "existem no mercado; os números ilustram o método.")
    L.append(f"\n_{DISCLAIMER}_\n")
    L.append(f"Entradas: capital {_money(q.capital_inicial, ccy)}; aporte {_money(q.aporte_periodico, ccy)} "
             f"({q.periodicidade}); objetivo {_money(q.valor_alvo, ccy)} em {q.horizonte_anos:g} anos; perfil "
             f"declarado {q.perfil_risco.value} (final após limites: {r.analysis.profile.final_profile.value}); "
             f"slider {1 - q.peso_alinhamento:.0%} quant / {q.peso_alinhamento:.0%} alinhamento; ativos sob revisão "
             f"{'INCLUÍDOS por opção explícita' if q.incluir_sob_revisao else 'excluídos (padrão)'}.")

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
    for th in r.theses:
        L.append(f"### {th['ticker']} — {th['nome']}\n")
        L.append(f"**(a) Tese financeira.** {th['tese_financeira']}\n")
        L.append(f"**(b) Tese de propósito.** {th['tese_proposito']}\n")

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

    # appendix
    L.append("\n## Apêndice — processo (etapas 1 a 9, nesta ordem)\n")
    for s in r.steps:
        L.append(f"{s['etapa']}. **{s['titulo']}**: {_cell(s['resultado'])}")
    L.append("\nNota: a etapa 5 (governança) é aplicada como exclusão dura do universo ANTES do otimizador rodar "
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
        "objetivo": r.goal, "sensibilidade": r.sensitivity.reset_index().to_dict(orient="records"),
        "alertas": [a.__dict__ for a in r.alerts], "etapas": r.steps, "universo": r.universe,
        "rf_por_moeda": r.rf_by_currency, "fonte_rf": r.rf_source,
        "sleeves": [s.__dict__ for s in r.sleeves_final],
        "temas": {"confianca": r.themes.confidence, "porque": r.themes.why,
                  "temas": [t.__dict__ for t in r.themes.themes], "negados": r.themes.negated,
                  "notas": r.themes.notes},
        "governanca": {"removidos": r.governance.removed, "incluidos_sob_revisao": r.governance.included_under_review},
        "estrategia_quant": r.analysis.selected_name,
        "validacao_quant": r.analysis.validation.overall.value if r.analysis.validation else None,
    })  # type: ignore[return-value]


def write_purpose_outputs(r: PurposeResult, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = r.analysis.run_id
    md = out_dir / f"{rid}_proposito.md"
    js = out_dir / f"{rid}_proposito.json"
    md.write_text(render_markdown(r), encoding="utf-8")
    js.write_text(json.dumps(to_dict(r), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": str(md), "json": str(js)}
