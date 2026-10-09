"""Markdown report builder (sections 1-14 of the specification)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import APP_NAME, APP_VERSION


def pct(x: object, d: int = 2) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/d"
    return "n/d" if not np.isfinite(v) else f"{v * 100:.{d}f}%"


def num(x: object, d: int = 3) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/d"
    return "n/d" if not np.isfinite(v) else f"{v:.{d}f}"


def money(x: object) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/d"
    s = f"{v:,.0f}".replace(",", ".")
    return f"R$ {s}"


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, index: bool = True, max_rows: int = 60) -> str:
    fmt = fmt or {}
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    head = [h.replace("|", "\\|") for h in
            (["" if d.index.name is None else str(d.index.name)] if index else []) + [str(c) for c in cols]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, row in d.iterrows():
        cells = [str(idx)] if index else []
        for c in cols:
            v = row[c]
            f = fmt.get(str(c), "auto")
            if f == "pct":
                cells.append(pct(v))
            elif f == "money":
                cells.append(money(v))
            elif f.startswith("num"):
                cells.append(num(v, int(f[3:] or 3)))
            elif isinstance(v, (float, np.floating)):
                cells.append(num(v, 4))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines)


def build_report(res, charts: dict[str, str], tables: dict[str, str]) -> str:  # noqa: ANN001, C901
    cfg = res.config
    ds = res.dataset
    prof = res.profile
    pin = res.profile_input
    v = res.validation
    S = res.sections
    L: list[str] = []
    a = L.append
    status = v.overall.value if v else "N/D"
    a(f"# Relatório de análise — {APP_NAME}")
    a("")
    if ds.is_synthetic:
        a("> **⚠ DADOS SINTÉTICOS.** Esta execução usou o gerador determinístico offline. Nenhum número abaixo "
          "descreve ativos ou mercados reais; a finalidade é demonstrar e testar a mecânica da plataforma.")
        a("")
    if v and not v.validated:
        a("> **⛔ RESULTADO NÃO VALIDADO.** O motor de validação independente encontrou falha crítica "
          "(ver seção 13). Os números não devem ser usados como resultado validado.")
    else:
        a(f"> Status da validação independente: **{status}** (PASS = sem ressalvas; WARNING = ressalvas não "
          "bloqueantes; FAIL = falha crítica). Veja a seção 13.")
    a("")
    a("> Ferramenta de pesquisa e simulação. Não é recomendação de investimento, não constitui suitability "
      "regulatório e não garante resultados. Não há execução de ordens.")
    a("")
    a("Legenda de natureza dos números: **[H]** histórico observado · **[E]** estimativa estatística · "
      "**[S]** simulação · **[C]** cenário hipotético.")
    a("")

    # 1
    a("## 1. Identificação")
    a(md_table(pd.DataFrame({"valor": {
        "ID da execução": res.run_id, "início (UTC)": res.started_at_utc, "fim (UTC)": res.finished_at_utc,
        "versão": APP_VERSION, "semente": cfg.seed, "modo": "offline (sintético)" if cfg.offline else "online",
        "hash SHA-256 dos preços": ds.content_hash()[:16] + "…"}})))
    a("")
    # 2
    a("## 2. Perfil e restrições")
    lim = prof.research_limits
    a(f"- Investidor (pseudônimo): `{prof.investor_id}`" + (" — **exemplo hipotético**" if prof.is_synthetic_example else ""))
    a(f"- Capital: {money(pin.capital)} · aporte mensal: {money(pin.monthly_contribution)} · horizonte: "
      f"{pin.horizon_years:g} anos · moeda-base: {pin.base_currency}")
    a(f"- Pontuação de tolerância {pin.risk_tolerance_score:g}/100 sugere **{prof.score_band_suggestion.value}**; "
      f"perfil final de pesquisa: **{prof.final_profile.value}**.")
    for c in prof.caps_applied:
        a(f"  - limite aplicado: {c}")
    for c in prof.conflicts:
        a(f"  - ⚠ conflito: {c}")
    for c in prof.missing_information:
        a(f"  - dado ausente: {c}")
    a(f"- Limites de pesquisa (configuráveis, não promessas): peso máximo por ativo {pct(lim.max_weight_per_asset, 0)}; "
      f"máximos por classe {', '.join(f'{k} {pct(x, 0)}' for k, x in lim.max_class_weight.items())}; mínimos por classe "
      f"{', '.join(f'{k} {pct(x, 0)}' for k, x in lim.min_class_weight.items()) or 'nenhum'}; faixa de volatilidade "
      f"de referência {pct(lim.target_volatility_band[0], 0)}–{pct(lim.target_volatility_band[1], 0)}; drawdown de "
      f"referência {pct(lim.max_drawdown_reference, 0)}.")
    a(f"- Venda a descoberto: {'permitida' if cfg.allow_short else 'desabilitada'}; alavancagem: desabilitada.")
    fe = res.feasibility
    a(f"- Viabilidade das restrições: **{'viável' if fe['feasible'] else 'INVIÁVEL'}**"
      + ("" if fe["feasible"] else f" — conflitos: {'; '.join(fe['conflicts'])}; relaxamentos possíveis (não aplicados): "
         f"{'; '.join(fe['suggestions'])}"))
    a(f"- {prof.disclaimer}")
    a("")
    # 3/4
    a("## 3. Universo de ativos e moeda")
    meta = pd.DataFrame({s: {"classe": m.asset_class, "setor": m.sector, "país": m.country, "moeda": m.currency,
                             "fonte dos metadados": m.metadata_source} for s, m in ds.meta.items()}).T
    a(md_table(meta))
    a("")
    a("## 4. Fontes, período e qualidade dos dados")
    prov = pd.DataFrame({s: {"provedor": p.provider, "origem": p.origin.value, "campo": p.price_field,
                             "ajuste": p.adjusted_for, "início": p.first_observation, "fim": p.last_observation,
                             "n": p.n_observations} for s, p in ds.provenance.items()}).T
    a(md_table(prov))
    a("")
    a(f"- Alinhamento: {ds.alignment}. Retornos nunca são preenchidos artificialmente.")
    a(f"- Qualidade agregada: **{ds.quality.overall.value}**. Símbolos excluídos: "
      f"{ds.quality.excluded_symbols or 'nenhum'}.")
    qf = ds.quality.to_frame()
    if not qf.empty:
        a("")
        a(md_table(qf, index=False))
    a("- Vieses: universo definido hoje e aplicado ao passado (survivorship bias potencial); sem dados "
      "point-in-time; eventos corporativos dependem do ajuste do provedor.")
    a("")
    # 5
    a("## 5. Retornos, volatilidade, correlações e estimadores")
    a("Estatísticas por ativo **[H]** (média aritmética anualizada = 252 × média diária; CAGR geométrico; "
      "volatilidade = desvio-padrão diário × √252):")
    a(md_table(res.asset_stats, {"arith_mean_ann": "pct", "cagr": "pct", "vol_ann": "pct", "max_drawdown": "pct",
                                 "sharpe": "num2", "sortino": "num2", "beta_vs_benchmark": "num2", "skew": "num2",
                                 "excess_kurtosis": "num2"}))
    a("")
    mu_u = S.get("mu_uncertainty")
    a(f"Retornos esperados **[E]** pelo método `{cfg.expected_return_method.value}` com erro-padrão (EP) da média; "
      "|t| < 2 indica que o retorno esperado não é distinguível de zero:")
    if isinstance(mu_u, pd.DataFrame):
        a(md_table(mu_u, {"mu": "pct", "se_annual": "pct", "t_stat": "num2"}))
    a("")
    a(f"Covariância **[E]**: `{res.cov.method}` {res.cov.params}; número de condição "
      f"{res.cov.diagnostics.condition_number:.1f}; menor autovalor {res.cov.diagnostics.min_eigenvalue:.2e}; "
      f"PSD: {res.cov.diagnostics.is_psd}.")
    es = S.get("estimator_selection", {})
    if es:
        a("")
        a("Comparação fora da amostra de estimadores (janela 252, horizonte 63 dias, passos de 63):")
        a("")
        a(md_table(es["cov_oos"], {"gmv_realized_vol": "pct", "gmv_realized_vol_se": "pct", "frobenius_to_realized": "num4"}))
        a(f"\nSeleção (regra 1-EP, mais simples): **{es['cov_selection'].get('selected')}** (melhor bruto: "
          f"{es['cov_selection'].get('best')}).\n")
        a(md_table(es["mu_oos"], {"mse": "num4", "mse_se": "num4", "rank_ic_mean": "num3"}))
        a(f"\nAvaliação fora da amostra sugere (regra 1-EP, informativo): **{es['mu_selection'].get('selected')}**; "
          f"método USADO nesta execução: **{cfg.expected_return_method.value}** (configuração). Médias "
          "amostrais são ruidosas; nenhum método deve ser escolhido por gerar retorno esperado maior.")
    dist = S.get("distribution")
    if dist:
        a("")
        a(f"Distribuição dos retornos diários da carteira selecionada **[H/E]**: assimetria {num(dist['skew'], 2)}, "
          f"excesso de curtose {num(dist['excess_kurtosis'], 2)}, Jarque-Bera p={num(dist['jarque_bera_p'], 4)}. "
          f"Student-t: ν={num(dist['student_t']['df'], 2)}, BIC {num(dist['student_t']['bic'], 1)} vs normal "
          f"{num(dist['normal']['bic'], 1)} (menor é melhor). Estabilidade: ν por metade "
          f"{[round(x, 2) for x in dist['stability_df_halves']]}, vol. por metade "
          f"{[pct(x) for x in dist['stability_vol_halves']]}.")
    vm = S.get("volatility_models")
    if isinstance(vm, dict) and "table" in vm:
        a("")
        a("Modelos de volatilidade — previsões de 1 passo fora da amostra (perda QLIKE e MSE; baseline EWMA λ=0,94):")
        a("")
        a(md_table(vm["table"], {"mse": "num4", "qlike": "num4"}))
        dm = pd.DataFrame(vm["diebold_mariano"]).T
        a("")
        a(md_table(dm, {"dm_stat": "num2", "p_value": "num3", "mean_loss_diff": "num4", "n": "num0"}))
        a(f"\nSeleção: **{vm['selection']['selected']}** — {vm['selection']['rule']}.")
    st_ = S.get("synthetic_truth")
    if isinstance(st_, pd.DataFrame):
        a("")
        a("Verdade do gerador sintético vs estimativas (demonstra o erro de estimação com ~5 anos de dados; o "
          "regime de estresse eleva a volatilidade efetiva acima da 'vol. calma'):")
        a(md_table(st_, {c: "pct" for c in st_.columns}))
    a("")
    if "correlation" in charts:
        a(f"![Correlação]({charts['correlation']})")
    a("")
    # 6/7/8
    a("## 6. Benchmarks e carteiras candidatas")
    rows = {}
    for k, c in res.candidates.items():
        rows[k] = {"sucesso": c.success, "viável": c.feasible, "retorno esp. [E]": c.expected_return,
                   "vol. [E]": c.volatility, "Sharpe [E]": c.sharpe if c.sharpe is not None else np.nan,
                   "observações": "; ".join((c.violations + c.warnings)[:2])}
    a(md_table(pd.DataFrame(rows).T, {"retorno esp. [E]": "pct", "vol. [E]": "pct", "Sharpe [E]": "num2"}))
    a("")
    a("`equal_weight` é a projeção viável (mínimos quadrados) de 1/N; `equal_weight_naive` e `hrp` são "
      "referências sem restrições e não são elegíveis quando violam limites.")
    if "frontier" in charts:
        a(f"\n![Fronteira eficiente]({charts['frontier']})\n")
    a("## 7. Pesos, retorno esperado e volatilidade da carteira selecionada")
    sel = res.selection
    a(f"Carteira selecionada: **{res.selected_name}**. Regra: {sel.get('rule', '')}.")
    if "excluded_reference_strategies" in sel:
        a(f"Estratégias não elegíveis (referências ou violação de limites): {sel['excluded_reference_strategies']}.")
    if sel.get("excluded_by_volatility_band"):
        a(f"Excluídas por volatilidade estimada acima do teto da faixa do perfil ({pct(sel['volatility_cap'], 0)}): "
          f"{sel['excluded_by_volatility_band']}.")
    w = res.selected.weights
    wt = pd.DataFrame({"peso": w, "contribuição de risco (%)": res.risk_contrib["rc_pct"],
                       "classe": [ds.meta[s].asset_class for s in w.index]})
    a(md_table(wt, {"peso": "pct", "contribuição de risco (%)": "pct"}))
    a("")
    a(f"Retorno esperado **[E]** {pct(res.selected.expected_return)}, volatilidade estimada **[E]** "
      f"{pct(res.selected.volatility)}, Sharpe estimado **[E]** {num(res.selected.sharpe, 2)} (rf "
      f"{pct(cfg.risk_free_rate)}{' — FALLBACK de demonstração' if cfg.rf_is_fallback else ''}).")
    if "weights" in charts:
        a(f"\n![Pesos]({charts['weights']})\n")
    a("## 8. Sharpe e métricas de risco")
    hp = S.get("portfolio_hist_performance", {})
    a("Desempenho histórico **[H]** dos pesos selecionados mantidos constantes (rebalanceamento diário implícito; "
      "inclui look-ahead dos pesos — apenas descritivo, ver backtest na seção 13):")
    a(md_table(pd.DataFrame({"valor": hp}), {"valor": "num4"}))
    conc = S.get("concentration", {})
    a("")
    a(f"Concentração: HHI {num(conc.get('hhi'), 3)}, N efetivo {num(conc.get('effective_n'), 2)}, razão de "
      f"diversificação {num(conc.get('diversification_ratio'), 2)}; exposição por classe "
      f"{ {k: round(x, 4) for k, x in conc.get('by_group', {}).items()} }.")
    a("")
    # 9
    a("## 9. Monte Carlo e percentis")
    dr = S.get("mc_drift")
    if dr:
        a(f"Deriva dos modelos: **{dr['mode']}** — {dr['note']}. Deriva contínua da carteira "
          f"{pct(dr['portfolio_drift_continuous'])} a.a. (retorno esperado [E] {pct(dr['portfolio_expected_return_E'])}; "
          f"deriva histórica seria {pct(dr['portfolio_drift_historical'])}).")
    for lab, mc in (("Horizonte configurado", res.mc_main), ("Horizonte do perfil", res.mc_profile)):
        if mc is None:
            continue
        sm = mc.summary
        a(f"### {lab} — {mc.model.get('model')} **[S]**")
        a(f"{sm['n_paths']} trajetórias, {sm['horizon_steps']} passos ({num(sm['horizon_years'], 2)} anos), semente "
          f"{mc.settings['seed']}, rebalanceamento a cada {mc.settings['rebalance_every']} passos, custos de "
          f"rebalanceamento {mc.settings['rebalance_cost_bps']} bps, aporte mensal "
          f"{money(mc.settings['cashflows']['monthly_contribution'])}, inflação **hipotética** "
          f"{pct(sm['inflation_annual_assumed'])} a.a.")
        tp = sm["terminal_wealth_percentiles"]
        a(md_table(pd.DataFrame({"patrimônio final nominal": tp}), {"patrimônio final nominal": "money"}))
        a("")
        a(f"- Média {money(sm['terminal_wealth_mean'])} (erro Monte Carlo ±{money(sm['terminal_wealth_mean_mc_se'])}); "
          f"mediana real (deflacionada) {money(sm['terminal_real_wealth_median'])}; total líquido aportado "
          f"{money(sm['net_invested'])}.")
        a(f"- P(patrimônio final < capital inicial) = {pct(sm['p_below_initial'])} (EP {pct(sm['p_below_initial_mc_se'])}); "
          f"P(patrimônio final < total aportado) = {pct(sm['p_below_net_invested'])}; P(retorno da estratégia "
          f"no horizonte < 0, sem fluxos) = {pct(sm['p_negative_return'])}.")
        if "p_goal" in sm:
            a(f"- Objetivo explícito: {money(sm['goal'])} ({sm.get('goal_definition', '')}). "
              f"P(atingir) = {pct(sm['p_goal'])} (EP {pct(sm['p_goal_mc_se'])}).")
        a(f"- Drawdown máximo da cota: mediana {pct(sm['max_drawdown_unit_median'])}; 5% piores trajetórias "
          f"≤ {pct(sm['max_drawdown_unit_p05'])}. Probabilidade de esgotamento: {pct(sm['p_ruin'])}.")
        a("")
    if res.mc_main is not None and res.mc_main.convergence is not None:
        a("Convergência (subconjuntos aninhados de trajetórias):")
        a(md_table(res.mc_main.convergence, {"mean_terminal": "money", "mc_se_mean": "money", "median_terminal": "money",
                                             "p_negative_return": "pct", "var95_horizon": "pct", "es95_horizon": "pct"}))
    cmp_ = S.get("mc_model_comparison")
    if isinstance(cmp_, pd.DataFrame):
        a("")
        a("Incerteza de MODELO (mesma carteira, 5.000 trajetórias por modelo, horizonte configurado) **[S]**:")
        fm_ = {"median_terminal": "money", "p05_terminal": "money", "p_negative_return": "pct", "mdd_median": "pct"}
        fm_.update({c: "pct" for c in cmp_.columns if c.startswith(("var", "es"))})
        a(md_table(cmp_, fm_))
        a("\nErro Monte Carlo (EP acima) ≠ incerteza de parâmetros (seção 12) ≠ incerteza de modelo (tabela acima).")
    for k in ("mc_fan", "mc_hist"):
        if k in charts:
            a(f"\n![{k}]({charts[k]})\n")
    # 10
    a("## 10. VaR e Expected Shortfall")
    a("Perda L = −R. VaR_α = quantil α de L; ES = média de cauda (Acerbi–Tasche, trata empates). Variável: "
      "retorno diário da carteira selecionada **[H/E]**, horizonte 1 dia útil:")
    vt = res.var_table[["method", "confidence", "horizon", "var", "es", "n_obs"]].copy()
    vt["avisos"] = res.var_table["warnings"].apply(lambda x: "; ".join(x))
    a(md_table(vt, {"var": "pct", "es": "pct", "confidence": "num2"}, index=False))
    vm_ = cfg.var_method.value
    if vm_ == "simulated" and res.mc_main is not None:
        lv = res.mc_main.summary["var_es_1step"]["levels"]
        a(f"\nMétodo principal configurado: **simulado ({res.mc_main.model.get('model')}, 1 passo)** — "
          + "; ".join(f"{k}: VaR {pct(x['var'])}, ES {pct(x['es'])}" for k, x in lv.items()))
    else:
        prim = res.var_table[res.var_table["method"] == vm_]
        if not prim.empty:
            a(f"\nMétodo principal configurado: **{vm_}** — " + "; ".join(
                f"{num(r['confidence'], 2)}: VaR {pct(r['var'])}, ES {pct(r['es'])}" for _, r in prim.iterrows()))
    if res.mc_main is not None:
        sm = res.mc_main.summary
        a("")
        a(f"Simulado **[S]** — {sm['var_es_horizon']['variable']} ({sm['var_es_horizon']['horizon_steps']} passos): "
          + "; ".join(f"{k}: VaR {pct(x['var'])}, ES {pct(x['es'])}" for k, x in sm['var_es_horizon']['levels'].items()))
        a(f"Simulado **[S]** — {sm['var_es_1step']['variable']}: "
          + "; ".join(f"{k}: VaR {pct(x['var'])}, ES {pct(x['es'])}" for k, x in sm['var_es_1step']['levels'].items()))
    evt = S.get("evt", {})
    if evt:
        a("")
        a("EVT/POT (GPD sobre perdas diárias) **[E]**:")
        for al, e in evt.items():
            if e.refused:
                a(f"- {pct(al, 0)}: **recusado** — {e.refusal_reason}")
            else:
                a(f"- {pct(al, 0)}: VaR {pct(e.var, 3)} (IC90% bootstrap {pct(e.ci_var[0], 3) if e.ci_var else 'n/d'}–"
                  f"{pct(e.ci_var[1], 3) if e.ci_var else 'n/d'}), ES {pct(e.es, 3)}; empírico VaR {pct(e.empirical_var, 3)} / "
                  f"ES {pct(e.empirical_es, 3)}; ξ={num(e.fit.xi, 3)}, β={num(e.fit.beta, 5)}, limiar = quantil "
                  f"{pct(e.fit.threshold_quantile, 1)} com {e.fit.n_exceed} excedências; KS p≈{num(e.fit.ks_pvalue_approx, 3)}.")
                sens = pd.DataFrame(e.sensitivity)
                if not sens.empty:
                    a("  - sensibilidade ao limiar: " + "; ".join(
                        f"q={r['threshold_quantile']}: " + (f"VaR {pct(r['var'], 3)}" if "var" in r and pd.notna(r.get("var")) else f"recusado ({r.get('refused', '')[:60]}…)")
                        for _, r in sens.iterrows()))
                for w_ in e.warnings:
                    a(f"  - ⚠ {w_}")
    vb = S.get("var_backtest")
    if isinstance(vb, pd.DataFrame):
        a("")
        a("Backtest de VaR fora da amostra (janela 500, Kupiec / Christoffersen; Z2 de Acerbi–Szekely < 0 indica "
          "ES subestimado):")
        a(md_table(vb, {"alpha": "num2", "expected": "num1", "kupiec_p": "num3", "christoffersen_cc_p": "num3",
                        "independence_p": "num3", "z2_es": "num3", "avg_var": "pct"}))
    a("")
    # 11
    a("## 11. Cenários de estresse e reverse stress **[C]**")
    st = S.get("stress")
    if isinstance(st, pd.DataFrame):
        t = st[["description", "portfolio_return", "loss_money", "liquidity_cost", "recovery_required",
                "worst_contributor", "baseline_return"]]
        a(md_table(t, {"portfolio_return": "pct", "loss_money": "money", "liquidity_cost": "pct",
                       "recovery_required": "pct", "baseline_return": "pct"}))
        a("\nChoques são HIPÓTESES ilustrativas, não previsões. `baseline_return` = carteira de pesos iguais viável. "
          "Recuperação necessária = 1/(1−d) − 1.")
        sh = S.get("stress_shocks")
        if isinstance(sh, pd.DataFrame):
            a("\nChoques por ativo (hipóteses):")
            a(md_table(sh, {c: "pct" for c in sh.columns}))
    if "stress" in charts:
        a(f"\n![Estresse]({charts['stress']})\n")
    hs = S.get("historical_stress")
    if isinstance(hs, pd.DataFrame) and not hs.empty:
        a("Piores janelas históricas da carteira selecionada **[H]**:")
        a(md_table(hs, {"return": "pct", "recovery_required": "pct"}, index=False))
    nc = S.get("named_crises")
    if isinstance(nc, pd.DataFrame):
        a("")
        a("Crises históricas nomeadas:")
        a(md_table(nc, index=False))
    ps = S.get("parametric_stress")
    if isinstance(ps, pd.DataFrame):
        a("")
        a("VaR 99% 1 dia gaussiano sob correlação/volatilidade estressadas **[C]** (correlação ← (1−b)·ρ + b):")
        a(md_table(ps, {"vol_1d": "pct", "var_1d": "pct"}, index=False))
    rs = S.get("reverse_stress")
    if rs:
        g = rs["gaussian"]
        a("")
        a(f"Reverse stress (limite de perda = drawdown de referência {pct(g['loss_limit'], 0)} em "
          f"{rs['horizon_days']} dias):")
        a(f"- Choque gaussiano mais provável que atinge o limite: distância de Mahalanobis {num(g['mahalanobis_distance'], 2)}; "
          f"probabilidade sob normal {g['prob_normal']:.2e}" + (f", sob Student-t (ν={num(g['t_df'], 1)}) {g['prob_student_t']:.2e}"
                                                                if "prob_student_t" in g else "") + ".")
        if g.get("warning"):
            a(f"- ⚠ {g['warning']}")
        cont = pd.DataFrame({"choque": g["shock"], "contribuição": g["contributions"]})
        a(md_table(cont, {"choque": "pct", "contribuição": "pct"}))
        a("")
        a(md_table(rs["scenario_multipliers"], {"scenario_return": "pct", "multiplier_to_breach": "num2"}))
        emp = rs["empirical"]
        a(f"\nPlausibilidade empírica **[H]**: {emp['n_breaches']} de {emp['n_windows_overlapping']} janelas "
          f"(sobrepostas) de {emp['horizon_days']} dias ultrapassaram o limite; pior observado {pct(emp['worst_observed'])}. "
          "Possibilidade matemática ≠ plausibilidade empírica.")
    td = S.get("tail_dependence", {})
    if isinstance(td, dict) and "t_copula" in td:
        tc = td["t_copula"]
        a("")
        a(f"Dependência de cauda — t-cópula **[E]**: ν = {num(tc.nu, 1)}; log-verossimilhança t {num(tc.loglik_t, 1)} vs "
          f"gaussiana {num(tc.loglik_gauss, 1)} (LR = {num(tc.lr_stat, 1)}). λ teórico médio fora da diagonal: "
          f"{num(tc.lambda_theoretical.to_numpy()[~np.eye(len(tc.lambda_theoretical), dtype=bool)].mean(), 3)}.")
        for w_ in tc.warnings:
            a(f"- ⚠ {w_}")
    liq = S.get("liquidity")
    if isinstance(liq, pd.DataFrame):
        a("")
        a("Liquidez (participação máxima de 10% do volume médio diário em valor, 63 dias):")
        a(md_table(liq, {"position_value": "money", "days_to_liquidate": "num2", "spread_bps_assumed": "num1"}))
        if ds.is_synthetic:
            a("Volumes também são SINTÉTICOS.")
    a("")
    # 12
    a("## 12. Contribuições de risco, papel dos ativos e robustez")
    a("MRC_i = (Σw)_i/σ_p; RC_i = w_i·MRC_i; Σ RC_i = σ_p (verificado na seção 13).")
    a(md_table(res.risk_contrib, {"weight": "pct", "mrc": "pct", "rc": "pct", "rc_pct": "pct"}))
    roles = S.get("asset_roles")
    if isinstance(roles, pd.DataFrame):
        a("")
        a("Papel de cada ativo (observações históricas + hipóteses de cenário):")
        a(md_table(roles[["class", "weight", "rc_pct", "corr_with_portfolio", "worst_hypothetical_scenario",
                          "worst_scenario_shock", "data_quality", "role"]],
                   {"weight": "pct", "rc_pct": "pct", "corr_with_portfolio": "num2", "worst_scenario_shock": "pct"}))
    rob = S.get("robustness", {})
    if isinstance(rob, dict) and rob:
        a("")
        a("Robustez — sensibilidade do **máximo Sharpe** (o otimizador mais dependente de μ):")
        sa = rob.get("max_sharpe_sensitivity")
        if sa:
            a(md_table(sa["mu_perturbation_summary"], {c: "pct" for c in ("base", "mean", "std", "min", "max")}))
            a("\nVariação por estimador e janela (distância L1 aos pesos-base; 2 = carteiras disjuntas):")
            t = pd.concat([sa["estimators"], sa["windows"]])[["success", "exp_ret_base_inputs", "vol_base_inputs", "l1_vs_base"]]
            a(md_table(t, {"exp_ret_base_inputs": "pct", "vol_base_inputs": "pct", "l1_vs_base": "num3"}))
        rr = rob.get("max_sharpe_resampling")
        if rr is not None and not rr.summary.empty:
            a(f"\nReamostragem em blocos ({len(rr.weights)} reamostragens válidas, {rr.n_failed} falhas): distância L1 "
              f"média aos pesos-base = {num(rr.stability_l1, 3)}.")
            a(md_table(rr.summary, {c: "pct" for c in rr.summary.columns}))
        ss = rob.get("selected_sensitivity")
        if ss:
            a(f"\nSensibilidade da carteira selecionada ({res.selected_name}) a perturbações de μ:")
            a(md_table(ss["mu_perturbation_summary"], {c: "pct" for c in ("base", "mean", "std", "min", "max")}))
    reg = S.get("regimes", {})
    if isinstance(reg, dict) and "selection_table" in reg:
        a("")
        a(f"Regimes (sinal: {reg['signal']}) — HMM gaussiano, K=1 é o baseline sem regimes **[E]**:")
        a(md_table(reg["selection_table"].drop(columns=[c for c in ("error",) if c in reg["selection_table"]]),
                   {"loglik": "num1", "bic": "num1"}))
        if "hmm2" in reg:
            a(f"\nHMM 2 estados: vol. anualizada por estado {[pct(x) for x in reg['hmm2']['vol_ann']]}, duração esperada "
              f"{[round(x, 1) for x in reg['hmm2']['expected_duration']]} dias. Concordância filtrado (tempo real) vs "
              f"suavizado (retrospectivo): {pct(reg.get('filtered_vs_smoothed_agreement'))} — a classificação "
              "suavizada NÃO estava disponível em tempo real.")
        if "accuracy_vs_truth_filtered" in reg:
            a(f"Contra o regime verdadeiro do gerador sintético: acurácia filtrada {pct(reg['accuracy_vs_truth_filtered'])}, "
              f"suavizada {pct(reg['accuracy_vs_truth_smoothed'])}, regra de volatilidade "
              f"{pct(reg.get('vol_rule_accuracy_vs_truth'))} (participação real de estresse {pct(reg['truth_stress_share'])}).")
    fac = S.get("factors", {})
    if isinstance(fac, dict) and "pca_explained" in fac:
        a("")
        a(f"Fatores: PCA explica {', '.join(pct(x) for x in fac['pca_explained'])} da variância (3 primeiros componentes).")
        if "market_exposures" in fac:
            a(f"{fac['factor_note']}. Beta da carteira ≈ {num(fac['portfolio_beta'], 2)}.")
            a(md_table(fac["market_exposures"], {"beta_MKT": "num2", "t_MKT": "num2", "alpha_annual": "pct",
                                                 "t_alpha": "num2", "r2": "num2"}))
    a("")
    # 13
    a("## 13. Backtesting e validação independente")
    bt = res.backtest
    if bt is not None:
        a(f"Walk-forward: lookback {bt.settings.lookback}, rebalanceamento a cada {bt.settings.rebalance_every} dias, "
          f"custos {bt.settings.cost_bps} bps + slippage {bt.settings.slippage_bps} bps sobre o turnover. "
          f"Validação: {bt.oos_start.date()} → {bt.validation_end.date()}; teste final: após {bt.validation_end.date()} "
          "(não usado na seleção).")
        cols = ["cagr", "volatility_ann", "sharpe", "max_drawdown", "turnover_annual", "cost_drag_annual"]
        fm = {"cagr": "pct", "volatility_ann": "pct", "sharpe": "num2", "max_drawdown": "pct",
              "turnover_annual": "num2", "cost_drag_annual": "pct"}
        a("\nValidação:")
        a(md_table(bt.metrics_validation[cols], fm))
        a("\nTeste final (somente relato):")
        a(md_table(bt.metrics_test[cols], fm))
        for w_ in bt.bias_warnings:
            a(f"- ⚠ {w_}")
        sel = res.selection
        a(f"\nSeleção: **{sel.get('selected')}** ({sel.get('rule')}). Melhor pela métrica "
          f"`{sel.get('ranking_metric')}`: {sel.get('best_by_metric')}; EP {num(sel.get('metric_se'), 4)}; "
          f"a 1 EP: {sel.get('within_one_se')}; dominadas na validação: {sel.get('dominated_in_validation') or 'nenhuma'}.")
        if sel.get("warning"):
            a(f"- ⚠ {sel['warning']}")
        if sel.get("p_holm"):
            ph = pd.DataFrame({"p (Sharpe>0)": sel["p_sharpe_gt_0_validation"], "p Holm": sel["p_holm"],
                               "p BH": sel["p_bh"]})
            a("\nTestes múltiplos (H0: Sharpe excedente <= 0 na validação; p = 1 − PSR):")
            a(md_table(ph, {c: "num3" for c in ph.columns}))
        d = sel.get("dsr_validation", {})
        if isinstance(d, dict) and "dsr" in d:
            a(f"Deflated Sharpe Ratio (N={int(d['n_trials'])} estratégias testadas): {num(d['dsr'], 3)}; "
              f"PSR no teste final: {num(sel.get('psr_test'), 3)}. Pressupostos: retornos estacionários e pouco "
              "autocorrelacionados.")
        cs = S.get("backtest_cost_sensitivity")
        if isinstance(cs, pd.DataFrame):
            a("\nSensibilidade a custos da estratégia selecionada (período fora da amostra completo):")
            a(md_table(cs, {"cost_bps": "num1", "cagr": "pct", "sharpe": "num2", "max_drawdown": "pct"}, index=False))
        if "backtest" in charts:
            a(f"\n![Backtest]({charts['backtest']})\n")
    if v is not None:
        a("")
        a(f"### Validação independente — status geral: **{v.overall.value}**")
        a(md_table(v.to_frame(), index=False))
    a("")
    # 14
    a("## 14. Premissas, alertas, limitações e falhas")
    for w_ in res.warnings:
        a(f"- ⚠ {w_}")
    a("- Retornos esperados históricos são estimativas ruidosas; resultados dependem do período amostral.")
    a("- Volatilidade anualizada por √252 assume retornos i.i.d.; VaR multi-período não é escalado por √h salvo "
      "rotulagem explícita.")
    a("- Liquidez real e impacto de mercado não são inferíveis apenas de preços diários; custos são paramétricos.")
    a("- Cenários de estresse são hipóteses; não há previsão de cisnes negros.")
    a("- Sem dados point-in-time, deslistagens ou mudanças históricas de universo: backtests potencialmente enviesados.")
    a("")
    a("### Tempos de execução (s)")
    a(md_table(pd.DataFrame({"segundos": res.timings}), {"segundos": "num2"}))
    a("")
    a("### Arquivos gerados")
    for k, p in {**{f"gráfico {k}": v for k, v in charts.items()}, **{f"tabela {k}": v for k, v in tables.items()}}.items():
        a(f"- `{k}`: `{p}`")
    return "\n".join(L) + "\n"
