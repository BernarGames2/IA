# Matriz de rastreabilidade de requisitos

Legenda de status — **OK**: implementado e coberto por teste automatizado; **PARCIAL**: implementado com
limitação documentada ou cobertura apenas indireta (via teste ponta a ponta); **NÃO VALIDADO**: implementado e
testado com mocks, mas sem validação com dados reais; **PENDENTE**: não implementado.
Nenhum item é marcado OK apenas porque um arquivo existe: cada OK aponta para o(s) teste(s) que o exercitam.
Testes executados e resultados: [`test_results.md`](test_results.md).

| ID | Requisito (seção da especificação) | Módulo(s) | Teste(s) | Status | Critério de aceitação |
|---|---|---|---|---|---|
| R0.1 | Auditoria e plano (§1) | `docs/architecture.md`, `docs/implementation_plan.md` | — | OK | auditoria e ondas documentadas |
| R0.2 | Matriz de rastreabilidade (§1) | este arquivo | — | OK | cada requisito → módulo/teste/status |
| R3.1 | Estrutura modular, type hints, exceções específicas (§3) | todo o projeto | toda a suíte | OK | módulos com responsabilidade única; sem `except Exception: pass` |
| R3.2 | Dependências reproduzíveis (§3) | `requirements.txt`, `requirements-lock.txt` | — | OK | versões testadas fixadas no lock |
| R4.1 | Configuração validada e centralizada (§4) | `config.py` | `test_profile_config::test_config_validation_and_rf_fallback_flagged` | OK | entradas inválidas rejeitadas |
| R4.2 | Rf nunca 0 silencioso; fallback marcado (§4) | `config.py`, validador | idem; `test_validator` | OK | `risk_free_rate_source=demo_fallback` + aviso + WARNING |
| R4.3 | CLI real (`--profile/--period/--simulations/--offline` etc.) (§4) | `main.py` | `test_e2e_offline` (3 testes + 1 lento) | OK | códigos de saída 0/2/3; opções documentadas são funcionais |
| R4.4 | Offline ponta a ponta sem internet/credenciais (§4) | `core/market_data/synthetic.py`, `main.py` | `test_e2e_offline::test_exit_code_and_files`, `test_full_offline_run_with_heavy_analyses` | OK | relatório, CSV, JSON, PNG válidos |
| R5.1 | Modelos tipados de perfil, sem dados pessoais (§5) | `models/investor.py` | `test_pseudonym_enforced` | OK | pseudônimo; campos extras proibidos |
| R5.2 | Conservador/Moderado/Arrojado; pontuação só sugere faixa (§5) | `core/profiler.py` | `test_score_bands`, `test_score_alone_does_not_decide`, `test_independent_caps` | OK | perfil final = mais restritivo entre faixa e limites |
| R5.3 | Conflitos sinalizados; dados ausentes explícitos (§5) | `core/profiler.py` | `test_missing_information_explicit_and_declared_drawdown_used` | OK | conflitos/ausências listados |
| R5.4 | Inviabilidade → solução viável mais próxima + violações + relaxamentos (§5/§9) | `core/optimization/constraints.py` | `test_infeasible_max_weight_detected_with_nearest_and_relaxation`, `test_infeasible_group_minimum_detected` | OK | LP elástico; preferências não alteradas |
| R5.5 | Exemplo sintético R$100k/5 anos/R$1k (§5) | `demo_profile` | `test_demo_profiles_are_marked_synthetic` | OK | marcado `is_synthetic_example` |
| R6.1 | yfinance individual e em lote, normalização de formatos (§6) | `yfinance_provider.py` | `test_normalize_download_layouts`, `test_invalid_ticker_vs_valid_in_batch` | NÃO VALIDADO | testado com mocks; Yahoo bloqueado no ambiente |
| R6.2 | Cache com expiração, rate limit, retries/backoff, timeout (§6) | `cache.py`, `yfinance_provider.py` | `test_cache_expiry_and_tamper_detection`, `test_rate_limiter_waits`, `test_retries_with_backoff_then_failure` | OK (mock) | expiração, detecção de adulteração, backoff 1s/2s |
| R6.3 | Duplicatas, preços inválidos, congelados, suspeitos, faltantes sem fabricar retorno (§6) | `quality.py`, `metrics.simple_returns` | `test_duplicates_invalid_prices_frozen_and_jumps_flagged`, `test_missing_price_yields_missing_return_not_zero` | OK | problemas reportados; nada interpolado |
| R6.4 | Histórico mínimo e calendários distintos (§6) | `quality.py` | `test_insufficient_history_and_missing_fraction_fail`, `test_different_calendars_align_on_prices_without_ffill` | OK | exclusão documentada; alinhamento por preço |
| R6.5 | Falha de download ≠ fechamento; nunca trocar por sintético (§6) | `data_loader.py` | `test_download_failure_not_replaced_by_synthetic`, `test_partial_failure_excluded_and_documented` | OK | `DataUnavailableError`; exclusões registradas |
| R6.6 | Proveniência, moeda, unidade, ajuste, hash do conjunto (§6) | `models/market_data.py`, `data_loader.py` | `test_synthetic_deterministic_and_labelled`, `test_fx_conversion_and_missing_fx` | OK | SHA-256 no manifesto; origem por série |
| R6.7 | Conversão cambial (§6) | `data_loader.convert_currency` | `test_fx_conversion_and_missing_fx`, `test_convert_currency_staleness_limit` | OK (mock) | câmbio ≤3 dias; sem câmbio → exclusão |
| R6.8 | Survivorship/look-ahead/point-in-time documentados (§6) | relatório §4/§13, validador | `test_walk_forward_split_and_selection_prefers_simple_when_equal` | OK | aviso em todo backtest sem point-in-time |
| R7.1 | Retornos simples/log; aritmético vs geométrico; vol anualizada (§7) | `core/metrics.py` | `test_simple_and_log_returns_manual`, `test_annualization_conventions` | OK | valores manuais, tolerância 1e-12 |
| R7.2 | Sharpe, Sortino, beta, drawdown, turnover, custos, concentração (§7) | `core/metrics.py` | `test_sharpe_manual_and_undefined`, `test_sortino_manual`, `test_beta_known_value`, `test_drawdown_manual`, `test_turnover_costs_concentration` | OK | fórmulas documentadas |
| R7.3 | Métricas indefinidas com mensagem explícita (§7) | `UndefinedMetricError`, `performance_summary` | `test_performance_summary_reports_undefined` | OK | `None` + motivo |
| R7.4 | VaR/ES histórico, paramétrico, simulado; ES com empates (§7) | `core/extreme_risk/var_es.py` | `test_empirical_var_es_hand_computed`, `test_es_handles_ties_and_atoms`, `test_normal_closed_form`, `test_student_t_matches_numerical_integration` | OK | ES ≥ VaR; método/nível/horizonte |
| R7.5 | Sem √T automático (§7) | `sqrt_time_scaled` | `test_sqrt_time_only_for_normal` | OK | só normal, rotulado |
| R8.1 | Retornos esperados comparados fora da amostra (§8.1) | `core/estimators/expected_returns.py` | `test_expected_return_estimators` | OK | MSE/IC OOS; shrinkage em direção ao alvo |
| R8.2 | Covariâncias (amostral, LW, OAS, EWMA, fatorial) + diagnósticos + reparo documentado (§8.2) | `core/covariance/estimators.py`, `utils/numerical.py` | `test_all_estimators_symmetric_psd`, `test_ledoit_wolf_matches_sklearn`, `test_lw_constant_corr_matches_loop_formula`, `test_nearest_psd_repairs_and_documents`, `test_oos_evaluation_runs` | OK | PSD; reparo com variação Frobenius |
| R8.3 | Volatilidade: histórica, EWMA, GARCH, GJR; comparação OOS (§8.3) | `core/volatility/models.py` | `test_garch_recovers_parameters`, `test_garch_insufficient_data_and_oos_no_lookahead`, `test_diebold_mariano_sign`, `test_vectorised_garch_filter_matches_reference_loop` | OK | QLIKE/MSE + DM; EGARCH pendente |
| R8.4 | Distribuições: normal, t, empírica, bootstrap (§8.4) | `var_es.py`, `portfolio_engine` (seção distribuição) | `test_fit_student_t_recovers_df`; e2e | PARCIAL | assimétricas não implementadas |
| R8.5 | Fatores (§8.5) | `core/factors/models.py` | `test_factor_regression_recovers_beta` | PARCIAL | PCA + mercado; demais fatores pendentes |
| R8.6 | Regimes: regra, Markov, HMM; filtrado vs suavizado; baseline K=1 (§8.6) | `core/regimes/models.py` | `test_hmm_recovers_two_regimes`, `test_hmm_filtered_is_real_time`, `test_vol_rule_uses_past_only_and_markov_summary` | OK | BIC vs K=1; acurácia vs verdade sintética |
| R9.1 | Markowitz: GMV, máx. Sharpe, retorno-alvo, vol-alvo (§9) | `core/optimization/optimizers.py` | `test_gmv_two_asset_closed_form`, `test_min_variance_long_only_valid_and_optimal`, `test_max_sharpe_matches_convex_reformulation`, `test_target_return_and_volatility` | OK | solução verificada; ótimo vs referência |
| R9.2 | Fronteira só com pontos válidos; exporta retorno/vol/Sharpe/pesos (§9) | `efficient_frontier`, reporter | `test_frontier_only_valid_monotone_points` | OK | CSV + PNG com GMV/máx. Sharpe/pesos iguais |
| R9.3 | Pesos iguais, ERC, máx. diversificação, HRP, mín. ES (§9) | `optimizers.py` | `test_risk_parity_equal_contributions`, `test_max_diversification_equals_correlation_gmv_rescaled`, `test_hrp_two_assets_inverse_variance`, `test_min_cvar_lp_equals_empirical_es` | OK | propriedades analíticas |
| R9.4 | Restrições: soma, limites, grupos (classe), turnover, sem short/alavancagem (§9) | `constraints.py`, `optimizers.solve` | `test_bounds_and_groups_respected`, `test_turnover_constraint_respected` | PARCIAL | limites por fator/liquidez não implementados como restrição |
| R9.5 | Status do solver, multi-start, finitude, sensibilidade a pontos iniciais (§9) | `optimizers.solve` | `test_convergence_failure_is_reported`, `test_singular_covariance_still_feasible`, `test_non_psd_covariance_rejected` | OK | sem pesos fabricados em falha |
| R9.6 | Otimização robusta: shrinkage, penalizações, custos, cenários, conjunto de incerteza, sensibilidade (§9.1) | `core/optimization/robust.py` | `test_robust_mv_more_conservative_than_nominal`, `test_turnover_penalty_reduces_trading`; e2e (sensibilidade/reamostragem) | OK | dispersão de pesos reportada; "robusto" nunca só por limites |
| R10.1 | Monte Carlo GBM, bootstrap, blocos, t, regimes, vol variável, correlação alterada (§10) | `core/simulation/models.py` | `test_gbm_positive_and_matches_theory`, `test_student_t_unit_variance_scaling`, `test_bootstrap_uses_only_historical_vectors`, `test_block_bootstrap_contiguous_blocks`, `test_regime_model_transitions`, `test_stress_covariance_raises_correlation_keeps_psd` | OK | teoria do GBM dentro de 4 EP |
| R10.2 | Aportes, retiradas, custos, inflação, semente, lotes (§10) | `core/simulation/engine.py` | `test_contributions_fees_and_probabilities_accounting`, `test_withdrawals_can_ruin`, `test_batches_do_not_change_results_distributionally`, `test_reproducible_with_seed_and_shapes` | OK | contabilidade exata em caso determinístico |
| R10.3 | Percentis, probabilidades definidas, drawdowns, VaR/ES 95/99 com variável e horizonte (§10) | `engine.summarize` | `test_var_es_not_from_terminal_wealth_and_labelled` | OK | VaR de 1 passo nunca derivado de W_T |
| R10.4 | Convergência; erro MC vs incerteza de parâmetros vs de modelo (§10) | `convergence_table`, comparação de modelos | e2e; `test_zero_volatility_is_deterministic` | OK | tabela de convergência e de modelos no relatório |
| R11.1 | VaR/ES; t com ν, loc, escala, variância (§11) | `var_es.py` | `test_t_scale_vs_std`, `test_student_t_matches_numerical_integration` | OK | escala ≠ desvio-padrão documentada |
| R11.2 | EVT/POT-GPD com limiar, ajuste, sensibilidade, incerteza, recusa (§11) | `core/extreme_risk/evt.py` | `test_evt_recovers_gpd_tail_with_sufficient_sample`, `test_evt_formula_matches_scipy_quantile`, `test_evt_refuses_small_sample`, `test_evt_level_below_threshold_refused` | OK | recusa < 50 excedências |
| R11.3 | Estresse histórico e hipotético, compostos, juros/câmbio/ações/commodities, liquidez (§11) | `core/extreme_risk/stress.py` | `test_scenario_deterministic_contributions_and_recovery`, `test_default_scenarios_rules`, `test_historical_windows_and_breach_frequency`, `test_named_crises_require_real_covering_data`, `test_parametric_stress_var_monotone` | OK | choques rotulados como hipóteses |
| R11.4 | Dependência de cauda / t-cópula (§11) | `tail_dependence.py` | `test_t_copula_detects_tail_dependence_and_refuses_small` | OK | ν baixo detectado; recusa < 250 obs. |
| R11.5 | Reverse stress: combinações, contribuições, possibilidade vs plausibilidade (§11) | `reverse_stress.py` | `test_reverse_stress_shock_hits_limit_exactly`, `test_scenario_multipliers_and_possibility` | OK | choque atinge exatamente o limite |
| R11.6 | Recuperação necessária com domínio validado (§11) | `metrics.recovery_required` | `test_recovery_required_domain` | OK | erro para d ≥ 1 ou d < 0 |
| R12.1 | MRC/RC e identidade ΣRC = σ_p (§12) | `metrics.risk_contributions`, validador | `test_risk_contribution_identity`, `test_detects_wrong_reported_volatility_and_rc` | OK | recalculado independentemente |
| R12.2 | Análise por ativo e papel (§12) | `core/recommendation/asset_analysis.py` | e2e (seção 12 do relatório) | PARCIAL | sem teste unitário dedicado; sem score multicritério |
| R13.1 | Walk-forward, rebalanceamento, custos/slippage, turnover, benchmarks (§13) | `core/backtesting/walk_forward.py` | `test_costs_and_turnover_accounting`, `test_weights_drift_between_rebalances` | OK | custos cobrados sobre turnover |
| R13.2 | Somente informação disponível na data (§13) | `run_strategy`, `lookahead_invariance_check` | `test_strategy_sees_only_past_data`, `test_lookahead_strategy_detected_by_invariance_check` | OK | estratégia que espia o futuro é detectada |
| R13.3 | Treino/validação/teste final; seleção sem usar o teste (§13) | `walk_forward`, `model_selection` | `test_walk_forward_split_and_selection_prefers_simple_when_equal` | OK | teste final só relatado |
| R13.4 | PSR, DSR, correção múltipla, registro de estratégias (§13) | `core/backtesting/overfitting.py` | `test_psr_matches_formula_and_dsr_penalises_trials`, `test_multiple_testing_corrections`, `test_registry_logs_every_trial` | OK | DSR < PSR com muitas tentativas |
| R13.5 | Purged CV/embargo (§13) | — | — | PENDENTE (justificado) | não aplicável: sem rótulos sobrepostos |
| R14.1 | Model selection por OOS, estabilidade, complexidade, baseline (§14) | `model_selection.py` | `test_select_simplest_one_se_rule`, `test_walk_forward_split_and_selection_prefers_simple_when_equal` | OK | regra 1-EP |
| R14.2 | Backtest de VaR/ES (§14) | `var_backtest.py` | `test_var_coverage_tests` | OK | Kupiec, Christoffersen, Z2 |
| R14.3 | Validador independente PASS/WARNING/FAIL com bloqueio (§14) | `independent_validator.py`, `main.py` (código 3) | `test_validator` (7 testes) | OK | FAIL crítico → "NÃO VALIDADO" |
| R15.1 | IA com ferramentas validadas, esquemas, limites, logs (§15) | `core/ai_analyst/tools.py` | `test_tool_schema_validation_and_limits` | OK | entradas inválidas rejeitadas; orçamento de chamadas |
| R15.2 | Números só de ferramentas executadas; rastreáveis (§15) | `analyst.verify_answer` | `test_every_number_traceable_and_tampering_detected`, `test_claim_values_equal_independent_computation` | OK | número adulterado é detectado |
| R15.3 | "Não é possível concluir"; sem previsão de preço/causalidade de notícia (§15) | `analyst.py` | `test_order_and_price_forecast_refused`, `test_unknown_question_cannot_conclude` | OK | recusas explícitas |
| R15.4 | Prompt injection, segredos, sem ordens (§15) | `core/ai_analyst/safety.py` | `test_injection_patterns_detected`, `test_external_content_never_changes_tools_or_triggers_calls`, `test_secrets_redacted` | OK | registro de ferramentas imutável |
| R15.5 | Pesquisa web com URLs/timestamps (§15) | `UntrustedContent` (fonte + timestamp) | `test_untrusted_content_rendered_as_quoted_data` | PARCIAL | busca web não implementada; apenas documentos fornecidos |
| R16.1 | Relatório com 14 seções; MD/CSV/JSON/PNG (§16) | `core/reporting/report_md.py`, `core/reporter.py` | `test_exit_code_and_files` | OK | 14 seções; PNG válidos; CSV legíveis |
| R16.2 | Gráficos obrigatórios com títulos/eixos/unidades/fonte/rótulo sintético (§16) | `core/reporting/charts.py` | e2e (7 PNGs) | OK | inspeção visual registrada em test_results |
| R16.3 | Manifesto: config, versões, commit, dados, hash, semente, tempos, avisos (§16) | `reporter.write_outputs` | `test_exit_code_and_files`, `test_reproducible_numbers` | OK | mesma semente → mesmos números |
| R16.4 | Logs estruturados (§16) | `utils/logging_config.py` | e2e | PARCIAL | JSON-lines; sem teste dedicado |
| R16.5 | PDF (opcional) | — | — | PENDENTE | — |
| R17.1 | Testes sem internet; tolerâncias explícitas (§17) | `tests/` | suíte completa | OK | 158 testes coletados; rede só no teste opcional |
| R17.2 | Tempo, memória, escalabilidade, reprodutibilidade; profiling antes de otimizar (§17) | `scripts/profile_run.py` | execução registrada em test_results | OK | 95 s → 68 s após otimização medida |
| R19.1 | README com instalação Windows/Linux, CLI, fórmulas, limitações (§19) | `README.md` | — | OK | só opções funcionais documentadas |
| R19.2 | `.env.example` sem segredos (§19) | `.env.example` | — | OK | — |
