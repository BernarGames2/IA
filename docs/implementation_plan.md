# Plano incremental por ondas (e estado real)

Cada onda terminou com o sistema executável e testes passando; commits no branch `claude/serene-wright-3bkyx5`.

| Onda | Escopo | Estado | Commit |
|---|---|---|---|
| 1 — Base confiável | estrutura, configuração, modelos tipados, perfil, ingestão/qualidade/proveniência, métricas, Markowitz, Monte Carlo GBM, relatórios, modo offline, testes | concluída | `f39ed1b`, `c989737`, `ded1093` |
| 2 — Robustez | shrinkage (LW, LW correlação constante, OAS, James-Stein), custos, sensibilidade, otimização robusta, reamostragem, bootstrap, Student-t, cenários de estresse | concluída | `c989737`, `ded1093` |
| 3 — Pesquisa avançada | fatores (PCA + mercado), GARCH/GJR comparados a EWMA, regimes (regra, Markov, HMM), EVT/POT, t-cópula, reverse stress | concluída (fatores parcial) | `c989737` |
| 4 — Validação institucional | walk-forward, benchmarks, PSR/DSR, Holm/BH, registro de estratégias, backtest de VaR, seleção de modelos 1-EP, validador independente, testes de viés/look-ahead | concluída | `c989737`, `ded1093` |
| 5 — IA analítica | ferramentas com esquemas/limites/logs, analista determinístico, verificação de números, segurança contra injeção | concluída (sem LLM, sem busca web) | `b1a8df9` |
| 6 — Escalabilidade | cache com expiração/hash, lotes no Monte Carlo, profiling, otimização medida, reprodutibilidade, logs estruturados | concluída | `7ccad34` |

Próximos passos priorizados: ver [`assumptions_and_limitations.md`](assumptions_and_limitations.md#itens-pendentes--próximos-passos).
