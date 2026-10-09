# Arquitetura

## Auditoria inicial (registro)

* Repositório `BernarGames2/IA`: **vazio** (sem commits) no início do trabalho — nada a preservar; a estrutura
  sugerida na especificação foi criada.
* Ambiente: Linux, Python 3.13.16; dependências instaladas via pip (versões em `requirements-lock.txt`).
* Rede: o proxy do ambiente bloqueia `query1.finance.yahoo.com`/`fc.yahoo.com` (HTTP 403 no `CONNECT`). Logo, o
  conector yfinance só pôde ser testado com provedores simulados; o modo offline é o caminho validado.
* Conflitos entre requisitos e decisões:
  * "Usar yfinance" × rede bloqueada → arquitetura de provedores + modo offline sintético rotulado; nenhum
    fallback silencioso de dados reais para sintéticos.
  * "Rf nunca 0 silenciosamente" × modo offline sem fonte de juros → fallback de demonstração por moeda
    (BRL 10%, USD 4%, EUR 2,5%) marcado em `risk_free_rate_source="demo_fallback"`, nos avisos e no validador.
  * "GARCH/EVT/HRP só se justificados" → implementados sem dependências extras (scipy), sempre comparados a
    baselines (EWMA, empírico, pesos iguais) e sujeitos a recusa por amostra insuficiente.
  * "IA analítica" × ausência de credenciais de LLM → analista determinístico baseado em ferramentas; nenhuma
    chamada externa.

## Fluxo de uma execução (`core/portfolio_engine.run_analysis`)

```
perfil (profiler) ─┐
                   ├─> dados (data_loader: provedor -> qualidade -> câmbio -> alinhamento -> proveniência/hash)
config (pydantic) ─┘          │
                              ├─> retornos simples/log ─> estatísticas por ativo [H]
                              ├─> avaliação OOS de estimadores (μ, Σ) ─> seleção 1-EP (Σ "auto")
                              ├─> restrições do perfil ─> teste de viabilidade (LP) / diagnóstico de conflitos
                              ├─> carteiras candidatas (SLSQP multi-start / LP / formulações convexas)
                              ├─> fronteira eficiente (somente pontos viáveis e não dominados)
                              ├─> backtest walk-forward ─> seleção (validação, 1-EP, elegíveis, faixa de vol.)
                              ├─> risco: RC, VaR/ES, distribuição, concentração
                              ├─> Black Swan Lab: EVT, estresse, reverse stress, liquidez, t-cópula
                              ├─> pesquisa: volatilidade (GARCH×EWMA), backtest de VaR, regimes (HMM), fatores
                              ├─> robustez: sensibilidade (estimadores, janelas, μ perturbado), reamostragem
                              ├─> Monte Carlo (modelo principal + comparação de modelos + horizonte do perfil)
                              └─> validação independente (PASS/WARNING/FAIL) ─> reporter (MD/JSON/CSV/PNG)
```

`main.py` só faz parsing de argumentos, monta configuração/perfil, chama o orquestrador, grava saídas e
(opcionalmente) aciona o analista de IA.

## Interfaces principais

* `MarketDataProvider.fetch(symbols, period, interval) -> dict[str, RawSeries]` (Protocol). Implementações:
  `YFinanceProvider`, `SyntheticProvider`; testes usam provedores falsos.
* Estimadores: funções `f(returns: DataFrame, periods) -> Series/CovarianceResult` registradas em dicionários
  (`ESTIMATORS`), com `evaluate_oos` comum.
* Otimizadores: recebem `mu`, `cov`, `PortfolioConstraints`, retornam `OptimizationResult` (pesos, status,
  viabilidade verificada, violações, avisos, diagnósticos de multi-start).
* Modelos de simulação: `ReturnModel.sample(paths, steps, rng) -> retornos simples (paths, steps, ativos)`.
* Estratégias de backtest: `f(train_returns, prev_weights) -> weights`; o motor só entrega dados passados.
* Ferramentas do analista: esquema pydantic de entrada (extra proibido), função pura sobre `AnalysisContext`,
  registro de chamadas com ID e digest.

## Convenções numéricas e tolerâncias

| Item | Valor |
|---|---|
| Soma dos pesos | `|Σw − 1| ≤ 1e-6` |
| Limites/grupos | violação tolerada ≤ 1e-6 |
| Simetria da covariância | `max|Σ−Σ'| ≤ 1e-10·max(1,|Σ|)` |
| PSD | menor autovalor ≥ −1e-10·max(1,λ_max); senão reparo documentado (corte de autovalores) |
| Condicionamento | aviso acima de 1e6 |
| Monte Carlo vs teoria | |z| < 4 erros-padrão |
| SLSQP | `ftol=1e-12`, `maxiter=1000`, 3–6 pontos iniciais determinísticos; dispersão L1 entre ótimos reportada |

## Decisões técnicas registradas

1. Retornos alinhados por **preço** (interseção de datas) — preserva retornos de fim de semana de cripto e
   nunca fabrica retorno zero.
2. ES de Acerbi–Tasche em vez de "média dos piores k" — correto com empates/átomos.
3. Ledoit–Wolf com alvo de correlação constante adicionado após a avaliação fora da amostra mostrar que o alvo
   identidade distorce a variância de ativos quase sem risco (GMV realizada 2,79% vs 0,62% amostral na demo).
4. Seleção de estratégia: somente estratégias que satisfazem as restrições e a faixa de volatilidade do perfil;
   regra de 1 erro-padrão a favor da menor complexidade; baseline e DSR reportados; teste final não usado.
5. Filtro GARCH vetorizado com `scipy.signal.lfilter` após profiling (equivalência testada contra o laço).
6. Lotes de 2.000 trajetórias no Monte Carlo (pico ~110 MB para 20.000×252×9; resultados invariantes ao lote).
7. Omega diagonal na otimização robusta (com Ω ∝ Σ o máximo Sharpe robusto coincidiria com o nominal).
8. HRP e 1/N ingênuo são referências: não suportam limites nativamente; violações são reportadas, não cortadas.
