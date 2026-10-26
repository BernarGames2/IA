# Resultados de testes e execuções registrados

Ambiente: Linux x86_64, Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1,
matplotlib 3.11.2, pydantic 2.13.5 (ver `requirements-lock.txt`). Data: 2026-10-21.

## 1. Suíte automatizada

Comando: `python -m pytest -o addopts="" -q -rs`

```
526 passed, 1 skipped (527 testes)
SKIPPED [1] tests/test_network_optional.py:14: online test disabled (QPI_NETWORK_TESTS!=1)
```

(Registros anteriores: 518 passed, 1 skipped antes da página da competição — seção 23; 514 passed, 1 skipped antes do acompanhamento sob demanda — seção 22; 509 passed, 1 skipped antes dos laboratórios com dados reais — seção 21; 493 passed, 1 skipped antes de fundamentos e cenários — seções 19 e 20; 484 passed, 1 skipped antes das previsões — seção 18; 476 passed, 1 skipped antes de "Minha carteira" e do risco — seção 17; 464 passed, 1 skipped antes da correção de valores pequenos — seção 15; 452 passed, 1 skipped
antes da "Ação do dia" — seção 14; 414 passed, 1 skipped antes dos
logos — seção 13; 361 passed, 1 skipped antes das correções
da revisão adversarial — seção 12; 328 passed, 1 skipped antes do motor dinâmico.)

(Em uma execução anterior, 2 testes de `tests/test_webapp.py` falharam porque ainda assumiam o modo sintético
como padrão do site; foram ajustados ao novo padrão — dados reais — e a suíte completa voltou a passar.)

* 0 falhas. O único teste pulado é o teste **online opcional**, que não pôde ser executado porque a política de
  rede do ambiente bloqueia o Yahoo Finance (o proxy responde HTTP 403 ao `CONNECT`).
* Cobertura de linhas (`--cov=core --cov=models --cov=utils --cov=config`): **93%** (5.153 instruções, 360 não
  cobertas). Módulos-fachada (`core/markowitz.py`, `core/monte_carlo.py`, `core/black_swan.py`) só reexportam
  funções testadas diretamente e aparecem com 0%.
* Testes mais lentos: execução ponta a ponta completa (39,9 s), reprodutibilidade (14,8 s).

Falhas encontradas pelos testes durante o desenvolvimento e corrigidas (não mascaradas):

| Teste | Problema real | Correção |
|---|---|---|
| `test_contributions_fees_and_probabilities_accounting` | taxa anual de 1% deixava 0,99010 em vez de 0,99 do patrimônio (composição errada) | fator por passo `(1−f)^{1/252}` |
| comparação OOS de covariância (relatório) | Ledoit-Wolf com alvo identidade inflava a variância do ativo quase sem risco (GMV realizada 2,79% vs 0,62%) | adicionado LW correlação constante + seleção `auto` |
| execução ponta a ponta | benchmark de ativo único (viola o perfil) era selecionável; EVT 95% usava limiar = nível | seleção só entre elegíveis; limiar abaixo do nível |
| `test_retries_with_backoff_then_failure`, `test_convergence_failure_is_reported`, `test_robust_mv_more_conservative_than_nominal` | expectativas do teste estavam erradas (comportamento do código era o correto) | testes corrigidos com a propriedade matemática certa |
| revisão manual do relatório "Vale a pena?" (PGMN3 com dublês) | a manchete dizia "evidência de que o ativo supera o CDI: forte" com t = −2,53, ou seja, quando ele rendeu MENOS | texto e critério conforme o sinal de t |
| tabela de indicadores no celular (Playwright, 390 px) | a coluna de valores não quebrava linha e ficava fora da tela | coluna de valores com quebra de linha; rótulos mais curtos |
| `test_engine_complete_universe_reports_the_funnel` (em 26/10/2026) | o dublê da B3 só servia o arquivo de 16/10/2026, e o motor procura a listagem de hoje voltando até 8 dias: com o calendário andando, a B3 "sumia" (0 em vez de 7 ativos) | o dublê desse teste serve a mesma listagem para o dia pedido; nada mudou no código do motor |
| tabela de ordens da competição no celular (Playwright, 390 px) | com 7 colunas, a do ativo encolhia a uma letra por linha | no celular a tabela mostra ativo, quantidade e peso (o CSV tem todas); a conferência de regras virou lista |

## 2. Execução ponta a ponta offline

Comando: `python main.py --offline` (perfil moderado de demonstração, 20.000 trajetórias, semente 42).

* Código de saída 0; duração 54,2 s (tempo de parede) — antes da otimização medida eram ~81–95 s.
* Dados: 9 séries **SINTÉTICAS** (`SINT_*`), 1.261 datas (2021-03-03 a 2025-12-31), SHA-256 dos preços
  `382578cbd3737914…`; qualidade OK; nenhuma exclusão.
* Covariância escolhida fora da amostra (`auto`, regra 1-EP): **`sample`** (melhor bruto: `lw_constant_corr`,
  dentro de 1 erro-padrão; o identidade-LW ficou entre os piores). Número de condição 16.555 (aviso se > 1e6).
* Carteira selecionada pelo walk-forward (validação, regra 1-EP, somente estratégias que respeitam limites e
  faixa de volatilidade do perfil): **risk_parity** — retorno esperado [E] 6,43%, volatilidade [E] 8,93%.
* Monte Carlo [S] (GBM, 1 ano, aportes R$ 1.000/mês): percentis do patrimônio final P5 R$ 99.533, mediana
  R$ 114.147, P95 R$ 131.245; P(retorno da estratégia < 0) = 40,8%. Horizonte do perfil (5 anos): P(atingir o
  objetivo de R$ 191.610 — capital e aportes corrigidos por inflação hipotética de 4,5%) = 28,1%.
* Validação independente: **WARNING** — 32 PASS, 10 WARNING, 0 FAIL. Os WARNINGs são: dados sintéticos,
  backtest sem point-in-time, rf de demonstração, HRP e 1/N ingênuo violam limites (referências), alta
  sensibilidade do máximo Sharpe a erros em μ (desvio de peso 13,75% > 10%), ganho da estratégia escolhida
  sobre o baseline não significativo (DM p=0,57) e ressalva do teste KS na EVT.
* Saídas: relatório Markdown (649 linhas, 14 seções), manifesto/resultados JSON, 20 CSVs, 7 PNGs, log JSON-lines e
  registro de estratégias. Cópia em [`docs/examples/offline_demo/`](examples/offline_demo/report.md).

Importante: os números acima vêm de **dados sintéticos** e só demonstram a mecânica — não descrevem nenhum ativo
real. O próprio relatório mostra que, com ~5 anos de dados, os retornos realizados divergem muito dos parâmetros
verdadeiros do gerador (erro de estimação).

## 3. Analista de IA (mesma execução, `--ask` + documento externo com injeção)

* "Qual o risco da carteira e como ela se comporta em estresse?" → 10 afirmações numéricas, todas com ID de
  chamada de ferramenta; verificação de rastreabilidade **OK**.
* "O SINT_CRIPTO vai subir amanhã? Devo comprar?" → recusa de ordens, "não é possível concluir" para previsão
  de preço, apenas métricas históricas rotuladas [H].
* Documento externo com "IGNORE ALL PREVIOUS INSTRUCTIONS… rm -rf… API_KEY… ordem de compra" → padrões
  detectados (`command_execution, order_execution, override_instructions, role_hijack, secret_exfiltration`),
  nenhuma instrução seguida, nenhuma ferramenta adicionada. Saída em
  [`docs/examples/offline_demo/analyst.md`](examples/offline_demo/analyst.md).

## 4. Profiling, memória e escalabilidade (`python scripts/profile_run.py`)

Etapas mais caras da execução completa sob cProfile (antes → depois da otimização):

| Etapa | Antes (s) | Depois (s) | Causa / ação |
|---|---|---|---|
| total (sob cProfile) | 95,1 | 67,8 | — |
| monte_carlo | 39,5 | 18,9 | ajuste CCC-GARCH usava o filtro GARCH em laço Python |
| volatility_models | 15,2 | 1,7 | filtro GARCH vetorizado com `scipy.signal.lfilter` (idêntico a 1e-16, ~120× mais rápido) |
| var_backtest | 10,0 | 11,3 | ajustes Student-t por MLE (não otimizado) |
| regimes | 9,7 | 11,0 | forward-backward do HMM em laço (não otimizado) |

Monte Carlo GBM, 9 ativos, 252 passos (tempo e pico de memória medidos com `tracemalloc`):

| trajetórias | lote | segundos | pico MB | mediana final |
|---|---|---|---|---|
| 5.000 | 5.000 | 0,50 | 269 | 92.548 |
| 20.000 | 5.000 | 2,28 | 386 | 92.286 |
| 20.000 | 1.000 | 2,39 | 109 | 92.286 |
| 20.000 | 20.000 | 4,04 | 651 | 92.286 |
| 50.000 | 5.000 | 6,37 | 445 | 92.199 |

O resultado é idêntico para qualquer tamanho de lote (mesma sequência do gerador); o padrão passou a 2.000
trajetórias por lote. Reprodutibilidade: duas execuções com a mesma semente produziram o mesmo digest
(`42d81e5236a98b1d`). Paralelização não foi adotada: nenhum ganho foi medido que a justificasse.

## 5. Carteira com propósito (`python main.py --offline --proposito …`)

Começar do zero (`tests/test_zero_capital.py`, 5 testes): capital inicial 0 com R$ 2.000 mensais por 10 anos
registra R$ 240.000 aportados; o CAGR necessário reproduz o valor-alvo pela fórmula de valor futuro; o motor
completo roda sem NaN/infinito; os aportes por ativo somam R$ 2.000; o site recusa "nada a investir" com mensagem
simples. Página conferida a 390 px: tabela "Onde investir e por quê" sem colunas cortadas (motivo de cada ativo em
linha própria).

Comando registrado (saída completa em [`docs/examples/purpose_demo/`](examples/purpose_demo/report.md)):

```
python main.py --offline --proposito --objetivo 900000 --capital 100000 --aporte 2000 --periodicidade mensal \
  --horizon-years 10 --profile moderado --rf BRL=0.105 --rf USD=0.043 \
  --motivo "Quero apoiar saúde e energia limpa porque perdi meu pai para o câncer e me preocupo com o planeta. O objetivo é pagar a faculdade dos meus filhos."
```

* Código de saída 0; ~43 s (10.000 trajetórias × 10 anos + backtest walk-forward). Universo **sintético**
  multimoeda (9 ativos BRL + 5 USD com câmbio sintético); `SINT_BANCO4` removido pela governança (exemplo
  sintético "sob revisão").
* Etapa 1: CAGR necessário **14,60% a.a.** (R$ 340.000 aportados no total para chegar a R$ 900.000).
* Etapa 6: temas saúde ("saúde", "câncer"), sustentabilidade ("planeta", "energia limpa"), educação
  ("faculdade") e família/legado ("filhos", citado mas não pontuado); porquê literal: "objetivo é pagar a
  faculdade dos meus filhos"; confiança alta.
* Etapa 8: tracking error da carteira final em relação à quantitativa = 1,50% (= 30% × orçamento de 5%);
  volatilidade [E] 8,3% → 9,6% (teto do perfil moderado: 14%).
* Saída: CAGR esperado (mediana simulada) **16,93%**; P(objetivo) **73,5%** (erro padrão MC 0,44%) contra
  74,2% da carteira só-quantitativa nas mesmas trajetórias (custo do alinhamento ≈ 0,7 p.p.).
* Sensibilidade: com o retorno esperado da carteira −1 / +1 erro-padrão (±4,4% a.a.), P(objetivo) vai de
  **23,1% a 98,3%** — a incerteza de estimação domina o ruído do Monte Carlo, e o relatório diz isso.
* Caso de borda registrado: perfil conservador com universo só de ações (`--classes acoes_br`) → restrições
  inviáveis; o motor não esconde: alertas FAIL (inviabilidade, violação de limites, validador) e código de
  saída 3.

Defeito encontrado pelos testes novos e corrigido: "Não me importo com sustentabilidade" era contado como tema
(janela de negação de 3 palavras). A negação agora vale até 5 palavras antes, dentro da mesma oração, e é
interrompida por "mas/porém/contudo…" (`test_negated_themes_are_not_counted`).

Web (`POST /api/proposito`, 2.000 trajetórias): ~33–37 s localmente; página verificada com Playwright em
1280 px e 390 px — sem erros de console e sem rolagem horizontal.

## 6. Dados reais na carteira com propósito

* `core/market_data/yahoo_chart.py` (biblioteca padrão) + `core/market_data/risk_free.py` (Selic BCB, T-bill
  `^IRX`): 12 testes em `tests/test_real_data.py` com dublês de HTTP/provedor — parsing (preço ajustado, fuso da
  bolsa, nulos), 429 → nova tentativa no outro host, 404 definitivo, disjuntor de rede, cache, Selic/T-bill, fluxo
  completo do motor e do site em modo real e falha explícita sem dados.
* Neste ambiente o proxy bloqueia o Yahoo e o BCB (`CONNECT tunnel failed, response 403`): `python main.py
  --proposito …` sem `--offline` termina em ~7 s com código 2 e a lista de tickers sem dados, sem substituir por
  sintético. Portanto **nenhuma carteira com cotações reais foi gerada aqui**.

## 7. "Vale a pena?" (análise de uma ação)

* `tests/test_stock_check.py`: 27 testes (resolução de nomes, regras do veredito, encolhimento, coerência da
  projeção, linguagem, caminho real com dublês, endpoint e CLI).
* Execução sintética registrada (`python main.py --offline --acao SINT_TECH_US --horizon-years 5 --capital 20000`):
  veredito FAVORÁVEL, 61% de chance de superar o T-bill em 5 anos, 29% de chance de prejuízo; o histórico pesa
  30% no retorno esperado; a chance vai de 37% (sem prêmio) a 87% (se o passado se repetir).
* Com cotações reais neste ambiente: `python main.py --acao PETR4` termina com código 2 ("erro de rede … 403"),
  sem estimar nada no lugar. Página verificada com Playwright (1280 px e 390 px, sem erros de console).

* O que a empresa faz e por que pode (ou não) crescer — `tests/test_company.py` (56 testes): base de
  conhecimento sem linguagem de garantia, segmento para todos os 231 ativos dos catálogos real e sintético, exemplos
  (farmácias, bancos, saneamento, energia por atividade, semicondutores…), leitura da B3 (payload base64,
  vírgulas exportadas como ". "), fluxo cookie + crumb do Yahoo com renovação após 401, perfil com fontes e
  falhas reportadas (sem cache de falhas), regras de crescimento com base e fonte, escopo BR/US, bancos sem regra
  de dívida/margem e a análise completa de "pague menos" com dublês. Página verificada com Playwright a 390 px
  (blocos novos, tabela de indicadores sem rolagem lateral, sem erros de console).

## 8. "Load failed" no site (celular) — diagnóstico e correção

* Sintoma relatado: no iPhone, "Montar carteira" com cotações reais terminava em "Erro: Load failed" (o Safari
  mostra isso quando a conexão cai sem resposta HTTP). Não foi possível acessar o deploy daqui (proxy 403), então
  a causa foi medida localmente: com 31 tickers e dados instantâneos, o cálculo levava ~98 s (sob cProfile) —
  Monte Carlo diário 63 s e backtest 26 s —, antes mesmo de ~30 downloads sequenciais com novas tentativas.
* Correções: simulação do objetivo em passos mensais (P(objetivo) 72,4% vs 71,8% na diária, z = 1,2;
  teste `test_monthly_goal_simulation_matches_daily`), modo rápido no site (backtest trimestral), downloads
  paralelos com sondagem, disjuntor (rede, 401/403/429/5xx) e orçamento total de tempo, respostas em stream com
  progresso e `ping`, e mensagem clara no lugar de "Load failed". Medido: 31 ativos ~15 s (rápido) / ~27 s
  (completo); carteira sintética no site 8 s (antes ~35 s). Testes novos: probe/paralelo/disjuntor/orçamento,
  progresso e o formato do stream (resultado e erro).

## 9. Universo completo (B3 + EUA)

* `tests/test_universe.py` (10 testes): COTAHIST montado com o layout oficial de posições fixas (ações, FII, ETF;
  BDR, fracionário e derivativos descartados), arquivo do último dia útil, screener Nasdaq (warrants/preferenciais
  descartados, BRK/B → BRK-B), diretório de símbolos, cache de 24 h, busca por nome, funil e o motor com universo
  completo.
* Tempo de cálculo medido (dados de teste, modo rápido do site): 25 ativos ~7 s, 40 ativos ~12 s, 60 ativos ~22 s
  (antes de otimizar o modo rápido, 50 ativos levavam ~51 s e 80 ativos ~87 s).
* Neste ambiente as listagens (B3, Nasdaq) e o Yahoo estão bloqueados (proxy 403): o funil caiu no catálogo curado
  e avisou; nenhum dado foi inventado.


## 10. Erro "cannot reindex on an axis with duplicate labels" no site — diagnóstico e correção

* Sintoma relatado: no site, "Montar carteira" com cotações reais e universo **Completo** terminava em
  "Erro: análise não pôde ser executada: cannot reindex on an axis with duplicate labels".
* Causa: o Yahoo às vezes devolve o pregão do dia **duas vezes** (a barra diária e uma barra "ao vivo" com a mesma
  data). A camada de qualidade já removia a data repetida dos preços, mas o **volume** (`core/data_loader.py`) e a
  série do **dólar** usada na conversão para reais (`convert_currency`) eram realinhados com a data repetida — o
  pandas recusa isso. O universo Completo mistura B3 e EUA, então sempre passa pela conversão; a análise de uma ação
  ("Vale a pena?") tinha o mesmo risco no volume.
* Reprodução: dados de teste com a barra repetida reproduziram exatamente o erro (linha 113 de `data_loader.py`).
* Correção: o parser do Yahoo mantém uma cotação por dia (a última) e registra isso nas notas da série; volume e
  câmbio passam por `last_per_date`, o que também cobre séries gravadas no cache antes da correção. Erros que
  surgem dentro de bibliotecas (pandas, numpy) passam a ser informados como **erro interno** (HTTP 500, "falha do
  sistema, não dos valores informados"), separados de entradas inválidas (HTTP 400).
* Depois do primeiro ajuste, o erro ainda apareceu no site — provavelmente antes de a nova versão terminar de ser
  publicada. Para não depender de suposição: o cache de cotações também passou a descartar datas repetidas ao
  gravar e ao ler (defensivo: o código antigo do cache não falhava, pois preço e volume tinham a mesma data
  repetida), a classificação "erro interno" passou a usar o nome do módulo onde o erro surgiu (independe de onde
  as bibliotecas estão instaladas na Vercel) e a página mostra a **versão publicada** (commit) no rodapé e nas
  mensagens de erro interno.
* `tests/test_production_like.py` (7 testes): listagens completas da B3 e dos EUA, respostas no formato do Yahoo com
  barra ao vivo repetida, volume ausente, preço nulo, feriados diferentes no Brasil e nos EUA e tickers inexistentes
  (404), passando pelo endpoint em stream — carteira com ativos em reais e em dólares, começando com capital 1 e com
  capital 0 —, a análise de uma ação (PGMN3 e AAPL), o cache com datas repetidas e a versão publicada.


## 11. Motor dinâmico (carteira a partir de dados, regras pessoais e plano executável)

Auditoria da repetição e tabela antes/depois: [`docs/audit_repeticao.md`](audit_repeticao.md)
(`python scripts/audit_repetition.py`). Testes da missão em `tests/test_dynamic_portfolio.py` (31 testes; caminho de
dados reais com dublês: listagens da B3/Nasdaq, mercado estruturado por família de ativos, Selic e CDI do Banco
Central; nenhuma rede):

| # | Comportamento exigido | Teste(s) | Resultado |
|---|---|---|---|
| 1 | mudar o perfil muda as restrições e pode mudar a carteira | `test_1_profile_changes_the_constraints_and_can_change_the_portfolio` | passa: tetos 8/14/25%, λ 8/4/2, 3 carteiras diferentes, conservador ≥ 40% em renda fixa |
| 2 | mudar o capital recalcula a viabilidade | `test_2_capital_change_recalculates_feasibility` | passa: com R$ 1.000 menos ativos cabem; o plano soma o valor e cada posição compra ≥ 1 lote |
| 3 | mudar o horizonte muda a avaliação quando se justifica | `test_3_horizon_changes_the_evaluation_and_the_portfolio_only_when_justified` | passa: 10 → 15 anos muda o CAGR necessário e mantém a carteira; 2 anos: caixa ≥ 20%, ações ≤ 30%, sem renda fixa longa |
| 4 | mudar o universo pode mudar os candidatos | `test_4_universe_change_changes_the_candidates`, `test_4b_…` | passa: só Brasil, listagens fora do ar, ativo excluído; nenhum "ativo por variedade" |
| 5 | dados novos geram nova análise | `test_5_new_market_data_triggers_a_new_analysis` | passa: outra impressão digital dos dados e da carteira; ativo cujo histórico melhora entra |
| 6 | ativos indisponíveis, duplicados ou com dados insuficientes | `test_6_…`, `test_6b_…` | passa: sem cotação, histórico curto e código desconhecido saem com motivo; duplicata de exposição descartada |
| 7 | falha de API não gera recomendação falsa | `test_7_…`, `test_7b_…` | passa: erro explícito (nada baixado / mais da metade sem cotação), CDI fora do ar avisado; resultado real expira em 15 min; a página esconde a carteira antiga |
| 8 | pesos, limites e somas validados | `test_8_weights_limits_and_sums_are_validated` (5 casos) | passa: soma 1, sem negativos, todas as restrições, teto por ativo, plano ≤ capital |
| 9 | reprodutível com os mesmos dados, parâmetros e versão | `test_9_same_data_parameters_and_version_give_the_same_result` | passa: pesos, plano e impressões digitais idênticos |
| 10 | "diferente" só quando posições/pesos mudam de verdade | `test_10_…`, `test_10b_…` | passa: 5 dias a mais de dados com as mesmas entradas → recomendação mantida (ganho < erro + custo); entradas mudadas → recalculada, cada mudança com motivo |

Também: regras de horizonte/renda/liquidez/queda, hipóteses escritas para o que não foi informado, cripto só com
permissão, capital zero (alvo = 12 meses de aportes), carteira atual (patrimônio, lacunas, "acima do alvo" sem
mandar vender), utilidade × aversão ao risco/teto/giro, seleção pessoal (mantém a carteira do investidor salvo
derrota clara), unidades de viabilidade e de concentração, relatório/JSON com as novas seções e verificador de
linguagem, site e linha de comando com os campos novos. Comparação com a linha de base 1/N: registrada em toda
execução (Sharpe no período de teste da escolhida × 1/N) e coberta por `test_equal_weight_is_a_baseline_never_the_choice`.

Problemas reais encontrados pelos testes e pela verificação no navegador durante o desenvolvimento (corrigidos):

| Onde | Problema | Correção |
|---|---|---|
| seleção (perfil moderado, 20 anos) | quando a carteira pessoal perdia por pouco, a regra "menos complexa" escolhia paridade de risco com 86% em caixa | ranking pela utilidade realizada do investidor e erro-padrão da diferença pareada |
| seleção (queda tolerada 25%) | a carteira pessoal mira o teto de volatilidade; na validação passou 13,2% contra 12,5% e era descartada, caindo numa carteira com 67% em caixa | tolerância de 10% na volatilidade realizada (o teto continua valendo ex ante na carteira final) |
| viabilidade com R$ 1.000 | sobravam posições de R$ 1 a R$ 12; ao tentar retirá-las, o ajuste espalhava peso para outros ativos até ficar inviável | regra semicontínua: mantém as posições que alcançam a compra mínima e devolve as próximas maiores no mínimo |
| caixa CDI | a série não tem volume e quebrava o recorte do conjunto de dados | volume reindexado (caixa sem volume) |
| relatório | o texto "o retorno passado não garante o futuro" era bloqueado pelo próprio verificador de linguagem | reescrito ("pode não se repetir") |
| apostas independentes | a versão por componentes principais dava 1,6 para 3 ativos independentes (componentes não únicos) | (razão de diversificação)² |

Verificação no navegador (Playwright, iPhone 13, dados sintéticos): formulário com os campos novos, resultado com
plano de compra, "o que mudou", diversificação, triagem, regras/hipóteses e fontes; segunda execução com as mesmas
entradas mostra "nenhuma mudança material — recomendação anterior mantida"; sem erros de JavaScript e sem rolagem
horizontal.

## 12. Revisão adversarial do motor dinâmico (achados reproduzidos e corrigidos)

Três revisões independentes (execução e estabilidade; universo e regras pessoais; otimizador, seleção, site e
linha de comando) reproduziram cada achado com os dublês de mercado. Cada correção tem um teste em
`tests/test_portfolio_review_fixes.py` (53 testes), além dos testes anteriores.

| Área | Achado reproduzido | Correção |
|---|---|---|
| execução | ETFs de caixa listados (SHY, LFTS11) eram tratados como dinheiro: preço 1, lote de R$ 0,01 e destino da sobra | só o CDI é dinheiro; ETFs de caixa são comprados por cota ao preço deles |
| execução | ETF de cripto da B3 (HASH11) recebia lote de cripto (0,00000001) | ETFs da B3 negociam cotas inteiras; frações só para cripto comprada direto |
| execução | ativo cujo lote custa mais que um aporte nunca era comprado ("entra nos próximos aportes" para sempre) | a sobra vai para o caixa e o caixa acima do alvo completa a compra (o plano mostra o resgate) |
| execução | só carteira atual, sem capital e sem aportes: vendas sugeridas sem dizer onde pôr o dinheiro; etapa 12 dizia "sem capital" | plano "se vender o excesso" e texto da etapa 12 conforme o caso |
| execução | regra de estabilidade mantinha a carteira anterior mas listava um ativo dela como "retirado" | lista limpa quando a anterior é mantida |
| execução | "valor inicial = capital × peso" contradizia o plano quando havia carteira atual; "peso depois da compra" ignorava posições fora do alvo | valor-alvo e valor atual por ativo; diferenças medidas sobre tudo o que se tem |
| execução | quantidades com 2 casas (bitcoin aparecia como 0,00) | casas decimais conforme a quantidade (até 8) no relatório e no site |
| execução | impressão digital dos parâmetros mudava com a ordem digitada e com o número de simulações | JSON canônico, listas ordenadas, sem o número de trajetórias |
| execução | mensagens: compra mínima × preço do lote; "todos cabem" no caso contrário; "próximos aportes" sem aportes; "12 meses" com prazo menor; plano vazio escondia a sobra; CDI listado como Yahoo e com o horário dele; "melhora em −0,00 p.p." | textos conforme o que de fato aconteceu; CDI como fonte separada |
| execução | dinheiro novo com carteira fora do alvo ia todo para uma posição (a primeira das maiores diferenças) | preenchimento por nível: as maiores diferenças diminuem juntas |
| universo | ativo com pouco mais de 1 ano de histórico (inclusive da carteira atual) encurtava a janela comum e derrubava o pedido | candidato precisa de ~3 anos (735 preços); o da carteira atual fica "fora da análise" com o motivo |
| universo | carteira atual enchia o limite de candidatos antes dos grupos exigidos pelas regras | escolhas obrigatórias podem passar do limite |
| universo | Sharpe extremo do CDI (volatilidade quase zero) distorcia as notas da renda fixa | CDI fora das notas comparativas e sempre avaliado pelo otimizador |
| universo | SHY contava como caixa para quem gasta reais; IVVB11, NASD11 e GOLD11 não contavam como dólar | caixa só na moeda do investidor; limite de moeda pela exposição |
| universo | exclusão e limite de setor só funcionavam com a chave exata (Nasdaq em inglês × catálogo em português) | chaves únicas, mapeamento da Nasdaq, normalização e aviso para nome desconhecido |
| universo | "só EUA" com objetivo em reais colidia com o limite de moeda estrangeira sem aviso | o limite não se aplica e a regra aparece |
| universo | códigos B3SA3/B5P211, PETR4F e BRK.B não eram reconhecidos; "caixa" na lista de classes quebrava | normalização dos códigos; "caixa" tratado como o CDI |
| universo | governança aplicada depois da triagem (ativo excluído ocupava vaga e aparecia como alternativa); IMA-B 5+ descrito como prazo curto; momento ausente virava 0 | governança antes da triagem; IMA-B 5+ é prazo longo; ausente é neutro |
| seleção | quando só a linha de base (1/N) passava no filtro de volatilidade, a escolha caía em 1/N | o filtro só vale se alguma estratégia escolhível passa |
| seleção | estratégia escolhida podia ficar acima do teto de volatilidade ex ante | projeção para a carteira mais próxima dentro do teto (dito nos alertas) |
| site | títulos "[object Object]" na diversificação; "1,500.00" lido como 1,5; resultado sintético reaproveitado rotulado "há menos de 15 min" | variável renomeada; leitor de valores com as duas convenções; rótulo correto |
| linha de comando | `--carteira-atual PETR4=1.500,00` quebrava e "1.500" virava 1,5; código vazio era ignorado | mesmo leitor de valores do site; código vazio é erro |
| dados | séries não tentadas porque o tempo de download acabou contavam como falha da fonte | contadas à parte, com alerta |
| dados | com o CDI fora do ar, o mínimo de caixa tornava toda a otimização inviável (antes, o SHY em dólar "resolvia") | o mínimo é retirado com aviso explícito para separar essa parte em caixa |
| covariância | o CDI quase sem oscilação deixava a matriz quase singular (otimização de 1 s → 0,03 s após a correção) | variância mínima (0,3% a.a.)² e covariância zero para essas séries |

Verificações feitas depois das correções:

* Suíte completa: 414 passed, 1 skipped (o teste online, bloqueado pela rede do ambiente).
* `ruff`: nenhum aviso novo nos arquivos alterados (os avisos que existem são anteriores, em outros trechos).
* Tempo de uma montagem do tamanho do site (40 candidatos, dublês de mercado, máquina sem outra carga): 13,8 s e
  11,3 s (antes desta rodada: 15,0 s e 9,4 s — a versão anterior escolhia sempre pesos iguais com 38 ativos).
* Navegador (Playwright, 390 px, dados sintéticos): colunas "Alvo hoje"/"Você tem", títulos da diversificação
  corretos, plano "se vender o excesso" depois da lista do que está acima do alvo, "1,500.00" lido como 1.500; sem
  erros de JavaScript e sem rolagem horizontal.
* Linha de comando: `--carteira-atual SINT_ENERGIA3=30.000,00 --carteira-atual SINT_SELIC=2,500.50
  --excluir-setores "Petróleo e Gás" xyz` roda; o setor é excluído, "xyz" gera aviso, sem "nan"/"None" e sem
  linguagem de garantia no relatório.
* Auditoria de repetição refeita (`docs/audit_repeticao.md`).

## 13. Logos das empresas

`tests/test_logos.py` (38 testes, sem rede; o serviço de favicons é um dublê):

| O que se verifica | Resultado |
|---|---|
| ativos do catálogo apontam para o site oficial (PETR3/PETR4 → petrobras.com.br; BOVA11 → iShares; BTC-USD → bitcoin.org) e toda ação do catálogo (B3 e EUA) tem site | passa |
| sem site verificado (SINT_*, CAIXA_CDI, código desconhecido, BDR) → sem logo (iniciais na página) | passa |
| site da fonte reduzido ao domínio da empresa; IP, localhost, `javascript:`, plataformas de RI e redes sociais recusados | passa |
| ícone buscado uma vez e guardado; falta (404), ícone de 16 px, SVG e arquivo grande recusados; falta tentada de novo após 1 h; falha de rede não fica guardada | passa |
| `/api/logo/site/<domínio>` e `/api/logo/ativo/<código>`: imagem com cache de CDN, `nosniff` e CSP; 404 sem chamar ninguém para código desconhecido ou domínio inválido | passa |
| resultado da carteira (`logos` e `onde_investir[].logo`) e da ação (`logo` pelo site da B3) com o endereço do logo; demonstração sintética sem logos | passa |

Navegador (Playwright, 390 px, servidor local com dublês: serviço de ícones simulado e mercado de teste): atalhos de
"Vale a pena?" com logo (BOVA11 nas iniciais quando a gestora não tem ícone), cabeçalho da ação com logo de 40 px,
carteira com 11 posições (7 logos + 4 iniciais: CAIXA_CDI, GOLD11 sem site, iShares e Kinea simulados sem ícone),
teses, plano de compra e mudanças com o mesmo logo; sem erros de JavaScript e sem rolagem horizontal. A tabela da
carteira passou a mostrar cada ativo numa linha própria (logo + nome) com os números embaixo: com as colunas "Você
tem" e "Por aporte", o nome ficava espremido no celular. Os ícones reais não puderam ser vistos neste ambiente: o
serviço de favicons é bloqueado pela rede (aparecem no site publicado).

## 14. Ação do dia

`tests/test_daily_pick.py` (12 testes, caminho de dados reais com dublês):

| O que se verifica | Resultado |
|---|---|
| a ação escolhida e as 5 alternativas cabem no valor (custo ≤ valor, quantidade inteira na B3) e respeitam o limite de risco de cada perfil | passa (3 perfis) |
| mesmos dados e entradas → mesma escolha; texto sem linguagem de garantia, com "não prevê o preço de amanhã" e "não é recomendação" | passa |
| "só Brasil" escolhe ação da B3; "só EUA", ação dos EUA | passa |
| ação que você já tem não é sugerida; a correlação com a sua carteira aparece; sem carteira informada, o aviso diz que ela não foi considerada | passa |
| valor pequeno demais → nenhuma ação, com a mais barata e a sugestão de frações; frações dos EUA só quando a corretora permite | passa |
| demonstração sintética rotulada; entradas inválidas recusadas; `/api/acao-do-dia` com cache de 15 min, 422/400 para valor ou carteira inválidos | passa |

Navegador (Playwright, 390 px, servidor local com dublês): o atalho "Ação do dia" leva à seção; com R$ 1.000 aparece
a ação com logo, "Comprar 4 ações a R$ 209,53 = R$ 838,11 (sobra R$ 161,89)", por quê, fatos, riscos, 5 alternativas
com logo, regras, avisos e fontes; com R$ 5, "nenhuma ação dentro das regras" com a mais barata (R$ 22,80); sem erros
de JavaScript e sem rolagem horizontal. Ações reais não puderam ser consultadas aqui (Yahoo bloqueado pela rede).

## 15. Valor pequeno demais (relato "ERRO" com R$ 1 + R$ 1/mês)

Relato: com capital de R$ 1 e aporte de R$ 1/mês o site mostrou uma carteira de 7 ativos (PETR4, GOLD11, CAIXA_CDI,
SBSP3, NVDA, WELL, LLY) com "R$ 0" em todas as linhas e "LLY: entra nos próximos aportes (… compra mínima de
6.465,80)". Causa: quando quase nada cabia numa compra mínima, um plano B montava a carteira-alvo como "referência para
os próximos aportes" com o universo inteiro — inclusive ativos que nem 12 meses de aportes compram; valores abaixo de
R$ 1 apareciam arredondados como "R$ 0". Reproduzido também em outros valores: R$ 400 (sintético) dava 60% numa ação
só (FAIL de limites); R$ 1.000 (dados reais simulados) mantinha 12 posições, algumas de R$ 4, porque a regra que
escolhe as posições compráveis gastava o dinheiro em lotes caros antes de incluir a renda fixa mínima do perfil.

`tests/test_small_amounts.py` (12 testes):

| O que se verifica | Resultado |
|---|---|
| R$ 1 + R$ 1/mês → recusa imediata ("Valor pequeno demais…", R$ 13 em 12 meses, pede pelo menos R$ 400), sem pedir nenhum dado de mercado | passa |
| o primeiro mínimo segue o limite por ativo e a compra mínima: conservador e arrojado R$ 300, moderado R$ 400, compra mínima de R$ 50 → R$ 200, com mínimo de caixa → R$ 200 | passa (5 casos) |
| abaixo do mínimo pelos preços do dia (sintético, R$ 600) a recusa diz o menor valor; com esse valor sai uma carteira em que toda posição compra ≥ 1 lote e ≥ R$ 100, dentro do limite por ativo, comprada inteira na compra inicial | passa |
| R$ 1.000 com dados reais simulados: só posições compráveis, sem aviso de execução nem violação de limites | passa |
| só aportes (R$ 50/mês): carteira para R$ 600; o próximo aporte compra e os demais "entram nos próximos aportes" com "R$ 100,00" | passa |
| escolha exata das posições (inteira-mista): onde ficar com as maiores falha, acha a combinação mais próxima; não repete uma combinação descartada; sem renda fixa possível ou com lotes somando 120% → nenhuma | passa |
| site: `/api/proposito` responde 400 com a mensagem (também no modo streaming), sem "não pôde ser executada"; a página mostra "Atenção:" e valores pequenos com centavos | passa |

Exploração com a mesma regra (fora dos testes): sintético — recusa até R$ 790 e carteira de 4 posições a partir daí;
dados reais simulados, moderado — R$ 400 → 4 posições de 25% (CDI, KLBN11, IMAB11, IRFM11), R$ 600 e R$ 50/mês → 4
posições, R$ 1.000 → 6 (conservador 4, arrojado 5), R$ 2.000 → 8. A recusa pelos preços leva ≈ 0,3 s depois dos dados
baixados. Observação não alterada aqui: no caso de R$ 400 o validador independente da etapa quantitativa acusa FAIL de
consistência da fronteira (ponto com Sharpe −0,3036 contra −0,3090 do máximo de Sharpe; com todos os Sharpes
negativos, o máximo de Sharpe é mal definido) — o aviso aparece na lista de alertas.

## 16. Redesign do site (identidade visual, navegação por telas)

Lógica de cada ferramenta preservada (mesmas funções de renderização e de pedido; só a apresentação mudou): os testes
que conferem a página (`tests/test_webapp.py`, `test_logos.py`, `test_daily_pick.py`, `test_small_amounts.py`,
`test_portfolio_review_fixes.py`, `test_zero_capital.py`, `test_dynamic_portfolio.py`) passam sem alteração, e a
suíte completa deu 476 aprovados e 1 pulado depois do redesign.

Navegador (Playwright, servidor local com dublês): computador 1440 px, tablet 1024 px e celular 390 px. Navegação
por todas as telas pelo menu lateral, pela barra inferior e pela folha "Mais"; busca global (PETR4 → análise pronta
em 0,8 s); Ação do dia (0,5 s); laboratório sintético (17 s); perfil salvo e aplicado aos formulários (perfil
arrojado; horizonte 8 anos → opção mais próxima, 10 anos); tema claro e escuro; link antigo `#d-title` abre a tela
Oportunidades. Sem erros de JavaScript e sem rolagem horizontal em nenhum dos tamanhos.

## 17. Minha carteira e risco

`tests/test_portfolio_analytics.py` (8 testes, dublês de mercado e do BCB):

| O que se verifica | Resultado |
|---|---|
| método das cotas e TIR em casos feitos à mão (aporte no meio do período; 10% em um ano) | passa |
| movimentações viram quantidades e fluxos; venda sem preço usa o fechamento do dia; venda maior que a posição é recusada | passa |
| patrimônio = Σ quantidade × preço (dólar × USDBRL=X do dia) + caixa; pesos e distribuições somam 100%; índices na mesma janela da carteira; alerta do limite por ativo do perfil | passa |
| com uma compra: rentabilidade = preço final / preço da compra − 1; resultado e TIR conferidos | passa |
| ativo sem dados listado e fora do patrimônio; com data de referência anterior, nenhum preço posterior é usado | passa |
| risco: contribuições somam 100%; VaR ≤ CVaR; crise da COVID só com dados de 2020 (10 anos) e "sem dados suficientes" com 5 anos; choques = beta × choque; texto sem linguagem de garantia | passa |
| liquidez = valor / (20% do volume médio em reais); sem volume → "não medido" | passa |
| rotas `/api/carteira/painel` e `/api/carteira/risco`: 200, cache de 15 min, 422 sem posições, 400 para venda maior que a posição | passa |

Navegador (servidor local com dublês, 1440 e 390 px): lista colada ("PETR4 100 28,50", "CAIXA_CDI 2.000"…), painel em
0,2 s (patrimônio, rentabilidade contra CDI/Ibovespa/S&P, evolução em base 100, distribuição, alertas), risco em 3,2 s
(VaR/CVaR em reais, contribuição para o risco, correlação normal × estresse, liquidez, sensibilidades, crises), resumo
no Painel; o link antigo `#p-title` abre a aba "Montar carteira"; sem erros de JavaScript e sem rolagem horizontal.
Cotações reais não puderam ser consultadas aqui (Yahoo e BCB bloqueados pela rede). Suíte completa depois desta etapa: 484 aprovados, 1 pulado (o teste online).

## 18. Previsões probabilísticas, calibração e registro auditável

`tests/test_forecasting.py` (9 testes, dublês de mercado e do BCB):

| O que se verifica | Resultado |
|---|---|
| o modelo é o do "Vale a pena?": retorno esperado, peso do histórico, prior, volatilidade e beta iguais aos de `_stats`/`_expected` (até 1e−12); a forma fechada reproduz o Monte Carlo do analisador (chance de superar a renda fixa e P10, diferença ≤ 1 p.p. com 40 mil trajetórias) | passa |
| Brier, log score, habilidade, tabela de confiabilidade e PIT conferidos à mão; índice de juros a partir de `^IRX` | passa |
| sem olhar o futuro: multiplicar os preços e os juros depois de uma data não muda nenhuma previsão feita antes dela; origem sem taxa conhecida é descartada (nunca juros zero) | passa |
| sem viés quando o modelo é o verdadeiro: em 12 mercados simulados, cobertura média da faixa de 80% entre 68% e 90% e PIT médio entre 0,42 e 0,58 | passa |
| distribuição: quantis em ordem, mediana = exp(m) − 1, chances iguais às fórmulas, cenários = P10/P50/P90, chance de superar o índice igual à fórmula e a 400 mil sorteios da normal bivariada | passa |
| sem histórico de juros: taxa de demonstração rotulada, calibração não calculada, prazo longo explicado | passa |
| registro: sobrevive à ida e volta pelo navegador (JSON); probabilidade alterada é recusada pela assinatura e apontada pela cadeia mesmo com o hash recalculado; remover ou trocar a ordem quebra a cadeia; sem chave, a reescrita completa não é detectada (limite declarado) | passa |
| estados: pendente antes do fim do prazo (com o retorno até agora), avaliada depois (retorno pela série atual, CDI pago, custo na compra e na venda, Brier e PIT conferidos), retroativa fora das notas | passa |
| rotas: previsão 200 com assinatura e calibração; registro 200; registro alterado 400; avaliação 200; SINT_* 400; prazo de 20 anos 422; código sem dados 503 | passa |

Navegador (servidor local com dublês terminando hoje e chave de auditoria, 1440 e 390 px): previsão de PETR4 em
2,6 s com distribuição, previsto × aconteceu, PIT, regimes e sensibilidade; registro guardado no aparelho e listado
como pendente; avaliação sob pedido; entrada alterada no aparelho → "conteúdo alterado" e "assinatura inválida";
9 entradas antigas (seladas com a mesma chave) → 8 avaliadas e 1 pendente, notas por evento e por classe; remover a
4ª entrada → "cadeia quebrada" na entrada seguinte; botão no "Vale a pena?" (VALE3) abre a previsão do mesmo ativo;
sem erros de JavaScript e sem rolagem horizontal. Cotações reais não puderam ser consultadas aqui. Suíte
completa depois desta etapa: 493 aprovados, 1 pulado (o teste online).

## 19. Fundamentos, armadilhas de valor, tese e radar

`tests/test_fundamentals.py` (9 testes, dublês de Yahoo — demonstrações, indicadores, estimativas e agenda —, B3, BCB
e cotações):

| O que se verifica | Resultado |
|---|---|
| leitura das demonstrações: ano sem valor fica vazio; valor reapresentado 2 dias depois conta como o mesmo ano; moeda | passa |
| razões conferidas à mão: margem, ROE pelo patrimônio médio, caixa livre = caixa operacional + capex, dívida líquida/EBITDA, cobertura dos dividendos, crescimento em 3 anos, diluição, P/VP, EV/EBITDA; múltiplos não misturam moedas | passa |
| Piotroski: 9 de 9 na empresa que melhora; 2 de 9 (com os testes certos) na que piora; poucos dados → não calculado | passa |
| estimativas e agenda: tendência do LPA, revisões, número de analistas, datas | passa |
| armadilha: barata e piorando → risco alto com os sinais certos; barata e saudável → baixo; cara → não se aplica; banco → indeterminado; texto sem linguagem de garantia | passa |
| tese: contra-tese da armadilha, só datas futuras na agenda, limite de invalidação da receita (95% da última), bases rotuladas, sem fato repetido | passa |
| rota `/api/tese`: empresa completa (4 anos, F-score 9, analistas, a favor/contra, previsão); B3 fora → informado; Yahoo fora → nada inventado (sem anos nem analistas) e a tese segue só com preços; cripto e SINT_* → 400 | passa |
| radar: ordem pela nota, pesos somam 1, evidências com números, fonte lenta → "tempo esgotado" sem travar, mesma entrada → mesma ordem, ação barata e piorando paga a penalidade; rota `/api/radar` 200 e 422 para mercado ou estilo inválido | passa |
| cache das fontes por 24 h (segunda consulta não acessa a rede; vencido → consulta de novo) | passa |

Navegador (servidor local com os mesmos dublês, 1440 e 390 px): tese de PETR4 em 2,6 s (a favor/contra com fontes,
armadilha com risco alto, gráfico e tabela de 4 anos, F-score 2 de 9, múltiplos, datas futuras, 5 condições de
invalidação); radar do Brasil com 86 ações e 10 candidatos com barras por fator, evidências, contra e o que desfaz;
botão "Tese" de um candidato abre a tese dele; o link antigo `#d-title` abre a aba "Ação do dia"; sem erros de
JavaScript e sem rolagem horizontal. A fonte real (Yahoo) não pôde ser consultada aqui.

## 20. Cenários: futuros da empresa e efeitos em cadeia

`tests/test_scenarios.py` (7 testes):

| O que se verifica | Resultado |
|---|---|
| valor à mão: caixa constante (10 por ação), perpetuidade crescente (5 anos + valor terminal = fluxo / (k − g)), margem em linha reta, banco (lucro × parcela distribuída) | passa |
| DCF reverso devolve o preço; pessimista < base < otimista; base = fórmula; percentis em ordem; tornado do maior para o menor; custo de capital maior → valor menor; mesma semente → mesmos sorteios | passa |
| premissas inválidas recusadas (perpetuidade a menos de 1 p.p. do custo de capital, fora dos limites, pessimista maior que a base, banco sem payout) | passa |
| premissas iniciais vêm dos balanços (crescimento de 3 anos, mediana da margem de caixa livre, renda fixa + beta × 4%) com a origem escrita | passa |
| rota `/api/cenarios`: 200; premissas do investidor marcadas como "informada por você" e mudam o valor; inválidas → 400; premissa desconhecida → 422; cripto → 400 | passa |
| efeitos em cadeia: betas plantados (+0,8, −0,8, 0) recuperados; efeito = beta × choque dentro da faixa de 95%; mineração e consumo "dados confirmam", sinal invertido "dados contrariam", setor ambíguo dito; juros +1 p.p. → IRFM11 −2% | passa |
| rota `/api/cadeia`: 200 (petróleo), choque desconhecido → 422, indicador indisponível → 503 | passa |

Navegador (servidor local com dublês, 1440 e 390 px): cenários de WEGE3 em 0,6 s (premissas com origem, histograma,
tornado, cenários, projeção); premissas editadas (crescimento −5/2/6%) recalculadas e marcadas; perpetuidade de 30%
recusada com a explicação; efeitos do dólar +10% e dos juros −1,5 p.p. com cadeia, setores e ações; sem erros de
JavaScript e sem rolagem horizontal. Fonte real não consultada aqui (dublês: o preço de demonstração não corresponde
aos balanços de teste). Suíte completa depois das seções 19 e 20: 509 aprovados, 1 pulado (o teste online).

## 21. Laboratório com cotações reais e laboratório de estratégias

`tests/test_strategy_lab.py` (5 testes):

| O que se verifica | Resultado |
|---|---|
| sinal de momento plantado (metade das ações com +30% ao ano, metade com −10%): "há evidência", com os três critérios; a variante escolhida é a de melhor Sharpe na validação; decisões não mudam quando o futuro é alterado; Sharpe deflacionado ≥ 0,95 | passa |
| mercado sem sinal: momento, reversão e baixa volatilidade não chegam a "há evidência" | passa |
| custos: mais custo → mais custo anual (derrapagem de 5 pb sempre); giro medido; tendência usa o caixa e é recusada sem a taxa de juros; sem juros, o Sharpe contra zero é dito; hipótese desconhecida recusada | passa |
| janela: 10 anos quando há ações completas; ação com 6 anos fica de fora e listada; sem ninguém com 10 anos, usa 5 | passa |
| rota `/api/estrategias`: 200 com veredito, caixa, índice, curvas e conferência; hipótese desconhecida 422; ativos de dois mercados 400; índice indisponível 503 | passa |
| laboratório de otimização com cotações reais: 200, dados reais, Selic do BCB como taxa, BOVA11 de referência, assistente funciona; menos de 3 ativos 400; cotações indisponíveis 503; a demonstração sintética continua | passa |

O texto de `/api/health` mudou para dizer que o laboratório também usa dados reais; o teste da página foi atualizado
para o texto novo.

Navegador (servidor local com dublês, 1440 e 390 px): otimização com 5 ativos reais em 10,7 s (aviso de restrições
inviáveis do perfil moderado sem renda fixa; o gráfico da fronteira, que quebrava com fronteira vazia, foi corrigido
para mostrar os ativos e a explicação); estratégia de momento em 1,9 s com veredito, curva fora da amostra marcando o
teste final, critérios, terços e variantes; sem erros de JavaScript e sem rolagem horizontal. Suíte completa
depois desta etapa: 514 aprovados, 1 pulado (o teste online).

## 22. Acompanhamento sob demanda

`tests/test_monitor.py` (4 testes):

| O que se verifica | Resultado |
|---|---|
| condições do balanço ficam "aguardando o próximo balanço" enquanto a fonte não publica um ano fiscal posterior; com o ano novo (receita 10% menor), a condição de receita é medida e acionada (378 bi < 400 bi) | passa |
| estimativa de analistas comparada a qualquer momento; preço pelo menor fechamento desde a tese (acionada e dentro do limite conferidas à mão); tese recém-criada → "aguardando novas cotações"; aviso de que nada roda em segundo plano | passa |
| fonte fora → "sem dados" e o motivo, nada preenchido | passa |
| rota `/api/monitorar`: 200; lista vazia ou condição desconhecida → 422 | passa |

Navegador (servidor local com dublês, 1440 e 390 px): Painel sem nada acompanhado mostra o convite e o botão
desativado; tese de PETR4 marcada (5 condições) e uma previsão registrada; "Conferir agora" em 1,2 s mostra cada
condição com situação, valor de hoje e limite, e a avaliação das previsões; "Última conferência" guardada; parar de
acompanhar remove a tese; sem erros de JavaScript e sem rolagem horizontal. Suíte completa depois desta
etapa: 518 aprovados, 1 pulado (o teste online).


## 23. Competição de investimentos

`tests/test_competition.py` (8 testes):

| O que se verifica | Resultado |
|---|---|
| caso em inglês e em português: idade, patrimônio, meta, prazo (o mais longo; a faculdade em 8 anos vira meta intermediária), aportes (o salário não conta), perfil pelo rótulo do cliente, objetivo, queda aceitável, liquidez (US$ 25 mil de 250 mil = 10%), exclusões e valores; toda frase citada é um trecho literal do texto colado | passa |
| regras: capital e moeda, preço mínimo de US$ 5, pelo menos 1 ETF da lista aprovada, teto de 25%, pelo menos 10 posições, sem margem nem venda a descoberto, 200 operações, 2× o volume diário, negociação de 28/09 a 04/12/2026 (inscrição e prazo do relatório ignorados); regras em português | passa |
| nada é adivinhado: "not an aggressive investor", "not interested in technology", salário anual e "works in the oil industry" não viram perfil, tema, aporte nem exclusão; texto vazio → itens faltando; lista de ETFs com nomes junto dos códigos | passa |
| regras impossíveis recusadas com a conta (3 posições de até 20% = 60%; mínimo acima do máximo; mais ETFs que a lista; menor posição acima do teto) | passa |
| meta na escala do capital (×2: US$ 1 mi com US$ 250 mil → US$ 2 mi com US$ 500 mil, aportes também ×2); retorno exigido; poder de compra como hipótese dita; 35 pregões até 04/12/2026; posição acima do teto perde ações uma a uma | passa |
| carteira com dublês do mercado: todas as regras conferidas cumprem (12 a 20 posições, teto de 20%, ações inteiras, preço ≥ US$ 5, ≥ 2 ETFs só da lista aprovada, ≥ 5 setores, caixa ≤ 2%, exclusões do cliente fora); duas ações plantadas abaixo de US$ 5 ficam fora pela regra do preço; IPS e resumo sem linguagem de promessa | passa |
| regra que não fecha (lista aprovada sem nenhum ETF existente) → "não cumpre" na conferência, nunca escondido | passa |
| rotas `/api/competicao/ler` (200; vazio → 400) e `/api/competicao/montar` (200; regras impossíveis → 422; sem cotações → 503) | passa |

Navegador (servidor local com dublês do mercado terminando hoje, 1440 e 390 px): caso e regras colados → "Ler
textos" encontra 27 itens e preenche o formulário (horizonte 20, perfil moderado, US$ 250 mil, capital US$ 500 mil,
preço mínimo 5, teto 25%, 10 posições, fim em 04/12/2026, exclusões e valores marcados); "Montar carteira" mostra
"Cumpre as 11 regras conferidas", 10 ordens, a ligação com o cliente (8 itens), o IPS (6,5 mil caracteres) e o
resumo; recarregar a página mantém os textos e o formulário (guardados só no aparelho); no celular a conferência
vira lista e a tabela de ordens mostra ativo, quantidade e peso; sem erros de JavaScript (só os 404 dos logos do
dublê) e sem rolagem horizontal. Suíte completa depois desta etapa: 526 aprovados, 1 pulado (o teste online); um
teste antigo que dependia da data de hoje foi corrigido (tabela de falhas da seção 1).
