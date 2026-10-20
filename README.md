# Quant Portfolio Intelligence Engine

Plataforma Python de **pesquisa quantitativa, construção de carteiras e inteligência de risco**: coleta e valida
dados de mercado, perfila objetivos e restrições do investidor, estima retornos/risco, constrói e compara
carteiras, executa Monte Carlo, testes de estresse e backtests walk-forward, submete tudo a um **motor de
validação independente** e explica os resultados com rastreabilidade.

> **Ferramenta de pesquisa e simulação.** Não é recomendação de investimento, não é suitability regulatório,
> não garante resultados e **não executa ordens**. Antes de oferecer recomendações individualizadas no Brasil,
> identifique os requisitos regulatórios aplicáveis (ex.: CVM) à atividade concreta.

Estado atual e evidências: [`docs/requirements_traceability.md`](docs/requirements_traceability.md) (matriz
requisito → módulo → teste → status) e [`docs/test_results.md`](docs/test_results.md) (resultados reais
registrados).

---

## 1. Arquitetura (motores)

| Motor | Módulos |
|---|---|
| Market Intelligence | `core/market_data/` (yfinance, gerador sintético, cache, qualidade), `core/data_loader.py` |
| Quant Research | `core/estimators/`, `core/covariance/`, `core/volatility/`, `core/factors/`, `core/regimes/` |
| Portfolio Construction | `core/optimization/` (`optimizers.py`, `constraints.py`, `robust.py`), `core/markowitz.py` |
| Risk Intelligence / Black Swan Lab | `core/extreme_risk/` (VaR/ES, EVT/POT, estresse, reverse stress, t-cópula), `core/black_swan.py` |
| Simulation | `core/simulation/` (modelos + motor de patrimônio), `core/monte_carlo.py` |
| Backtesting & Model Validation | `core/backtesting/` (walk-forward, PSR/DSR, registro), `core/model_validation/var_backtest.py` |
| Model Selection | `core/model_validation/model_selection.py` |
| Independent Validation | `core/model_validation/independent_validator.py` |
| AI Research Analyst | `core/ai_analyst/` (ferramentas validadas, segurança, analista) |
| Purpose Alignment (carteira com propósito) | `core/purpose/` (CAGR necessário, catálogo de tickers, sleeves por moeda, governança, temas, alinhamento, motor e relatório) |
| Orquestração / relatórios | `core/portfolio_engine.py`, `core/reporter.py`, `core/reporting/`, `main.py` |

Detalhes em [`docs/architecture.md`](docs/architecture.md).

## 2. Instalação

Requer Python 3.11+ (desenvolvido e testado com **Python 3.13.16**).

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # ou requirements-lock.txt para as versões exatas testadas
```

Windows (PowerShell):
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Execução

```bash
python main.py --offline                          # ponta a ponta, dados SINTÉTICOS, sem internet (~50–70 s)
python main.py --offline --profile conservador
python main.py --offline --simulations 5000 --fast  # rápido: pula backtest, GARCH, regimes, robustez
python main.py --profile moderado --period 5y --risk-free-rate 0.105    # online (yfinance)
python main.py --simulations 20000
python main.py --offline --ask "Qual o risco da carteira?" --ask "Compare as carteiras."
python main.py --offline --proposito --objetivo 900000 --aporte 2000 --horizon-years 10 \
    --rf BRL=0.105 --rf USD=0.043 --motivo "Quero apoiar saúde e energia limpa. O objetivo é pagar a faculdade dos meus filhos."
```

No Windows use `python` ou `py -3` com os mesmos argumentos.

Saídas (por execução, prefixadas pelo ID): `outputs/reports/*_report.md` (relatório com as 14 seções),
`*_manifest.json` (configuração, versões, commit git, hash SHA-256 dos dados, semente, tempos, avisos),
`*_results.json`, `outputs/tables/*.csv`, `outputs/charts/*.png`, `outputs/logs/run.jsonl` (log estruturado) e
`outputs/logs/strategy_registry_*.jsonl` (todas as estratégias testadas).

Códigos de saída: `0` concluído e validado (PASS/WARNING); `2` dados indisponíveis/entrada inválida;
`3` concluído mas **não validado** (falha crítica no validador independente).

### Opções de CLI (todas funcionais)

| Opção | Efeito |
|---|---|
| `--offline` | dados sintéticos determinísticos (`SINT_*`), claramente rotulados |
| `--profile {conservador,moderado,arrojado}` | perfil hipotético de demonstração (capital R$ 100.000, aporte R$ 1.000/mês) |
| `--profile-file arq.json` | perfil próprio (`InvestorProfileInput`, ver `models/investor.py`) |
| `--tickers ... --benchmark X` | universo e benchmark (modo online) |
| `--period 5y --interval 1d` | histórico e frequência do provedor |
| `--base-currency BRL` | moeda-base; ativos em outra moeda são convertidos com câmbio do provedor |
| `--risk-free-rate 0.105` | taxa livre de risco anual (sem ela: fallback de demonstração **sinalizado**) |
| `--simulations N --horizon-days D --seed S` | Monte Carlo principal |
| `--sim-model {gbm,student_t,bootstrap,block_bootstrap,regime,garch}` | modelo da simulação principal |
| `--cov-method {auto,sample,ledoit_wolf,lw_constant_corr,oas,ewma,factor_pca}` | `auto` = escolhido fora da amostra |
| `--mu-method {historical,ewma,james_stein}` | estimador de retorno esperado |
| `--var-method {historical,normal,student_t,cornish_fisher,simulated}` | método destacado no relatório |
| `--max-weight`, `--cost-bps`, `--spread-bps` | limites e custos |
| `--inflation`, `--goal`, `--capital`, `--monthly-contribution`, `--horizon-years` | parâmetros da projeção |
| `--resamples N` | reamostragens da análise de robustez (0 desativa) |
| `--fast`, `--no-charts`, `--output-dir`, `--log-level` | execução |
| `--ask "pergunta"`, `--external-file doc.txt` | analista de IA; documentos externos tratados como dados não confiáveis |
| `--proposito --objetivo V` | modo carteira com propósito (seção 10); `--objetivo` é o valor-alvo nominal |
| `--aporte A --periodicidade {mensal,trimestral,semestral,anual,nenhum}` | aporte periódico do modo propósito |
| `--motivo "texto"` / `--motivo-file arq.txt` | motivo do investidor (vazio = só a etapa quantitativa) |
| `--peso-alinhamento 0.3`, `--incluir-sob-revisao`, `--rf MOEDA=TAXA` (repetível), `--classes ...` | slider, governança, uma rf por moeda, macroclasses |
| `--acao PETR4` (ou nome: `--acao Petrobras`) | "vale a pena?" de uma ação (seção 11); usa `--horizon-years`, `--capital` (valor investido), `--profile` |

Variáveis de ambiente com prefixo `QPI_` (ver `.env.example`) também configuram `config.py`.

## 4. Fontes de dados

* **Online:** Yahoo Finance via `yfinance` (preços `Close` com `auto_adjust=True`, i.e. ajustados por
  desdobramentos e dividendos pelo provedor; `Adj Close` nunca é misturado). Cache com expiração e hash,
  rate limiting, retries com backoff, timeouts. Falhas de download **não** são substituídas por dados sintéticos.
  Respeite os termos de uso do provedor.
* **Offline:** gerador sintético determinístico (`core/market_data/synthetic.py`) com parâmetros verdadeiros
  conhecidos (regimes calmo/estresse, choques t multivariados). Todos os símbolos começam com `SINT_`; relatórios
  e gráficos exibem "DADOS SINTÉTICOS".
* **Neste ambiente de desenvolvimento o Yahoo Finance estava bloqueado pela política de rede (HTTP 403)**: o
  modo online foi implementado e testado com provedores simulados (mocks), mas **não foi validado com dados
  reais**. Há um teste opcional (`QPI_NETWORK_TESTS=1 pytest tests/test_network_optional.py`).

## 5. Convenções e fórmulas

* Retorno simples `r_t = P_t/P_{t-1} - 1`; log `ln(P_t/P_{t-1})`. Preços alinhados por interseção de datas,
  **sem forward-fill** (um preço ausente gera retorno ausente).
* Média aritmética anualizada `252·média`; CAGR `Π(1+r)^{252/N} − 1`; volatilidade `σ_diária·√252` (hipótese
  i.i.d.). Taxa livre de risco convertida por `(1+rf)^{1/252} − 1`, mesma moeda dos retornos.
* Sharpe = média do excesso / desvio do excesso · √252; Sortino com MAR = rf e desvio negativo sobre todas as
  observações; beta = cov/var; drawdown sobre a riqueza incluindo o ponto inicial.
* Carteira: `w'μ`, `w'Σw`, `√(w'Σw)`; `MRC_i = (Σw)_i/σ_p`, `RC_i = w_i·MRC_i`, `ΣRC_i = σ_p` (verificado).
* Perda `L = −R`; `VaR_α` = quantil α de L (`inverted_cdf`); ES de Acerbi–Tasche (trata empates/átomos).
  Todo VaR/ES informa método, nível e horizonte; escala √h só para o modelo normal e com rótulo.
* GBM: `S_{t+dt} = S_t·exp((μ − σ²/2)dt + σ√dt·Z)`, com μ = deriva contínua (`E[S_T] = S_0 e^{μT}`), estimada
  como `252·média(log r) + σ²/2`.
* Recuperação necessária após perda `d ∈ [0,1)`: `1/(1−d) − 1`.

## 6. Modelos implementados

Retorno esperado: histórico, EWMA, Bayes-Stein (Jorion), média comum (baseline) — com avaliação fora da amostra.
Covariância: amostral, Ledoit-Wolf (identidade), Ledoit-Wolf correlação constante, OAS, EWMA, fatorial PCA —
diagnósticos de simetria/autovalores/condicionamento e seleção fora da amostra (regra 1-EP).
Volatilidade: histórica, EWMA, GARCH(1,1), GJR-GARCH (QML), comparação QLIKE/MSE + Diebold-Mariano.
Distribuições: normal vs Student-t (AIC/BIC, Jarque-Bera, estabilidade por subperíodo), Cornish-Fisher.
Otimização: mínima variância, máximo Sharpe, retorno-alvo, volatilidade-alvo, fronteira eficiente (somente pontos
viáveis e não dominados), pesos iguais, risk parity (Spinu), máxima diversificação, HRP, mínimo CVaR (LP
Rockafellar-Uryasev), média-variância robusta (incerteza elipsoidal), penalizada (L2/turnover), reamostrada.
Simulação: GBM, Student-t multivariada, bootstrap, bootstrap em blocos, Markov-switching, CCC-GARCH,
correlações estressadas; aportes, retiradas, taxa de administração, custos de rebalanceamento, inflação.
Risco extremo: VaR/ES histórico/normal/t/Cornish-Fisher/simulado, EVT/POT-GPD (com recusa, sensibilidade ao
limiar e IC bootstrap), cenários hipotéticos e compostos, piores janelas históricas, crises nomeadas (só com
dados reais), VaR sob correlação estressada, liquidez, reverse stress (Mahalanobis + escala de cenários +
frequência empírica), t-cópula e dependência de cauda empírica.
Regimes: regra de volatilidade em tempo real, cadeia de Markov, HMM gaussiano (K=1..3 por BIC; filtrado vs
suavizado). Fatores: PCA e regressão com erros HAC.
Backtest: walk-forward com custos/slippage/turnover, divisão validação/teste final, teste de invariância a
look-ahead, PSR/DSR, Holm/BH, registro de estratégias, sensibilidade a custos; backtest de VaR (Kupiec,
Christoffersen, Z2 de Acerbi-Szekely).

## 7. Testes

```bash
python -m pytest                       # suíte completa (offline, determinística; ~10 min)
python scripts/audit_repetition.py     # a carteira reage ao perfil/dados/universo — e só a eles?
python -m pytest --cov=core --cov=models --cov=utils --cov=config
python -m pytest -m "not slow"         # sem o teste ponta a ponta completo
python scripts/profile_run.py          # profiling, escalabilidade e reprodutibilidade
```

Os testes não dependem de internet. Resultados registrados em [`docs/test_results.md`](docs/test_results.md).

## 8. Limitações e problemas conhecidos

Resumo (lista completa em [`docs/assumptions_and_limitations.md`](docs/assumptions_and_limitations.md)):

* Modo online não validado com dados reais neste ambiente (rede bloqueada).
* Sem dados point-in-time, deslistagens ou histórico de composição de índices → backtests potencialmente
  enviesados (sinalizado em todo relatório).
* Metadados de ativos (classe/setor/país/moeda) vêm de uma tabela estática só para os símbolos de exemplo;
  outros símbolos ficam "unknown" (sem limites por classe).
* Liquidez e impacto de mercado são paramétricos; impostos não são modelados.
* Limites por fator não são restrições do otimizador (exposições são apenas reportadas).
* O analista de IA é determinístico (regras + ferramentas); nenhum LLM externo é chamado.
* Retornos esperados históricos são ruidosos; a demo sintética mostra explicitamente o erro de estimação.

## 9. Site (Vercel)

A versão web fica em `webapp/`: API FastAPI (`webapp/server.py`) + página única (`webapp/static/index.html`)
com formulário, gráficos interativos (tooltips, tema claro/escuro, layout para celular), tabelas e o analista
de IA. A **Carteira com propósito** usa por padrão **cotações reais**: preços diários ajustados de ~30 ações e
fundos da B3 e dos EUA baixados na hora do endpoint de gráficos do Yahoo Finance (biblioteca padrão, sem
`yfinance`; cache em `/tmp`), Selic do Banco Central (SGS 432) e T-bill (`^IRX`) como taxas livres de risco. O
site mostra só métricas derivadas (não republica cotações) e, se o provedor não responder, mostra o erro — nunca
troca por dados sintéticos. A análise geral (`/api/analisar`) continua sobre o mercado sintético.

| Endpoint | Função |
|---|---|
| `GET /` | página |
| `POST /api/analisar` | roda a análise (modo `rapido` ou `completo`) e devolve um resumo JSON |
| `POST /api/perguntar` | pergunta ao analista de IA (mesmas recusas e verificação de números da CLI) |
| `POST /api/proposito` | carteira com propósito (seção 10): `fonte_dados="real"` (padrão, Yahoo Finance) ou `"sintetico"` |
| `POST /api/acao` | "vale a pena?" de uma ação digitada (seção 11): veredito por regras + projeção |
| `GET /api/health`, `GET /api/docs` | saúde e documentação OpenAPI |

Entradas são validadas e limitadas (≤ 20.000 simulações, horizonte ≤ 30 anos, pergunta ≤ 500 caracteres).

**Pedidos longos sem "Load failed":** `/api/analisar`, `/api/proposito` e `/api/acao` aceitam `?stream=1` (usado
pela página) e respondem em JSON por linha — eventos de progresso ("baixando cotações 12/31…", "otimizando…",
"simulando…"), um `ping` a cada 5 s de silêncio e, no fim, exatamente um `resultado` ou `erro`. A conexão nunca
fica ociosa (navegadores de celular e proxies derrubam pedidos parados) e a tela mostra a etapa e os segundos.
Sem `?stream=1` a resposta continua sendo um JSON único. No site, a carteira com propósito roda em modo rápido
(backtest com rebalanceamento trimestral; a CLI usa mensal), as cotações são baixadas em paralelo com limite
total de 45 s, e um disjuntor encerra em segundos se o Yahoo recusar ou limitar o acesso (HTTP 401/403/429).
A simulação do objetivo usa passos mensais (exata para o GBM com rebalanceamento mensal; testada contra a
diária): a carteira com 31 ativos caiu de ~98 s para ~15 s de cálculo.

**Publicar na Vercel:** envie o repositório ao GitHub → em vercel.com/new importe o repositório → Deploy.
A Vercel lê:
* `pyproject.toml` — dependências **só do runtime web** (sem matplotlib, yfinance ou pytest) e
  `[tool.vercel] entrypoint = "webapp.server:app"` (o `main.py` da raiz é a CLI, não um app web);
* `uv.lock` — versões exatas; `.python-version` — Python 3.13;
* `.vercelignore` — exclui CLI, testes e documentação do pacote da função.

Medido localmente com exatamente essas dependências: pacote de dependências **256 MB** (limite da Vercel para
Python: 500 MB); modo rápido ~15 s e completo ~45 s numa máquina de 4 núcleos (no plano gratuito da Vercel a
duração máxima é 300 s). O deploy na própria Vercel não pôde ser testado a partir deste ambiente.

**Rodar o site localmente:**
```bash
pip install -r requirements.txt
uvicorn webapp.server:app --reload      # http://127.0.0.1:8000
```

## 10. Carteira com propósito (`--proposito`, `POST /api/proposito`)

Motor em `core/purpose/` (com `core/universe/`) que monta uma carteira **a partir de dados, das regras pessoais do
investidor e de um plano de compra executável** — nunca de uma lista fixa de ativos — e a combina com o motivo
declarado. As 13 etapas, nesta ordem, ficam registradas no relatório:

1. **CAGR necessário** (`required_cagr.py`): taxa efetiva anual que leva o que você já tem (capital inicial +
   carteira atual) e os aportes (fim de cada período, `n = ⌊m·T⌋`) ao valor-alvo; raiz única por Brent. O capital
   inicial é opcional (com 0, o plano começa só com os aportes).
2. **Perfil e regras pessoais** (`personalization.py`): perfil declarado limitado pela capacidade de perda,
   horizonte, liquidez, queda tolerada e objetivo (`core/profiler.py`); aversão ao risco λ (conservador 8, moderado
   4, arrojado 2; ×1,5 preservação, ×1,25 renda); regras de horizonte (< 1 ano: ações ≤ 10%, caixa ≥ 50%; 1–3 anos:
   ações ≤ 30%, caixa ≥ 20%; 3–5 anos: ações ≤ 50%; sem renda fixa longa abaixo de 3 anos); objetivo renda (≥ 50% em
   ativos que pagam renda) ou preservação (caixa ≥ 20%); queda máxima D → volatilidade ≤ D/2; necessidade de liquidez
   L → caixa ≥ L; teto por setor e por moeda estrangeira; cripto só com permissão. Toda informação não dada vira uma
   **hipótese escrita** no resultado (nada é presumido em silêncio).
3. **Descoberta e filtro** (`core/universe/screen.py`, etapa A, **sem usar desempenho passado**): catálogo curado +
   listagens completas (B3: COTAHIST do dia — ações, FIIs e ETFs; EUA: screener da Nasdaq) + o que o investidor já
   tem; classe identificada (ETFs da B3 fora do catálogo são classificados pelo código/nome, ou ficam de fora);
   preferências (mercados, ativos/setores excluídos, cripto, renda fixa longa incompatível com o prazo); pisos de
   liquidez/tamanho; uma classe de ação por empresa (PETR3/PETR4, GOOG/GOOGL); amostra estratificada por classe e
   setor com presença mínima de cada classe e espaço para os temas do motivo. Baixa até 2× os candidatos (máx. 100).
4. **Dados e triagem** (`core/universe/scoring.py`, etapa B): cada ativo é medido no **próprio histórico** (uma
   listagem recente não encurta o dos outros): volatilidade, queda máxima, Sharpe encolhido para a média da classe,
   momento 12-1 e liquidez, em z-scores dentro da classe com pesos do perfil; duplicatas de exposição (correlação
   ≥ 0,95 na mesma classe) e grupos que andam juntos (≥ 0,70, cota por grupo); primeiro o que os limites exigem
   (caixa, renda, mínimo de renda fixa), depois temas e as melhores pontuações. Cada ativo recebe decisão e motivo.
   Caixa em reais = **CDI diário do Banco Central (SGS 12)**, rotulado como modelo de Tesouro Selic/CDB de liquidez
   diária. Mais da metade do universo sem cotação → **nenhuma carteira** (erro claro, sem resultado antigo).
5. **Sleeves por moeda** (`sleeves.py`): estatísticas na moeda nativa com a rf de cada moeda (nunca rf = 0).
6. **Governança** (`governance.py`): exclusão dura antes do otimizador.
7. **Tamanho mínimo de compra** (`feasibility.py`): a carteira-alvo é dimensionada para o que você tem hoje + os
   aportes dos próximos 12 meses; ativo cuja menor compra possível (1 lote/cota/ação, ou a compra mínima, padrão
   R$ 100) passa do peso máximo fica de fora, com o motivo.
8. **Otimização** (`core/portfolio_engine.py` com `OptimizationPolicy`): os limites pessoais viram restrições; a
   carteira de **utilidade pessoal** `max w'μ − λ/2·w'Σw` (com o teto de volatilidade) é a escolha padrão e só é
   trocada se outra estratégia a superar, na validação fora da amostra, por mais de 1 erro-padrão da diferença,
   medindo pela **utilidade realizada do próprio investidor**. Pesos iguais e o benchmark viram **linhas de base**
   (comparadas no relatório, nunca escolhidas). Com carteira atual, trocar tem custo (penalidade de giro).
9–10. **Temas e score de alinhamento** (`themes.py`, `alignment.py`): léxico determinístico com citação literal.
11. **Combinação e carteira executável**: `w_final = (1 − a)·w_quant + a·w_alinhamento` dentro de todos os limites;
   depois, posições menores que uma compra mínima saem (ou sobem até ela, se os limites exigirem) e os pesos são
   recalculados como a carteira executável mais próxima. **Regra de estabilidade**: com a recomendação anterior e as
   mesmas entradas, ela é mantida se a nova não a superar por mais que o erro de estimativa + o custo de trocar.
12. **Plano de compra e dupla justificativa**: quantidades por lote (B3: fracionário de 1 ação, FIIs/ETFs 1 cota;
   EUA: ação inteira ou frações se a corretora permitir; CDI: qualquer valor), sobra, o que fica para os próximos
   aportes e o que está acima do alvo na carteira atual (sem mandar vender: custo e imposto). Por ativo: mercado,
   classe, peso, valor, papel, riscos, por que virou candidato, fatos [H], estimativas [E], hipóteses,
   contribuição para a diversificação e alternativas consideradas.
13. **Mudanças e diversificação real** (`changes.py`, `concentration.py`): o que entrou/saiu/subiu/caiu desde a
   carteira atual ou a recomendação anterior, com o motivo (variações < 1 p.p. contam como "mantido"); apostas
   independentes = (razão de diversificação)², grupos que andam juntos, sobreposição ETF × ação estimada pela
   correlação, exposição por moeda/classe/setor/mercado em dinheiro e em risco.

Saída (Markdown + JSON em `outputs/proposito/`): CAGR necessário × esperado, tabela de pesos, teses, cenários
P90/P50/P10, P(objetivo) com erro padrão e IC, sensibilidade, alertas e as seções 7–12 (plano de compra, mudanças,
diversificação real, triagem completa, regras/hipóteses, fontes e reprodutibilidade: impressão digital da carteira,
dos parâmetros e dos dados + versão). Todo texto gerado passa pelo verificador de linguagem de garantia.

**Por que a carteira se repetia e o que mudou:** ver [`docs/audit_repeticao.md`](docs/audit_repeticao.md) (causas
com arquivo e linha, tabela antes/depois gerada por `python scripts/audit_repetition.py`).

**Dados e cache:** preços diários do Yahoo Finance (podem trazer o pregão em andamento; nunca tratados como tempo
real) guardados por até 4 h; resultado do site reaproveitado por até 15 min para o mesmo pedido (o resultado diz
quando foi calculado). Um pedido novo esconde a carteira anterior na tela; se falhar, nada antigo é mostrado como
atual. O site guarda neste aparelho a última carteira (pesos + impressão digital das entradas) para a comparação.

**Limites honestos:** a cobertura é a B3 (ações, FIIs, ETFs negociados no dia) e as ações dos EUA do screener da
Nasdaq + os ETFs americanos do catálogo + caixa em CDI — não "todas as ações do mundo"; CDBs/LCIs específicos e
títulos individuais do Tesouro não são consultados. Classificação/temas dos ativos do catálogo são curadoria
editorial; o backtest escolhe o universo com dados de hoje (viés de sobrevivência/seleção, sinalizado). Impostos
não são modelados. O modo offline usa um universo **sintético** e o resultado é marcado "NÃO É UMA CARTEIRA FINAL".

## 11. "Vale a pena?" — análise de uma ação (`--acao`, `POST /api/acao`)

Você digita o código ou o nome de **qualquer ação da B3 ou dos EUA** (PETR4, VALE3, WEGE3, AAPL, "Petrobras",
"Banco do Brasil"…; nomes que não estão na lista conhecida são procurados nas listagens completas da B3 e da
Nasdaq; nomes curtos ambíguos como "vale" só valem sozinhos, e o que não for encontrado pede o código — nada é
adivinhado). O módulo
`core/stock_check/` baixa ~5 anos de preços ajustados (Yahoo Finance), compara com a renda fixa **da mesma moeda**
(Selic do BCB para BRL, T-bill `^IRX` para USD) e devolve:

* **Veredito por regras fixas**, iguais para todos os ativos e mostradas no resultado:
  FAVORÁVEL se a chance de superar a renda fixa no horizonte é ≥ 60% **e** o retorno histórico acima dela tem
  t ≥ 1; DESFAVORÁVEL se a chance é ≤ 40% **ou** o retorno histórico abaixo dela tem t ≤ −1; NEUTRO nos demais
  casos (o histórico não separa o ativo da renda fixa). A adequação ao perfil (volatilidade e queda máxima contra
  os limites do perfil) é informada à parte.
* **Critérios**: chance de superar a renda fixa, retorno acima dela (com estatística t), Sharpe, volatilidade,
  queda máxima, VaR 95% diário, % de janelas de 12 meses acima da renda fixa, correlação com Ibovespa/S&P 500,
  giro financeiro diário (liquidez) e variação de 12 meses (informativa).
* **Projeção** (Monte Carlo, horizonte e valor escolhidos): leque de percentis com a linha da renda fixa,
  cenários P90/P50/P10 com retorno anual e "preço equivalente" (reescala, não é preço-alvo), chance de prejuízo e
  sensibilidade a quatro premissas de retorno.
* **Retorno esperado da projeção** com encolhimento documentado: referência = rf + beta × prêmio de 4% a.a.;
  peso do histórico = τ²/(τ² + erro-padrão²) com τ = 10% a.a. Cinco anos de uma ação com 30% de volatilidade
  pesam ~35%: um passado excepcional não é projetado como se fosse se repetir.

* **O que a empresa faz** (`core/stock_check/company.py`): descrição curada do catálogo (ex.: Pague Menos — "rede
  de farmácias e drogarias (medicamentos, higiene pessoal e beleza)"), atividade registrada e classificação setorial
  da **B3** (serviço público de empresas listadas), sede, funcionários e resumo do **Yahoo Finance** (em inglês,
  rotulado), o **segmento** identificado e "como o segmento costuma ganhar dinheiro". Fonte e data em cada item;
  o que a fonte não traz aparece como "indisponível".
* **Por que pode crescer / por que pode não crescer tanto**: cada motivo vem com a sua base e a fonte —
  *Fundamentos* (crescimento de receita e lucro, margem, ROE contra a renda fixa, dívida/patrimônio, P/L,
  dividendos, fluxo de caixa livre, com limites fixos), *Analistas* (P/L projetado e preço-alvo médio, só com ≥ 3
  analistas, sempre como opinião de terceiros), *Histórico* (tendência de 12 meses com a média de 200 pregões,
  retorno contra o CDI com t, volatilidade, queda máxima, beta, liquidez), *Simulação*, *Juros*/*Câmbio* e *Setor*
  (base de conhecimento editorial com 56 segmentos — farmácias, bancos, saneamento, energia… —, com regras
  brasileiras como CMED, ANS, ANEEL, Marco do Saneamento e FIES marcadas por país). Os motivos são contexto: o
  veredito continua usando só as regras acima.

Não avalia notícias e não é recomendação de compra ou venda. Indicadores fundamentalistas dependem do Yahoo
Finance (não oficial; exige o "crumb" de sessão) e a descrição oficial do serviço da B3: ambos podem falhar e,
então, são marcados como indisponíveis. Neste ambiente de desenvolvimento só o modo sintético (`SINT_*`) pôde ser
executado; o caminho real (preços, Selic, B3, Yahoo) foi testado com dublês (rede bloqueada).

