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
| Market Intelligence | `core/market_data/` (yfinance, gerador sintético, cache, qualidade, cotações do dia em `quotes.py`), `core/data_loader.py`, `core/market_watch/` (mercado hoje) |
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

A versão web fica em `webapp/`: API FastAPI (`webapp/server.py`) + página única (`webapp/static/index.html`, sem
etapa de build). Identidade visual própria: fundo azul-marinho/grafite, fonte Inter servida pelo próprio app
(`/static/fonts/`, SIL OFL), ícones Lucide (ISC) embutidos, tokens de cor/espaçamento/raio, tema escuro (padrão) e
claro, `prefers-reduced-motion` respeitado. Navegação por telas (`#/painel`, `#/mercado`, `#/acoes`, `#/oportunidades`,
`#/previsoes`, `#/cenarios`, `#/carteira`, `#/competicao`, `#/laboratorio`, `#/perfil`, com abas como `#/acoes/tese` ou
`#/oportunidades/radar`): menu lateral no computador, trilho de ícones no tablet e barra inferior
no celular (Painel, Mercado, Radar, Carteira e "Mais"; a análise de ações fica em "Mais" e na lupa do topo); links
antigos por âncora (`#d-title`, `#p-title`…) abrem a tela certa. O **Painel** mostra o que existe
de fato neste aparelho (última carteira montada, histórico de análises, perfil); o **Perfil** guarda as respostas do
investidor no navegador (`localStorage`, com exportar/importar/apagar) e preenche os formulários de todas as
ferramentas. Nada do investidor vai para o servidor fora dos pedidos de análise. A **Carteira com propósito** usa por padrão **cotações reais**: preços diários ajustados de ~30 ações e
fundos da B3 e dos EUA baixados na hora do endpoint de gráficos do Yahoo Finance (biblioteca padrão, sem
`yfinance`; cache em `/tmp`), Selic do Banco Central (SGS 432) e T-bill (`^IRX`) como taxas livres de risco. O
site mostra só métricas derivadas (não republica cotações) e, se o provedor não responder, mostra o erro — nunca
troca por dados sintéticos. O laboratório (`/api/analisar`) roda sobre o mercado sintético de demonstração ou,
com `fonte_dados: "real"`, sobre as cotações dos ativos escolhidos (seção 10g).

| Endpoint | Função |
|---|---|
| `GET /` | página |
| `POST /api/analisar` | roda a análise (modo `rapido` ou `completo`) e devolve um resumo JSON |
| `POST /api/perguntar` | pergunta ao analista de IA (mesmas recusas e verificação de números da CLI) |
| `POST /api/proposito` | carteira com propósito (seção 10): `fonte_dados="real"` (padrão, Yahoo Finance) ou `"sintetico"` |
| `POST /api/acao` | "vale a pena?" de uma ação digitada (seção 11): veredito por regras + projeção |
| `POST /api/acao-do-dia` | uma ação para o valor de hoje, por regras fixas, no preço de agora (seção 10b) |
| `POST /api/mercado` | mercado hoje: índices, maiores altas e quedas, mais negociadas e preços, com atraso (seção 10j) |
| `POST /api/carteira/painel`, `/api/carteira/risco` | painel e risco da carteira do investidor (seção 10c) |
| `POST /api/previsoes/prever`, `/registrar`, `/avaliar` | previsões, calibração e registro auditável (seção 10d) |
| `POST /api/tese`, `/api/radar` | tese e fundamentos, armadilhas de valor, radar (seção 10e) |
| `POST /api/cenarios`, `/api/cadeia` | futuros da empresa e efeitos em cadeia (seção 10f) |
| `POST /api/estrategias` | laboratório de estratégias (seção 10g) |
| `POST /api/monitorar` | acompanhamento sob demanda das teses marcadas (seção 10h) |
| `POST /api/competicao/ler`, `/api/competicao/montar` | competição de investimentos: leitura do caso e das regras, carteira que cumpre as regras (seção 10i) |
| `GET /api/logo/...` | logos das empresas pelo próprio app |
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
   L → caixa ≥ L; teto por setor e por moeda estrangeira (pela moeda de exposição: IVVB11, ouro e cripto contam
   como dólar; "só EUA" dispensa esse teto, e a regra aparece); caixa só na moeda do investidor; cripto só com
   permissão; setores digitados são normalizados ("Petróleo e gás", "oil") e nome desconhecido gera aviso. Toda
   informação não dada vira uma **hipótese escrita** no resultado (nada é presumido em silêncio).
3. **Descoberta e filtro** (`core/universe/screen.py`, etapa A, **sem usar desempenho passado**): catálogo curado +
   listagens completas (B3: COTAHIST do dia — ações, FIIs e ETFs; EUA: screener da Nasdaq) + o que o investidor já
   tem; classe identificada (ETFs da B3 fora do catálogo são classificados pelo código/nome, ou ficam de fora);
   preferências (mercados, ativos/setores excluídos, cripto, renda fixa longa incompatível com o prazo); pisos de
   liquidez/tamanho; uma classe de ação por empresa (PETR3/PETR4, GOOG/GOOGL; fica a do catálogo, se houver);
   amostra estratificada por classe e setor com presença mínima de cada classe e espaço para os temas do motivo.
   Baixa até 2× os candidatos (máx. 100). Setores da Nasdaq viram as mesmas chaves do catálogo.
4. **Dados e triagem** (`core/universe/scoring.py`, etapa B): cada ativo é medido no **próprio histórico** (uma
   listagem recente não encurta o dos outros): volatilidade, queda máxima, Sharpe encolhido para a média da classe,
   momento 12-1 e liquidez, em z-scores dentro da classe com pesos do perfil; duplicatas de exposição (correlação
   ≥ 0,95 na mesma classe) e grupos que andam juntos (≥ 0,70, cota por grupo); primeiro o que os limites exigem
   (caixa, renda, mínimo de renda fixa — podem passar do limite de candidatos), depois temas e as melhores
   pontuações. Candidato precisa de ~3 anos de preços (735); ativo da carteira atual com menos fica "fora da análise"
   com o motivo. Governança é aplicada antes da triagem. Cada ativo recebe decisão e motivo.
   Caixa em reais = **CDI diário do Banco Central (SGS 12)**, rotulado como modelo de Tesouro Selic/CDB de liquidez
   diária; fica fora das notas comparativas e é sempre avaliado pelo otimizador. Mais da metade das séries tentadas
   sem cotação → **nenhuma carteira** (erro claro, sem resultado antigo).
5. **Sleeves por moeda** (`sleeves.py`): estatísticas na moeda nativa com a rf de cada moeda (nunca rf = 0).
6. **Governança** (`governance.py`): exclusão dura antes da triagem e do otimizador.
7. **Tamanho mínimo de compra** (`feasibility.py`): a carteira-alvo é dimensionada para o que você tem hoje + os
   aportes dos próximos 12 meses; ativo cuja menor compra possível (1 lote/cota/ação, ou a compra mínima, padrão
   R$ 100) passa do peso máximo fica de fora, com o motivo. **Valor pequeno demais**: se nenhuma combinação de ativos
   tem cada posição com pelo menos uma compra mínima dentro dos limites do perfil (problema inteiro-misto com os
   preços do dia), a resposta é uma recusa (HTTP 400, "Atenção:" no site) com o menor valor que forma uma carteira;
   antes de baixar dados vale a conta rápida ⌈1 / limite por ativo⌉ × compra mínima (R$ 400 no moderado, R$ 300 no
   conservador e no arrojado). Nunca sai uma carteira com posições que o dinheiro não compra; para valores pequenos,
   a seção "Ação do dia" aponta uma ação.
8. **Otimização** (`core/portfolio_engine.py` com `OptimizationPolicy`): os limites pessoais viram restrições; a
   carteira de **utilidade pessoal** `max w'μ − λ/2·w'Σw` (com o teto de volatilidade) é a escolha padrão e só é
   trocada se outra estratégia a superar, na validação fora da amostra, por mais de 1 erro-padrão da diferença,
   medindo pela **utilidade realizada do próprio investidor**. Pesos iguais e o benchmark viram **linhas de base**
   (comparadas no relatório, nunca escolhidas). Com carteira atual, trocar tem custo (penalidade de giro). Se a
   estratégia escolhida passar do teto de volatilidade, usa-se a carteira mais próxima dela dentro do teto.
9–10. **Temas e score de alinhamento** (`themes.py`, `alignment.py`): léxico determinístico com citação literal.
11. **Combinação e carteira executável**: `w_final = (1 − a)·w_quant + a·w_alinhamento` dentro de todos os limites;
   depois, posições menores que uma compra mínima saem (ou sobem até ela, se os limites exigirem) e os pesos são
   recalculados como a carteira executável mais próxima; se ficar com as maiores posições não fechar os limites (o
   dinheiro acaba em lotes caros antes da renda fixa mínima, por exemplo), as posições são escolhidas de forma exata
   (inteira-mista: a combinação mais próxima da teórica). **Regra de estabilidade**: com a recomendação anterior e as
   mesmas entradas, ela é mantida se a nova não a superar por mais que o erro de estimativa + o custo de trocar.
12. **Plano de compra e dupla justificativa**: quantidades por lote (B3: fracionário de 1 ação, FIIs/ETFs 1 cota,
   inclusive ETFs de cripto e de Tesouro Selic; EUA: ação inteira ou frações se a corretora permitir; cripto direta:
   0,00000001; CDI: qualquer valor), sobra, o que fica para os próximos aportes e o que está acima do alvo na
   carteira atual (sem mandar vender: custo e imposto). O dinheiro novo nivela as posições mais abaixo do alvo; a
   sobra vai para o caixa, e o caixa acima do alvo completa um lote mais caro que um aporte (o plano mostra o
   resgate). Sem dinheiro novo, o plano "se vender o excesso" mostra onde o dinheiro da venda entraria. Por ativo: mercado,
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

**Primeiro, o que comprar:** no site, o resultado abre com um cartão curto — para cada ativo da compra inicial (ou
do primeiro aporte, quando não há capital), quantas ações ou cotas comprar, o preço de cada uma (ativos dos EUA
também em dólar, `execucao.precos_originais`: o último fechamento na moeda do ativo, do mesmo dia do preço em
reais), o total e uma linha de "por quê" (o papel na carteira em palavras simples e o tema do seu motivo, quando há).
Embaixo, o total, a sobra e a chance estimada de chegar à meta. A análise completa (pesos, KPIs, plano, mudanças,
diversificação, teses, triagem, regras, cenários, alertas e fontes) continua inteira, recolhida em "Ver a análise
completa". Os preços são do último dado diário (com a data), não a cotação de agora.

**Logos das empresas** (`core/market_data/logos.py`): na carteira (alocação, teses, plano de compra, mudanças) e
em "Vale a pena?" (atalhos e cabeçalho do resultado), cada ativo aparece com o ícone que a empresa publica no próprio
site. O site vem da B3 ou do Yahoo Finance (ação pesquisada) ou de uma lista curada de sites oficiais do catálogo
(fundos e ETFs mostram a gestora: iShares, Vanguard, Kinea…). O servidor do app busca o ícone num serviço público de
favicons (Google), aceita só imagens rasterizadas e guarda por uma semana; o navegador só fala com o próprio app.
Sem logo verificado — ativos fora do catálogo, demonstração sintética, caixa CDI — aparecem as iniciais do código,
nunca o logo de outra empresa. Logos são marcas dos donos, mostradas só para identificar o ativo.

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

## 10b. "Ação do dia" — quanto você tem hoje e qual ação as regras apontam (`POST /api/acao-do-dia`)

Você diz quanto tem para investir hoje (e, se quiser, o que já tem) e o app aponta **uma ação** por regras fixas,
mostradas no resultado (`core/stock_check/daily_pick.py`):

1. universo: as ações do catálogo curado (grandes empresas da B3 e dos EUA), nos mercados escolhidos; sem ETFs,
   fundos ou cripto;
2. dados: ~5 anos de preços diários ajustados, em reais; a ação precisa de ~3 anos de histórico;
3. pontuação do perfil (a mesma da triagem da carteira): Sharpe encolhido, volatilidade, queda máxima, momento de
   12 meses e liquidez;
4. limite de risco para UMA ação: conservador — volatilidade até 25% ao ano e queda máxima até 40%; moderado —
   volatilidade até 35%; arrojado — sem limite;
5. precisa caber no valor: 1 lote (B3: 1 ação no fracionário; EUA: 1 ação, ou frações se a corretora permitir);
6. o que você já tem não é sugerido de novo, e a pontuação cai 0,5 × a correlação com a sua carteira.

Resultado: a ação, quantas comprar, custo e sobra, por que ela (fatos do histórico), o que a empresa faz, riscos, 5
alternativas que também cabem nas regras, o que ficou de fora e por quê, fontes e avisos. Com os mesmos dados a
escolha é a mesma (pode repetir em dias seguidos). Usa só o histórico: não prevê o preço de amanhã e não é
recomendação; para quem está começando, distribuir os aportes numa carteira costuma reduzir o risco.

**Preço de agora** (`apply_live_prices`): a escolha vem do histórico diário (reaproveitada por até 15 min); a cada
pedido, a ação escolhida e as alternativas recebem a última cotação (`core/market_data/quotes.py`, a mesma da tela
Mercado, guardada por 60 s), e a quantidade, o custo e a sobra são refeitos com ela — ações dos EUA pelo dólar da
mesma consulta. O cartão mostra primeiro quantas comprar, o preço agora com a variação do dia e o total; embaixo, a
hora do último negócio (horário de Brasília), pregão aberto ou fechado e o atraso (cerca de 15 min na B3: não é tempo
real), e o fechamento usado na escolha. Se o valor não comprar mais 1 lote no preço de agora, o site diz e aponta a
primeira alternativa que cabe; se a cotação de agora não vier, fica o último fechamento, com aviso na linha.

## 10c. Minha carteira e risco (`POST /api/carteira/painel`, `POST /api/carteira/risco`)

O investidor cadastra o que tem (código e quantidade; preço médio e movimentações opcionais). Não há conta nem
banco de dados: as posições ficam no navegador (`localStorage`, com exportar/importar) e vão no pedido.

* **Painel**: patrimônio com cotações reais (dólar convertido pelo `USDBRL=X` do mesmo dia), posições, distribuição
  por ativo/classe/setor/moeda, evolução em base 100 contra CDI (BCB SGS 12), Ibovespa (BOVA11) e S&P 500 em reais
  (IVVB11), rentabilidade por período pelo **método das cotas** (TWR, fluxos no início do dia) e, com movimentações,
  a **rentabilidade do seu dinheiro** (TIR) e o resultado (patrimônio − compras + vendas + proventos). Sem
  movimentações, a evolução é uma **hipótese** rotulada (quantidades de hoje mantidas no período). Alertas por regras
  com os números: limite por ativo do perfil, setor acima de 35%, moeda estrangeira acima de 50%, queda além da
  referência do perfil, cotação velha, ativo sem dados.
* **Risco** (pesos de hoje, até 5 anos de cotações; 10 anos para crises): volatilidade histórica e EWMA; VaR e CVaR a
  95% e 99% (histórico, normal e teoria de valores extremos) em 1 ou 10 dias; maior queda; correlação em dias normais
  e de estresse; contribuição de cada ativo para o risco (Euler) por ativo, setor, classe e moeda; liquidez (dias para
  vender a 20% do volume médio); sensibilidade ao Ibovespa, ao S&P 500 em reais, ao dólar e aos juros (IRFM11; IEF
  com ativos em dólar), com t de Newey–West; crises reais (COVID-19, pior mês e pior trimestre do Ibovespa nos dados)
  e choques hipotéticos pelos betas medidos. Estimativas do passado, não limites de perda.

## 10d. Previsões probabilísticas e registro auditável (`POST /api/previsoes/prever`, `/registrar`, `/avaliar`)

Em vez de preço-alvo, a **distribuição do retorno** de um ativo no prazo escolhido (3 meses a 5 anos), com o mesmo
modelo do "Vale a pena?" escrito em forma fechada (`core/forecasting/model.py`): retorno esperado = renda fixa +
beta × prêmio de risco, puxado para a média histórica conforme a precisão dela (encolhimento), volatilidade
histórica, retorno log-normal. Mostra quantis de 5% a 95%, cenários P10/P50/P90 e as chances de subir, de superar a
renda fixa (Selic/CDI ou T-bill), de cair mais de 20% e de superar o índice de referência (BOVA11 ou SPY; fórmula
exata da normal bivariada com a correlação medida), mais uma tabela de sensibilidade (prêmio 2/6%, tau 5/20%,
volatilidade ±25%, só o histórico, sem prêmio).

* **Calibração histórica no próprio ativo** (`calibration.py`): o modelo é refeito todo mês dos últimos 10 anos
  usando só os dados até cada data (nada do futuro — testado) e as previsões de 12 meses (ou do prazo, se menor)
  são comparadas com o que aconteceu: Brier e habilidade contra a frequência do passado, log score, tabela de
  confiabilidade (previsto × aconteceu), cobertura da faixa de 80%, histograma PIT e teste KS, por regime de
  volatilidade, com o aviso de que janelas sobrepostas reduzem o tamanho efetivo da amostra.
* **Registro auditável** (`audit.py`): a previsão sai do servidor com data e assinatura HMAC (chave
  `QPI_AUDIT_KEY`, quando configurada); ao registrar, recebe a hora do servidor e um hash encadeado ao anterior
  (`sha256` do JSON canônico). O registro fica no aparelho do investidor. A avaliação, só quando pedida, confere a
  cadeia (alteração, remoção ou troca de ordem aparecem), marca como **retroativa** a entrada registrada mais de 3
  dias úteis depois da data dos dados (fora das notas), deixa **pendente** até o dia seguinte ao fim do prazo e mede o
  resultado com a série ajustada atual (proventos contam), contra a renda fixa efetivamente paga, o índice e após
  custos; notas por evento, classe, prazo e regime. Limites ditos no resultado: apagar as últimas entradas não é
  detectável pela cadeia, e sem a chave do servidor quem reescreve tudo não é detectado.

## 10e. Tese e fundamentos, armadilhas de valor e radar de oportunidades (`POST /api/tese`, `POST /api/radar`)

* **Fundamentos ano a ano** (`core/fundamentals/sources.py`, `analysis.py`): demonstrações anuais publicadas pelo
  Yahoo Finance (`fundamentals-timeseries`: receita, lucros, EBITDA, caixa operacional, capex, caixa livre, dívida,
  caixa, patrimônio, ativos, ações, dividendos pagos, LPA) e estimativas de analistas com a agenda (`quoteSummary`:
  `earningsTrend`, `recommendationTrend`, `calendarEvents`), pelo mesmo cliente com cookie e *crumb* do "Vale a
  pena?", com cache de 24 h. Ano sem valor fica sem valor. Margens, ROE, dívida líquida/EBITDA, cobertura dos
  dividendos pelo caixa livre, crescimento em 3 anos, diluição e o **Piotroski F-score** (9 testes; não calculado
  para bancos e seguradoras, como no estudo original). Múltiplos (P/L, P/VP, EV/EBITDA, caixa livre e dividendos
  sobre o valor de mercado) nunca misturam moedas.
* **Detector de armadilhas de valor** (`value_trap.py`): primeiro, "parece barata?" (P/L ≤ 8, P/VP ≤ 1, dividendos ≥
  8%, caixa livre ≥ 10% do valor de mercado, EV/EBITDA ≤ 5 ou queda ≥ 40% do topo de 3 anos); depois, 11 sinais de que
  o preço baixo pode ser merecido (F-score ≤ 3, receita e lucro em queda, margem encolhendo, caixa livre negativo,
  dividendos acima do caixa livre, dívida > 3× EBITDA, diluição > 5%, analistas cortando estimativas, preço em
  tendência de queda, dividendos ≥ 15%). Risco alto com 4 ou mais sinais (ou F-score ≤ 2); cada sinal mostra a medida,
  a regra e a fonte, ou "não medido".
* **Tese estruturada** (`thesis.py`, aba "Tese e fundamentos" em Análise de ações): fatos medidos [H], estimativas
  [E] (modelo de preços e, rotuladas, as de analistas), pontos a favor e contra com a base de cada um (medido, modelo,
  terceiros ou setor), o que precisa ser verdade, datas futuras publicadas (resultados, data com e pagamento de
  dividendos), condições de invalidação medíveis (receita −5%, margem −3 p.p., dívida > 3,5× EBITDA, estimativa de
  LPA −10%, cenário dos 5% piores, dividendos pela metade), a contra-tese e a qualidade dos dados.
* **Radar** (`radar.py`, aba "Radar" em Oportunidades): varre as ações do catálogo do(s) mercado(s) escolhido(s) e
  ordena por notas z de valor, qualidade, crescimento, momento e risco, com pesos por estilo (equilibrado, valor,
  qualidade, crescimento, dividendos, momento); ação que parece barata perde pontos por sinal de armadilha. Cada
  candidato mostra as evidências com números, o que pesa contra e o que desfaria a sua posição; a fonte de balanços
  tem orçamento de 35 s (quem não responde a tempo é avaliado só pelo preço, e isso é dito).

## 10f. Cenários: futuros da empresa e efeitos em cadeia (`POST /api/cenarios`, `POST /api/cadeia`)

* **Futuros da empresa** (`core/scenarios/dcf.py`): valor por ação pelo caixa ao acionista em 5 anos + perpetuidade,
  na moeda do balanço. Premissas: crescimento da receita, margem de caixa livre no ano 5 (a margem anda em linha reta
  da atual até ela), custo de capital (padrão: renda fixa + beta × 4%, o mesmo modelo das previsões) e crescimento
  perpétuo (4% em BRL, 2,5% em USD); bancos e seguradoras usam lucro × parcela distribuída. Os valores iniciais vêm
  dos balanços da própria empresa, com a origem escrita, e o investidor pode editar pessimista/base/otimista.
  Resultado: três cenários, 20 mil sorteios triangulares (parcela acima do preço de hoje), o que mais pesa no valor
  (tornado), a projeção do cenário base, a parcela do valor que vem da perpetuidade e o **crescimento implícito no
  preço** (DCF reverso). Recusa quando faltam balanços ou as moedas do balanço e da cotação diferem.
* **Efeitos em cadeia** (`core/scenarios/chain.py`): para juros no Brasil (IRFM11, duração ≈ 2), dólar (USDBRL=X),
  petróleo (Brent, BZ=F), bolsa americana (SPY) e juros nos EUA (IEF, duração ≈ 7,5), a cadeia de transmissão e o
  efeito esperado por setor (hipótese, com o motivo) ao lado do beta medido de cada ação (retornos semanais de 5 anos,
  t de Newey–West) e do efeito estimado = beta × movimento do indicador; por setor, a mediana e se os dados confirmam,
  contrariam ou não permitem dizer. Pode usar as ações do catálogo (Brasil ou EUA) ou as da carteira do investidor.

## 10g. Laboratório com cotações reais e laboratório de estratégias (`POST /api/analisar`, `POST /api/estrategias`)

* **Otimização com os seus ativos**: o laboratório (`POST /api/analisar`) aceita `fonte_dados: "real"` com 3 a 15
  ativos; baixa 5 anos de cotações (dólar convertido para reais), usa a Selic do BCB como taxa livre de risco (ou a
  informada), inclui o BOVA11 como referência e roda o mesmo motor da demonstração (fronteira, Monte Carlo, VaR/ES,
  EVT, estresse, backtest e validação independente, assistente). A demonstração sintética continua disponível.
  Restrições do perfil inviáveis para os ativos escolhidos (ex.: perfil moderado sem renda fixa) são ditas, nunca
  contornadas em silêncio.
* **Estratégias** (`core/strategy_lab/`): quatro ideias publicadas — momento (Jegadeesh e Titman), baixa
  volatilidade (Baker, Bradley e Wurgler), tendência com média móvel (Faber) e reversão de curto prazo (Jegadeesh) —
  em variantes, sobre até 10 anos de cotações das ações do catálogo do mercado escolhido. Walk-forward mensal com
  custos (+5 pb de derrapagem), escolha da variante só na validação (60% iniciais), teste final nos 40% restantes,
  Sharpe deflacionado considerando todas as variantes, consistência por terços e conferência de que as decisões não
  olham o futuro. Veredito estrito: "há evidência", "evidência fraca" ou "sem evidência".

## 10h. Acompanhamento sob demanda (`POST /api/monitorar`)

O site roda como função sem servidor (sem banco nem agendador): **não existe monitoramento contínuo, e nada é
simulado como se existisse**. Na tese, "Acompanhar esta tese" guarda no aparelho as condições de invalidação com os
seus limites numéricos; no Painel, "Conferir agora" busca os dados naquele momento e confere cada condição
(`core/fundamentals/monitor.py`): as do balanço anual (receita, margem, dívida, dividendos) ficam "aguardando o
próximo balanço" até a fonte publicar um ano fiscal posterior ao da tese; a estimativa de analistas é comparada a
qualquer momento; a condição de preço usa o menor fechamento desde a tese (ou "aguardando novas cotações"). As
previsões registradas são avaliadas junto. O resultado diz quando foi conferido.

## 10i. Competição de investimentos (`POST /api/competicao/ler`, `POST /api/competicao/montar`)

Para competições como a Wharton Global High School Investment Competition, em que a equipe recebe o caso de um
cliente e administra uma carteira virtual com as regras da sua edição (tela **Competição**):

* **Leitura dos textos** (`core/competition/reader.py`, sem modelo de linguagem): o caso do cliente e as regras, em
  inglês ou português, são lidos por padrões fixos; cada item traz a frase exata de onde saiu (conferida por teste) e
  o que não está no texto fica em branco. Do caso: idade, patrimônio, meta, aportes, retiradas, prazo (o mais longo;
  os mais curtos viram metas intermediárias), tolerância a risco (vale o rótulo que o cliente usa; com rótulos
  diferentes, o mais conservador), objetivo, queda aceitável, retorno exigido, liquidez em 12 meses, exclusões
  (combustíveis fósseis, tabaco, armas, álcool, jogos — só com uma palavra de recusa antes do termo, na mesma frase)
  e valores (sustentabilidade, saúde, tecnologia, educação, comunidade, legado). Das regras: capital e moeda, preço
  mínimo por ação, mínimo de ETFs e lista aprovada, peso máximo por ativo, número de posições e de setores, caixa
  máximo, margem e venda a descoberto, limite de operações, tamanho da ordem frente ao volume diário, frações e a
  janela de negociação (datas de inscrição e de relatório são ignoradas). A lista de ETFs aprovados é colada à parte.
* **Montagem** (`core/competition/build.py`): o motor da carteira com propósito (seção 10) com o perfil do cliente,
  cotações reais e as regras como restrições duras — preço mínimo (último fechamento), só ETFs da lista aprovada,
  teto por ativo (vale também para o quase-caixa) e quantas posições, ETFs e setores a carteira precisa ter (escolha
  exata das posições por programação inteira mista, `semicontinuous_weights`). A meta do cliente é reproduzida na
  escala do capital da competição (multiplicar todos os valores pelo mesmo fator não muda a chance de atingi-la); sem
  meta em dinheiro, vale o retorno exigido ou, como hipótese dita, manter o poder de compra. As ordens usam o capital
  inteiro em ações inteiras (frações só se as regras permitirem), e uma posição acima do teto nos preços de compra
  perde ações uma a uma.
* **Conferência regra a regra**, medida nas ordens e nunca suposta: o que o site não consegue medir aparece como
  "conferir"; se uma regra não fecha, a página diz qual.
* **Cliente atendido**: cada necessidade do caso com a citação e a medida da carteira; resultados até o fim das
  negociações e no horizonte do cliente; **rascunho do IPS em inglês** (perfil, objetivo de retorno, tolerância a
  risco, restrições, alocação estratégica, seleção, gestão de risco, monitoramento) e resumo em português, os dois
  com linguagem probabilística verificada; ordens em CSV.

Nada é executado: a equipe digita as ordens no simulador da competição. As regras mudam a cada edição, e o site não
traz regras prontas de nenhuma competição.

## 10j. Mercado hoje (`POST /api/mercado`)

Uma tela para ver o que o mercado está fazendo no dia (tela **Mercado hoje**, a primeira de "Pesquisar"):

* **Índices:** Ibovespa (`^BVSP`), dólar (`USDBRL=X`), S&P 500 (`^GSPC`), Nasdaq (`^IXIC`) e petróleo Brent
  (`BZ=F`) — valor, variação do dia (em % e, nos índices, em pontos), hora do último negócio e o caminho do dia em
  barras de 5 minutos, com o fechamento anterior tracejado. Um índice que a fonte não responder aparece como "sem
  cotação agora"; nada é estimado no lugar.
* **Ações acompanhadas** (30, 60 ou 100; `core/market_watch/overview.py`): no Brasil, as de maior volume financeiro
  no último pregão do arquivo diário da B3 (a mesma listagem da carteira); nos EUA, as maiores por valor de mercado
  no screener da Nasdaq. Sem a listagem, as ações do catálogo do site, dito na tela.
* **Listas:** quantas sobem, caem e ficam estáveis (variação menor que 0,05%), com a média simples (não é o
  Ibovespa); as 10 maiores altas e as 10 maiores quedas; as 10 mais negociadas (ações negociadas no dia × último
  preço, aproximado); e todas as cotações, com busca e ordem. Tocar numa ação abre a análise "Vale a pena?".
* **Cotações** (`core/market_data/quotes.py`): metadados do gráfico de 1 dia do Yahoo Finance (o mesmo endpoint das
  séries diárias, com as mesmas tentativas, ritmo, disjuntor e limite de tempo): último preço, fechamento do pregão
  anterior, hora do último negócio, pregão aberto ou fechado e o atraso que a fonte informa. **Não é tempo real:** na
  B3 o atraso costuma ser de 15 minutos; cada linha mostra a hora do último negócio (horário de Brasília). Fora do
  pregão, os números são os do último pregão. Uma ação sem resposta é contada e deixada de fora.
* **Atualização:** a tela busca ao abrir e, se a opção estiver marcada, a cada 2 minutos enquanto está aberta e
  visível (nada roda em segundo plano depois que você sai). O servidor guarda cada cotação por 60 s e a tela pronta
  pelo mesmo tempo, para muitos acessos não multiplicarem os pedidos à fonte.
* **Sem motivos inventados:** o site mostra os números, não a causa. Uma queda forte do Ibovespa aparece como
  queda, com o tamanho e a hora; para entender o motivo, veja as notícias do dia — notícias não são lidas aqui.

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

