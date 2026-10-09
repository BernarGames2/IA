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
python -m pytest                       # suíte completa (offline, determinística; ~1,5 min)
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
de IA. **O site usa apenas os dados sintéticos** (os termos do Yahoo restringem redistribuir cotações num site
público, e o modo offline é o caminho validado).

| Endpoint | Função |
|---|---|
| `GET /` | página |
| `POST /api/analisar` | roda a análise (modo `rapido` ou `completo`) e devolve um resumo JSON |
| `POST /api/perguntar` | pergunta ao analista de IA (mesmas recusas e verificação de números da CLI) |
| `GET /api/health`, `GET /api/docs` | saúde e documentação OpenAPI |

Entradas são validadas e limitadas (≤ 20.000 simulações, horizonte ≤ 30 anos, pergunta ≤ 500 caracteres).

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
