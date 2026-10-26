# Inventário de funcionalidades (auditoria de 22/10/2026)

Estado de cada recurso pedido no plano de redesign e inteligência financeira, verificado no código antes de qualquer
mudança. **Funcional** = implementado e testado; **Parcial** = existe com limitação; **Ausente** = não existe.
A coluna "Mudança" registra o que foi feito e a verificação; a regra é não reescrever o que já funciona.

## Rotas da API (antes da auditoria)

| Rota | O que faz | Dados |
|---|---|---|
| `GET /` | página única do site | — |
| `GET /api/health` | versão e commit publicados | — |
| `POST /api/acao` | "Vale a pena?": veredito por regras, empresa, fundamentos, projeção Monte Carlo | reais (Yahoo chart + quoteSummary, B3) ou sintéticos rotulados |
| `POST /api/acao-do-dia` | uma ação para o valor de hoje, por regras fixas | reais ou sintéticos rotulados |
| `POST /api/proposito` | carteira personalizada com plano de compra em lotes | reais (B3, Nasdaq, Yahoo, BCB) ou sintéticos rotulados |
| `POST /api/analisar` | laboratório: fronteira, Monte Carlo, VaR/ES, EVT, estresse, backtest, validação | **somente sintéticos** |
| `POST /api/perguntar` | analista determinístico baseado em ferramentas (sobre o laboratório) | sintéticos |
| `GET /api/logo/...` | logos das empresas pelo próprio app | Google favicons |

## Funcionalidades

| Funcionalidade | Estado inicial | Onde | Mudança |
|---|---|---|---|
| Análise de uma ação ("Vale a pena?") | Funcional | `core/stock_check/analyzer.py`, `company.py`, `/api/acao` | lógica mantida; só apresentação nova (tela "Análise de ações") |
| Ação do dia | Funcional | `core/stock_check/daily_pick.py`, `/api/acao-do-dia` | lógica mantida; tela "Oportunidades" |
| Carteira com propósito (montar carteira) | Funcional | `core/purpose/*`, `/api/proposito` | lógica mantida; tela "Carteira" |
| Laboratório quantitativo (Markowitz, fronteira, Monte Carlo, VaR/ES, EVT, estresse, backtest, validação) | Parcial: só mercado sintético | `core/portfolio_engine.py`, `core/optimization`, `core/extreme_risk`, `core/backtesting`, `/api/analisar` | lógica mantida; agora também com cotações reais dos ativos escolhidos (3 a 15, Selic do BCB); demonstração sintética mantida e rotulada |
| Assistente de IA | Parcial: só sobre o laboratório sintético; regras determinísticas, sem LLM | `core/ai_analyst/*`, `/api/perguntar` | mantido dentro do Laboratório |
| Logos das empresas | Funcional (mecanismo) | `core/market_data/logos.py` | mantido |
| Fundamentos da empresa (indicadores) | Parcial: um retrato atual (quoteSummary); sem séries de demonstrações, pares ou revisões | `core/stock_check/company.py` | implementado: demonstrações anuais, estimativas e agenda (Yahoo, cache 24 h), razões, crescimento, F-score e múltiplos (`core/fundamentals/`); sem comparação com pares (sem base setorial confiável) |
| Dashboard (patrimônio, evolução, benchmarks, distribuição, alertas, movimentações) | Ausente: não há cadastro de carteira | — | implementado: "Minha carteira" (posições e movimentações no aparelho) + `/api/carteira/painel` (cotações reais, TWR, TIR, CDI/Ibovespa/S&P, alertas) e resumo no Painel; `tests/test_portfolio_analytics.py` |
| Radar de oportunidades com evidências | Parcial: a Ação do dia ordena ações por histórico de preços; sem fundamentos, contra-evidências ou invalidação | `daily_pick.py`, `core/universe/scoring.py` | implementado: aba Radar + `/api/radar` (fatores explicáveis por estilo, evidências com números, contra, armadilha, o que desfaz a posição); Ação do dia mantida na outra aba |
| Previsões probabilísticas | Parcial: projeção com percentis, P(superar a renda fixa) e sensibilidade; sem calibração histórica | `analyzer.py`, `core/simulation` | implementado: tela "Previsões" + `/api/previsoes/prever` (mesmo modelo em forma fechada; quantis, cenários, chances de subir/superar a renda fixa/cair 20%/superar o índice, sensibilidade) com a calibração histórica do modelo no próprio ativo; `tests/test_forecasting.py` |
| Auditoria das próprias previsões | Ausente | — | implementado: registro assinado e encadeado por hash (no aparelho), retroativas marcadas, pendentes até o fim do prazo, avaliação sob pedido com Brier, log score, PIT, cobertura, índice, após custos e grupos (`audit.py`, `/api/previsoes/registrar`, `/api/previsoes/avaliar`) |
| Simulador de futuros empresariais (cenários, valuation) | Ausente | — | implementado: Cenários → Futuros da empresa + `/api/cenarios` (DCF com premissas dos balanços editáveis, 3 cenários, Monte Carlo, tornado, DCF reverso) |
| Efeitos econômicos em cadeia | Ausente (o setor tem motivos gerais de crescimento, sem medição) | `core/stock_check/sectors.py` | implementado: Cenários → Efeitos em cadeia + `/api/cadeia` (cadeia por setor como hipótese ao lado dos betas medidos; confirma/contraria/inconclusivo) |
| Risco avançado da carteira do investidor (dados reais) | Parcial: os motores existem (VaR, CVaR, EVT, estresse, cauda, regimes, concentração) mas só rodam no laboratório sintético; a carteira com propósito mede concentração | `core/extreme_risk`, `core/purpose/concentration.py` | implementado: aba Risco + `/api/carteira/risco` reutilizando os motores (VaR/CVaR/EVT, correlação em estresse, Euler, liquidez, betas HAC, crises reais, choques); testes no mesmo arquivo |
| Detector de armadilhas de valor | Ausente (só regras soltas no "Vale a pena?", ex.: P/L baixo com ressalva) | `company.growth_case` | implementado: "parece barata?" + 11 sinais medidos com F-score (`value_trap.py`), na tese e no radar |
| Tese de investimento estruturada | Parcial: motivos a favor/contra e teses duplas da carteira; sem invalidação, catalisadores datados ou refutação | `company.py`, `core/purpose/report.py` | implementado: aba "Tese e fundamentos" + `/api/tese` (fatos, estimativas, a favor/contra com base, premissas, datas, invalidação medível, contra-tese) |
| Descoberta e validação de estratégias | Parcial: walk-forward com custos, PSR/DSR e correção de testes múltiplos existem para carteiras do laboratório; sem laboratório de hipóteses | `core/backtesting/*` | implementado: Laboratório → Estratégias + `/api/estrategias` (4 hipóteses publicadas, variantes, validação × teste final, Sharpe deflacionado, terços, conferência de olhar o futuro, veredito estrito) |
| Monitoramento contínuo | Ausente; a infraestrutura (função sem servidor, sem banco nem agendador) só atualiza dados quando o usuário pede | — | não simulado; implementado o acompanhamento **sob demanda** (Painel → "Conferir agora": teses marcadas e previsões registradas; `/api/monitorar`) |
| Carteira para competição de investimentos (caso do cliente + regras) | Ausente (pedido de 24/10/2026) | — | implementado: tela "Competição" + `/api/competicao/ler` e `/montar` (leitura do caso e das regras com a frase de origem, carteira do motor com propósito com as regras como restrições duras, ordens em ações inteiras, conferência regra a regra, rascunho do IPS em inglês); `tests/test_competition.py` |
| Personalização do investidor | Funcional na carteira com propósito; ausente como perfil salvo para todas as ferramentas | `core/purpose/personalization.py` | perfil salvo neste aparelho, que preenche os formulários de todas as ferramentas |
| Histórico de análises | Ausente | — | histórico neste aparelho, no painel |
| Configurações | Parcial: só o tema | — | tela "Perfil e dados": perfil, tema, exportar/importar/apagar os dados deste aparelho, fontes e licenças |

## Dependências e riscos identificados

* **Persistência**: o site roda como uma função sem servidor (Vercel) sem banco de dados nem agendador. Tudo o que é do
  investidor fica no navegador dele (`localStorage`), com exportar/importar; não há conta nem sincronização entre aparelhos.
* **Rede neste ambiente de desenvolvimento**: Yahoo, BCB, B3, Nasdaq, SEC e CVM estão bloqueados aqui; integrações novas
  só podem ser testadas com dublês e precisam ser conferidas no site publicado.
* **Limite de tempo por pedido** (~60 s na plataforma): cada recurso novo precisa de orçamento de tempo, cache e limite
  de ativos.
* **Fundamentos**: a fonte disponível sem custo é o Yahoo Finance (quoteSummary com cookie + crumb, já usado em produção);
  cobertura de estimativas de analistas é pequena para ações brasileiras.
