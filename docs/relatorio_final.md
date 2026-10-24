# Relatório final — redesign premium e inteligência financeira

Data: 24/10/2026. Branch: `claude/serene-wright-3bkyx5`. Este relatório cobre o pedido "PROMPT MASTER DEFINITIVO"
(redesign completo do site e auditoria/implementação da inteligência financeira). Ele separa o que foi verificado do
que não pôde ser: **neste ambiente de desenvolvimento, Yahoo Finance, Banco Central, B3, SEC e CVM estão bloqueados
pela rede**, então todas as integrações novas foram testadas com dublês (respostas no mesmo formato das fontes) e
precisam ser conferidas no site publicado.

## 1. Funcionalidades

| Funcionalidade | Estado inicial | Alteração realizada | Teste ou verificação |
|---|---|---|---|
| Identidade visual, navegação e telas | Página única longa, sem navegação por telas | Redesign: azul-marinho/grafite, verde/vermelho, acento azul, fonte Inter servida pelo próprio app, ícones Lucide, menu lateral, barra inferior no celular, telas e abas, tema claro/escuro, gráficos com o mesmo padrão | Testes da página sem alteração de lógica; navegador a 1440, 1024 e 390 px sem erros de JavaScript nem rolagem horizontal (`test_results.md` §16) |
| "Vale a pena?" (análise de uma ação) | Funcional | Lógica mantida; nova apresentação; botões para Previsão, Tese e Cenários do mesmo ativo | Testes existentes passam; navegador |
| Ação do dia | Funcional | Lógica mantida; agora na aba "Ação do dia" de Oportunidades (o link antigo abre a aba) | Testes existentes; navegador |
| Montar carteira com propósito | Funcional | Lógica mantida; aba "Montar carteira" | Testes existentes; navegador |
| Valores pequenos demais | Erro genérico ("ERRO") | Recusa clara com o menor valor viável; só posições compráveis | 12 testes (`test_small_amounts.py`) |
| Painel / dashboard da carteira | Ausente | Minha carteira: posições e movimentações no aparelho; patrimônio, evolução, TWR e TIR, CDI/Ibovespa/S&P, distribuição, alertas; resumo no Painel | 8 testes (`test_portfolio_analytics.py`), navegador |
| Risco avançado | Motores só no laboratório sintético | Aba Risco: VaR/CVaR (histórico, normal, EVT), drawdown, correlação em estresse, contribuições (Euler), liquidez, betas HAC, crises reais, choques | Mesmo arquivo de testes; navegador |
| Previsões probabilísticas | Projeção sem calibração | Tela Previsões: distribuição, cenários P10/P50/P90, chances (subir, superar a renda fixa, cair 20%, superar o índice), sensibilidade, calibração histórica sem olhar o futuro | 9 testes (`test_forecasting.py`), navegador |
| Auditoria das próprias previsões | Ausente | Registro assinado (HMAC com `QPI_AUDIT_KEY`) e encadeado por hash; retroativas fora das notas; pendentes até o fim do prazo; avaliação sob pedido (Brier, log score, PIT, cobertura, índice, após custos) | Mesmos testes; navegador com entradas antigas e adulteradas |
| Fundamentos ano a ano | Retrato único | Demonstrações anuais, estimativas e agenda; razões, crescimento, F-score de Piotroski, múltiplos | 9 testes (`test_fundamentals.py`) |
| Detector de armadilhas de valor | Ausente | "Parece barata?" + 11 sinais medidos, nível e regras | Mesmos testes |
| Tese estruturada | Motivos soltos | Fatos, estimativas, a favor/contra com base, premissas, datas, invalidação medível, contra-tese, qualidade dos dados | Mesmos testes; navegador |
| Radar de oportunidades | Ação do dia só por preço | Fatores explicáveis por estilo, evidências com números, contra, armadilha, o que desfaz | Mesmos testes; navegador |
| Simulador de futuros da empresa | Ausente | Fluxo de caixa ao acionista com premissas dos balanços editáveis, 3 cenários, Monte Carlo, tornado, DCF reverso | 7 testes (`test_scenarios.py`), navegador |
| Efeitos econômicos em cadeia | Ausente | Juros, dólar, petróleo, bolsa e juros dos EUA: cadeia por setor (hipótese) ao lado dos betas medidos | Mesmos testes (betas plantados recuperados) |
| Laboratório quantitativo | Só dados sintéticos | Também com cotações reais dos ativos escolhidos (Selic do BCB); a demonstração sintética continua | 5 testes (`test_strategy_lab.py`), navegador |
| Descoberta e validação de estratégias | Ferramentas soltas | Laboratório de estratégias: 4 hipóteses publicadas, variantes, validação × teste final, Sharpe deflacionado, terços, conferência de olhar o futuro, veredito estrito | Mesmos testes (sinal plantado encontrado; sem sinal não vira evidência) |
| Acompanhamento | Ausente (sem servidor contínuo) | Acompanhamento **sob demanda**: teses marcadas e previsões registradas conferidas quando o investidor pede; nada roda em segundo plano | 4 testes (`test_monitor.py`), navegador |
| Personalização | Só na carteira com propósito | Perfil salvo no aparelho preenche os formulários; histórico de análises; exportar/importar/apagar dados | Navegador |
| Logos das empresas | Funcional | Mantido em todas as telas novas | Testes existentes |

## 2. Testes executados

Suíte automatizada completa (`python -m pytest -o addopts="" -q -rs`): **518 aprovados, 1 pulado** (o teste online,
desativado porque a rede do ambiente bloqueia as fontes), sem falhas; `ruff` sem avisos nos módulos novos. Testes
novos desta entrega: `test_small_amounts.py` (12), `test_portfolio_analytics.py` (8), `test_forecasting.py` (9),
`test_fundamentals.py` (9), `test_scenarios.py` (7), `test_strategy_lab.py` (5), `test_monitor.py` (4). As seções 15 a
22 de `docs/test_results.md` detalham cada verificação, inclusive no navegador (Playwright, 1440 e 390 px). Um teste
antigo foi atualizado: o texto de `/api/health`, que dizia que o laboratório era só sintético.

## 3. Páginas e componentes

* **Páginas/telas modificadas ou novas**: Painel (resumo da carteira, acompanhamento, atalhos), Análise de ações
  (abas Vale a pena? / Tese e fundamentos), Oportunidades (abas Radar / Ação do dia), Previsões (abas Prever /
  Registro e acertos), Cenários (abas Futuros da empresa / Efeitos em cadeia), Carteira (abas Minha carteira / Risco /
  Montar carteira), Laboratório (abas Otimização / Estratégias), Perfil e dados.
* **Componentes novos**: sistema de abas acessível, cartões de indicadores, gráficos SVG próprios (linhas com marca,
  distribuição, confiabilidade, PIT, barras por ano, tornado, efeitos por setor, barras de fatores), editores de
  posições/movimentações/premissas, listas de registro e de acompanhamento.
* **Reutilizados**: motores de otimização, risco extremo, backtest e sobreajuste; cliente do Yahoo com cookie e
  *crumb*; resolvedor de ativos; catálogo; base de conhecimento por segmento; filtro de linguagem de garantia.

## 4. Integrações

* Funcionando no código e testadas com dublês: Yahoo Finance (cotações, indicadores, demonstrações, estimativas,
  agenda), Banco Central (Selic SGS 432 e CDI SGS 12), B3 (empresas listadas), favicons para logos.
* **Não verificadas contra as fontes reais neste ambiente** (rede bloqueada): conferir no site publicado.

## 5. Dependências externas pendentes

* **`QPI_AUDIT_KEY`** precisa ser configurada nas variáveis de ambiente da Vercel para as previsões saírem assinadas
  pelo servidor; sem ela, o registro funciona, mas só com hashes (o resultado diz isso).
* O formato do `fundamentals-timeseries` do Yahoo não é oficial: se mudar, as telas mostram "indisponível" (nada é
  inventado), e o leitor precisa de ajuste.
* Cobertura de estimativas de analistas para ações brasileiras é pequena na fonte gratuita.

## 6. Problemas não resolvidos e limitações

* Monitoramento contínuo não existe (a plataforma é uma função sem servidor, sem banco nem agendador): o
  acompanhamento é sob demanda, e isso é dito na tela.
* O radar e o laboratório de estratégias usam o catálogo curado (165 ações), não a bolsa inteira, e o universo de hoje
  aplicado ao passado (viés de sobrevivência, dito no resultado).
* "Parece barata" não compara com o setor (sem base setorial confiável gratuita).
* Lint: há avisos antigos do `ruff` em módulos que não foram tocados (ex.: `core/black_swan.py`); os módulos novos
  passam sem avisos.

## 7. Próximos passos sugeridos

1. Conferir todas as telas no site publicado com dados reais (sobretudo demonstrações e estimativas do Yahoo).
2. Configurar `QPI_AUDIT_KEY` na Vercel.
3. Base setorial (medianas de múltiplos por setor) para "parece barata" relativo.
4. Calendário oficial de pregões (B3/NYSE) e dados *point-in-time* para reduzir o viés de sobrevivência dos testes.
