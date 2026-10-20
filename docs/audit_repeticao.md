# Auditoria: por que a carteira se repetia

Data: 19/10/2026. Escopo: o caminho que gera carteiras com dados reais — "Carteira com propósito"
(`core/purpose/engine.run_purpose`, `POST /api/proposito`) e o motor quantitativo que ele chama
(`core/portfolio_engine.run_analysis`). A "Análise geral" do site usa um mercado **sintético** fixo de
propósito (demonstração rotulada) e não gera recomendações reais.

## Método

Execuções controladas do caminho de dados reais com dublês de teste (listagens da B3 e dos EUA e preços
com parâmetros diferentes por classe de ativo), variando **uma** entrada por vez. Script reproduzível:
`python scripts/audit_repetition.py` (resultado completo em JSON no diretório de saída).

## Evidência (antes da correção)

| Variação (partindo de moderado, 10 anos, R$ 100 mil, alvo R$ 500 mil) | Estratégia escolhida | Candidatos iguais? | Carteira igual? |
|---|---|---|---|
| perfil conservador | pesos iguais | sim | não (só pelos limites) |
| perfil arrojado | pesos iguais | sim | não (só pelos limites) |
| horizonte 30 anos | pesos iguais | sim | **sim** |
| horizonte 6 anos | pesos iguais | sim | **sim** |
| capital R$ 1.000 | pesos iguais | sim | **sim** |
| capital R$ 5 milhões | pesos iguais | sim | **sim** |
| alvo R$ 5 milhões | pesos iguais | sim | **sim** |
| outro conjunto de preços de mercado | pesos iguais | sim | **sim** |
| mais um conjunto de preços | pesos iguais | sim | **sim** |

A carteira-base era 5% ou 3% em cada um dos 28 ativos (pesos iguais projetados nos limites do perfil),
**inclusive 5% em BTC-USD** sem que o investidor tivesse permitido cripto.

## Causas, em ordem de impacto

1. **Regra de desempate que favorece pesos iguais** — `core/model_validation/model_selection.py:101` com a
   complexidade da linha 27: entre as estratégias a até 1 erro-padrão da melhor na validação, escolhe-se a
   "menos complexa", e pesos iguais têm complexidade 0. Com ~2 anos de validação o erro-padrão do Sharpe é
   ~0,9 ao ano, então quase tudo empata e **pesos iguais vencem**. Pesos iguais não usam dado de mercado:
   a carteira passava a depender só da lista de candidatos e dos limites.
2. **Lista fixa de ativos sempre incluída** — `core/universe/screen.py:28` (`CORE`: IMAB11, B5P211, IRFM11,
   BOVA11, IVVB11, SPY, AGG, SHY, TIP, GLD) entra primeiro em toda carteira (linha 155).
3. **Pré-seleção só por tamanho** — o restante dos candidatos era escolhido por volume/valor de mercado
   (os mesmos maiores nomes sempre), sem perfil, objetivo, horizonte, liquidez necessária ou qualidade de
   dados; ETFs fora do catálogo curado eram descartados.
4. **Entradas do investidor que não chegavam ao otimizador** — horizonte (só limitava o perfil abaixo de 5
   anos, `core/profiler.py:35-36`), capital e valor-alvo só entravam na simulação da meta; tolerância a
   perdas, necessidade de liquidez, objetivo (renda/preservação/crescimento), países, setores e ativos
   excluídos, permissão para cripto e carteira atual não existiam no pedido. O perfil virava três valores
   fixos (`core/purpose/engine.py:221`).
5. **Cripto sem permissão** — o limite moderado admite 5% em cripto (`core/profiler.py:50`), e a projeção de
   pesos iguais preenchia esse limite.
6. **Cache por um dia inteiro** — `webapp/server.py:109-112`: o mesmo pedido devolvia o resultado guardado
   até a virada do dia (UTC), mesmo com novos fechamentos.
7. **Carteira antiga visível depois de um erro** — `webapp/static/index.html:967`: quando um novo pedido
   falhava, só a linha de status mudava e a carteira anterior continuava na tela.
8. **Viabilidade ignorada** — a alocação era apresentada só em percentuais e valores teóricos, sem lotes,
   preços, quantidade comprável, posição mínima ou caixa residual.
9. **Sobreposição não tratada** — ETFs do mesmo índice (ex.: SPY e IVVB11, BOVA11 e ações do Ibovespa) e
   ativos que respondem aos mesmos choques eram contados como diversificação.

## O que foi alterado

| Causa | Correção | Onde |
|---|---|---|
| 1. desempate a favor de pesos iguais | com um investidor, as estratégias são comparadas pela **utilidade realizada dele** na validação (retorno − λ/2 × variância); a carteira de utilidade pessoal é a padrão e só é trocada se outra a superar por mais de 1 erro-padrão da diferença; pesos iguais e benchmark viram linhas de base (comparadas, nunca escolhidas) | `core/model_validation/model_selection.py`, `core/portfolio_engine.py` (`OptimizationPolicy`, `REFERENCE_ONLY`), `core/optimization/optimizers.max_utility` |
| 2. lista fixa `CORE` | removida; nenhum ticker é forçado; cada classe tem presença mínima pelo tamanho/liquidez | `core/universe/screen.py` |
| 3. pré-seleção só por tamanho | etapa A (preferências, classe identificada, liquidez, uma classe de ação por empresa, amostra estratificada) + etapa B (risco, qualidade, liquidez, correlação, duplicatas, cobertura do que os limites exigem), medida no histórico de cada ativo | `core/universe/screen.py`, `core/universe/scoring.py` |
| 4. entradas que não chegavam ao otimizador | objetivo, queda tolerada, liquidez, mercados, exclusões, cripto, carteira atual, frações e compra mínima viram restrições, aversão ao risco ou o plano de compra; o que não for informado vira hipótese escrita | `core/purpose/personalization.py`, `PurposeRequest`, site e CLI |
| 5. cripto sem permissão | fora, a menos que o investidor permita | `personalization.py` |
| 6. cache do dia inteiro | resultado reaproveitado por até 15 min (com a hora do cálculo); cotações por até 4 h | `webapp/server.py`, `core/purpose/engine.py` |
| 7. carteira antiga após erro | o pedido novo esconde a carteira na tela; falha de dados → erro, nunca carteira antiga | `index.html`, `engine.run_purpose` |
| 8. viabilidade ignorada | carteira-alvo para hoje + 12 meses de aportes; posições menores que uma compra mínima saem; plano por lotes, sobra, próximos aportes | `core/purpose/feasibility.py`, `engine._make_executable` |
| 9. sobreposição | duplicatas por correlação na triagem; apostas independentes, grupos e exposição em risco no resultado | `scoring.py`, `core/purpose/concentration.py` |
| — "novidade" artificial | mudanças < 1 p.p. contam como mantidas; com as mesmas entradas, a recomendação anterior só é trocada se a nova a superar por mais que o erro de estimativa + o custo | `core/purpose/changes.py` |

## Evidência (depois da correção)

Mesmas variações, mesmo script (`python scripts/audit_repetition.py`), mercado de teste em que as famílias de ativos
têm retorno e risco diferentes, mas ativos da mesma família têm o mesmo retorno verdadeiro (quem "ganhou" no passado
ganhou por sorte):

| Variação | Estratégia | Candidatos iguais à base? | Carteira igual à base? | Posições | Volatilidade |
|---|---|---|---|---|---|
| base (moderado, 10 anos, R$ 100 mil, alvo R$ 500 mil) | paridade de risco | — | — | 30 | 4,5% |
| perfil conservador | utilidade pessoal | não | não | 9 | 4,0% |
| perfil arrojado | utilidade pessoal | não | não | 9 | 11,6% |
| horizonte 30 anos | paridade de risco | sim | **sim** (justificado) | 30 | 4,5% |
| horizonte 6 anos | paridade de risco | sim | **sim** (justificado) | 30 | 4,5% |
| horizonte 2 anos | utilidade pessoal | não | não | 8 | 5,6% |
| capital R$ 1.000 | paridade de risco | sim | não | 16 | 3,3% |
| capital R$ 5 milhões | paridade de risco | sim | **sim** (justificado) | 30 | 4,5% |
| alvo R$ 5 milhões | paridade de risco | sim | **sim** (justificado) | 30 | 4,5% |
| objetivo renda | paridade de risco | não | não | 30 | 4,3% |
| só Brasil | utilidade pessoal | não | não | 10 | 7,5% |
| outro conjunto de preços de mercado | utilidade pessoal | não | não | 12 | 7,2% |
| mais um conjunto de preços | paridade de risco | não | não | 30 | 5,1% |

Leitura: perfil, prazo curto, objetivo, mercados, capital pequeno e dados novos mudam candidatos e/ou carteira.
Prazo acima de 5 anos, capital grande e valor-alvo **não** mudam a alocação, de propósito: não alteram nenhum limite
de risco nem o que dá para comprar (o valor-alvo muda a avaliação — CAGR necessário, P(objetivo) — e não aumenta o
risco além do perfil). Quando a estratégia é a paridade de risco, é porque a carteira de utilidade pessoal perdeu
para ela na validação fora da amostra por mais de 1 erro-padrão: neste mercado de teste os "vencedores" do passado
são sorte, e a otimização que confia nas médias históricas é punida — exatamente o caso que a regra deve detectar.

Os testes que comprovam cada comportamento exigido estão em `tests/test_dynamic_portfolio.py`
(ver `docs/test_results.md`, seção 11).
