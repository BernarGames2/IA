# Carteira alinhada ao propósito — resultado

> **DEMONSTRAÇÃO COM DADOS SINTÉTICOS — NÃO É UMA CARTEIRA FINAL.** Os tickers `SINT_*` não existem no mercado; os números ilustram o método.

_Ferramenta de pesquisa e simulação. Não é recomendação de investimento, não executa ordens e não oferece garantia de resultado: todos os números são estimativas [E], históricos [H] ou simulações [S] sujeitas a erro de modelo e de estimação._

Entradas: capital BRL 100.000; aporte BRL 2.000 (mensal); objetivo BRL 900.000 em 10 anos; perfil declarado moderado (final após limites: moderado); slider 70% quant / 30% alinhamento; ativos sob revisão excluídos (padrão).

## 1. CAGR necessário vs. CAGR esperado da carteira

| CAGR necessário | CAGR esperado da carteira (mediana simulada [S]) | Diferença | Retorno esperado médio [E] (taxa efetiva a.a., ± 1 EP) |
|---|---|---|---|
| 14.60% | 16.93% | +2.32% | 17.49% ± 4.43% |

CAGR necessário de 14.60% a.a. para levar o capital inicial e os aportes (340.000 aportados no total) a 900.000 em 10 anos. Ambas as taxas da comparação usam a mesma convenção (taxa efetiva anual aplicada ao mesmo capital e aos mesmos aportes): o CAGR esperado é a taxa que leva esse fluxo ao patrimônio MEDIANO simulado, já líquido de custos e do arrasto de volatilidade. O retorno esperado médio [E] é a média estimada pelo otimizador convertida para taxa efetiva; fica acima da mediana porque a média de um processo multiplicativo supera a sua mediana.

## 2. Carteira

| ticker | classe | peso | score_alinhamento | moeda | peso só-quant |
|---|---|---|---|---|---|
| SINT_IPCA | renda_fixa_br | 25.3% | 0.00 | BRL | 30.0% |
| SINT_SELIC | renda_fixa_br | 25.2% | 0.00 | BRL | 30.0% |
| SINT_ENERGIA3 | acoes_br | 22.7% | 0.40 | BRL | 21.1% |
| SINT_SAUDE_US | acoes_eua | 20.1% | 0.40 | USD | 18.7% |
| SINT_SAUDE3 | acoes_br | 3.1% | 0.40 | BRL | 0.0% |
| SINT_TREAS_US | etf_eua | 2.2% | 0.00 | USD | 0.0% |
| SINT_LOGI11 | fii | 1.4% | 0.00 | BRL | 0.0% |

Motivo do score (uma frase por ativo elegível):

- **SINT_IPCA**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_SELIC**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_ENERGIA3**: atende a sustentabilidade ("planeta", "energia limpa") pelo modelo de negócio: [sintético] geração de energia eólica e solar
- **SINT_SAUDE_US**: atende a saúde ("saúde", "câncer") pelo modelo de negócio: [sintético] farmacêutica
- **SINT_SAUDE3**: atende a saúde ("saúde", "câncer") pelo modelo de negócio: [sintético] rede de hospitais e diagnósticos
- **SINT_TREAS_US**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_LOGI11**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_PETRO4**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_EDUCA3**: atende a educação ("faculdade") pelo modelo de negócio: [sintético] escolas e ensino superior
- **SINT_VAREJO3**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_TECH_US**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)
- **SINT_CLEAN_US**: atende a sustentabilidade ("planeta", "energia limpa") pelo modelo de negócio: [sintético] ETF global de energia limpa
- **SINT_OURO_US**: sem conexão com os temas declarados (saúde, educação, sustentabilidade)

## 3. Justificativa por ativo

Porquê declarado (citação literal): "objetivo é pagar a faculdade dos meus filhos"

### SINT_IPCA — Sintético Tesouro IPCA+

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: títulos públicos indexados à inflação. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 17.30%, volatilidade 5.88%, drawdown máximo -4.57%. Retorno esperado usado pelo otimizador [E]: 14.06% (estatística t = 5.35; distinto de zero a ~95%). Na carteira: peso 25.3%, contribuição de 4.8% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de taxa de juros (marcação a mercado) e de inflação implícita.

**(b) Tese de propósito.** Sem conexão temática direta com o que você declarou (saúde, educação, sustentabilidade); score 0,00. O ativo está na carteira pela função financeira descrita na tese financeira, não pelo propósito.

### SINT_SELIC — Sintético Tesouro Selic

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: títulos pós-fixados (caixa). Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 9.86%, volatilidade 0.59%, drawdown máximo -0.13%. Retorno esperado usado pelo otimizador [E]: 9.88% (estatística t = 37.32; distinto de zero a ~95%). Na carteira: peso 25.2%, contribuição de 0.1% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de taxa de juros (marcação a mercado) e de inflação implícita.

**(b) Tese de propósito.** Sem conexão temática direta com o que você declarou (saúde, educação, sustentabilidade); score 0,00. O ativo está na carteira pela função financeira descrita na tese financeira, não pelo propósito.

### SINT_ENERGIA3 — Sintética Energia Renovável

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: geração de energia eólica e solar. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 35.50%, volatilidade 30.61%, drawdown máximo -48.64%. Retorno esperado usado pelo otimizador [E]: 24.28% (estatística t = 1.77; estatisticamente indistinto de zero). Na carteira: peso 22.7%, contribuição de 56.6% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de ações: volatilidade elevada, risco de setor e de execução da empresa.

**(b) Tese de propósito.** Conexão com o seu motivo: sustentabilidade — você escreveu "planeta", "energia limpa". Atividade do ativo: geração de energia eólica e solar. Score de alinhamento 0.40 (fração dos seus temas setoriais atendidos).

### SINT_SAUDE_US — Sintética Saúde EUA

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: farmacêutica. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 25.66%, volatilidade 26.08%, drawdown máximo -32.44%. Retorno esperado usado pelo otimizador [E]: 18.75% (estatística t = 1.61; estatisticamente indistinto de zero). Na carteira: peso 20.1%, contribuição de 31.3% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de ações e cambial (cotado em USD).

**(b) Tese de propósito.** Conexão com o seu motivo: saúde — você escreveu "saúde", "câncer". Atividade do ativo: farmacêutica. Score de alinhamento 0.40 (fração dos seus temas setoriais atendidos).

### SINT_SAUDE3 — Sintética Saúde

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: rede de hospitais e diagnósticos. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 17.13%, volatilidade 32.46%, drawdown máximo -53.12%. Retorno esperado usado pelo otimizador [E]: 13.96% (estatística t = 0.96; estatisticamente indistinto de zero). Na carteira: peso 3.1%, contribuição de 5.4% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de ações: volatilidade elevada, risco de setor e de execução da empresa.

**(b) Tese de propósito.** Conexão com o seu motivo: saúde — você escreveu "saúde", "câncer". Atividade do ativo: rede de hospitais e diagnósticos. Score de alinhamento 0.40 (fração dos seus temas setoriais atendidos).

### SINT_TREAS_US — Sintético Títulos EUA

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: títulos do Tesouro dos EUA. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 6.48%, volatilidade 16.86%, drawdown máximo -33.03%. Retorno esperado usado pelo otimizador [E]: 7.98% (estatística t = 1.06; estatisticamente indistinto de zero). Na carteira: peso 2.2%, contribuição de 0.6% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco de juros dos EUA e cambial (cotado em USD).

**(b) Tese de propósito.** Sem conexão temática direta com o que você declarou (saúde, educação, sustentabilidade); score 0,00. O ativo está na carteira pela função financeira descrita na tese financeira, não pelo propósito.

### SINT_LOGI11 — Sintético FII Logística

**(a) Tese financeira.** [ATIVO SINTÉTICO: métricas de uma série simulada, não de uma empresa real] Modelo de negócio: fundo imobiliário de galpões logísticos. Métrica-chave [H] (2021-03-03 a 2025-12-31, em BRL): retorno médio anual 16.77%, volatilidade 16.87%, drawdown máximo -27.20%. Retorno esperado usado pelo otimizador [E]: 13.76% (estatística t = 1.82; estatisticamente indistinto de zero). Na carteira: peso 1.4%, contribuição de 1.1% do risco total. Moat/vantagem competitiva: não avaliado (exige dados fundamentalistas que esta ferramenta não possui; nada é inferido). Risco principal: risco imobiliário: vacância, inadimplência e sensibilidade a juros.

**(b) Tese de propósito.** Sem conexão temática direta com o que você declarou (saúde, educação, sustentabilidade); score 0,00. O ativo está na carteira pela função financeira descrita na tese financeira, não pelo propósito.


## 4. Cenários no horizonte de 10 anos (segundo a simulação)

| cenário | patrimônio final nominal | real (inflação hipotética) | CAGR equivalente | atinge o objetivo? |
|---|---|---|---|---|
| otimista (P90) | BRL 1.464.864 | BRL 943.266 | 21.80% | sim |
| base (P50) | BRL 1.053.662 | BRL 678.482 | 16.93% | sim |
| pessimista (P10) | BRL 769.068 | BRL 495.224 | 12.28% | não |

No cenário base (mediana), o patrimônio simulado é BRL 1.053.662; em 10% das trajetórias fica abaixo de BRL 769.068. Inflação hipotética usada para valores reais: 4.5% a.a.

## 5. Probabilidade de atingir o objetivo (Monte Carlo)

P(patrimônio final ≥ BRL 900.000) = **73.5%** (erro padrão Monte Carlo 0.44%; IC95% 72.6%–74.4%; 10000 trajetórias; GBM multivariado centrado no retorno esperado [E] do otimizador, rebalanceamento mensal com custos, aportes no fim de cada período).
Carteira só-quantitativa nas mesmas trajetórias: 74.2% (EP 0.44%). P(terminar abaixo do total aportado BRL 340.000) = 0.0%.

Sensibilidade a modelo e a erro de estimativa:

| modelo | P(objetivo) | EP | mediana final | CAGR equivalente mediano |
|---|---|---|---|---|
| GBM, retorno [E] (principal) | 72.7% | 1.00% | BRL 1.049.011 | 16.86% |
| GBM, retorno [E] − 1 erro-padrão | 23.1% | 0.94% | BRL 744.074 | 11.80% |
| GBM, retorno [E] + 1 erro-padrão | 98.2% | 0.29% | BRL 1.499.874 | 22.15% |
| bootstrap em blocos (21d), centrado em [E] | 73.5% | 0.99% | BRL 1.070.583 | 17.16% |
| GBM, deriva histórica | 98.9% | 0.23% | BRL 1.553.239 | 22.67% |

O erro padrão acima mede apenas o ruído do Monte Carlo; a incerteza de modelo e de estimação (tabela) costuma ser bem maior.

## 6. Alertas de validação

- **FAIL** [dados] DADOS SINTÉTICOS: os tickers SINT_* não existem no mercado. Este resultado é uma DEMONSTRAÇÃO do método e NÃO é uma carteira final; para tickers reais rode no modo online com acesso ao provedor.
- **WARNING** [backtest] backtest sobre dados SINTÉTICOS: valida a mecânica, não o desempenho real
- **WARNING** [backtest] universo atual aplicado ao passado e sem dados point-in-time: resultados potencialmente enviesados (survivorship/look-ahead de universo)
- **WARNING** [backtest] viés de sobrevivência/seleção: o catálogo é uma lista estática escolhida hoje; ativos deslistados ou que fracassaram não estão no histórico.
- **WARNING** [concentracao] maior peso 25.3% (SINT_IPCA); N efetivo 4.5; por moeda: BRL 78%, USD 22%; por classe: acoes_br 26%, acoes_eua 20%, etf_eua 2%, fii 1%, renda_fixa_br 51%
- **WARNING** [sensibilidade] P(objetivo) varia de 23.1% a 98.2% quando o retorno esperado da carteira muda ±1 erro-padrão (±4.43% a.a.): resultado SENSÍVEL a erro de estimativa.
- **INFO** [governanca] SINT_BANCO4: sob revisão (excluído por padrão): exemplo hipotético de investigação regulatória (ativo sintético, apenas para demonstrar o filtro) (fonte: exemplo sintético embutido no projeto)
- **INFO** [alinhamento] temas de valor sem correspondência setorial direta (citados na tese, não usados no score): família/legado
- **INFO** [alinhamento] SINT_SAUDE3 entrou pela componente de alinhamento (score 0.40; peso quant ~0), dentro de todos os limites de risco, governança e tracking error
- **INFO** [alinhamento] SINT_LOGI11 (score 0) entrou como contrapeso de risco da componente de alinhamento: mantém a volatilidade e o tracking error dentro dos limites ao abrir espaço para os ativos alinhados
- **INFO** [alinhamento] SINT_TREAS_US (score 0) entrou como contrapeso de risco da componente de alinhamento: mantém a volatilidade e o tracking error dentro dos limites ao abrir espaço para os ativos alinhados
- **INFO** [validacao] validador independente da etapa quantitativa: WARNING
- **INFO** [alinhamento] alinhamento médio ponderado 0.16 (quant) -> 0.18 (final); P(objetivo) 74.2% (quant) -> 73.5% (final), mesmas trajetórias aleatórias.
- **INFO** [estimacao] covariância escolhida fora da amostra: sample

## Apêndice — processo (etapas 1 a 9, nesta ordem)

1. **CAGR necessário**: CAGR necessário de 14.60% a.a. para levar o capital inicial e os aportes (340.000 aportados no total) a 900.000 em 10 anos
2. **Universo elegível**: 14 de 14 tickers com série validada (>= 252 observações); excluídos: nenhum
3. **Sleeves por moeda**: BRL: 9 ativos, rf 10.50% (informada pelo usuário); USD: 5 ativos, rf 4.30% (informada pelo usuário)
4. **Otimização quantitativa**: estratégia 'max_sharpe' escolhida por validação apenas: remove dominadas; ranking por Sharpe excedente; menor complexidade a 1 EP da melhor no walk-forward fora da amostra; perfil final moderado; [E] retorno 15.85%, vol 8.90%
5. **Filtro de governança**: removidos: {'SINT_BANCO4': 'sob revisão (excluído por padrão): exemplo hipotético de investigação regulatória (ativo sintético, apenas para demonstrar o filtro) (fonte: exemplo sintético embutido no projeto)'}
6. **Temas do motivo**: temas: ['saúde', 'educação', 'sustentabilidade', 'família/legado']; confiança alta; porquê: "objetivo é pagar a faculdade dos meus filhos"
7. **Score de alinhamento**: calculado para todos os ativos elegíveis
8. **Combinação quant + alinhamento**: 70% quant / 30% alinhamento; tracking error vs quant 1.48% (orçamento 1.50%); vol 8.90% -> 10.05%; pesos abaixo de 0.5% zerados e renormalizados (limites, teto de volatilidade e tracking error reverificados)
9. **Carteira final com dupla justificativa**: 7 ativos com peso; teses financeira e de propósito separadas

Nota: a etapa 5 (governança) é aplicada como exclusão dura do universo ANTES do otimizador rodar (equivale a impor peso zero); remover depois e renormalizar quebraria os limites do perfil.

### Sleeves por moeda (cada um na própria moeda, com a própria rf) [H]

| moeda | ativos | peso na carteira | rf | fonte da rf | retorno médio nativo | vol nativa | Sharpe nativo |
|---|---|---|---|---|---|---|---|
| BRL | SINT_ENERGIA3, SINT_SAUDE3, SINT_LOGI11, SINT_IPCA, SINT_SELIC | 77.7% | 10.50% | informada pelo usuário | 20.18% | 10.46% | 0.97 |
| USD | SINT_SAUDE_US, SINT_TREAS_US | 22.3% | 4.30% | informada pelo usuário | 21.33% | 18.36% | 0.93 |

A carteira agregada é avaliada na moeda do objetivo (com conversão cambial) e com a rf dessa moeda.

### Estatísticas por ativo na moeda nativa (antes da otimização) [H]

| ticker | moeda | rf da moeda | retorno médio | volatilidade | Sharpe | observações |
|---|---|---|---|---|---|---|
| SINT_ENERGIA3 | BRL | 10.50% | 35.50% | 30.61% | 0.83 | 1260 |
| SINT_BANCO4 | BRL | 10.50% | 11.14% | 32.75% | 0.04 | 1260 |
| SINT_PETRO4 | BRL | 10.50% | -5.38% | 40.84% | -0.38 | 1260 |
| SINT_SAUDE3 | BRL | 10.50% | 17.13% | 32.46% | 0.22 | 1260 |
| SINT_EDUCA3 | BRL | 10.50% | 18.69% | 46.04% | 0.19 | 1260 |
| SINT_VAREJO3 | BRL | 10.50% | 28.68% | 47.98% | 0.39 | 1260 |
| SINT_LOGI11 | BRL | 10.50% | 16.77% | 16.87% | 0.40 | 1260 |
| SINT_IPCA | BRL | 10.50% | 17.30% | 5.88% | 1.24 | 1260 |
| SINT_SELIC | BRL | 10.50% | 9.86% | 0.59% | -0.21 | 1260 |
| SINT_TECH_US | USD | 4.30% | 26.48% | 34.13% | 0.65 | 1260 |
| SINT_SAUDE_US | USD | 4.30% | 23.24% | 20.47% | 0.93 | 1260 |
| SINT_CLEAN_US | USD | 4.30% | 11.88% | 39.05% | 0.20 | 1260 |
| SINT_TREAS_US | USD | 4.30% | 4.12% | 6.14% | -0.01 | 1260 |
| SINT_OURO_US | USD | 4.30% | 14.12% | 18.45% | 0.54 | 1260 |

### Extração de temas (determinística, sem modelo de linguagem)

Confiança: **alta**.
- saude (ambos): evidências literais 'saúde', 'câncer'
- educacao (ambos): evidências literais 'faculdade'
- sustentabilidade (ambos): evidências literais 'planeta', 'energia limpa'
- familia_legado (valor): evidências literais 'filhos'

Universo: origem SINTÉTICA (demonstração offline); período 2021-03-03 a 2025-12-31; 14 validados de 14 candidatos.
