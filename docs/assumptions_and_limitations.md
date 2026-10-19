# Premissas, limitações e itens pendentes

## Premissas explícitas

| Premissa | Onde | Observação |
|---|---|---|
| 252 dias úteis/ano | `config.trading_days` | ajustável; cripto negocia 365 dias, mas o painel alinhado usa datas comuns |
| Retornos i.i.d. para anualizar volatilidade (√252) | `core/metrics.py` | hipótese simplificadora; GARCH/regimes mostram que não vale em estresse |
| Taxa livre de risco | `config.py` | se não informada, fallback de demonstração por moeda, sinalizado em todo relatório |
| Inflação 4,5% a.a. | `config.inflation_annual` | **hipótese** usada só para deflacionar patrimônio simulado e definir o objetivo demo |
| Durações por setor (pós-fixado 0,25; inflação 6; agregado 6) | `core/extreme_risk/stress.py` | aproximação para choques de juros |
| Choques de cenário | `default_scenarios()` | hipóteses ilustrativas, não previsões |
| Custos: 10 bps corretagem + 5 bps meio-spread + 5 bps slippage | config | paramétricos; não derivados de dados de book |
| Participação máx. de 10% do volume médio | `liquidity_analysis` | usado para dias de liquidação |
| Perfil por regras configuráveis | `core/profiler.py` | não é suitability regulatório |
| Limites de pesquisa por perfil | `ProfilerSettings.limits` | configurações, não promessas |

## Limitações conhecidas

1. **Modo online não validado com dados reais** (rede bloqueada no ambiente). Conector, cache, retries,
   câmbio e qualidade foram testados com provedores simulados.
2. **Vieses de backtest**: universo atual aplicado ao passado; sem point-in-time, deslistagens nem mudanças
   de índice; eventos corporativos dependem do ajuste do Yahoo. Todo relatório marca o backtest como
   potencialmente enviesado.
3. **Metadados** de classe/setor/país/moeda só para os símbolos de exemplo; demais ficam `unknown` e não entram
   em limites por classe. Moeda desconhecida é assumida como moeda-base **com aviso**.
4. **Calendário**: alinhamento por interseção; feriados e diferenças B3/NYSE/24x7 reduzem a amostra (o número de
   datas descartadas é reportado). Não há calendário oficial de feriados.
5. **Câmbio**: conversão com cotação do provedor carregada no máximo 3 dias corridos; sem custos de conversão.
6. **Liquidez/impacto de mercado**: paramétricos; dados diários não permitem inferir profundidade de book.
7. **Impostos** não modelados.
8. **Fatores**: apenas PCA e fator de mercado construído com um ativo do universo; sem Fama-French, juros,
   inflação, câmbio como séries externas; limites por fator não são restrições do otimizador.
9. **GARCH**: QML gaussiano, média constante, sem erros-padrão dos parâmetros; EGARCH não implementado.
10. **HMM**: univariado (sinal de mercado); covariâncias por estado estimadas por ponderação das probabilidades
    suavizadas (uso retrospectivo, rotulado). Simulação por regime parte das probabilidades filtradas.
11. **EVT**: GPD por máxima verossimilhança do scipy; IC por bootstrap não paramétrico (200 réplicas); teste KS
    aproximado (otimista). Recusa automática com menos de 50 excedências.
12. **Otimização por ES** usa cenários históricos (LP); não há versão com cenários simulados no pipeline.
13. **Purged CV/embargo** não implementados: não há rótulos sobrepostos; usa-se walk-forward cronológico.
14. **Analista de IA** sem LLM: entende apenas intenções por palavras-chave; perguntas fora do escopo recebem
    "não é possível concluir com os dados disponíveis". Pesquisa web não implementada.
15. **PDF** não gerado (opcional na especificação).
16. Dados sintéticos: os parâmetros do gerador são escolhas de projeto; a semente 42 produz uma amostra com
    retornos realizados bem abaixo dos verdadeiros — mantida de propósito para ilustrar erro de estimação.

### Carteira com propósito (`core/purpose/`)

17. **Tickers reais não validados neste ambiente**: o catálogo real (B3, EUA, ouro, cripto) é baixado e validado em
    tempo de execução (≥ 252 observações, exclusão sem proxy), mas o Yahoo Finance está bloqueado aqui; a regra
    foi testada com provedor simulado. A demonstração offline usa um universo **sintético** multimoeda e é
    marcada "NÃO É UMA CARTEIRA FINAL" (alerta FAIL de categoria dados).
18. **Extração de temas por léxico**: determinística e auditável (citação literal de cada evidência, negações
    descartadas), mas não entende sinônimos fora da lista, ironia ou contexto longo. Nenhum modelo de linguagem
    é usado; temas de valor (família, disciplina, superação…) são citados e não pontuados.
19. **Tags temáticas dos ativos** são curadoria editorial estática (`TAG_SOURCE_REAL`), não dado de mercado;
    devem ser revisadas periodicamente. O score de alinhamento mede apenas a aderência do negócio aos temas
    declarados — não é ESG nem avaliação de qualidade.
20. **Moat/vantagem competitiva** é sempre "não avaliado": não há dados fundamentalistas no projeto.
21. **Governança**: o arquivo `data/governance_flags.json` contém apenas um exemplo sintético; alegações sobre
    empresas reais só podem ser adicionadas com fonte citada (entradas sem fonte são rejeitadas).
22. **Simulação do objetivo**: GBM multivariado centrado no retorno esperado [E] do otimizador; o erro padrão
    reportado mede só o ruído do Monte Carlo. A sensibilidade a ±1 erro-padrão do retorno esperado (tabela do
    relatório) costuma mover P(objetivo) dezenas de pontos — é a incerteza dominante.
23. A otimização da carteira agregada é feita na moeda do objetivo (com câmbio) e com a rf dessa moeda; os
    sleeves por moeda (cada um com a própria rf) são calculados antes, para diagnóstico, e não alteram o
    otimizador.

## Itens pendentes / próximos passos

* Validar o modo online com dados reais (SPY/AGG/GLD e ativos B3) em ambiente com acesso ao Yahoo Finance,
  inclusive o modo propósito com o catálogo real.
* Extração de temas com modelo de linguagem opcional, mantendo a exigência de citação literal e a verificação.
* Calendários oficiais (B3/NYSE) e fonte de taxa livre de risco (ex.: CDI/SELIC via API do BCB) por moeda.
* Metadados de ativos via provedor e restrições lineares por fator.
* EGARCH, erros-padrão robustos para GARCH, cópulas com assimetria.
* Otimização de ES com cenários simulados; restrições de liquidez no otimizador.
* Backend opcional de LLM (somente sobre saídas de ferramentas, com a mesma verificação de números).
* Exportação PDF.
