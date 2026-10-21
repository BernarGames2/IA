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

17. **Tickers reais não validados neste ambiente**: o catálogo real (31 tickers: B3, EUA, ouro, cripto) é baixado
    e validado em tempo de execução (≥ 252 observações, exclusão sem proxy) pelo endpoint de gráficos do Yahoo
    (não oficial; termos restringem redistribuição — o site exibe só métricas derivadas), mas o Yahoo e o BCB estão
    bloqueados aqui; o caminho foi testado com dublês. A Selic é a META (o CDI efetivo fica ~0,1 p.p. abaixo) e o
    `^IRX` é rendimento em base desconto (aproximação da taxa anual). A demonstração offline usa um universo **sintético** multimoeda e é
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

### "Vale a pena?" — empresa e motivos (`core/stock_check/company.py`, `sectors.py`)

24. **Fontes de empresa não validadas ao vivo**: o serviço de empresas listadas da B3
    (`GetInitialCompanies`/`GetDetail`, parâmetros em base64) e o `quoteSummary` do Yahoo (exige cookie + crumb;
    não oficial) foram implementados conforme o formato público conhecido e testados com dublês; aqui a rede é
    bloqueada. Se o formato mudar, o item vira "indisponível" — nada é preenchido no lugar.
25. **Base de conhecimento setorial** (`sectors.py`) é curadoria editorial: descreve o modelo de negócio típico e
    fatores gerais do segmento (com regras brasileiras marcadas por país). Não é previsão para a empresa, não
    acompanha notícias e precisa de revisão periódica. Uma empresa pode fugir do padrão do seu segmento.
26. **Regras de fundamentos** usam limites fixos (ex.: dívida/patrimônio > 1,5×, P/L > 35, ROE abaixo da taxa
    livre de risco) sem ajuste por setor, exceto bancos/seguradoras/fundos (margem, dívida e caixa não se aplicam).
    O Yahoo publica crescimento trimestral contra o mesmo trimestre do ano anterior; bases pequenas exageram a
    variação (o texto avisa quando |variação| > 100%). Preços-alvo são opinião de terceiros.
27. Os motivos não entram no veredito, que continua usando só o histórico de preços e a simulação.

### Motor dinâmico da carteira com propósito (`core/purpose/personalization.py`, `core/universe/`, `feasibility.py`, `changes.py`, `concentration.py`)

28. **Regras pessoais são padrões documentados, não regulação** (CVM/suitability fora do escopo): aversão ao risco
    λ = 8/4/2 (conservador/moderado/arrojado), ×1,5 para preservação e ×1,25 para renda; horizonte < 1 ano: ações
    ≤ 10%, imóveis ≤ 5%, caixa ≥ 50%; 1–3 anos: ações ≤ 30%, imóveis ≤ 10%, caixa ≥ 20%; 3–5 anos: ações ≤ 50%,
    renda fixa longa ≤ 20%; sem renda fixa longa abaixo de 3 anos; renda: ≥ 50% em ativos que pagam renda;
    preservação: caixa ≥ 20%; teto por setor 25/35/45% e por moeda estrangeira 30/50/70%; cripto 0/5/10% e só com
    permissão. O horizonte nunca aumenta o risco além do perfil.
29. **Queda tolerada → volatilidade**: teto = D/2 (premissa: a queda máxima histórica de carteiras diversificadas
    costuma ficar entre 2 e 3 vezes a volatilidade anual). É uma aproximação, não uma garantia de perda máxima.
30. **O valor-alvo não aumenta o risco**: se o objetivo pede mais retorno do que o perfil comporta, o relatório mostra
    a diferença (CAGR necessário × esperado, P(objetivo)) e as alternativas (prazo, aportes, alvo); a carteira
    continua dentro do perfil. Por isso mudar só o valor-alvo (ou um horizonte acima de 5 anos) muda a avaliação,
    não a alocação.
31. **Etapa A sem desempenho passado**: a amostra baixada é estratificada por classe e setor, com presença mínima de
    3 nomes por classe, até 30% para temas do motivo e o restante pelos maiores (volume/valor de mercado). Ativos
    fora dessa amostra não são avaliados naquele pedido (o limite de 100 séries por pedido é de tempo/memória).
32. **Etapa B**: Sharpe encolhido para a média da classe (τ = 0,30), duplicata = correlação diária ≥ 0,95 na mesma
    classe, grupo = correlação ≥ 0,70 (ligação média), cota de max(2, candidatos/5) por grupo; pesos dos critérios
    por perfil fixos e documentados em `scoring.py`. O universo é escolhido com dados até hoje: o backtest sobre ele
    é otimista (viés de seleção/sobrevivência — alerta no relatório).
33. **Caixa em reais = CDI diário do Banco Central (SGS 12)**: modelo de Tesouro Selic/CDB/fundo DI de liquidez
    diária (risco de emissor e IR não modelados); não é um ticker de bolsa. Se a série falhar, a carteira sai sem ele
    e o alerta diz isso; um mínimo de caixa que nenhum instrumento disponível consegue cumprir é retirado com aviso
    explícito ("separe X% em caixa por conta própria"), em vez de tornar toda a otimização inviável. O Banco Central
    publica o CDI com 1 dia útil de atraso e o painel usa só as datas comuns a todas as séries: a última data usada
    pode ser o dia útil anterior (a data aparece no relatório).
34. **Seleção pessoal**: a carteira de utilidade é a escolha padrão e só é trocada se outra estratégia a superar na
    validação por mais de 1 erro-padrão da diferença diária pareada (anualizado), medindo pela utilidade realizada
    do investidor; volatilidade realizada na validação até 10% acima do teto é tolerada (erro de estimação). O filtro
    por essa volatilidade só vale quando alguma estratégia escolhível passa nele (senão a escolha cairia em 1/N). Se a
    estratégia escolhida tiver volatilidade estimada acima do teto, os pesos usados são os da carteira mais próxima
    dela (mínimo tracking error) dentro do teto e dos limites — dito nos alertas. Pesos iguais e benchmark são só
    linhas de base.
35. **Tamanho mínimo de compra**: carteira-alvo dimensionada para o que se tem hoje + aportes dos próximos 12 meses
    (ou dos meses que cabem no horizonte); compra mínima padrão R$ 100 (configurável); lotes: B3 ações 1
    (fracionário), FIIs/ETFs 1 cota (inclusive ETFs de cripto, ouro e Tesouro Selic listados na B3), EUA ação inteira
    ou 0,0001 se a corretora permitir, cripto comprada direto 0,00000001, CDI qualquer valor. Só o CDI é "dinheiro":
    ETFs de caixa listados (LFTS11, SHY) são comprados por cota ao preço deles. Regra gulosa (não é otimização
    inteira): mantém as posições que alcançam a compra mínima e, se os limites não fecharem, devolve as próximas
    maiores no mínimo.
36. **Preços para o plano**: último ponto da série diária do Yahoo (pode ser o pregão em andamento), convertido pela
    taxa de câmbio do dia; não é cotação em tempo real — confira na corretora. Custos e impostos de compra/venda
    não entram nas quantidades; a sugestão de "acima do alvo" não manda vender (IR sobre ganho de capital não é
    calculado).
37. **Estabilidade**: só com a recomendação anterior guardada neste aparelho e as MESMAS entradas; o erro-padrão do
    ganho de utilidade é aproximado por TE/√anos de dados (erro de uma média); custo = (corretagem + spread +
    slippage) × giro. Com entradas diferentes, a carteira é recalculada para as novas entradas.
38. **Diversificação real**: apostas independentes = (razão de diversificação)² (Choueifaty et al., 2013); a versão
    por componentes principais (Meucci, 2009) não é usada porque, com volatilidades parecidas, os componentes não
    são únicos. Sobreposição ETF × ação é **estimada pela correlação**: a composição dos ETFs não está nas fontes.
39. **Cobertura declarada**: B3 (ações, FIIs, ETFs negociados no dia de referência), ações dos EUA no screener da
    Nasdaq (NYSE/Nasdaq/AMEX), ETFs americanos do catálogo curado e caixa em CDI. Não cobre outros países, BDRs,
    fundos não listados, CDBs/LCIs específicos nem títulos individuais do Tesouro.
40. **Cache**: séries de preço reutilizadas por até 4 h; resultado do site por até 15 min para o mesmo pedido (o
    resultado informa quando foi calculado; a demonstração sintética não muda e é reaproveitada sem prazo). Mais de
    metade das séries tentadas sem cotação → erro, nunca uma carteira montada com o que sobrou ou um resultado
    antigo. Séries que nem foram tentadas porque o tempo de download do site acabou não contam como falha da fonte:
    a triagem usa as baixadas e o alerta diz quantas ficaram de fora.
41. **Moeda e caixa pela exposição**: para o limite de moeda estrangeira e a diversificação "por moeda", conta a
    moeda que o valor acompanha, não a de cotação: ETFs de índice estrangeiro cotados em reais (IVVB11, NASD11),
    ouro e cripto contam como dólar (país US ou GLOBAL no catálogo; nos ETFs da B3 fora do catálogo, pela regra do
    nome). Instrumento de caixa só conta como caixa na moeda do investidor (SHY não é caixa para quem gasta reais).
    Quem escolhe só mercados estrangeiros (ex.: só EUA com objetivo em reais) não tem o limite de moeda estrangeira
    (ele empurraria o resto para o CDI); a regra aparece no relatório.
42. **Setores**: uma chave por setor em português (`SECTOR_LABEL` em `catalog.py`); setores da Nasdaq são mapeados
    para elas, então o limite por setor junta Brasil e EUA (ex.: TOTS3 + AAPL em "tecnologia"). ETFs setoriais do
    catálogo (XLK, XLE, XLF, XLV, XLU, IBB, TAN) carregam o setor deles; ETFs amplos não são filtrados por setor. Nomes
    digitados são normalizados ("Petróleo e gás", "oil", "bancos"); nome não reconhecido gera aviso (nada é excluído
    em silêncio). Ações da B3 fora do catálogo não têm setor na fonte ("nao_classificado") e ficam sem limite de
    setor (continuam com o limite por ativo e a triagem de correlação).
43. **Histórico mínimo de candidato**: ~3 anos de preços diários (735). Abaixo disso o ativo é medido e listado, mas
    não vira candidato: as estimativas seriam ruidosas, a validação fora da amostra não teria período para julgá-lo
    e uma série curta encurtaria a janela comum de todos os outros. Um ativo da carteira atual nessa situação aparece
    como "fora da análise" com o motivo (o pedido não falha por causa dele).
44. **Triagem (etapa B) — ajustes**: o caixa CDI não entra nas notas comparativas da renda fixa (volatilidade quase
    zero geraria um Sharpe extremo) e é sempre avaliado pelo otimizador; a carteira atual e os grupos exigidos pelas
    suas regras podem passar do limite de candidatos; métrica ausente (ex.: momento) conta como neutra; a governança
    é aplicada antes da triagem (ativo excluído não ocupa vaga nem aparece como alternativa); na deduplicação por
    empresa (BRK-A/BRK-B) fica a classe de ação do catálogo.
45. **Aportes**: o dinheiro de cada aporte nivela as posições mais abaixo do alvo (preenchimento por nível: as
    maiores diferenças diminuem juntas), com a compra mínima por posição; a diferença é medida sobre tudo o que você
    tem, inclusive posições fora da carteira-alvo. A sobra vai para o caixa e o caixa acima do alvo completa compras
    que um aporte sozinho não faz (lote mais caro que o aporte) — o plano mostra o resgate. Sem dinheiro novo e com
    posições acima do alvo, o plano "se vender o excesso" mostra onde o dinheiro da venda entraria (vender é opção,
    não ordem; IR não calculado).
46. **Covariância com séries quase sem oscilação** (CDI): variância mínima de (0,3% a.a.)² e covariância zero para
    séries com volatilidade anual abaixo disso; evita uma matriz quase singular (otimização lenta e instável).
47. **Impressão digital dos parâmetros**: não depende da ordem em que listas e posições foram digitadas nem do número
    de trajetórias simuladas (que só muda a probabilidade do objetivo, não a carteira).
48. **Valores digitados** (site e linha de comando, mesma regra): "1.500,00", "1500,5" e "1,500.00" são aceitos; com
    um só tipo de separador, grupos de exatamente três dígitos ("1.500", "1,500") são milhar. Código de ativo vazio
    é erro. Códigos: PETR4F (fracionário) vira PETR4.SA; B3SA3 e B5P211 são reconhecidos; BRK.B vira BRK-B.
49. **Logos**: ícone que a empresa publica no próprio site, lido por um serviço público de favicons (Google,
    `s2/favicons`) pelo servidor do app — o navegador não fala com terceiros. O domínio vem da lista curada de sites
    oficiais (`B3_SITES` pela raiz do código, `US_SITES` por ticker; fundos e ETFs: a gestora) ou, numa ação
    pesquisada fora dela, do site informado pela B3/Yahoo (plataformas de RI e redes sociais são recusadas, porque o
    ícone seria de outra empresa). Só imagens rasterizadas (PNG, JPEG, GIF, WEBP, ICO; SVG recusado), até 200 kB e
    com lado ≥ 24 px; guardadas em memória por 7 dias (falta: 1 hora) e no cache da CDN. Sem logo: as iniciais do
    código. Os ícones reais não puderam ser vistos neste ambiente (o serviço de favicons é bloqueado pela rede); o
    mecanismo foi verificado com um serviço simulado. Um site oficial pode trocar de domínio: a lista curada precisa
    de revisão periódica. Logos são marcas dos donos, usados só para identificar o ativo.

## Itens pendentes / próximos passos

* Validar o modo online com dados reais (SPY/AGG/GLD e ativos B3) em ambiente com acesso ao Yahoo Finance,
  inclusive o modo propósito com o catálogo real.
* Extração de temas com modelo de linguagem opcional, mantendo a exigência de citação literal e a verificação.
* Calendários oficiais (B3/NYSE).
* Metadados de ativos via provedor e restrições lineares por fator.
* EGARCH, erros-padrão robustos para GARCH, cópulas com assimetria.
* Otimização de ES com cenários simulados; restrições de liquidez no otimizador.
* Tamanho mínimo de compra por otimização inteira (hoje: regra gulosa documentada) e composição real dos ETFs
  para medir sobreposição (hoje: correlação).
* Impostos (IR sobre ganho de capital, come-cotas) no plano de compra e na decisão de vender.
* Backend opcional de LLM (somente sobre saídas de ferramentas, com a mesma verificação de números).
* Exportação PDF.
