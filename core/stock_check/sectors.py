"""Segment knowledge base: how a business in each segment makes money and the GENERAL reasons
that tend to help or hold back its growth.

This is editorial, documented knowledge about segments (e.g. drugstores, banks, sanitation),
not a forecast for any company and not market data. Every bullet is written in probabilistic
language and is shown with the label "setor". Bullets scoped "BR" or "US" only apply to
companies of that country (Brazilian regulation such as CMED, ANS, ANEEL, FIES/MEC).

Classification (``classify_segment``) reads, in order of specificity, the catalogue one-liner,
the B3 registered activity and sector classification, and the Yahoo/Nasdaq industry and
sector names (Portuguese and English keywords, accent-insensitive). In a text, the segment
whose keyword appears FIRST wins ("banco de varejo, atacado e seguros" -> banks), ties go to
the rule listed first; generic sector names are only used when nothing specific matches.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from core.purpose.themes import _fold

Bullet = tuple[str, str]          # (text, scope: "*" | "BR" | "US")


@dataclass(frozen=True)
class Segment:
    key: str
    rotulo: str
    como_ganha: str
    cresce: tuple[Bullet, ...]
    limita: tuple[Bullet, ...]
    financeiro: bool = False      # margin/leverage rules do not apply (banks, insurers, funds)


def _seg(key: str, rotulo: str, como_ganha: str, cresce: tuple[Bullet, ...], limita: tuple[Bullet, ...],
         financeiro: bool = False) -> Segment:
    return Segment(key, rotulo, como_ganha, cresce, limita, financeiro)


_ALL = "*"

SEGMENTS: dict[str, Segment] = {s.key: s for s in (
    _seg("farmacias", "Farmácias e drogarias",
         "Compra medicamentos e produtos de higiene e beleza de laboratórios e distribuidores e revende em lojas e "
         "canais digitais; o lucro vem da margem de revenda multiplicada pelo volume. Os medicamentos têm preço "
         "máximo regulado; higiene, beleza e dermocosméticos costumam ter margem maior.",
         ((("Envelhecimento da população: pessoas mais velhas tendem a usar mais medicamentos de uso contínuo, o que "
           "sustenta a demanda por anos."), _ALL),
          (("Demanda pouco sensível a crises: remédio é gasto essencial, então as vendas costumam oscilar menos que as "
           "do varejo em geral."), _ALL),
          (("Abertura de lojas e ganho de escala: redes maiores tendem a comprar melhor dos fornecedores e a diluir "
           "custos fixos."), _ALL),
          (("Higiene, beleza e dermocosméticos (inclusive marcas próprias) têm margem maior e podem elevar a "
           "rentabilidade se ganharem espaço nas vendas."), _ALL),
          ("Vendas digitais e serviços na loja (vacinação, testes rápidos) podem aumentar a frequência de compra.",
           _ALL)),
         ((("Concorrência intensa entre grandes redes, farmácias independentes e comércio eletrônico pressiona preços "
           "e margens."), _ALL),
          ("Preço dos medicamentos controlado: o reajuste anual autorizado pela CMED limita o repasse de custos.",
           "BR"),
          (("Lojas novas levam alguns anos para amadurecer: expansão acelerada pode reduzir a rentabilidade no curto "
           "prazo."), _ALL),
          ("Juros altos encarecem o estoque e a expansão, financiados com capital de giro e dívida.", _ALL))),
    _seg("farmaceutica", "Indústria farmacêutica e biotecnologia",
         "Pesquisa, fabrica e vende medicamentos; o lucro vem de marcas fortes e de patentes, que dão exclusividade "
         "temporária, e da escala de produção e distribuição.",
         (("Lançamentos de medicamentos novos podem ampliar a receita, principalmente em doenças crônicas.", _ALL),
          ("Envelhecimento da população aumenta a demanda por tratamentos.", _ALL),
          ("Patentes e marcas fortes costumam sustentar margens altas enquanto duram.", _ALL)),
         (("Fim de patentes abre espaço para genéricos e costuma derrubar o preço do medicamento.", _ALL),
          ("Pesquisa cara e incerta: muitos medicamentos falham nos testes clínicos.", _ALL),
          (("Preços regulados ou negociados com governos (CMED no Brasil; negociações de preços nos EUA) limitam "
           "reajustes."), _ALL),
          ("Riscos de processos judiciais e de problemas de qualidade na fabricação.", _ALL))),
    _seg("hospitais", "Hospitais e serviços médicos",
         "Recebe de planos de saúde e de pacientes particulares por internações, cirurgias, exames e tratamentos; a "
         "ocupação dos leitos e os valores pagos pelos planos definem o resultado.",
         (("Envelhecimento da população e maior uso de serviços médicos.", _ALL),
          ("Novos hospitais e aquisições podem ampliar a escala da rede.", _ALL),
          ("Tratamentos de alta complexidade (oncologia, cardiologia) costumam gerar mais receita por paciente.",
           _ALL)),
         ((("Dependência dos planos de saúde, que pressionam preços, contestam contas (glosas) e podem atrasar "
           "pagamentos."), _ALL),
          ("Custos com pessoal e insumos médicos tendem a subir com a inflação.", _ALL),
          ("Expansão exige muito investimento e dívida, o que pesa com juros altos.", _ALL))),
    _seg("planos_saude", "Planos de saúde",
         "Recebe mensalidades de pessoas e empresas e paga as despesas médicas de quem usa; o lucro depende da "
         "diferença entre mensalidades e custos médicos (sinistralidade).",
         (("Emprego formal em alta tende a aumentar a venda de planos empresariais.", _ALL),
          ("Rede própria de hospitais e clínicas (verticalização) pode reduzir o custo médico.", _ALL),
          ("Reajustes de mensalidades acima da inflação médica melhoram a margem.", _ALL)),
         (("O uso dos planos (sinistralidade) pode subir mais que as mensalidades.", _ALL),
          ("Reajuste dos planos individuais é limitado pela ANS, e decisões judiciais podem ampliar coberturas.",
           "BR"),
          ("Desemprego reduz a base de clientes dos planos empresariais.", _ALL),
          ("Regulação e pressão política sobre preços e coberturas.", _ALL))),
    _seg("dispositivos_medicos", "Equipamentos e dispositivos médicos",
         "Fabrica equipamentos, dispositivos e insumos médicos vendidos a hospitais, laboratórios e clínicas; parte "
         "da receita vem de insumos e serviços recorrentes.",
         (("Envelhecimento da população e mais cirurgias e exames.", _ALL),
          ("Inovação (robótica, monitoramento) e venda recorrente de insumos para os equipamentos já instalados.",
           _ALL)),
         (("Aprovação regulatória demorada e risco de recall de produtos.", _ALL),
          ("Hospitais e governos pressionam preços.", _ALL),
          ("Concorrência e mudanças tecnológicas rápidas.", _ALL))),
    _seg("diagnosticos", "Medicina diagnóstica (laboratórios e exames)",
         "Cobra por exames laboratoriais e de imagem, pagos principalmente por planos de saúde e, em parte, por "
         "pacientes particulares.",
         (("Mais exames preventivos e envelhecimento da população.", _ALL),
          ("Exames de maior valor (genética, imagem avançada) e expansão de unidades.", _ALL),
          ("Consolidação do setor pode trazer ganhos de escala.", _ALL)),
         (("Planos de saúde pressionam os preços dos exames.", _ALL),
          ("Laboratórios próprios dos planos (verticalização) disputam os mesmos pacientes.", _ALL),
          ("Equipamentos caros e custos fixos altos.", _ALL))),
    _seg("saude_geral", "Saúde",
         "Atua em saúde (serviços, produtos ou planos); a receita costuma depender de planos de saúde, governos ou "
         "consumidores finais.",
         (("Envelhecimento da população e maior uso de serviços de saúde.", _ALL),
          ("Demanda pouco sensível a crises.", _ALL)),
         (("Regulação de preços e de coberturas.", _ALL),
          ("Custos médicos tendem a subir acima da inflação.", _ALL))),
    _seg("holding", "Holdings (participações em outras empresas)",
         "Tem participações em outras empresas; o resultado vem dos lucros e dividendos dessas empresas e da "
         "valorização das participações.",
         (("Acompanha o crescimento das empresas investidas.", _ALL),
          (("Holdings costumam negociar com desconto sobre o valor das participações; se o desconto diminuir, a ação "
           "pode se valorizar."), _ALL)),
         (("O desconto de holding pode persistir por muitos anos.", _ALL),
          ("Custos da estrutura, impostos e eventual dívida da holding reduzem o retorno.", _ALL),
          ("Concentração: o resultado depende de poucas empresas investidas.", _ALL)), financeiro=True),
    _seg("gestao_bolsa", "Bolsa, bancos de investimento e gestão de recursos",
         "Cobra taxas sobre negociações, emissões de ações e títulos e sobre o patrimônio administrado; a receita "
         "acompanha o volume negociado e o valor dos ativos.",
         (("Mais pessoas investindo e migração da poupança para o mercado de capitais.", _ALL),
          ("Mercados em alta aumentam o patrimônio administrado e as taxas cobradas.", _ALL),
          ("Ofertas de ações e de títulos de dívida geram comissões.", _ALL)),
         (("A receita cai quando os mercados caem ou o volume de negócios diminui.", _ALL),
          ("Concorrência e pressão por taxas menores (fundos de índice baratos).", _ALL),
          ("Com juros altos, investidores tendem a preferir a renda fixa, o que reduz o volume na bolsa.", "BR")),
         financeiro=True),
    _seg("pagamentos", "Pagamentos e cartões",
         "Cobra uma pequena taxa sobre cada transação processada na sua rede ou plataforma; o volume de compras "
         "define a receita.",
         (("Troca do dinheiro em espécie por pagamentos digitais.", _ALL),
          ("Comércio eletrônico em expansão.", _ALL),
          ("Viagens internacionais geram taxas maiores por transação.", _ALL)),
         (("Regulação das tarifas cobradas e processos antitruste.", _ALL),
          ("Concorrência de novos meios de pagamento (Pix no Brasil, carteiras digitais).", _ALL),
          ("Gastos do consumidor caem em recessões.", _ALL)), financeiro=True),
    _seg("bancos", "Bancos",
         "Empresta dinheiro cobrando juros maiores do que paga para captar (spread) e cobra tarifas e comissões de "
         "cartões, seguros e investimentos.",
         (("Crescimento do crédito na economia amplia a carteira de empréstimos.", _ALL),
          ("Serviços e tarifas (cartões, seguros, investimentos) diversificam a receita.", _ALL),
          ("Bancos grandes costumam ter escala e captação barata.", _ALL)),
         (("A inadimplência sobe em crises e com juros altos, exigindo provisões que reduzem o lucro.", _ALL),
          ("Concorrência de bancos digitais e o Pix reduzem as tarifas.", "BR"),
          ("Concorrência de bancos digitais e fintechs reduz as tarifas.", "US"),
          ("Regulação e exigências de capital limitam a alavancagem.", _ALL)), financeiro=True),
    _seg("financeiro_geral", "Serviços financeiros",
         "Oferece crédito, seguros, investimentos ou serviços financeiros; a receita depende de juros, tarifas e "
         "volume de negócios.",
         (("Mais crédito e investimentos na economia.", _ALL),
          ("Digitalização reduz custos e amplia o alcance.", _ALL)),
         (("Inadimplência e juros altos.", _ALL),
          ("Regulação e concorrência de fintechs.", _ALL)), financeiro=True),
    _seg("seguros", "Seguros, previdência e resseguros",
         "Recebe os prêmios (o preço do seguro), paga os sinistros e aplica as reservas; ganha quando os prêmios "
         "superam sinistros e despesas e com o rendimento das aplicações.",
         (("Juros altos aumentam o ganho com a aplicação das reservas.", _ALL),
          ("A baixa penetração de seguros no país deixa espaço para crescer.", "BR"),
          ("Parcerias de distribuição (bancos, corretores, canais digitais) dão escala.", _ALL)),
         (("Sinistros acima do esperado (eventos climáticos, roubos, fraudes).", _ALL),
          ("Queda de juros reduz o resultado financeiro.", _ALL),
          ("Concorrência pressiona os preços dos seguros.", _ALL)), financeiro=True),
    _seg("saneamento", "Saneamento (água e esgoto)",
         "Cobra tarifas reguladas pelo fornecimento de água e pela coleta e tratamento de esgoto nas cidades onde "
         "tem concessão.",
         ((("O Marco Legal do Saneamento prevê universalizar água e esgoto até 2033, o que exige investimentos e "
           "amplia a base de clientes."), "BR"),
          ("Tarifas reajustadas pela inflação dão previsibilidade à receita.", _ALL),
          ("Privatizações e novas concessões abrem espaço para crescer.", "BR")),
         (("As tarifas dependem de agências reguladoras e, às vezes, de decisões políticas.", _ALL),
          ("Investimentos obrigatórios elevados pressionam o caixa e a dívida.", _ALL),
          ("Secas e crises hídricas afetam o volume faturado.", _ALL))),
    _seg("energia_geracao", "Geração de energia elétrica",
         "Produz energia (hidrelétrica, eólica, solar ou térmica) e vende em contratos de longo prazo, muitas vezes "
         "corrigidos pela inflação, ou no mercado livre.",
         (("Contratos longos corrigidos pela inflação dão previsibilidade à receita.", _ALL),
          ("Demanda por energia renovável e eletrificação da economia (inclusive data centers).", _ALL),
          ("A abertura do mercado livre de energia traz novos clientes.", "BR")),
         (("O preço da energia no mercado de curto prazo pode cair com excesso de oferta.", _ALL),
          ("Falta de chuvas reduz a geração hidrelétrica, e cortes de geração afetam parques eólicos e solares.",
           "BR"),
          ("Juros altos encarecem novos projetos financiados com dívida.", _ALL))),
    _seg("energia_transmissao", "Transmissão de energia elétrica",
         "Recebe uma receita anual definida em leilão (RAP) pela disponibilidade das linhas, corrigida pela inflação "
         "e independente do consumo.",
         (("Receita contratada e corrigida pela inflação, pouco sensível à economia.", _ALL),
          ("Novos leilões e reforços na rede permitem crescer.", _ALL)),
         (("Crescer depende de vencer leilões concorridos, em geral com retornos menores.", _ALL),
          ("Revisões periódicas podem reduzir a receita de parte das linhas.", _ALL),
          ("Costuma distribuir boa parte do lucro: sobra menos para reinvestir.", _ALL))),
    _seg("energia_distribuicao", "Distribuição de energia elétrica (concessionárias)",
         "Leva energia até os consumidores da sua área de concessão e é remunerada por uma tarifa regulada, revista "
         "periodicamente pelo regulador (ANEEL, no Brasil).",
         (("O consumo de energia tende a crescer com a economia e a população.", _ALL),
          ("Investimentos na rede aumentam a base de ativos remunerada pela tarifa.", _ALL),
          ("Redução de perdas e de inadimplência melhora o resultado.", "BR")),
         (("A tarifa é definida pelo regulador: revisões podem reduzir a rentabilidade.", _ALL),
          ("Furto de energia e inadimplência.", "BR"),
          ("Geração própria dos consumidores (painéis solares) reduz o consumo da rede.", _ALL),
          ("Juros altos encarecem a dívida usada nos investimentos.", _ALL))),
    _seg("energia_eletrica", "Energia elétrica",
         "Gera, transmite ou distribui energia; a receita vem de contratos de longo prazo ou de tarifas reguladas.",
         (("Receitas contratadas ou reguladas, em geral corrigidas pela inflação.", _ALL),
          ("O consumo de energia tende a crescer com a economia e a eletrificação.", _ALL)),
         (("Regulação: revisões tarifárias e renovação de concessões podem reduzir a rentabilidade.", _ALL),
          ("Muito investimento e dívida: juros altos pesam.", _ALL))),
    _seg("combustiveis", "Distribuição de combustíveis",
         "Compra combustíveis de refinarias e importadores e revende a postos e empresas; ganha uma margem pequena "
         "por litro sobre um volume muito grande.",
         (("Crescimento da frota e da atividade econômica aumenta o volume vendido.", _ALL),
          ("Lojas de conveniência e serviços nos postos complementam a receita.", _ALL),
          ("O combate à sonegação no setor tende a melhorar a concorrência.", "BR")),
         (("Margem por litro pequena e sensível a oscilações no preço dos combustíveis.", _ALL),
          ("Concorrência com distribuidoras menores e importadores.", _ALL),
          ("Carros elétricos e híbridos podem reduzir a demanda no longo prazo.", _ALL))),
    _seg("petroleo", "Petróleo e gás",
         "Extrai e vende petróleo e gás (e, em alguns casos, refina e vende combustíveis); a receita acompanha o "
         "preço internacional do petróleo e o volume produzido.",
         (("Novos campos e aumento de produção podem elevar o volume vendido.", _ALL),
          ("Petróleo caro aumenta a receita e o caixa.", _ALL),
          ("Empresas do setor costumam pagar dividendos altos quando o petróleo está caro.", _ALL)),
         (("O preço do petróleo é volátil e fora do controle da empresa.", _ALL),
          ("A transição energética pode reduzir a demanda por combustíveis fósseis no longo prazo.", _ALL),
          ("Investimentos muito grandes e riscos ambientais e regulatórios.", _ALL))),
    _seg("quimica", "Química e petroquímica",
         "Produz resinas, plásticos e insumos químicos vendidos a outras indústrias; o lucro depende da diferença "
         "entre o preço dos produtos e o custo das matérias-primas (nafta, gás).",
         (("A retomada da indústria e do consumo aumenta a demanda.", _ALL),
          ("Matéria-prima mais barata amplia a margem.", _ALL)),
         (("Setor cíclico, com excesso de oferta global, especialmente da Ásia.", _ALL),
          ("Custos de energia e de matérias-primas voláteis.", _ALL),
          ("Exigências ambientais e pressão contra plásticos.", _ALL))),
    _seg("mineracao", "Mineração",
         "Extrai e vende minérios (como ferro, níquel e cobre); a receita depende do preço internacional dos metais, "
         "muito influenciado pela demanda da China.",
         (("Demanda por metais para infraestrutura e para a transição energética (cobre, níquel).", _ALL),
          ("Minério de alta qualidade costuma ter preço maior.", _ALL),
          ("Custos baixos de produção em grandes minas dão vantagem competitiva.", _ALL)),
         (("Dependência da economia chinesa e do setor imobiliário da China.", _ALL),
          ("O preço dos metais é cíclico e volátil.", _ALL),
          ("Riscos ambientais (barragens) e processos judiciais.", _ALL))),
    _seg("siderurgia", "Siderurgia (aço)",
         "Produz e vende aço para construção, indústria e montadoras; o lucro depende do preço do aço e do custo de "
         "minério, carvão e energia.",
         (("Obras de infraestrutura e construção civil aumentam a demanda.", _ALL),
          ("Medidas contra importações a preços muito baixos (antidumping) podem sustentar os preços.", _ALL)),
         (("Concorrência com aço importado, especialmente da China.", _ALL),
          ("Setor cíclico: a demanda cai em recessões.", _ALL),
          ("Custos de energia e de matérias-primas voláteis.", _ALL))),
    _seg("materiais_basicos", "Materiais básicos",
         "Produz matérias-primas (metais, químicos, papel, materiais de construção) vendidas a outras indústrias; "
         "preços e margens são cíclicos.",
         (("A retomada da indústria e da construção aumenta a demanda.", _ALL),
          ("Exportações em dólar ajudam quando o câmbio está favorável.", "BR")),
         (("Preços cíclicos e concorrência global.", _ALL),
          ("Custos de energia e de matérias-primas.", _ALL))),
    _seg("celulose_papel", "Celulose, papel e embalagens",
         "Produz celulose, papel e embalagens a partir de florestas plantadas; grande parte é exportada em dólar, "
         "então o preço da celulose e o câmbio pesam no resultado.",
         (("Dólar alto aumenta a receita em reais das exportações.", "BR"),
          ("Demanda por embalagens com o comércio eletrônico e a substituição do plástico.", _ALL),
          ("O eucalipto cresce rápido no Brasil, o que tende a dar custo de produção baixo.", "BR")),
         (("O preço da celulose é cíclico, e novas fábricas no mundo aumentam a oferta.", _ALL),
          ("Dólar mais fraco reduz a receita em reais.", "BR"),
          ("Fábricas novas exigem investimentos grandes e elevam a dívida.", _ALL))),
    _seg("telecom", "Telecomunicações",
         "Cobra assinaturas de telefonia móvel, internet e TV; a receita é recorrente, mas exige investimento "
         "contínuo na rede.",
         (("Mais uso de dados móveis e de fibra óptica.", _ALL),
          ("5G e serviços digitais para empresas.", _ALL),
          ("Mercado concentrado em poucas operadoras tende a reduzir guerras de preço.", _ALL)),
         (("Investimentos altos e contínuos em rede.", _ALL),
          ("Concorrência de provedores regionais de internet.", "BR"),
          ("Crescimento limitado: quase todo mundo já tem celular.", _ALL))),
    _seg("semicondutores", "Semicondutores (chips)",
         "Projeta ou fabrica chips (ou as máquinas que os fabricam) usados em data centers, computadores, celulares "
         "e carros.",
         (("Demanda por inteligência artificial e data centers.", _ALL),
          ("Mais chips em carros, celulares e equipamentos industriais.", _ALL),
          ("Barreiras tecnológicas altas protegem os líderes.", _ALL)),
         (("Setor cíclico: excesso de estoques derruba preços e vendas.", _ALL),
          ("Tensões geopolíticas (EUA x China, Taiwan) e restrições de exportação.", _ALL),
          ("Fábricas e pesquisa exigem investimentos bilionários.", _ALL),
          ("Expectativas altas já embutidas no preço: resultados abaixo do esperado costumam provocar quedas fortes.",
           _ALL))),
    _seg("hardware", "Equipamentos e eletrônicos",
         "Fabrica e vende computadores, celulares ou equipamentos eletrônicos, muitas vezes com serviços e software "
         "associados.",
         (("Ciclos de troca de aparelhos e novos produtos.", _ALL),
          ("Serviços e assinaturas vendidos sobre a base de aparelhos instalada.", _ALL)),
         (("Concorrência e queda de preços dos equipamentos.", _ALL),
          ("Componentes importados encarecem com a alta do dólar.", "BR"),
          ("Dependência de cadeias de suprimento na Ásia.", _ALL))),
    _seg("software", "Software e serviços de tecnologia",
         "Vende licenças e, cada vez mais, assinaturas de software e serviços na nuvem; a receita recorrente e a "
         "margem alta favorecem ganhos de escala.",
         (("Digitalização das empresas e migração para a nuvem.", _ALL),
          ("A inteligência artificial abre espaço para novos produtos.", _ALL),
          ("Assinaturas geram receita recorrente e previsível.", _ALL)),
         (("Concorrência e mudanças tecnológicas rápidas.", _ALL),
          ("Ações de tecnologia costumam ter preço alto em relação ao lucro, o que as torna sensíveis a juros.", _ALL),
          ("Regulação de dados e de concorrência.", _ALL))),
    _seg("tecnologia_geral", "Tecnologia",
         "Vende produtos ou serviços de tecnologia para empresas ou consumidores.",
         (("Digitalização e inteligência artificial.", _ALL),),
         (("Concorrência e mudanças tecnológicas rápidas.", _ALL),
          ("Preço alto em relação ao lucro torna a ação sensível a juros.", _ALL))),
    _seg("internet", "Internet, mídia e plataformas digitais",
         "Ganha com publicidade, assinaturas ou comissões sobre transações em plataformas digitais com milhões de "
         "usuários.",
         (("Efeito de rede: quanto mais usuários, mais valiosa a plataforma.", _ALL),
          ("Publicidade digital e comércio online continuam ganhando espaço.", _ALL),
          ("Novos produtos com inteligência artificial.", _ALL)),
         (("Regulação (privacidade, concorrência) e multas.", _ALL),
          ("Concorrência pela atenção dos usuários.", _ALL),
          ("A receita de publicidade cai em desacelerações da economia.", _ALL))),
    _seg("educacao", "Educação",
         "Cobra mensalidades de cursos presenciais e a distância (EAD) ou assinaturas de plataformas de ensino.",
         (("O ensino a distância permite atender mais alunos com custo menor.", _ALL),
          ("Cursos de medicina e saúde têm alta demanda e mensalidades altas.", "BR"),
          ("Demanda por qualificação profissional e cursos online.", _ALL)),
         (("Evasão e inadimplência de alunos.", _ALL),
          ("Dependência de programas de financiamento do governo, como o FIES.", "BR"),
          ("Concorrência e queda das mensalidades no EAD.", _ALL),
          ("Regras do MEC podem limitar cursos e vagas.", "BR"))),
    _seg("varejo_alimentar", "Supermercados e atacarejo",
         "Vende alimentos e produtos básicos com margem pequena e giro alto; o lucro vem do volume vendido e do "
         "controle de custos.",
         (("O atacarejo costuma ganhar clientes quando o orçamento das famílias aperta.", _ALL),
          ("Abertura de lojas em regiões com pouca concorrência.", _ALL),
          ("A demanda por alimentos é estável.", _ALL)),
         (("Margens muito pequenas: aumentos de custo pesam muito no lucro.", _ALL),
          ("Concorrência intensa.", _ALL),
          ("Juros altos encarecem a expansão e o capital de giro.", _ALL))),
    _seg("beleza_higiene", "Cosméticos, higiene e beleza",
         "Fabrica e vende cosméticos, perfumes e produtos de higiene por lojas, revendedoras, varejo e canais "
         "digitais; marca e inovação sustentam a margem.",
         (("Categoria com demanda resiliente e espaço para produtos premium.", _ALL),
          ("Marcas fortes permitem repassar custos ao preço.", _ALL),
          ("Vendas digitais e por revendedoras ampliam o alcance.", _ALL)),
         (("Concorrência intensa, com muitas marcas novas.", _ALL),
          ("Embalagens e insumos cotados em dólar pressionam custos.", _ALL),
          ("Itens não essenciais perdem espaço quando a renda aperta.", _ALL))),
    _seg("alimentos_bebidas", "Alimentos, bebidas e bens de consumo básico",
         "Fabrica e vende produtos de consumo diário com marcas conhecidas; a força da marca ajuda a repassar custos "
         "ao preço.",
         (("Marcas fortes ajudam a reajustar preços com a inflação.", _ALL),
          ("Demanda estável mesmo em crises.", _ALL),
          ("Expansão para novos mercados e produtos premium.", _ALL)),
         (("Custos de matérias-primas (grãos, alumínio, embalagens) e câmbio pressionam margens.", _ALL),
          ("O volume tende a crescer devagar em mercados maduros.", _ALL),
          ("Concorrência de marcas próprias e mudanças de hábitos (menos açúcar e álcool).", _ALL))),
    _seg("varejo", "Varejo e consumo",
         "Vende produtos ou serviços a consumidores em lojas e no digital; o resultado depende do consumo das "
         "famílias, da margem e do giro de estoque.",
         (("Renda e emprego em alta aumentam o consumo.", _ALL),
          ("Queda de juros barateia o crédito ao consumidor.", _ALL),
          ("Ganho de participação de mercado e vendas online.", _ALL)),
         (("O consumo tende a cair quando os juros estão altos e a renda aperta.", _ALL),
          ("Concorrência de plataformas estrangeiras de comércio eletrônico.", "BR"),
          ("Concorrência intensa, inclusive do comércio eletrônico.", "US"),
          ("Estoques e crédito ao consumidor exigem capital.", _ALL))),
    _seg("agronegocio", "Agronegócio",
         "Produz e vende produtos agrícolas, carnes ou insumos, em grande parte cotados em dólar; o resultado "
         "depende dos preços internacionais, do câmbio e do clima.",
         (("Demanda mundial por alimentos e biocombustíveis.", _ALL),
          ("Dólar alto aumenta a receita em reais das exportações.", "BR"),
          ("Produtividade crescente do agronegócio.", _ALL)),
         (("O clima (secas, geadas, excesso de chuva) afeta a produção.", _ALL),
          ("Preços de commodities agrícolas são voláteis.", _ALL),
          ("Barreiras comerciais e sanitárias em países importadores.", _ALL))),
    _seg("construcao", "Construção e incorporação imobiliária",
         "Compra terrenos, constrói e vende imóveis; o lucro depende das vendas, do custo da obra e das condições de "
         "financiamento dos compradores.",
         (("Programas habitacionais como o Minha Casa, Minha Vida sustentam a demanda por moradias econômicas.",
           "BR"),
          ("O déficit habitacional ainda é elevado no país.", "BR"),
          ("Queda de juros barateia o financiamento imobiliário.", _ALL)),
         (("Juros altos encarecem o crédito imobiliário e reduzem as vendas.", _ALL),
          ("Custos de construção (materiais e mão de obra) sobem com a inflação.", _ALL),
          ("Distratos e imóveis encalhados prendem capital.", "BR"))),
    _seg("shoppings", "Shopping centers",
         "Aluga lojas em shopping centers, cobrando aluguel mínimo mais um percentual das vendas dos lojistas, além "
         "de estacionamento e serviços.",
         (("Aluguéis corrigidos pela inflação.", _ALL),
          ("Vendas dos lojistas em alta elevam o aluguel percentual.", _ALL),
          ("Expansões e projetos de uso misto nos terrenos dos shoppings.", _ALL)),
         (("O comércio eletrônico reduz o movimento nas lojas físicas.", _ALL),
          ("Vacância e inadimplência de lojistas em crises.", _ALL),
          ("Juros altos tendem a reduzir o valor dos imóveis e encarecem a dívida.", _ALL))),
    _seg("fii_tijolo", "Fundo imobiliário de imóveis (tijolo)",
         "Aluga imóveis (galpões, escritórios, lojas) e distribui a maior parte do aluguel aos cotistas, geralmente "
         "todo mês.",
         (("Aluguéis reajustados pela inflação.", _ALL),
          ("Demanda por galpões logísticos com o comércio eletrônico.", _ALL),
          ("Queda de juros costuma valorizar as cotas.", _ALL)),
         (("Vacância (imóveis vazios) e inadimplência de inquilinos.", _ALL),
          ("Juros altos tornam a renda fixa concorrente e costumam derrubar o valor das cotas.", _ALL),
          ("Crescer depende de emitir novas cotas para comprar imóveis.", _ALL)), financeiro=True),
    _seg("fii_papel", "Fundo imobiliário de recebíveis (papel)",
         "Investe em títulos de dívida imobiliária (CRI) e distribui os juros recebidos; a renda acompanha o CDI ou "
         "a inflação, conforme a carteira.",
         (("Com juros e inflação altos, a renda distribuída tende a ser maior.", _ALL),
          ("Carteira com muitos devedores dilui o risco de cada um.", _ALL)),
         (("Risco de calote dos devedores dos CRI.", _ALL),
          ("Pouco potencial de valorização das cotas: o retorno vem sobretudo da renda.", _ALL),
          ("Queda dos juros ou da inflação reduz a renda distribuída.", _ALL)), financeiro=True),
    _seg("fii_generico", "Fundo imobiliário",
         "Investe em imóveis para aluguel e/ou em títulos imobiliários (CRI) e distribui a maior parte da renda aos "
         "cotistas, geralmente todo mês.",
         (("Aluguéis e juros dos títulos corrigidos pela inflação ou pelo CDI.", _ALL),
          ("Queda de juros costuma valorizar as cotas.", _ALL)),
         (("Vacância, inadimplência ou calote de devedores.", _ALL),
          ("Juros altos costumam derrubar o valor das cotas.", _ALL)), financeiro=True),
    _seg("imoveis_renda", "Imóveis para renda (REITs e empresas de imóveis)",
         "É dona de imóveis (lojas, galpões, torres, data centers) e ganha com aluguéis; REITs distribuem a maior "
         "parte do lucro como dividendos.",
         (("Contratos longos com reajustes periódicos.", _ALL),
          ("Demanda estrutural em segmentos como data centers, logística e torres.", _ALL)),
         (("Juros altos tendem a reduzir o valor dos imóveis e encarecem a dívida.", _ALL),
          ("Vacância e inquilinos em dificuldade.", _ALL),
          ("Crescer depende de captar dinheiro novo, já que a maior parte do lucro é distribuída.", _ALL)),
         financeiro=True),
    _seg("aviacao", "Aviação comercial",
         "Vende passagens aéreas e transporte de cargas; o lucro depende da ocupação dos voos, do preço das "
         "passagens e do custo do combustível.",
         (("A demanda por viagens cresce com a renda e o turismo.", _ALL),
          ("Aviões mais novos consomem menos combustível.", _ALL)),
         ((("Combustível, dívidas e parte dos custos atrelados ao dólar: alta do dólar ou do petróleo aperta a "
           "margem."), _ALL),
          ("Setor muito competitivo e com histórico de recuperações judiciais.", _ALL),
          ("Choques de demanda (crises, pandemias) derrubam a receita rapidamente.", _ALL))),
    _seg("logistica", "Logística e concessões de transporte",
         "Cobra pelo transporte de cargas ou pelo uso de rodovias e ferrovias; concessões têm contratos longos com "
         "tarifas corrigidas pela inflação.",
         (("Safras maiores e exportações aumentam o volume transportado.", "BR"),
          ("Contratos de concessão longos com reajuste pela inflação.", _ALL),
          ("Novas concessões e extensões de contratos.", _ALL)),
         (("Investimentos altos e dívida elevada.", _ALL),
          ("O volume transportado cai com a economia.", _ALL),
          ("Riscos regulatórios nas renovações de contratos.", _ALL))),
    _seg("locacao", "Locação de veículos e equipamentos",
         "Compra veículos ou máquinas, aluga por diárias ou contratos longos e depois revende os usados; ganha no "
         "aluguel e na revenda.",
         (("Empresas e motoristas de aplicativo preferem alugar a comprar.", _ALL),
          ("Terceirização de frotas por empresas.", _ALL),
          ("Escala para comprar veículos com desconto.", _ALL)),
         (("Juros altos encarecem a frota, financiada com dívida.", _ALL),
          ("Queda no preço dos carros usados reduz o ganho na revenda.", _ALL),
          ("Depreciação maior que a esperada.", _ALL))),
    _seg("aeroespacial", "Aeroespacial e defesa",
         "Fabrica aviões, motores ou equipamentos de defesa, com contratos longos e uma carteira de pedidos que dá "
         "visibilidade de receita.",
         (("A carteira de pedidos dá visibilidade de receita por anos.", _ALL),
          ("Aumento de gastos com defesa em vários países.", _ALL),
          ("Expansão da aviação comercial e serviços de manutenção recorrentes.", _ALL)),
         (("Problemas de produção e qualidade atrasam entregas.", _ALL),
          ("Dependência de orçamentos de governos.", _ALL),
          ("Cadeia de suprimentos e custos pressionados.", _ALL))),
    _seg("automotivo", "Veículos",
         "Fabrica e vende veículos; o lucro depende do volume vendido, do preço médio e do custo de baterias, aço e "
         "componentes.",
         (("Adoção de carros elétricos.", _ALL),
          ("Software, direção autônoma e armazenamento de energia podem abrir novas fontes de receita.", _ALL)),
         (("Concorrência crescente, inclusive de montadoras chinesas.", _ALL),
          ("Guerras de preço reduzem margens.", _ALL),
          ("Subsídios e tarifas mudam com a política.", _ALL))),
    _seg("energia_solar", "Equipamentos de energia solar",
         "Fabrica painéis ou equipamentos para sistemas solares vendidos a empresas de energia e residências.",
         (("Expansão da energia solar no mundo.", _ALL),
          ("Incentivos governamentais à energia limpa.", _ALL)),
         (("Concorrência chinesa e queda de preços dos painéis.", _ALL),
          ("Mudanças em incentivos e tarifas de importação.", _ALL),
          ("Juros altos encarecem projetos financiados.", _ALL))),
    _seg("bens_capital", "Indústria e bens de capital",
         "Fabrica máquinas, motores e equipamentos vendidos a empresas; a receita acompanha os investimentos da "
         "indústria, do agronegócio e da infraestrutura.",
         (("Investimentos em infraestrutura, automação e transição energética.", _ALL),
          ("Exportações diversificam a receita.", _ALL),
          ("Serviços e peças de reposição geram receita recorrente.", _ALL)),
         (("Setor cíclico: empresas adiam investimentos em crises e com juros altos.", _ALL),
          ("Custos de aço, cobre e outros insumos.", _ALL),
          ("Concorrência internacional.", _ALL))),
    _seg("etf_acoes", "Fundo de índice de ações (ETF)",
         "Replica um índice de ações: o retorno acompanha o conjunto das empresas do índice, menos a taxa de "
         "administração.",
         (("Diversificação: acompanha o crescimento de muitas empresas ao mesmo tempo.", _ALL),
          ("Custo baixo em relação a fundos de gestão ativa.", _ALL)),
         (("Não supera o índice: acompanha também as quedas do mercado.", _ALL),
          ("Concentração nas maiores empresas do índice.", _ALL)), financeiro=True),
    _seg("etf_renda_fixa", "Fundo de índice de renda fixa",
         "Investe em títulos de renda fixa que pagam juros; o preço das cotas sobe quando os juros de mercado caem e "
         "cai quando eles sobem.",
         (("Queda dos juros de mercado valoriza os títulos.", _ALL),
          ("Títulos atrelados à inflação preservam o poder de compra, quando é o caso.", _ALL)),
         (("Alta dos juros de mercado desvaloriza os títulos.", _ALL),
          ("O retorno fica limitado aos juros dos títulos.", _ALL)), financeiro=True),
    _seg("etf_generico", "Fundo de índice (ETF)",
         "Replica um índice; o retorno acompanha os ativos do índice, menos a taxa de administração.",
         (("Diversificação e custo baixo.", _ALL),),
         (("Acompanha as quedas do índice.", _ALL),
          ("A classe de ativos do índice não foi identificada aqui: confira o regulamento do fundo.", _ALL)),
         financeiro=True),
    _seg("ouro", "Ouro",
         "Não gera lucro nem dividendos; o preço depende da oferta e da procura, dos juros reais e da busca por "
         "proteção.",
         (("Procura por proteção em crises e compras de bancos centrais.", _ALL),
          ("Juros reais em queda tornam o ouro mais atraente.", _ALL)),
         (("Não gera fluxo de caixa.", _ALL),
          ("Juros reais altos e dólar forte costumam pesar no preço.", _ALL)), financeiro=True),
    _seg("commodities", "Commodities",
         "Acompanha preços de matérias-primas; não gera lucro nem dividendos.",
         (("Demanda global e restrições de oferta.", _ALL),),
         (("Preços muito voláteis e cíclicos.", _ALL),
          ("Não gera fluxo de caixa.", _ALL)), financeiro=True),
    _seg("cripto", "Criptoativo",
         "Não gera lucro nem dividendos; o preço depende da adoção e da especulação.",
         (("Adoção institucional (como ETFs) e oferta limitada em algumas moedas.", _ALL),),
         (("Volatilidade extrema.", _ALL),
          ("Riscos regulatórios e de custódia.", _ALL),
          ("Não gera fluxo de caixa.", _ALL)), financeiro=True),
)}

# Structural, factual notes about specific companies (kept short and conservative).
COMPANY_NOTES: dict[str, tuple[str, ...]] = {
    "PETR3.SA": (("Controle estatal (a União é a acionista controladora): decisões sobre preços de combustíveis, "
                 "investimentos e dividendos podem considerar objetivos do governo além do lucro."),),
    "PETR4.SA": (("Controle estatal (a União é a acionista controladora): decisões sobre preços de combustíveis, "
                 "investimentos e dividendos podem considerar objetivos do governo além do lucro."),),
    "BBAS3.SA": (("Controle estatal (a União é a acionista controladora): decisões de crédito e de dividendos podem "
                 "considerar objetivos do governo além do lucro."),),
}

# ------------------------------------------------------------------ classification rules
# Specific rules (tier 1), in priority order for ties. Patterns run on accent-free lower-case text.
_TIER1: tuple[tuple[str, str], ...] = (
    ("farmacias", (r"\bfarmacias?\b|drogari|pharmaceutical retail|retail[- ]drug|drug ?stores?"
                  r"|varejista de (produtos )?(farmaceutic|medicament)|comercio e distribuicao / medicament")),
    ("planos_saude", (r"planos? (de saude|odontolog)|healthcare plans?|health ?plans?|managed (health ?)?care"
                     r"|operadoras? de (planos|saude)")),
    ("hospitais", r"hospita|medical care facilit|medico[- ]hospitalar|serv\.? ?med\.? ?hospit|nursing"),
    ("dispositivos_medicos", (r"dispositivos medicos|medical (devices?|instruments?|electronics)"
                             r"|robos? para cirurgia|medical/dental")),
    ("diagnosticos", r"diagnostic|laboratori|analises clinicas"),
    ("farmaceutica", r"medicament|drug manufacturer|pharmaceutic|farmaceutic|biotechnolog|biotecnolog|vacina"),
    ("holding", r"\bholding|conglomerad|participac(ao|oes)( relevante)? (em|no|na|de)\b|holdings diversificadas"),
    ("gestao_bolsa", (r"bolsa de valores|banco de investimento|gestao de (ativos|patrimonio|recursos)"
                     r"|asset management|capital markets|investment bank|stock exchange|financial data|corretora")),
    ("pagamentos", r"pagamento|\bcart(ao|oes)\b|credit services|\bpayments?\b|transaction processing"),
    ("bancos", r"\bbancos?\b(?! de dados)|\bbanks?\b|intermediarios financeiros|instituic(ao|oes) financeira"),
    ("seguros", r"\bsegur(os?|adoras?|idade)\b|ressegur|previdencia|insurance|insurers|capitalizacao|reinsurance"),
    ("saneamento", r"saneamento|agua e esgoto|\bwater\b"),
    ("energia_geracao", r"\bgera(cao|dora)\b|renewable|renovave|independent power|eolica|power produc"),
    ("energia_transmissao", r"\btransmissao\b"),
    ("energia_distribuicao", (r"distribuicao(,| e| de energia)|regulated electric|electric utilit"
                             r"|energia eletrica regulada|concessionaria de energia")),
    ("combustiveis", r"combustive|refining & marketing|oil & gas refining|fuel distribut"),
    ("petroleo", r"petrole|\boil\b|oil & gas|oil and gas|gas natural|integrated oil|crude"),
    ("quimica", r"\bquimic|petroquimic|chemical"),
    ("mineracao", r"minera(cao|dora)|minerio|\bmining\b|minerais metalicos|metal mining|precious metals"),
    ("siderurgia", r"siderurg|\baco\b|\bsteel\b|metalurgia"),
    ("celulose_papel", r"celulose|\bpapel\b|embalage|\bpaper\b|packaging|madeira e papel|lumber"),
    ("aviacao", r"transporte aereo|\bairlines?\b|companhias? aereas?|linhas aereas"),
    ("telecom", r"telefon|telecom|banda larga|wireless|\b5g\b"),
    ("semicondutores", (r"semicondut|\bchips?\b|processador|maquinas de litografia|litografia|semiconductor"
                       r"|placas graficas")),
    ("hardware", (r"eletronicos de consumo|computador|consumer electronics|computer hardware"
                 r"|communication equipment|computer manufactur|equipamentos de comunicacao"
                 r"|telecommunications equipment")),
    ("educacao", (r"\bensino|educac|educat|\bescolas?\b|\bcursos?\b|faculdade|universidade|idiomas"
                 r"|aplicativo de ensino|plataforma de cursos|training services")),
    ("software", (r"software|\bnuvem\b|\bcloud\b|\berp\b|hospedagem|programas e servicos"
                 r"|information technology services|edp services|\bsaas\b|banco de dados|consultoria")),
    ("internet", (r"redes sociais|publicidade|streaming|aplicativo|plataforma|internet|entretenimento|entertainment"
                 r"|reservas de viagens|travel services|lodging|interactive media")),
    ("varejo_alimentar", (r"atacarejo|supermercad|hipermercad|grocery|discount stores|clubes de compras"
                         r"|comercio e distribuicao / alimentos|food distribution")),
    ("beleza_higiene", (r"cosmetic|perfum|higiene|household & personal|personal products|produtos de uso pessoal"
                       r"|\bbeleza\b")),
    ("alimentos_bebidas", (r"bebida|cerveja|refrigerante|biscoito|\bmassas\b|beverage|brewer|packaged foods"
                          r"|confection|salgadinho|laticin|alimentos processados")),
    ("varejo", (r"varejo|varejista|\bmoda\b|vestuario|calcados|joalheri|comercio eletronico|e-commerce|apparel"
               r"|footwear|specialty retail|internet retail|department stores?|home improvement|restaurant"
               r"|cafeteria|fast food|luxury|tecidos|eletrodomestic")),
    ("agronegocio", (r"acucar|etanol|\bsoja\b|algodao|\bmilho\b|\bcarnes?\b|frigorific|agricol|agropecuar"
                    r"|farm products|agricultural|\bmeat\b|\bgraos\b|fertilizant")),
    ("construcao", (r"incorporac|construcao de moradias|moradias|residential construction|homebuild"
                   r"|construcao civil|real estate - development")),
    ("fii_papel", r"fundo imobiliario de recebiveis|recebiveis \(cri\)|\bcri\b"),
    ("fii_tijolo", r"fundo imobiliario"),
    ("shoppings", r"shopping"),
    ("imoveis_renda", (r"\breits?\b|real estate investment trust|exploracao de imoveis|aluguel de imoveis"
                      r"|imoveis comerciais|data centers?|torres de telecom|galpoes logisticos")),
    ("logistica", (r"ferrovi|rodovi|logistic|railroad|trucking|freight|transporte|courier|infrastructure operations"
                  r"|aeroport|airport")),
    ("locacao", r"aluguel de carros|locacao|rental|leasing|gestao de frotas"),
    ("aeroespacial", r"motores de avi|\bavi(ao|oes)\b|aeroespac|aerospace|defesa|defense|aeronave"),
    ("automotivo", r"carros eletricos|automove|veiculos|auto manufactur|montadora|auto parts|autopecas"),
    ("energia_solar", r"paineis solares|microinversor|\bsolar\b"),
    ("bens_capital", (r"maquinas|motores|automacao|equipamentos (de|industriais|eletricos|para)|industrial machinery"
                     r"|heavy construction machinery|specialty industrial|electrical equipment|paineis de madeira"
                     r"|loucas|metais sanitarios|building products|conglomerates|engineering & construction"
                     r"|material de construcao")),
)
# Generic sector names (tier 2): only when nothing specific matched in any text.
_TIER2: tuple[tuple[str, str], ...] = (
    ("energia_eletrica", r"energia eletrica|\butilities\b|utilidade publica|electric"),
    ("saude_geral", r"health|saude"),
    ("tecnologia_geral", r"technolog|tecnologia"),
    ("financeiro_geral", r"financ"),
    ("varejo", r"consumer cyclical|consumer discretionary|consumo ciclico"),
    ("alimentos_bebidas", r"consumer defensive|consumer staples|consumo nao ciclico"),
    ("petroleo", r"\benergy\b|petroleo"),
    ("materiais_basicos", r"basic materials|materiais basicos"),
    ("bens_capital", r"industrials|bens industriais|industria"),
    ("internet", r"communication services"),
    ("telecom", r"telecomunica|comunicacoes"),
    ("imoveis_renda", r"real estate|imobiliari"),
)
_RX1 = tuple((k, re.compile(p)) for k, p in _TIER1)
_RX2 = tuple((k, re.compile(p)) for k, p in _TIER2)
_ENERGY_ONLY = {"energia_geracao", "energia_transmissao", "energia_distribuicao"}
_ENERGY_CTX = re.compile(r"energi|eletric|electric|utilit|power|hidreletric|termeletric")


def _first_match(text: str, rules: tuple[tuple[str, re.Pattern[str]], ...]) -> str | None:
    t = _fold(text)
    best: tuple[int, str] | None = None
    for key, rx in rules:
        if key in _ENERGY_ONLY and not _ENERGY_CTX.search(t):
            continue
        m = rx.search(t)
        if m and (best is None or m.start() < best[0]):      # strict: ties keep the earlier rule
            best = (m.start(), key)
    return best[1] if best else None


def classify_segment(texts: Iterable[tuple[str, str | None]]) -> tuple[Segment, str] | None:
    """``texts``: (source label, text) from most to least specific. Returns (segment, source)."""
    items = [(src, t) for src, t in texts if t and t.strip()]
    for rules in (_RX1, _RX2):
        for src, t in items:
            key = _first_match(t, rules)
            if key is not None:
                return SEGMENTS[key], src
    return None


def segment_for_class(macro_class: str, asset_class: str, business: str) -> Segment | None:
    """Instrument types whose segment follows from the asset class (ETFs, gold, crypto)."""
    b = _fold(business)
    if asset_class == "crypto" or macro_class == "cripto":
        return SEGMENTS["cripto"]
    if macro_class == "ouro" or (asset_class == "commodity" and re.search(r"ouro|gold", b)):
        return SEGMENTS["ouro"]
    if asset_class == "commodity":
        return SEGMENTS["commodities"]
    if macro_class in ("etf_br", "etf_eua") or (macro_class == "renda_fixa_br" and "etf" in b):
        if asset_class == "fixed_income":
            return SEGMENTS["etf_renda_fixa"]
        if asset_class == "real_estate":
            return SEGMENTS["imoveis_renda"]
        return SEGMENTS["etf_acoes"]
    if macro_class == "renda_fixa_br":
        return SEGMENTS["etf_renda_fixa"]
    return None


def applies(scope: str, country: str) -> bool:
    return scope == _ALL or scope == country
