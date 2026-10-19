"""Asset catalogue: macro classes expanded into INDIVIDUAL tickers with factual metadata.

* ``REAL_CATALOG`` lists real tickers (B3, US exchanges, crypto). Descriptions are short,
  factual statements about the line of business; theme tags are an editorial,
  documented curation (``tag_source``) used only for the purpose-alignment score, never
  for financial estimates. Every price series must still be downloaded and validated at
  run time; a ticker without >= 252 valid observations is EXCLUDED (never proxied).
* ``SYNTHETIC_CATALOG`` mirrors the same structure with clearly synthetic tickers for
  the offline demonstration (multi-currency, one synthetic governance flag).

Theme tag vocabulary (asset side): energia_limpa, sustentabilidade, saude, educacao,
tecnologia, inovacao, infraestrutura, imoveis, renda_recorrente, preservacao_capital,
protecao_inflacao, reserva_valor, diversificacao_global, consumo, financeiro,
petroleo_gas, mineracao, florestal, industria, especulativo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TAG_SOURCE_REAL = "curadoria editorial estática do catálogo (revisar periodicamente); não é dado de mercado"


@dataclass(frozen=True)
class CatalogAsset:
    ticker: str
    name: str
    macro_class: str          # acoes_br, fii, renda_fixa_br, etf_br, acoes_eua, etf_eua, ouro, cripto
    asset_class: str          # equity, fixed_income, real_estate, commodity, crypto (constraint classes)
    sector: str
    country: str
    currency: str
    business: str             # factual one-liner
    tags: tuple[str, ...] = field(default_factory=tuple)
    class_risk: str = ""      # generic risk of the instrument type (documented, not company-specific)


_EQ_BR = "risco de ações: volatilidade elevada, risco de setor e de execução da empresa"
_EQ_US = "risco de ações e cambial (cotado em USD)"
_FII = "risco imobiliário: vacância, inadimplência e sensibilidade a juros"
_RF_BR = "risco de taxa de juros (marcação a mercado) e de inflação implícita"
_RF_US = "risco de juros dos EUA e cambial (cotado em USD)"
_GOLD = "commodity sem fluxo de caixa; risco cambial (cotado em USD)"
_CRYPTO = "volatilidade extrema, risco regulatório e de custódia; sem fluxo de caixa"

BASE_CATALOG: tuple[CatalogAsset, ...] = (
    CatalogAsset("PETR4.SA", "Petrobras PN", "acoes_br", "equity", "petroleo_gas", "BR", "BRL",
                 "exploração, produção e refino de petróleo e gás", ("petroleo_gas", "industria"), _EQ_BR),
    CatalogAsset("VALE3.SA", "Vale ON", "acoes_br", "equity", "mineracao", "BR", "BRL",
                 "mineração de minério de ferro, níquel e cobre", ("mineracao", "industria"), _EQ_BR),
    CatalogAsset("ITUB4.SA", "Itaú Unibanco PN", "acoes_br", "equity", "financeiro", "BR", "BRL",
                 "banco de varejo e atacado", ("financeiro",), _EQ_BR),
    CatalogAsset("WEGE3.SA", "WEG ON", "acoes_br", "equity", "industria", "BR", "BRL",
                 "fabricação de motores elétricos, equipamentos de geração/transmissão de energia e automação",
                 ("industria", "energia_limpa", "infraestrutura"), _EQ_BR),
    CatalogAsset("EGIE3.SA", "Engie Brasil ON", "acoes_br", "equity", "utilidades", "BR", "BRL",
                 "geração e comercialização de energia elétrica (hidrelétrica, eólica e solar)",
                 ("energia_limpa", "sustentabilidade", "infraestrutura"), _EQ_BR),
    CatalogAsset("RADL3.SA", "Raia Drogasil ON", "acoes_br", "equity", "saude", "BR", "BRL",
                 "rede de farmácias e drogarias (Droga Raia e Drogasil): medicamentos, higiene pessoal e beleza",
                 ("saude", "consumo"), _EQ_BR),
    CatalogAsset("FLRY3.SA", "Fleury ON", "acoes_br", "equity", "saude", "BR", "BRL",
                 "medicina diagnóstica e serviços de saúde", ("saude",), _EQ_BR),
    CatalogAsset("COGN3.SA", "Cogna ON", "acoes_br", "equity", "educacao", "BR", "BRL",
                 "educação superior, básica e soluções educacionais", ("educacao",), _EQ_BR),
    CatalogAsset("SUZB3.SA", "Suzano ON", "acoes_br", "equity", "florestal", "BR", "BRL",
                 "celulose e papel a partir de florestas plantadas de eucalipto", ("florestal", "industria"), _EQ_BR),
    CatalogAsset("BBAS3.SA", "Banco do Brasil ON", "acoes_br", "equity", "financeiro", "BR", "BRL",
                 "banco múltiplo com forte atuação em crédito ao agronegócio", ("financeiro",), _EQ_BR),
    CatalogAsset("ABEV3.SA", "Ambev ON", "acoes_br", "equity", "consumo", "BR", "BRL",
                 "fabricação e distribuição de cervejas e outras bebidas", ("consumo",), _EQ_BR),
    CatalogAsset("TOTS3.SA", "Totvs ON", "acoes_br", "equity", "tecnologia", "BR", "BRL",
                 "software de gestão empresarial (ERP) e serviços digitais para empresas",
                 ("tecnologia", "inovacao"), _EQ_BR),
    CatalogAsset("RDOR3.SA", "Rede D'Or ON", "acoes_br", "equity", "saude", "BR", "BRL",
                 "rede de hospitais privados", ("saude",), _EQ_BR),
    CatalogAsset("HAPV3.SA", "Hapvida ON", "acoes_br", "equity", "saude", "BR", "BRL",
                 "planos de saúde e odontológicos com rede própria de hospitais e clínicas", ("saude",), _EQ_BR),
    CatalogAsset("YDUQ3.SA", "Yduqs ON", "acoes_br", "equity", "educacao", "BR", "BRL",
                 "ensino superior presencial e a distância", ("educacao",), _EQ_BR),
    CatalogAsset("BOVA11.SA", "iShares Ibovespa", "etf_br", "equity", "indice", "BR", "BRL",
                 "ETF que replica o Ibovespa", ("diversificacao_local",), _EQ_BR),
    CatalogAsset("IVVB11.SA", "iShares S&P 500 (BRL)", "etf_br", "equity", "indice", "US", "BRL",
                 "ETF listado na B3 que replica o S&P 500 com exposição cambial", ("diversificacao_global",), _EQ_BR),
    CatalogAsset("KNRI11.SA", "Kinea Renda Imobiliária", "fii", "real_estate", "imobiliario", "BR", "BRL",
                 "fundo imobiliário de lajes corporativas e galpões logísticos", ("imoveis", "renda_recorrente"), _FII),
    CatalogAsset("HGLG11.SA", "CSHG Logística", "fii", "real_estate", "imobiliario", "BR", "BRL",
                 "fundo imobiliário de galpões logísticos", ("imoveis", "renda_recorrente"), _FII),
    CatalogAsset("IMAB11.SA", "iShares IMA-B", "renda_fixa_br", "fixed_income", "titulos_publicos", "BR", "BRL",
                 "ETF de títulos públicos federais indexados ao IPCA (IMA-B)",
                 ("preservacao_capital", "protecao_inflacao"), _RF_BR),
    CatalogAsset("B5P211.SA", "It Now IMA-B5", "renda_fixa_br", "fixed_income", "titulos_publicos", "BR", "BRL",
                 "ETF de títulos públicos IPCA+ de prazo até 5 anos (IMA-B5)",
                 ("preservacao_capital", "protecao_inflacao"), _RF_BR),
    CatalogAsset("AAPL", "Apple", "acoes_eua", "equity", "tecnologia", "US", "USD",
                 "eletrônicos de consumo, software e serviços", ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    CatalogAsset("MSFT", "Microsoft", "acoes_eua", "equity", "tecnologia", "US", "USD",
                 "software, computação em nuvem e serviços corporativos",
                 ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    CatalogAsset("NVDA", "NVIDIA", "acoes_eua", "equity", "tecnologia", "US", "USD",
                 "semicondutores: processadores gráficos para data centers, inteligência artificial e jogos",
                 ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    CatalogAsset("GOOGL", "Alphabet (Google)", "acoes_eua", "equity", "tecnologia", "US", "USD",
                 "busca e publicidade digital, computação em nuvem e software",
                 ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    CatalogAsset("JNJ", "Johnson & Johnson", "acoes_eua", "equity", "saude", "US", "USD",
                 "medicamentos e dispositivos médicos", ("saude", "diversificacao_global"), _EQ_US),
    CatalogAsset("ICLN", "iShares Global Clean Energy", "etf_eua", "equity", "energia", "GLOBAL", "USD",
                 "ETF de empresas globais de energia limpa",
                 ("energia_limpa", "sustentabilidade", "diversificacao_global"), _EQ_US),
    CatalogAsset("SPY", "SPDR S&P 500", "etf_eua", "equity", "indice", "US", "USD",
                 "ETF que replica o S&P 500", ("diversificacao_global",), _EQ_US),
    CatalogAsset("AGG", "iShares Core US Aggregate Bond", "etf_eua", "fixed_income", "titulos", "US", "USD",
                 "ETF de títulos de renda fixa grau de investimento dos EUA",
                 ("preservacao_capital", "diversificacao_global"), _RF_US),
    CatalogAsset("GLD", "SPDR Gold Shares", "ouro", "commodity", "ouro", "GLOBAL", "USD",
                 "ETF lastreado em ouro físico", ("reserva_valor",), _GOLD),
    CatalogAsset("BTC-USD", "Bitcoin", "cripto", "crypto", "cripto", "GLOBAL", "USD",
                 "criptoativo descentralizado", ("especulativo", "inovacao"), _CRYPTO),
)

# --------------------------------------------------------------------------------------------
# Extended curated universe (fallback when the full exchange listings are unavailable, and the
# source of curated sectors/tags for well-known names). Listed roughly by size/liquidity within
# each block; every ticker is still validated against market data at run time and excluded
# (never proxied) when it has no usable series.
_REIT_US = "risco imobiliário e de juros dos EUA e cambial (cotado em USD)"
_GOLD_BR = "commodity sem fluxo de caixa; exposição ao dólar embutida no preço"


def _br(t: str, name: str, sector: str, business: str, tags: tuple[str, ...]) -> CatalogAsset:
    return CatalogAsset(t, name, "acoes_br", "equity", sector, "BR", "BRL", business, tags, _EQ_BR)


def _us(t: str, name: str, sector: str, business: str, tags: tuple[str, ...]) -> CatalogAsset:
    return CatalogAsset(t, name, "acoes_eua", "equity", sector, "US", "USD", business,
                        (*tags, "diversificacao_global"), _EQ_US)


def _fii(t: str, name: str, business: str) -> CatalogAsset:
    return CatalogAsset(t, name, "fii", "real_estate", "imobiliario", "BR", "BRL", business,
                        ("imoveis", "renda_recorrente"), _FII)


def _etf_us(t: str, name: str, cls: str, business: str, tags: tuple[str, ...], risk: str) -> CatalogAsset:
    return CatalogAsset(t, name, "etf_eua", cls, "etf", "US", "USD", business, tags, risk)


EXTENDED_CATALOG: tuple[CatalogAsset, ...] = (
    # B3 — financeiro
    _br("BBDC4.SA", "Bradesco PN", "financeiro", "banco de varejo, atacado e seguros", ("financeiro",)),
    _br("ITSA4.SA", "Itaúsa PN", "financeiro", "holding com participação relevante no Itaú Unibanco",
        ("financeiro",)),
    _br("BPAC11.SA", "BTG Pactual units", "financeiro", "banco de investimento e gestão de patrimônio",
        ("financeiro",)),
    _br("SANB11.SA", "Santander Brasil units", "financeiro", "banco de varejo e atacado", ("financeiro",)),
    _br("B3SA3.SA", "B3 ON", "financeiro", "bolsa de valores e infraestrutura do mercado financeiro",
        ("financeiro",)),
    _br("BBSE3.SA", "BB Seguridade ON", "financeiro", "seguros, previdência e capitalização", ("financeiro",)),
    _br("PSSA3.SA", "Porto Seguro ON", "financeiro", "seguros de automóvel, residência e saúde", ("financeiro",)),
    _br("CXSE3.SA", "Caixa Seguridade ON", "financeiro", "seguros e previdência vendidos pela Caixa",
        ("financeiro",)),
    _br("IRBR3.SA", "IRB Re ON", "financeiro", "resseguros", ("financeiro",)),
    # B3 — petróleo, mineração, siderurgia, papel
    _br("PETR3.SA", "Petrobras ON", "petroleo_gas", "exploração, produção e refino de petróleo e gás",
        ("petroleo_gas", "industria")),
    _br("PRIO3.SA", "PRIO ON", "petroleo_gas", "exploração e produção de petróleo", ("petroleo_gas",)),
    _br("RECV3.SA", "PetroReconcavo ON", "petroleo_gas", "produção de petróleo e gás em campos terrestres",
        ("petroleo_gas",)),
    _br("UGPA3.SA", "Ultrapar ON", "petroleo_gas", "distribuição de combustíveis (Ipiranga) e de gás (Ultragaz)",
        ("petroleo_gas",)),
    _br("VBBR3.SA", "Vibra Energia ON", "petroleo_gas", "distribuição de combustíveis", ("petroleo_gas",)),
    _br("CSAN3.SA", "Cosan ON", "petroleo_gas", "holding de energia, combustíveis e logística", ("petroleo_gas",)),
    _br("ENEV3.SA", "Eneva ON", "petroleo_gas", "geração termelétrica a gás e produção de gás natural",
        ("petroleo_gas", "infraestrutura")),
    _br("GGBR4.SA", "Gerdau PN", "siderurgia", "produção de aço", ("mineracao", "industria")),
    _br("CSNA3.SA", "CSN ON", "siderurgia", "siderurgia, mineração e cimento", ("mineracao", "industria")),
    _br("USIM5.SA", "Usiminas PNA", "siderurgia", "produção de aço plano", ("mineracao", "industria")),
    _br("CMIN3.SA", "CSN Mineração ON", "mineracao", "mineração de minério de ferro", ("mineracao",)),
    _br("BRAP4.SA", "Bradespar PN", "mineracao", "holding com participação na Vale", ("mineracao",)),
    _br("KLBN11.SA", "Klabin units", "florestal", "papel e embalagens a partir de florestas plantadas",
        ("florestal", "industria")),
    _br("DXCO3.SA", "Dexco ON", "industria", "painéis de madeira, louças e metais sanitários", ("industria",)),
    # B3 — energia elétrica e saneamento
    _br("ELET3.SA", "Eletrobras ON", "utilidades", "geração (majoritariamente hidrelétrica) e transmissão de energia",
        ("energia_limpa", "infraestrutura")),
    _br("AURE3.SA", "Auren Energia ON", "utilidades", "geração de energia renovável (hídrica, eólica e solar)",
        ("energia_limpa", "sustentabilidade", "infraestrutura")),
    _br("EQTL3.SA", "Equatorial ON", "utilidades", "distribuição e transmissão de energia e saneamento",
        ("infraestrutura",)),
    _br("CPFE3.SA", "CPFL Energia ON", "utilidades", "distribuição, geração e transmissão de energia elétrica",
        ("infraestrutura",)),
    _br("NEOE3.SA", "Neoenergia ON", "utilidades", "distribuição, transmissão e geração de energia elétrica",
        ("infraestrutura",)),
    _br("ENGI11.SA", "Energisa units", "utilidades", "distribuição de energia elétrica", ("infraestrutura",)),
    _br("CMIG4.SA", "Cemig PN", "utilidades", "geração, transmissão e distribuição de energia em Minas Gerais",
        ("infraestrutura",)),
    _br("TAEE11.SA", "Taesa units", "utilidades", "transmissão de energia elétrica", ("infraestrutura",)),
    _br("ALUP11.SA", "Alupar units", "utilidades", "transmissão e geração de energia elétrica", ("infraestrutura",)),
    _br("ISAE4.SA", "ISA Energia Brasil PN", "utilidades", "transmissão de energia elétrica", ("infraestrutura",)),
    _br("SBSP3.SA", "Sabesp ON", "saneamento", "saneamento básico: água e esgoto em São Paulo",
        ("saneamento", "infraestrutura")),
    _br("SAPR11.SA", "Sanepar units", "saneamento", "saneamento básico no Paraná", ("saneamento", "infraestrutura")),
    _br("CSMG3.SA", "Copasa ON", "saneamento", "saneamento básico em Minas Gerais", ("saneamento", "infraestrutura")),
    # B3 — consumo, agro, saúde, educação, tecnologia
    _br("MGLU3.SA", "Magazine Luiza ON", "consumo", "varejo físico e comércio eletrônico", ("consumo",)),
    _br("LREN3.SA", "Lojas Renner ON", "consumo", "varejo de moda", ("consumo",)),
    _br("ASAI3.SA", "Assaí ON", "consumo", "atacarejo de alimentos", ("consumo",)),
    _br("GMAT3.SA", "Grupo Mateus ON", "consumo", "supermercados e atacarejo no Norte e Nordeste", ("consumo",)),
    _br("VIVA3.SA", "Vivara ON", "consumo", "joalheria", ("consumo",)),
    _br("AZZA3.SA", "Azzas 2154 ON", "consumo", "moda e calçados", ("consumo",)),
    _br("MDIA3.SA", "M. Dias Branco ON", "consumo", "massas e biscoitos", ("consumo",)),
    _br("SMTO3.SA", "São Martinho ON", "agronegocio", "açúcar, etanol e bioeletricidade", ("agronegocio",)),
    _br("SLCE3.SA", "SLC Agrícola ON", "agronegocio", "produção de soja, algodão e milho", ("agronegocio",)),
    _br("BEEF3.SA", "Minerva ON", "agronegocio", "processamento e exportação de carne bovina", ("agronegocio",)),
    _br("HYPE3.SA", "Hypera ON", "saude", "medicamentos", ("saude",)),
    _br("PGMN3.SA", "Pague Menos ON", "saude", "rede de farmácias e drogarias (medicamentos, higiene pessoal e "
        "beleza), com forte presença no Norte e no Nordeste", ("saude", "consumo")),
    _br("PNVL3.SA", "Panvel ON", "saude", "rede de farmácias Panvel (medicamentos, higiene e beleza, inclusive "
        "produtos de marca própria), com lojas concentradas no Sul", ("saude", "consumo")),
    _br("ODPV3.SA", "Odontoprev ON", "saude", "planos odontológicos", ("saude",)),
    _br("QUAL3.SA", "Qualicorp ON", "saude", "administração de planos de saúde coletivos", ("saude",)),
    _br("CSED3.SA", "Cruzeiro do Sul Educacional ON", "educacao", "ensino superior presencial e a distância",
        ("educacao",)),
    _br("ANIM3.SA", "Ânima ON", "educacao", "ensino superior, inclusive medicina", ("educacao",)),
    _br("SEER3.SA", "Ser Educacional ON", "educacao", "ensino superior no Norte e Nordeste", ("educacao",)),
    _br("LWSA3.SA", "Locaweb ON", "tecnologia", "hospedagem, software e comércio eletrônico para empresas",
        ("tecnologia",)),
    _br("POSI3.SA", "Positivo Tecnologia ON", "tecnologia", "computadores e soluções de tecnologia",
        ("tecnologia",)),
    _br("INTB3.SA", "Intelbras ON", "tecnologia", "equipamentos de comunicação, segurança e energia solar",
        ("tecnologia",)),
    # B3 — infraestrutura, transporte, telecom, imobiliário
    _br("RAIL3.SA", "Rumo ON", "logistica", "logística ferroviária", ("infraestrutura",)),
    _br("ECOR3.SA", "Ecorodovias ON", "logistica", "concessões de rodovias", ("infraestrutura",)),
    _br("RENT3.SA", "Localiza ON", "servicos", "aluguel de carros e gestão de frotas", ()),
    _br("VAMO3.SA", "Vamos ON", "servicos", "locação de caminhões e máquinas", ()),
    _br("VIVT3.SA", "Telefônica Brasil (Vivo) ON", "telecom", "telefonia móvel, fixa e banda larga",
        ("infraestrutura",)),
    _br("TIMS3.SA", "TIM ON", "telecom", "telefonia móvel e banda larga", ("infraestrutura",)),
    _br("CYRE3.SA", "Cyrela ON", "construcao", "incorporação imobiliária residencial", ("imoveis",)),
    _br("MRVE3.SA", "MRV ON", "construcao", "construção de moradias econômicas", ("imoveis",)),
    _br("DIRR3.SA", "Direcional ON", "construcao", "construção de moradias econômicas", ("imoveis",)),
    _br("CURY3.SA", "Cury ON", "construcao", "construção de moradias econômicas", ("imoveis",)),
    _br("EZTC3.SA", "EZTEC ON", "construcao", "incorporação imobiliária em São Paulo", ("imoveis",)),
    _br("MULT3.SA", "Multiplan ON", "imobiliario", "shopping centers", ("imoveis",)),
    _br("ALOS3.SA", "Allos ON", "imobiliario", "shopping centers", ("imoveis",)),
    _br("IGTI11.SA", "Iguatemi units", "imobiliario", "shopping centers", ("imoveis",)),
    # B3 — fundos imobiliários
    _fii("XPML11.SA", "XP Malls", "fundo imobiliário de shopping centers"),
    _fii("VISC11.SA", "Vinci Shopping Centers", "fundo imobiliário de shopping centers"),
    _fii("BTLG11.SA", "BTG Pactual Logística", "fundo imobiliário de galpões logísticos"),
    _fii("XPLG11.SA", "XP Log", "fundo imobiliário de galpões logísticos"),
    _fii("HGRU11.SA", "CSHG Renda Urbana", "fundo imobiliário de imóveis urbanos de varejo e educação"),
    _fii("MXRF11.SA", "Maxi Renda", "fundo imobiliário de recebíveis (CRI) e imóveis"),
    _fii("KNCR11.SA", "Kinea Rendimentos Imobiliários", "fundo imobiliário de recebíveis (CRI) atrelados ao CDI"),
    _fii("KNIP11.SA", "Kinea Índice de Preços", "fundo imobiliário de recebíveis (CRI) atrelados à inflação"),
    _fii("CPTS11.SA", "Capitânia Securities II", "fundo imobiliário de recebíveis (CRI)"),
    # B3 — ETFs
    CatalogAsset("SMAL11.SA", "iShares Small Cap", "etf_br", "equity", "indice", "BR", "BRL",
                 "ETF de ações de empresas de menor capitalização da B3", ("diversificacao_local",), _EQ_BR),
    CatalogAsset("DIVO11.SA", "It Now IDIV", "etf_br", "equity", "indice", "BR", "BRL",
                 "ETF de ações que pagam mais dividendos na B3", ("renda_recorrente", "diversificacao_local"), _EQ_BR),
    CatalogAsset("NASD11.SA", "Trend Nasdaq-100 (BRL)", "etf_br", "equity", "indice", "US", "BRL",
                 "ETF listado na B3 que replica o Nasdaq-100 com exposição cambial",
                 ("tecnologia", "inovacao", "diversificacao_global"), _EQ_BR),
    CatalogAsset("IRFM11.SA", "iShares IRF-M", "renda_fixa_br", "fixed_income", "titulos_publicos", "BR", "BRL",
                 "ETF de títulos públicos prefixados (IRF-M)", ("preservacao_capital",), _RF_BR),
    CatalogAsset("GOLD11.SA", "Trend Ouro", "ouro", "commodity", "ouro", "GLOBAL", "BRL",
                 "ETF listado na B3 que acompanha o preço do ouro em reais", ("reserva_valor",), _GOLD_BR),
    CatalogAsset("HASH11.SA", "Hashdex Nasdaq Crypto Index", "cripto", "crypto", "cripto", "GLOBAL", "BRL",
                 "ETF listado na B3 de uma cesta de criptoativos", ("especulativo", "inovacao"), _CRYPTO),
    # EUA — ações
    _us("AMZN", "Amazon", "tecnologia", "comércio eletrônico e computação em nuvem (AWS)",
        ("tecnologia", "inovacao", "consumo")),
    _us("META", "Meta Platforms", "tecnologia", "redes sociais e publicidade digital", ("tecnologia", "inovacao")),
    _us("TSLA", "Tesla", "automotivo", "carros elétricos e armazenamento de energia",
        ("tecnologia", "inovacao", "energia_limpa")),
    _us("AVGO", "Broadcom", "tecnologia", "semicondutores e software de infraestrutura", ("tecnologia", "inovacao")),
    _us("TSM", "Taiwan Semiconductor (ADR)", "tecnologia", "fabricação de chips sob encomenda",
        ("tecnologia", "inovacao")),
    _us("ASML", "ASML (ADR)", "tecnologia", "máquinas de litografia para fabricar chips", ("tecnologia", "inovacao")),
    _us("AMD", "Advanced Micro Devices", "tecnologia", "processadores e placas gráficas", ("tecnologia", "inovacao")),
    _us("ORCL", "Oracle", "tecnologia", "banco de dados, software corporativo e nuvem", ("tecnologia",)),
    _us("CRM", "Salesforce", "tecnologia", "software de gestão de clientes na nuvem", ("tecnologia",)),
    _us("ADBE", "Adobe", "tecnologia", "software criativo e de marketing digital", ("tecnologia",)),
    _us("CSCO", "Cisco", "tecnologia", "equipamentos e software de redes", ("tecnologia",)),
    _us("QCOM", "Qualcomm", "tecnologia", "chips para celulares e comunicação sem fio", ("tecnologia",)),
    _us("INTC", "Intel", "tecnologia", "processadores e fabricação de chips", ("tecnologia",)),
    _us("IBM", "IBM", "tecnologia", "software, nuvem híbrida e consultoria", ("tecnologia",)),
    _us("NFLX", "Netflix", "tecnologia", "streaming de vídeo por assinatura", ("tecnologia", "consumo")),
    _us("UBER", "Uber", "tecnologia", "mobilidade e entregas por aplicativo", ("tecnologia",)),
    _us("SHOP", "Shopify", "tecnologia", "plataforma de comércio eletrônico para lojistas", ("tecnologia",)),
    _us("BKNG", "Booking Holdings", "tecnologia", "reservas de viagens online", ("tecnologia", "consumo")),
    _us("ABNB", "Airbnb", "tecnologia", "plataforma de hospedagem", ("tecnologia", "consumo")),
    _us("PYPL", "PayPal", "financeiro", "pagamentos digitais", ("financeiro", "tecnologia")),
    _us("BRK-B", "Berkshire Hathaway B", "financeiro", "conglomerado de seguros, ferrovia, energia e participações",
        ("financeiro",)),
    _us("JPM", "JPMorgan Chase", "financeiro", "banco de varejo, atacado e investimento", ("financeiro",)),
    _us("V", "Visa", "financeiro", "rede de pagamentos com cartão", ("financeiro", "tecnologia")),
    _us("MA", "Mastercard", "financeiro", "rede de pagamentos com cartão", ("financeiro", "tecnologia")),
    _us("BAC", "Bank of America", "financeiro", "banco de varejo e investimento", ("financeiro",)),
    _us("WFC", "Wells Fargo", "financeiro", "banco de varejo", ("financeiro",)),
    _us("GS", "Goldman Sachs", "financeiro", "banco de investimento", ("financeiro",)),
    _us("MS", "Morgan Stanley", "financeiro", "banco de investimento e gestão de patrimônio", ("financeiro",)),
    _us("AXP", "American Express", "financeiro", "cartões e pagamentos", ("financeiro",)),
    _us("BLK", "BlackRock", "financeiro", "gestão de ativos", ("financeiro",)),
    _us("LLY", "Eli Lilly", "saude", "medicamentos para diabetes, obesidade, oncologia e neurologia", ("saude",)),
    _us("UNH", "UnitedHealth", "saude", "planos de saúde e serviços de saúde", ("saude",)),
    _us("NVO", "Novo Nordisk (ADR)", "saude", "medicamentos para diabetes e obesidade", ("saude",)),
    _us("ABBV", "AbbVie", "saude", "medicamentos (imunologia, oncologia, neurociência)", ("saude",)),
    _us("MRK", "Merck & Co.", "saude", "medicamentos e vacinas", ("saude",)),
    _us("PFE", "Pfizer", "saude", "medicamentos e vacinas", ("saude",)),
    _us("TMO", "Thermo Fisher Scientific", "saude", "equipamentos e insumos para laboratórios e diagnósticos",
        ("saude",)),
    _us("ABT", "Abbott", "saude", "dispositivos médicos, diagnósticos e nutrição", ("saude",)),
    _us("AMGN", "Amgen", "saude", "biotecnologia: medicamentos biológicos", ("saude", "inovacao")),
    _us("ISRG", "Intuitive Surgical", "saude", "robôs para cirurgia minimamente invasiva",
        ("saude", "tecnologia", "inovacao")),
    _us("MDT", "Medtronic", "saude", "dispositivos médicos", ("saude",)),
    _us("GILD", "Gilead Sciences", "saude", "medicamentos antivirais e oncológicos", ("saude",)),
    _us("CVS", "CVS Health", "saude", "farmácias e planos de saúde", ("saude",)),
    _us("DUOL", "Duolingo", "educacao", "aplicativo de ensino de idiomas", ("educacao", "tecnologia")),
    _us("COUR", "Coursera", "educacao", "plataforma de cursos online", ("educacao", "tecnologia")),
    _us("LRN", "Stride", "educacao", "educação online para ensino básico", ("educacao",)),
    _us("NEE", "NextEra Energy", "utilidades", "energia elétrica; grande geradora eólica e solar",
        ("energia_limpa", "sustentabilidade", "infraestrutura")),
    _us("FSLR", "First Solar", "energia", "fabricação de painéis solares", ("energia_limpa", "sustentabilidade")),
    _us("ENPH", "Enphase Energy", "energia", "microinversores e baterias para energia solar",
        ("energia_limpa", "sustentabilidade", "tecnologia")),
    _us("DUK", "Duke Energy", "utilidades", "energia elétrica regulada", ("infraestrutura",)),
    _us("SO", "Southern Company", "utilidades", "energia elétrica regulada", ("infraestrutura",)),
    _us("XOM", "Exxon Mobil", "petroleo_gas", "petróleo, gás e química", ("petroleo_gas",)),
    _us("CVX", "Chevron", "petroleo_gas", "petróleo e gás", ("petroleo_gas",)),
    _us("PG", "Procter & Gamble", "consumo", "produtos de higiene, limpeza e beleza", ("consumo",)),
    _us("KO", "Coca-Cola", "consumo", "bebidas não alcoólicas", ("consumo",)),
    _us("PEP", "PepsiCo", "consumo", "bebidas e salgadinhos", ("consumo",)),
    _us("WMT", "Walmart", "consumo", "supermercados e varejo", ("consumo",)),
    _us("COST", "Costco", "consumo", "clubes de compras (atacado)", ("consumo",)),
    _us("HD", "Home Depot", "consumo", "varejo de material de construção", ("consumo",)),
    _us("MCD", "McDonald's", "consumo", "restaurantes de fast food", ("consumo",)),
    _us("SBUX", "Starbucks", "consumo", "cafeterias", ("consumo",)),
    _us("NKE", "Nike", "consumo", "calçados e vestuário esportivo", ("consumo",)),
    _us("DIS", "Walt Disney", "consumo", "entretenimento, streaming e parques temáticos", ("consumo",)),
    _us("CAT", "Caterpillar", "industria", "máquinas para construção e mineração", ("industria",)),
    _us("DE", "Deere & Company", "industria", "máquinas agrícolas", ("industria", "agronegocio")),
    _us("HON", "Honeywell", "industria", "automação industrial e aeroespacial", ("industria",)),
    _us("GE", "GE Aerospace", "industria", "motores de avião", ("industria",)),
    _us("BA", "Boeing", "industria", "aviões comerciais e defesa", ("industria",)),
    _us("LMT", "Lockheed Martin", "industria", "defesa e aeroespacial", ("industria",)),
    _us("UPS", "UPS", "logistica", "entregas e logística", ("infraestrutura",)),
    CatalogAsset("O", "Realty Income", "acoes_eua", "real_estate", "imobiliario", "US", "USD",
                 "REIT de imóveis comerciais com aluguel mensal", ("imoveis", "renda_recorrente",
                                                                    "diversificacao_global"), _REIT_US),
    CatalogAsset("PLD", "Prologis", "acoes_eua", "real_estate", "imobiliario", "US", "USD",
                 "REIT de galpões logísticos", ("imoveis", "renda_recorrente", "diversificacao_global"), _REIT_US),
    CatalogAsset("AMT", "American Tower", "acoes_eua", "real_estate", "imobiliario", "US", "USD",
                 "REIT de torres de telecomunicação", ("imoveis", "infraestrutura", "diversificacao_global"),
                 _REIT_US),
    CatalogAsset("EQIX", "Equinix", "acoes_eua", "real_estate", "imobiliario", "US", "USD",
                 "REIT de data centers", ("imoveis", "tecnologia", "diversificacao_global"), _REIT_US),
    # EUA — ETFs
    _etf_us("QQQ", "Invesco QQQ (Nasdaq-100)", "equity", "ETF das 100 maiores empresas não financeiras da Nasdaq",
            ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    _etf_us("VOO", "Vanguard S&P 500", "equity", "ETF que replica o S&P 500", ("diversificacao_global",), _EQ_US),
    _etf_us("VTI", "Vanguard Total Stock Market", "equity", "ETF de todo o mercado acionário dos EUA",
            ("diversificacao_global",), _EQ_US),
    _etf_us("IWM", "iShares Russell 2000", "equity", "ETF de ações de menor capitalização dos EUA",
            ("diversificacao_global",), _EQ_US),
    _etf_us("EFA", "iShares MSCI EAFE", "equity", "ETF de ações de países desenvolvidos fora dos EUA",
            ("diversificacao_global",), _EQ_US),
    _etf_us("EEM", "iShares MSCI Emerging Markets", "equity", "ETF de ações de países emergentes",
            ("diversificacao_global",), _EQ_US),
    _etf_us("SCHD", "Schwab US Dividend Equity", "equity", "ETF de ações americanas pagadoras de dividendos",
            ("renda_recorrente", "diversificacao_global"), _EQ_US),
    _etf_us("VYM", "Vanguard High Dividend Yield", "equity", "ETF de ações americanas de alto dividendo",
            ("renda_recorrente", "diversificacao_global"), _EQ_US),
    _etf_us("XLV", "Health Care Select Sector SPDR", "equity", "ETF do setor de saúde do S&P 500",
            ("saude", "diversificacao_global"), _EQ_US),
    _etf_us("XLK", "Technology Select Sector SPDR", "equity", "ETF do setor de tecnologia do S&P 500",
            ("tecnologia", "diversificacao_global"), _EQ_US),
    _etf_us("XLE", "Energy Select Sector SPDR", "equity", "ETF do setor de petróleo e gás do S&P 500",
            ("petroleo_gas", "diversificacao_global"), _EQ_US),
    _etf_us("XLF", "Financial Select Sector SPDR", "equity", "ETF do setor financeiro do S&P 500",
            ("financeiro", "diversificacao_global"), _EQ_US),
    _etf_us("XLU", "Utilities Select Sector SPDR", "equity", "ETF do setor de energia elétrica e utilidades",
            ("infraestrutura", "diversificacao_global"), _EQ_US),
    _etf_us("IBB", "iShares Biotechnology", "equity", "ETF de empresas de biotecnologia",
            ("saude", "inovacao", "diversificacao_global"), _EQ_US),
    _etf_us("TAN", "Invesco Solar", "equity", "ETF de empresas de energia solar",
            ("energia_limpa", "sustentabilidade", "diversificacao_global"), _EQ_US),
    _etf_us("ARKK", "ARK Innovation", "equity", "ETF ativo de empresas de inovação disruptiva",
            ("inovacao", "tecnologia", "especulativo"), _EQ_US),
    _etf_us("VNQ", "Vanguard Real Estate", "real_estate", "ETF de REITs dos EUA",
            ("imoveis", "renda_recorrente", "diversificacao_global"), _REIT_US),
    _etf_us("BND", "Vanguard Total Bond Market", "fixed_income", "ETF de títulos de renda fixa dos EUA",
            ("preservacao_capital", "diversificacao_global"), _RF_US),
    _etf_us("SHY", "iShares 1-3 Year Treasury", "fixed_income", "ETF de títulos do Tesouro dos EUA de 1 a 3 anos",
            ("preservacao_capital", "diversificacao_global"), _RF_US),
    _etf_us("IEF", "iShares 7-10 Year Treasury", "fixed_income", "ETF de títulos do Tesouro dos EUA de 7 a 10 anos",
            ("preservacao_capital", "diversificacao_global"), _RF_US),
    _etf_us("TLT", "iShares 20+ Year Treasury", "fixed_income",
            "ETF de títulos longos do Tesouro dos EUA (sensível a juros)", ("diversificacao_global",), _RF_US),
    _etf_us("TIP", "iShares TIPS Bond", "fixed_income", "ETF de títulos do Tesouro dos EUA indexados à inflação",
            ("preservacao_capital", "protecao_inflacao", "diversificacao_global"), _RF_US),
    _etf_us("LQD", "iShares Investment Grade Corporate", "fixed_income",
            "ETF de títulos corporativos grau de investimento dos EUA", ("diversificacao_global",), _RF_US),
    _etf_us("HYG", "iShares High Yield Corporate", "fixed_income",
            "ETF de títulos corporativos de alto rendimento (maior risco de crédito)", ("diversificacao_global",),
            _RF_US),
    _etf_us("SLV", "iShares Silver Trust", "commodity", "ETF lastreado em prata", ("reserva_valor",), _GOLD),
    CatalogAsset("ETH-USD", "Ethereum", "cripto", "crypto", "cripto", "GLOBAL", "USD",
                 "criptoativo da rede Ethereum", ("especulativo", "inovacao"), _CRYPTO),
)

REAL_CATALOG: tuple[CatalogAsset, ...] = BASE_CATALOG + EXTENDED_CATALOG

SYNTHETIC_CATALOG: tuple[CatalogAsset, ...] = (
    CatalogAsset("SINT_ENERGIA3", "Sintética Energia Renovável", "acoes_br", "equity", "utilidades", "BR", "BRL",
                 "[sintético] geração de energia eólica e solar", ("energia_limpa", "sustentabilidade",
                                                                   "infraestrutura"), _EQ_BR),
    CatalogAsset("SINT_BANCO4", "Sintético Banco", "acoes_br", "equity", "financeiro", "BR", "BRL",
                 "[sintético] banco de varejo", ("financeiro",), _EQ_BR),
    CatalogAsset("SINT_PETRO4", "Sintética Petróleo", "acoes_br", "equity", "petroleo_gas", "BR", "BRL",
                 "[sintético] produção de petróleo e gás", ("petroleo_gas", "industria"), _EQ_BR),
    CatalogAsset("SINT_SAUDE3", "Sintética Saúde", "acoes_br", "equity", "saude", "BR", "BRL",
                 "[sintético] rede de hospitais e diagnósticos", ("saude",), _EQ_BR),
    CatalogAsset("SINT_EDUCA3", "Sintética Educação", "acoes_br", "equity", "educacao", "BR", "BRL",
                 "[sintético] escolas e ensino superior", ("educacao",), _EQ_BR),
    CatalogAsset("SINT_VAREJO3", "Sintético Varejo", "acoes_br", "equity", "consumo", "BR", "BRL",
                 "[sintético] varejo de bens de consumo", ("consumo",), _EQ_BR),
    CatalogAsset("SINT_LOGI11", "Sintético FII Logística", "fii", "real_estate", "imobiliario", "BR", "BRL",
                 "[sintético] fundo imobiliário de galpões logísticos", ("imoveis", "renda_recorrente"), _FII),
    CatalogAsset("SINT_IPCA", "Sintético Tesouro IPCA+", "renda_fixa_br", "fixed_income", "titulos_publicos", "BR",
                 "BRL", "[sintético] títulos públicos indexados à inflação", ("preservacao_capital",
                                                                               "protecao_inflacao"), _RF_BR),
    CatalogAsset("SINT_SELIC", "Sintético Tesouro Selic", "renda_fixa_br", "fixed_income", "pos_fixado", "BR", "BRL",
                 "[sintético] títulos pós-fixados (caixa)", ("preservacao_capital",), _RF_BR),
    CatalogAsset("SINT_TECH_US", "Sintética Tecnologia EUA", "acoes_eua", "equity", "tecnologia", "US", "USD",
                 "[sintético] software e nuvem", ("tecnologia", "inovacao", "diversificacao_global"), _EQ_US),
    CatalogAsset("SINT_SAUDE_US", "Sintética Saúde EUA", "acoes_eua", "equity", "saude", "US", "USD",
                 "[sintético] farmacêutica", ("saude", "diversificacao_global"), _EQ_US),
    CatalogAsset("SINT_CLEAN_US", "Sintético ETF Energia Limpa", "etf_eua", "equity", "energia", "GLOBAL", "USD",
                 "[sintético] ETF global de energia limpa", ("energia_limpa", "sustentabilidade",
                                                              "diversificacao_global"), _EQ_US),
    CatalogAsset("SINT_TREAS_US", "Sintético Títulos EUA", "etf_eua", "fixed_income", "titulos", "US", "USD",
                 "[sintético] títulos do Tesouro dos EUA", ("preservacao_capital", "diversificacao_global"), _RF_US),
    CatalogAsset("SINT_OURO_US", "Sintético Ouro", "ouro", "commodity", "ouro", "GLOBAL", "USD",
                 "[sintético] ouro físico", ("reserva_valor",), _GOLD),
)


def by_ticker(catalog: tuple[CatalogAsset, ...]) -> dict[str, CatalogAsset]:
    return {a.ticker: a for a in catalog}


def expand(catalog: tuple[CatalogAsset, ...], macro_classes: list[str] | None = None) -> list[CatalogAsset]:
    """Expand macro classes into individual tickers (all classes when None)."""
    if macro_classes is None:
        return list(catalog)
    unknown = set(macro_classes) - {a.macro_class for a in catalog}
    if unknown:
        raise ValueError(f"classes desconhecidas: {sorted(unknown)}")
    return [a for a in catalog if a.macro_class in macro_classes]
