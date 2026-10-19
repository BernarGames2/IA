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

REAL_CATALOG: tuple[CatalogAsset, ...] = (
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
                 "rede de farmácias e drogarias", ("saude", "consumo"), _EQ_BR),
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
