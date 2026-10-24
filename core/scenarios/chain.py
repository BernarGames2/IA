"""Economic chain effects: "what tends to happen to each sector if rates, the dollar, oil or the US market move?"

Two layers, always shown side by side and never mixed:

* the CHAIN [hipótese]: how the shock is usually transmitted (editorial, general economics) and the effect each
  sector is EXPECTED to feel (+ helps, − hurts, ? ambiguous), with the reason;
* the DATA [H/E]: how each stock actually moved with the shock's market proxy over the last 5 years — beta of weekly
  returns (Friday to Friday) on the proxy's weekly returns, with Newey–West t-statistic — and the implied effect of
  the chosen shock (beta × the proxy's move). Per sector: median effect, how many stocks, and whether the data agree
  with the chain ("confirmam": same sign and at least half of the stocks with |t| ≥ 2 in that direction;
  "contrariam": the opposite; otherwise "inconclusivo").

Proxies: Brazilian rates → IRFM11 (fixed-rate government bonds ETF; its return ≈ −2 × the change in rates, an
approximate duration [hipótese]); dollar → USDBRL=X; oil → Brent futures BZ=F; US market → SPY; US rates → IEF
(7–10-year Treasuries; return ≈ −7.5 × the change in rates [hipótese]). A beta is a past average relation: it can
change, it mixes every other thing that moved on the same weeks, and the combined effect of two shocks is not the sum.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.factors.models import factor_regression
from core.market_data.base import DataUnavailableError, last_per_date
from core.market_data.logos import logo_path
from core.purpose.catalog import REAL_CATALOG, SECTOR_LABEL, by_ticker
from core.purpose.engine import norm_ticker
from core.purpose.language import guarantee_phrases
from utils.br_format import num, pct
from utils.validation import InsufficientDataError

MIN_WEEKS = 104
T_SIG = 2.0
CATALOG = by_ticker(REAL_CATALOG)

SHOCKS: dict[str, dict[str, object]] = {
    "juros_br": {
        "nome": "Juros no Brasil sobem", "frase": ("os juros no Brasil subirem", "os juros no Brasil caírem"), "unidade": "p.p.", "padrao": 1.0, "limites": (-5.0, 5.0), "proxy": "IRFM11.SA",
        "conversao": -2.0, "conversao_nota": "IRFM11 varia cerca de −2 × a mudança dos juros (duração aproximada; hipótese)",
        "cadeia": [("O dinheiro fica mais caro", "empréstimos e financiamentos encarecem para empresas e famílias"),
                   ("Fluxos futuros valem menos", ("o valor presente de lucros distantes cai: empresas de crescimento e "
                                                   "de prazos longos tendem a sofrer mais")),
                   ("A renda fixa fica mais atraente", "parte do dinheiro tende a sair da bolsa"),
                   ("A economia desacelera depois de alguns meses", "consumo e investimento a prazo perdem força")],
        "setores": {"consumo": (-1, "crédito ao consumidor mais caro reduz vendas a prazo"),
                    "construcao": (-1, "financiamento imobiliário mais caro"),
                    "imobiliario": (-1, "imóveis e aluguéis competem com a renda fixa; dívida mais cara"),
                    "utilidades": (-1, "empresas endividadas e de retorno longo"), "saneamento": (-1, "endividadas, retorno longo"),
                    "energia": (-1, "projetos longos financiados com dívida"), "tecnologia": (-1, "lucros mais distantes valem menos"),
                    "educacao": (-1, "renda das famílias e financiamento estudantil"), "industria": (-1, "investimento e crédito mais caros"),
                    "logistica": (-1, "empresas endividadas e de investimento pesado"), "automotivo": (-1, "financiamento de veículos"),
                    "financeiro": (0, "bancos: a margem pode subir, mas inadimplência e crédito menor pesam"),
                    "saude": (0, "demanda pouco sensível, mas há empresas endividadas"),
                    "petroleo_gas": (0, "receita ligada ao petróleo em dólar"), "mineracao": (0, "exportadora, receita em dólar"),
                    "florestal": (0, "exportadora, receita em dólar"), "agronegocio": (0, "exportação e crédito rural"),
                    "siderurgia": (0, "demanda local e exportação"), "telecom": (0, "receita estável, mas endividadas"),
                    "servicos": (0, "efeitos mistos")}},
    "dolar": {
        "nome": "Dólar sobe frente ao real", "frase": ("o dólar subir frente ao real", "o dólar cair frente ao real"), "unidade": "%", "padrao": 10.0, "limites": (-30.0, 30.0), "proxy": "USDBRL=X",
        "conversao": 0.01, "conversao_nota": "o choque é a própria variação do USDBRL=X",
        "cadeia": [("Exportadoras recebem mais reais", "receita em dólar e custos em real aumentam a margem"),
                   ("Importados ficam mais caros", "insumos e produtos importados encarecem e a inflação tende a subir"),
                   ("Dívida em dólar pesa mais", "empresas endividadas em moeda forte pagam mais em reais"),
                   ("Juros podem subir para conter a inflação", "o que encadeia os efeitos do choque de juros")],
        "setores": {"mineracao": (1, "receita em dólar, custos em real"), "florestal": (1, "celulose exportada"),
                    "petroleo_gas": (1, "preços em dólar"), "agronegocio": (1, "exportação de alimentos"),
                    "siderurgia": (1, "exportação e proteção contra importados"),
                    "consumo": (-1, "importados e inflação reduzem a renda disponível"),
                    "saude": (-1, "insumos e equipamentos importados"), "automotivo": (-1, "peças importadas"),
                    "telecom": (-1, "equipamentos importados"), "industria": (0, "exportadoras ganham, importadoras perdem"),
                    "financeiro": (0, "efeitos mistos"), "utilidades": (0, "receita regulada em real"),
                    "saneamento": (0, "receita regulada em real"), "energia": (0, "receita em real"), "tecnologia": (0, "efeitos mistos"),
                    "educacao": (0, "receita em real"), "construcao": (0, "custos de insumos sobem"), "imobiliario": (0, "efeitos mistos"),
                    "logistica": (0, "mais volume exportado, diesel e dívida mais caros"), "servicos": (0, "efeitos mistos")}},
    "petroleo": {
        "nome": "Petróleo sobe", "frase": ("o petróleo subir", "o petróleo cair"), "unidade": "%", "padrao": 20.0, "limites": (-60.0, 60.0), "proxy": "BZ=F",
        "conversao": 0.01, "conversao_nota": "o choque é a própria variação do Brent (BZ=F)",
        "cadeia": [("Produtoras faturam mais", "o preço do barril eleva receita e margem de quem extrai"),
                   ("Combustíveis encarecem", "frete, transporte e produção ficam mais caros"),
                   ("Inflação tende a subir", "o que pode pressionar os juros e a renda das famílias")],
        "setores": {"petroleo_gas": (1, "produtoras e distribuidoras de combustíveis"),
                    "logistica": (-1, "diesel mais caro"), "consumo": (-1, "renda das famílias pressionada"),
                    "automotivo": (-1, "uso e venda de veículos"), "agronegocio": (0, "fertilizantes e diesel mais caros, mas biocombustíveis"),
                    "utilidades": (0, "pouco ligadas ao petróleo"), "industria": (0, "custos de energia e frete")}},
    "bolsa_eua": {
        "nome": "Bolsa americana cai", "frase": ("a bolsa americana subir", "a bolsa americana cair"), "unidade": "%", "padrao": -10.0, "limites": (-50.0, 50.0), "proxy": "SPY",
        "conversao": 0.01, "conversao_nota": "o choque é a própria variação do SPY (S&P 500)",
        "cadeia": [("Investidores reduzem risco no mundo todo", "emergentes costumam cair junto"),
                   ("Empresas globais sentem a demanda americana", "exportadoras e multinacionais")],
        "setores": {}},
    "juros_eua": {
        "nome": "Juros nos EUA sobem", "frase": ("os juros nos EUA subirem", "os juros nos EUA caírem"), "unidade": "p.p.", "padrao": 1.0, "limites": (-3.0, 3.0), "proxy": "IEF",
        "conversao": -7.5, "conversao_nota": "IEF varia cerca de −7,5 × a mudança dos juros de 7 a 10 anos (duração aproximada; hipótese)",
        "cadeia": [("Fluxos futuros valem menos", "empresas de crescimento e de prazos longos tendem a sofrer mais"),
                   ("O dólar tende a se fortalecer", "o que pesa sobre emergentes e encadeia o choque de câmbio"),
                   ("Imóveis e serviços públicos com dívida sofrem", "custo de capital mais alto")],
        "setores": {"tecnologia": (-1, "lucros mais distantes valem menos"), "imobiliario": (-1, "fundos imobiliários e REITs com dívida"),
                    "utilidades": (-1, "retorno longo e endividadas"), "financeiro": (0, "bancos ganham margem, mas o crédito encolhe")}},
}
PROXIES = {k: str(v["proxy"]) for k, v in SHOCKS.items()}


def weekly(s: pd.Series) -> pd.Series:
    s = last_per_date(s.dropna()).sort_index()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    return s.resample("W-FRI").last().pct_change(fill_method=None).dropna()


def _agreement(expected: int, effects: list[tuple[float, float]]) -> str:
    """``effects``: (effect of the shock, t-stat of the beta) per stock."""
    if not effects:
        return "sem dados"
    med = float(np.median([e for e, _ in effects]))
    sig_same = sum(1 for e, t in effects if abs(t) >= T_SIG and np.sign(e) == np.sign(med))
    strong = sig_same >= len(effects) / 2
    if expected == 0:
        return f"teoria ambígua; os dados indicam {'alta' if med > 0 else 'queda'}" + ("" if strong else " (fraco)")
    if not strong:
        return "inconclusivo"
    return "dados confirmam" if np.sign(med) == expected else "dados contrariam"


def chain_effects(shock: str, size: float | None, market: str, provider: object, tickers: list[str] | None = None,
                  progress: Callable[[str], None] | None = None) -> dict[str, object]:
    if shock not in SHOCKS:
        raise ValueError(f"choque desconhecido: {shock}")
    spec = SHOCKS[shock]
    size = float(spec["padrao"] if size is None else size)  # type: ignore[arg-type]
    lo, hi = spec["limites"]  # type: ignore[misc]
    if not lo <= size <= hi or size == 0:
        raise ValueError(f"tamanho do choque entre {num(lo, 0)} e {num(hi, 0)} {spec['unidade']} (diferente de zero)")
    proxy = str(spec["proxy"])
    conv = float(spec["conversao"])  # type: ignore[arg-type]
    move = size / 100 * conv if spec["unidade"] == "p.p." else size * conv     # the proxy's return for this shock
    own = [norm_ticker(t) for t in (tickers or [])]
    if own:
        names = own
    else:
        cls = {"BR": "acoes_br", "US": "acoes_eua"}.get(market)
        if cls is None:
            raise ValueError("mercado: BR ou US")
        names = [a.ticker for a in REAL_CATALOG if a.macro_class == cls]
    if progress:
        progress(f"baixando 5 anos de cotações de {len(names)} ações e do indicador do choque ({proxy})…")
    got = provider.fetch([*names, proxy], "5y", "1d")  # type: ignore[attr-defined]
    if proxy not in got or got[proxy].close is None or got[proxy].close.dropna().empty:
        why = getattr(provider, "failures", {}).get(proxy, "sem cotações")
        raise DataUnavailableError(f"indicador do choque ({proxy}) indisponível agora: {why}")
    f = weekly(got[proxy].close).rename("proxy").to_frame()
    rows, skipped = [], []
    for t in names:
        rs = got.get(t)
        if rs is None or rs.close is None:
            skipped.append(t)
            continue
        try:
            reg = factor_regression(weekly(rs.close), f, periods=52, lag=4)
        except InsufficientDataError:
            skipped.append(t)
            continue
        if reg["n_obs"] < MIN_WEEKS:
            skipped.append(t)
            continue
        b, tt = float(reg["coef"]["proxy"]), float(reg["t_hac"]["proxy"])
        se = float(reg["se_hac"]["proxy"])
        a = CATALOG.get(t)
        sector = a.sector if a else "nao_classificado"
        rows.append({"ticker": t, "nome": a.name if a else t, "logo": logo_path(t), "setor": sector,
                     "setor_nome": SECTOR_LABEL.get(sector, sector), "beta": b, "t": tt, "r2": float(reg["r2"]),
                     "semanas": int(reg["n_obs"]), "efeito": b * move,
                     "faixa_95": sorted([(b - 1.96 * se) * move, (b + 1.96 * se) * move])})
    if not rows:
        raise ValueError("nenhuma ação com 2 anos de cotações semanais em comum com o indicador do choque")
    df = pd.DataFrame(rows)
    sectors = []
    theory = spec["setores"]  # type: ignore[assignment]
    for sec, g in df.groupby("setor"):
        exp, why = theory.get(sec, (None, None)) if isinstance(theory, dict) else (None, None)  # type: ignore[union-attr]
        if shock == "bolsa_eua":
            exp, why = (1 if size > 0 else -1), "as bolsas costumam se mover juntas"
        eff = list(zip(g["efeito"], g["t"], strict=True))
        sectors.append({
            "setor": sec, "setor_nome": SECTOR_LABEL.get(str(sec), str(sec)), "n": len(g),
            "esperado": None if exp is None else {1: "+", -1: "−", 0: "?"}[exp], "motivo": why,
            "efeito_mediano": float(np.median(g["efeito"])), "efeito_p25": float(np.percentile(g["efeito"], 25)),
            "efeito_p75": float(np.percentile(g["efeito"], 75)),
            "significativos": int((g["t"].abs() >= T_SIG).sum()),
            "concordancia": "sem leitura teórica" if exp is None else _agreement(int(exp), eff),
            "acoes": g.reindex(g["efeito"].abs().sort_values(ascending=False).index)[["ticker", "efeito", "t"]].head(3).to_dict("records"),
        })
    sectors.sort(key=lambda x: x["efeito_mediano"])
    warn = [(f"beta de retornos semanais dos últimos 5 anos (mínimo {MIN_WEEKS} semanas), com erro-padrão de Newey–West; "
             "|t| ≥ 2 indica relação estatisticamente clara"),
            "uma relação passada pode mudar; ela também carrega tudo o mais que mudou nas mesmas semanas",
            f"{spec['conversao_nota']}"]
    if skipped:
        warn.append(f"sem dados suficientes, fora da análise: {', '.join(sorted(skipped))}")
    out = {
        "choque": {"chave": shock, "nome": spec["nome"], "tamanho": size, "unidade": spec["unidade"],
                   "descricao": describe(spec, size), "indicador": proxy,
                   "movimento_indicador": move},
        "cadeia": [{"etapa": a, "explicacao": b} for a, b in spec["cadeia"]],  # type: ignore[union-attr]
        "setores": sectors,
        "acoes": sorted(rows, key=lambda r: r["efeito"]),
        "mercado": market if not own else "seus ativos",
        "avisos": warn,
        "aviso": ("Efeitos estimados a partir de relações passadas e de uma cadeia econômica geral (hipótese): não são "
                  "previsões, não somam quando há mais de um choque e não são recomendação."),
    }
    texts = [*warn, out["aviso"], *(x["explicacao"] for x in out["cadeia"]), *(str(s["motivo"] or "") for s in sectors)]
    bad = guarantee_phrases(" ".join(texts))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    out["leitura"] = _reading(out, size, spec)
    return out


def describe(spec: dict[str, object], size: float) -> str:
    up, down = spec["frase"]  # type: ignore[misc]
    amount = num(abs(size), 0 if float(size).is_integer() else 1) + ("%" if spec["unidade"] == "%" else " p.p.")
    return f"se {up if size > 0 else down} {amount}"


def _reading(out: dict[str, object], size: float, spec: dict[str, object]) -> list[str]:
    secs = out["setores"]
    if not secs:
        return []
    worst, best = secs[0], secs[-1]  # type: ignore[index]
    what = describe(spec, size)
    lines = [(f"{what[0].upper()}{what[1:]}, o pior efeito estimado pelos dados passados é o de {worst['setor_nome']} "
              f"(mediana de {pct(worst['efeito_mediano'], 1, sign=True)}) e o melhor, o de {best['setor_nome']} "
              f"({pct(best['efeito_mediano'], 1, sign=True)}).")]
    agree = [s["setor_nome"] for s in secs if s["concordancia"] == "dados confirmam"]  # type: ignore[union-attr]
    contra = [s["setor_nome"] for s in secs if s["concordancia"] == "dados contrariam"]  # type: ignore[union-attr]
    if agree:
        lines.append("Os dados confirmam a cadeia em: " + ", ".join(agree) + ".")
    if contra:
        lines.append("Os dados contrariam a cadeia em: " + ", ".join(contra) + " — a teoria geral não descreveu bem esses casos.")
    return lines
