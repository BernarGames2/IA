"""Value-trap detector: a stock that LOOKS cheap may be cheap for a reason. Fixed, documented rules.

1. "Parece barata?" — any of: P/L between 0 and 8; P/VP up to 1; dividend yield of 8% or more; free-cash-flow
   yield of 10% or more; EV/EBITDA between 0 and 5 (not for banks and insurers); price 40% or more below the
   highest close of the last 3 years.
2. Signals that the cheapness may be deserved (each measured from published data, or "não medido"):
   * Piotroski F-score of 3 or less (weak fundamentals for the year; ``analysis.piotroski``);
   * revenue falling: 3-year growth below 0;
   * profit falling two years in a row, or a loss in the latest year;
   * operating margin down 3 percentage points or more in 3 years (not for banks/insurers);
   * negative free cash flow in the latest year (not for banks/insurers);
   * dividends paid above the free cash flow (not for banks/insurers);
   * net debt above 3x EBITDA (not for banks/insurers);
   * more than 5% new shares in 3 years (dilution);
   * analysts cutting: this year's EPS estimate 5% or more below 90 days ago, or more cuts than raises in 30 days
     (third-party estimates);
   * price in a downtrend: 12-month return excluding the last month below -20% AND below the 200-day average;
   * dividend yield of 15% or more (the market may be pricing a cut).
3. Level: "alto" with 4 or more signals (or F-score of 2 or less), "moderado" with 2–3, "baixo" with 0–1;
   "indeterminado" when fewer than 4 signals could be measured. The detector reads like a checklist, not a
   verdict: a cheap stock with no signal can still fall, and one with signals can recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.fundamentals.analysis import (
    dividends_paid,
    fcf,
    growth,
    piotroski,
    valuation,
    year_ratios,
)
from core.fundamentals.sources import Analysts, Annual
from utils.br_format import millions, num, pct

CHEAP_PE, CHEAP_PB, CHEAP_DY, CHEAP_FCF, CHEAP_EV, CHEAP_DD = 8.0, 1.0, 0.08, 0.10, 5.0, -0.40


def price_stats(px: pd.Series) -> dict[str, float | None]:
    """Momentum 12-1, position vs the 200-day average and distance from the 3-year high (adjusted closes)."""
    px = px.dropna()
    out: dict[str, float | None] = {"momento_12_1": None, "acima_media_200": None, "queda_do_topo_3a": None,
                                    "ultimo": float(px.iloc[-1]) if len(px) else None}
    if len(px) > 252:
        out["momento_12_1"] = float(px.iloc[-22] / px.iloc[-253] - 1.0)
    if len(px) >= 200:
        out["acima_media_200"] = float(px.iloc[-1] > px.iloc[-200:].mean())
    if len(px) >= 252:
        top = float(px.iloc[-756:].max())
        out["queda_do_topo_3a"] = float(px.iloc[-1] / top - 1.0)
    return out


def _sig(name: str, hit: bool | None, value: str, rule: str, src: str) -> dict[str, object]:
    return {"sinal": name, "acionado": hit, "valor": value, "regra": rule, "fonte": src}


def detect(a: Annual | None, snap: dict[str, object], an: Analysts | None, px_stats: dict[str, float | None],
           financial: bool = False) -> dict[str, object]:
    val = valuation(a, snap)
    cheap = []
    pe, pb, dy, fy, ev = val["pl"], val["p_vp"], val["dividend_yield"], val["fcl_yield"], val["ev_ebitda"]
    dd = px_stats.get("queda_do_topo_3a")
    if pe is not None and 0 < pe <= CHEAP_PE:
        cheap.append(f"P/L de {num(pe)} (até {CHEAP_PE:.0f})")
    if pb is not None and pb <= CHEAP_PB:
        cheap.append(f"P/VP de {num(pb, 2)} (até {CHEAP_PB:.0f})")
    if dy is not None and dy >= CHEAP_DY:
        cheap.append(f"dividendos de {pct(dy)} ao ano (8% ou mais)")
    if fy is not None and fy >= CHEAP_FCF:
        cheap.append(f"fluxo de caixa livre de {pct(fy)} do valor de mercado (10% ou mais)")
    if not financial and ev is not None and 0 < ev <= CHEAP_EV:
        cheap.append(f"EV/EBITDA de {num(ev)} (até {CHEAP_EV:.0f})")
    if dd is not None and dd <= CHEAP_DD:
        cheap.append(f"preço {pct(-dd, 0)} abaixo do maior fechamento de 3 anos (40% ou mais)")

    src_fin = a.fonte + f" (consultado em {a.consultado_em})" if a else "demonstrações indisponíveis"
    src_an = an.fonte + f" (consultado em {an.consultado_em})" if an else "estimativas indisponíveis"
    sig = []
    fs = piotroski(a) if a is not None and not financial else None
    if fs is not None and fs["medidos"] >= 6:
        sig.append(_sig("fundamentos fracos no ano (Piotroski F-score)", fs["pontos"] <= 3,  # type: ignore[operator]
                        f"{fs['pontos']} de {fs['medidos']} testes", "3 pontos ou menos", src_fin))
    else:
        sig.append(_sig("fundamentos fracos no ano (Piotroski F-score)", None,
                        "não se aplica a bancos e seguradoras" if financial else "dados insuficientes", "3 pontos ou menos", src_fin))
    gr = growth(a) if a is not None else {}
    rc = gr.get("receita_cagr_3a")
    sig.append(_sig("receita em queda", None if rc is None else rc < 0, f"{pct(rc)} ao ano em 3 anos", "abaixo de 0", src_fin))
    ni = [a.get("lucro_liquido", i) for i in (-3, -2, -1)] if a is not None and len(a.anos) >= 3 else []
    hit_ni = None
    if len(ni) == 3 and None not in ni:
        hit_ni = (ni[2] < ni[1] < ni[0]) or ni[2] < 0  # type: ignore[operator]
    elif a is not None and a.get("lucro_liquido", -1) is not None:
        hit_ni = True if a.get("lucro_liquido", -1) < 0 else None  # type: ignore[operator]
    sig.append(_sig("lucro em queda ou prejuízo", hit_ni,
                    " → ".join(millions(x) for x in ni) or "n/d",
                    "dois anos seguidos de queda, ou prejuízo no último", src_fin))
    if not financial:
        m_now = year_ratios(a, -1)["margem_operacional"] if a is not None else None
        m_old = year_ratios(a, -4)["margem_operacional"] if a is not None and len(a.anos) >= 4 else None
        d_m = m_now - m_old if m_now is not None and m_old is not None else None
        sig.append(_sig("margem operacional encolhendo", None if d_m is None else d_m <= -0.03,
                        "n/d" if d_m is None else f"{pct(m_old)} → {pct(m_now)}", "queda de 3 p.p. ou mais em 3 anos", src_fin))
        f_last = fcf(a, -1) if a is not None else None
        sig.append(_sig("fluxo de caixa livre negativo", None if f_last is None else f_last < 0,
                        millions(f_last), "abaixo de 0 no último ano", src_fin))
        dp = dividends_paid(a, -1) if a is not None else None
        sig.append(_sig("dividendos acima do caixa livre", None if f_last is None or dp is None else (dp > 0 and dp > f_last),
                        "n/d" if f_last is None or dp is None else f"pagou {millions(dp)}; caixa livre {millions(f_last)}",
                        "dividendos pagos maiores que o fluxo de caixa livre", src_fin))
        nde = year_ratios(a, -1)["divida_liquida_ebitda"] if a is not None else None
        sig.append(_sig("dívida alta", None if nde is None else nde > 3.0, "n/d" if nde is None else f"{num(nde)}× o EBITDA",
                        "dívida líquida acima de 3× o EBITDA", src_fin))
    dil = gr.get("acoes_variacao")
    sig.append(_sig("diluição dos acionistas", None if dil is None else dil > 0.05,
                    "n/d" if dil is None else f"{pct(dil, sign=True)} de ações em {gr.get('acoes_anos')} anos", "mais de 5% em 3 anos", src_fin))
    rev = None
    if an is not None:
        cur, old = an.eps_ano.get("current"), an.eps_ano.get("90daysAgo")
        up, down = an.revisoes_30d.get("alta"), an.revisoes_30d.get("baixa")
        cut = (cur / old - 1.0) if cur is not None and old is not None and old > 0 else None
        hits = [x for x in ((None if cut is None else cut <= -0.05), (None if up is None or down is None else down > up)) if x is not None]
        rev = any(hits) if hits else None
        txt = ("n/d" if cut is None else f"estimativa de LPA do ano {pct(cut, sign=True)} em 90 dias") + (
            "" if up is None or down is None else f"; {up} altas e {down} cortes em 30 dias")
        sig.append(_sig("analistas cortando estimativas (terceiros)", rev, txt, "queda de 5% ou mais em 90 dias, ou mais cortes que altas", src_an))
    else:
        sig.append(_sig("analistas cortando estimativas (terceiros)", None, "n/d", "queda de 5% ou mais em 90 dias, ou mais cortes que altas", src_an))
    mom, above = px_stats.get("momento_12_1"), px_stats.get("acima_media_200")
    sig.append(_sig("preço em tendência de queda", None if mom is None or above is None else (mom < -0.20 and above < 0.5),
                    "n/d" if mom is None else f"{pct(mom, sign=True)} em 12 meses (sem o último)" + ("" if above is None else
                                                                                        ("; acima" if above else "; abaixo") + " da média de 200 dias"),
                    "abaixo de -20% e abaixo da média de 200 dias", "cotações ajustadas"))
    sig.append(_sig("dividendos altos demais", None if dy is None else dy >= 0.15, pct(dy), "15% ou mais ao ano", "Yahoo Finance"))

    measured = [s for s in sig if s["acionado"] is not None]
    n_hit = sum(1 for s in measured if s["acionado"])
    weak = fs is not None and fs["pontos"] is not None and fs["medidos"] >= 6 and fs["pontos"] <= 2  # type: ignore[operator]
    if len(measured) < 4:
        level = "indeterminado"
    elif n_hit >= 4 or weak:
        level = "alto"
    elif n_hit >= 2:
        level = "moderado"
    else:
        level = "baixo"
    if not cheap:
        reading = (f"Não parece barata pelos critérios do detector. Mesmo assim, {n_hit} de {len(measured)} sinais de "
                   "deterioração foram acionados." if measured else "Não parece barata pelos critérios, e quase nada pôde ser medido.")
    elif level == "indeterminado":
        reading = "Parece barata, mas faltam dados para separar uma pechincha de uma armadilha."
    else:
        reading = {"alto": "Parece barata, e há vários sinais de que o preço baixo pode ser merecido (risco ALTO de armadilha).",
                   "moderado": "Parece barata, com alguns sinais de deterioração (risco MODERADO de armadilha).",
                   "baixo": "Parece barata e quase nenhum sinal de deterioração foi acionado (risco BAIXO de armadilha pelas regras)."}[level]
    return {"parece_barata": bool(cheap), "criterios_barata": cheap, "sinais": sig, "acionados": n_hit, "medidos": len(measured),
            "nivel": level if cheap else ("indeterminado" if level == "indeterminado" else "não se aplica (não parece barata)"),
            "nivel_sinais": level, "leitura": reading, "piotroski": fs, "multiplos": val,
            "regras": ("barata: P/L ≤ 8, P/VP ≤ 1, dividendos ≥ 8%, caixa livre ≥ 10% do valor de mercado, EV/EBITDA ≤ 5 ou "
                       "queda ≥ 40% do topo de 3 anos; risco alto com 4+ sinais ou F-score ≤ 2, moderado com 2–3, baixo com 0–1, "
                       "indeterminado com menos de 4 sinais medidos")}


def trend_table(a: Annual) -> list[dict[str, object]]:
    """One row per fiscal year with the published values and the derived ratios (for the page)."""
    rows = []
    for i, y in enumerate(a.anos):
        r = year_ratios(a, i)
        rows.append({"ano": y, "receita": a.get("receita", i), "lucro_liquido": a.get("lucro_liquido", i),
                     "ebitda": a.get("ebitda", i), "fcl": r["fcl"], "divida_liquida": r["divida_liquida"],
                     "margem_operacional": r["margem_operacional"], "margem_liquida": r["margem_liquida"], "roe": r["roe"],
                     "divida_liquida_ebitda": r["divida_liquida_ebitda"], "dividendos_pagos": dividends_paid(a, i),
                     "lpa": a.get("lpa", i)})
    return rows


def finite_or_none(x: object) -> object:
    return float(x) if isinstance(x, float | np.floating) and np.isfinite(x) else (None if isinstance(x, float) else x)
