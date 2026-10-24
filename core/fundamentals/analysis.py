"""Measured fundamentals from the annual statements: margins, growth, cash generation, leverage, the Piotroski
F-score and valuation multiples. Every number is a ratio of published values [H]; a ratio whose inputs are
missing is None ("não medido"), never estimated.

Sign conventions of the source: capital expenditure and dividends paid are cash outflows (negative); free cash
flow = operating cash flow + capex when the source does not publish it.

Piotroski F-score (Piotroski, 2000, "Value Investing: The Use of Historical Financial Statement Information to
Separate Winners from Losers", Journal of Accounting Research 38): nine yes/no tests on the latest fiscal year t
against t-1 — profitability (ROA > 0, operating cash flow > 0, ROA up, cash flow above net income), funding
(leverage down, current ratio up, no new shares) and efficiency (gross margin up, asset turnover up). Among cheap
stocks, low scores (0–3) flagged the ones that kept doing badly. ROA and turnover use assets at the start of the
year; leverage = long-term debt (total debt when the source has no split) / average assets. Not computed for banks
and insurers, as in the paper.
"""

from __future__ import annotations

from core.fundamentals.sources import Annual
from utils.br_format import num, pct

NEW_SHARES_TOL = 0.005          # +0.5% of diluted shares is rounding, not an issue of stock


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def fcf(a: Annual, i: int) -> float | None:
    v = a.get("fcl", i)
    if v is not None:
        return v
    cfo, capex = a.get("fco", i), a.get("capex", i)
    return cfo + capex if cfo is not None and capex is not None else None


def dividends_paid(a: Annual, i: int) -> float | None:
    v = a.get("dividendos_pagos", i)
    return abs(v) if v is not None else None


def year_ratios(a: Annual, i: int) -> dict[str, float | None]:
    """Ratios of fiscal year ``i`` (negative counts from the latest); start-of-year values come from year i-1."""
    i %= len(a.anos)
    g = lambda k, j: a.get(k, j) if j is not None else None
    prev = i - 1 if i >= 1 else None
    rev, ni, assets, cfo = g("receita", i), g("lucro_liquido", i), g("ativos", i), g("fco", i)
    a0 = g("ativos", prev)
    eq, eq0 = g("patrimonio", i), g("patrimonio", prev)
    avg_eq = (eq + eq0) / 2 if eq is not None and eq0 is not None else eq
    debt, cash, ebitda = g("divida_total", i), g("caixa", i), g("ebitda", i)
    net_debt = debt - cash if debt is not None and cash is not None else None
    f = fcf(a, i)
    dp = dividends_paid(a, i)
    return {
        "margem_bruta": _div(g("lucro_bruto", i), rev),
        "margem_operacional": _div(g("lucro_operacional", i), rev),
        "margem_liquida": _div(ni, rev),
        "margem_fcl": _div(f, rev),
        "roe": _div(ni, avg_eq) if avg_eq is not None and avg_eq > 0 else None,
        "roa": _div(ni, a0 if a0 is not None else assets),
        "divida_liquida": net_debt,
        "divida_liquida_ebitda": _div(net_debt, ebitda) if ebitda is not None and ebitda > 0 else None,
        "divida_patrimonio": _div(debt, eq) if eq is not None and eq > 0 else None,
        "liquidez_corrente": _div(g("ativo_circulante", i), g("passivo_circulante", i)),
        "cobertura_dividendos": _div(f, dp) if dp else None,
        "acumulacoes": _div(ni - cfo, assets) if ni is not None and cfo is not None else None,
        "fcl": f,
    }


def growth(a: Annual) -> dict[str, float | None]:
    n = len(a.anos)

    def cagr(key: str, years: int) -> float | None:
        if n <= years:
            return None
        x0, x1 = a.get(key, -1 - years), a.get(key, -1)
        if x0 is None or x1 is None or x0 <= 0 or x1 <= 0:
            return None
        return (x1 / x0) ** (1 / years) - 1

    def last(key: str) -> float | None:
        if n < 2:
            return None
        x0, x1 = a.get(key, -2), a.get(key, -1)
        return x1 / x0 - 1 if x0 is not None and x1 is not None and x0 > 0 else None

    sh0 = a.get("acoes", -4) if n >= 4 else a.get("acoes", 0)
    sh1 = a.get("acoes", -1)
    return {"receita_cagr_3a": cagr("receita", 3), "receita_ultimo_ano": last("receita"),
            "lucro_cagr_3a": cagr("lucro_liquido", 3), "lucro_ultimo_ano": last("lucro_liquido"),
            "acoes_variacao": (sh1 / sh0 - 1) if sh0 and sh1 else None,
            "acoes_anos": min(3, n - 1) if n >= 2 else 0}


def piotroski(a: Annual) -> dict[str, object]:
    """F-score of the latest year; each test is 1, 0 or None (inputs missing)."""
    n = len(a.anos)
    if n < 2:
        return {"pontos": None, "medidos": 0, "testes": [], "ano": a.anos[-1] if a.anos else None,
                "nota": "precisa de pelo menos dois anos de demonstrações"}
    g = a.get
    t, p = -1, -2
    a_t0 = g("ativos", -2)                               # assets at the start of year t
    a_p0 = g("ativos", -3) if n >= 3 else None           # assets at the start of year t-1
    roa_t = _div(g("lucro_liquido", t), a_t0)
    roa_p = _div(g("lucro_liquido", p), a_p0)
    cfo_t = _div(g("fco", t), a_t0)
    lev = lambda i, a_start: _div(g("divida_lp", i) if g("divida_lp", i) is not None else g("divida_total", i),
                                  (g("ativos", i) + a_start) / 2 if g("ativos", i) is not None and a_start is not None else None)
    lev_t, lev_p = lev(t, a_t0), lev(p, a_p0)
    cr_t = _div(g("ativo_circulante", t), g("passivo_circulante", t))
    cr_p = _div(g("ativo_circulante", p), g("passivo_circulante", p))
    sh_t, sh_p = g("acoes", t), g("acoes", p)
    gm_t, gm_p = _div(g("lucro_bruto", t), g("receita", t)), _div(g("lucro_bruto", p), g("receita", p))
    turn_t, turn_p = _div(g("receita", t), a_t0), _div(g("receita", p), a_p0)

    def test(name: str, ok: bool | None, detail: str) -> dict[str, object]:
        return {"teste": name, "ponto": None if ok is None else int(ok), "detalhe": detail}

    num2 = lambda x: num(x, 2)
    tests = [
        test("lucro sobre ativos positivo", None if roa_t is None else roa_t > 0, f"ROA {pct(roa_t)}"),
        test("geração de caixa operacional positiva", None if cfo_t is None else cfo_t > 0, f"caixa operacional/ativos {pct(cfo_t)}"),
        test("ROA subiu", None if roa_t is None or roa_p is None else roa_t > roa_p, f"{pct(roa_p)} → {pct(roa_t)}"),
        test("caixa operacional maior que o lucro", None if cfo_t is None or roa_t is None else cfo_t > roa_t,
             f"caixa {pct(cfo_t)} × lucro {pct(roa_t)} dos ativos"),
        test("alavancagem caiu", None if lev_t is None or lev_p is None else lev_t <= lev_p,
             f"dívida/ativos {pct(lev_p)} → {pct(lev_t)}"),
        test("liquidez corrente subiu", None if cr_t is None or cr_p is None else cr_t > cr_p, f"{num2(cr_p)} → {num2(cr_t)}"),
        test("sem emissão de ações", None if not sh_t or not sh_p else sh_t <= sh_p * (1 + NEW_SHARES_TOL),
             "ações diluídas " + ("n/d" if not sh_t or not sh_p else pct(sh_t / sh_p - 1, sign=True))),
        test("margem bruta subiu", None if gm_t is None or gm_p is None else gm_t > gm_p, f"{pct(gm_p)} → {pct(gm_t)}"),
        test("giro dos ativos subiu", None if turn_t is None or turn_p is None else turn_t > turn_p, f"{num2(turn_p)} → {num2(turn_t)}"),
    ]
    measured = [x["ponto"] for x in tests if x["ponto"] is not None]
    return {"pontos": sum(measured) if measured else None, "medidos": len(measured), "testes": tests, "ano": a.anos[-1],
            "ano_anterior": a.anos[-2]}


def valuation(a: Annual | None, snap: dict[str, object]) -> dict[str, float | None]:
    """Multiples at today's price: market cap and P/E from the source's snapshot, the rest from the latest annual
    statements. Ratios mixing currencies (statements in one, quote in another) are not computed."""
    mcap = snap.get("valor_mercado")
    out: dict[str, float | None] = {"pl": snap.get("pl"), "dividend_yield": snap.get("dividend_yield"),  # type: ignore[dict-item]
                                    "p_vp": None, "ev_ebitda": None, "fcl_yield": None, "lucro_yield": None}
    pl = out["pl"]
    out["lucro_yield"] = 1.0 / pl if isinstance(pl, float) and pl > 0 else None
    same_ccy = a is not None and (a.moeda is None or snap.get("moeda_cotacao") in (None, a.moeda))
    if a is None or not isinstance(mcap, float) or mcap <= 0 or not same_ccy:
        return out
    eq, f = a.get("patrimonio", -1), fcf(a, -1)
    debt, cash, ebitda = a.get("divida_total", -1), a.get("caixa", -1), a.get("ebitda", -1)
    out["p_vp"] = mcap / eq if eq is not None and eq > 0 else None
    if debt is not None and cash is not None and ebitda is not None and ebitda > 0:
        out["ev_ebitda"] = (mcap + debt - cash) / ebitda
    out["fcl_yield"] = f / mcap if f is not None else None
    return out
