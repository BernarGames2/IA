"""Fundamentals over time from Yahoo Finance — published values only; a missing year stays missing.

* Annual statements (``fundamentals-timeseries``): revenue, gross profit, operating income, net income, EBITDA,
  operating cash flow, capex, free cash flow, total and long-term debt, cash, equity, total and current assets,
  current liabilities, diluted average shares, dividends paid and diluted EPS, for the fiscal years the source
  publishes (usually 4–5).
* Analysts and calendar (``quoteSummary`` modules ``earningsTrend``, ``recommendationTrend``, ``calendarEvents``):
  EPS estimates and their revisions (current vs 7/30/60/90 days ago; how many analysts raised or cut them), the
  monthly recommendation counts, the next earnings date and the dividend dates. These are THIRD-PARTY estimates and
  are labelled so wherever they appear.

Both go through the cookie + crumb client of the stock check (``YahooFundamentals.get_json``). Results are cached
on disk for ``TTL_S`` per ticker (statements change a few times a year). Any failure is reported, never filled.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.stock_check.company import SourceError, YahooFundamentals

TS_URL = ("https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{sym}"
          "?symbol={sym}&type={types}&period1={p1}&period2={p2}&crumb={crumb}")
QS_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}?modules={mods}&crumb={crumb}"
MODULES = "earningsTrend,recommendationTrend,calendarEvents"
TTL_S = 24 * 3600
YEARS_BACK = 7
FIELDS = {  # our key -> Yahoo statement item (requested with the "annual" prefix)
    "receita": "TotalRevenue", "lucro_bruto": "GrossProfit", "lucro_operacional": "OperatingIncome",
    "lucro_liquido": "NetIncome", "ebitda": "EBITDA", "fco": "OperatingCashFlow", "capex": "CapitalExpenditure",
    "fcl": "FreeCashFlow", "divida_total": "TotalDebt", "divida_lp": "LongTermDebt", "caixa": "CashAndCashEquivalents",
    "patrimonio": "StockholdersEquity", "ativos": "TotalAssets", "ativo_circulante": "CurrentAssets",
    "passivo_circulante": "CurrentLiabilities", "acoes": "DilutedAverageShares", "dividendos_pagos": "CashDividendsPaid",
    "lpa": "DilutedEPS"}
_BY_TYPE = {"annual" + v: k for k, v in FIELDS.items()}


@dataclass
class Annual:
    """Annual statements: ``anos`` are fiscal year-end dates (ISO), ``valores[field][i]`` the value for ``anos[i]``
    (None when the source has no value for that year)."""
    anos: list[str]
    valores: dict[str, list[float | None]]
    moeda: str | None
    fonte: str
    consultado_em: str

    def get(self, key: str, i: int) -> float | None:
        v = self.valores.get(key)
        if v is None or not -len(v) <= i < len(v):
            return None
        return v[i]


@dataclass
class Analysts:
    """Third-party estimates and calendar (any part may be missing)."""
    eps_ano: dict[str, float | None] = field(default_factory=dict)       # current, 7/30/60/90 days ago (this FY)
    eps_proximo_ano: dict[str, float | None] = field(default_factory=dict)
    revisoes_30d: dict[str, int | None] = field(default_factory=dict)    # up / down, this FY
    n_analistas: int | None = None
    crescimento_esperado: float | None = None                            # this FY EPS growth (consensus)
    recomendacoes: dict[str, int] = field(default_factory=dict)          # latest month counts
    proximo_resultado: str | None = None
    data_com: str | None = None                                          # ex-dividend date
    pagamento_dividendo: str | None = None
    fonte: str = ""
    consultado_em: str = ""


def _raw(v: object) -> float | None:
    if isinstance(v, dict):
        v = v.get("raw")
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v) if math.isfinite(v) else None


def _date(v: object) -> str | None:
    x = _raw(v)
    return datetime.fromtimestamp(x, UTC).date().isoformat() if x is not None else None


def parse_timeseries(data: object, consulted: str) -> Annual | None:
    res = data.get("timeseries", {}).get("result") if isinstance(data, dict) else None
    if not isinstance(res, list):
        raise SourceError("Yahoo (demonstrações): resposta em formato inesperado")
    rows: dict[str, dict[str, float]] = {}
    ccys: set[str] = set()
    for item in res:
        if not isinstance(item, dict):
            continue
        types = (item.get("meta") or {}).get("type") or []
        t = types[0] if isinstance(types, list) and types else None
        key = _BY_TYPE.get(str(t))
        if key is None:
            continue
        for r in item.get(str(t)) or []:
            if not isinstance(r, dict):
                continue
            d, v = r.get("asOfDate"), _raw(r.get("reportedValue"))
            if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and v is not None:
                rows.setdefault(key, {})[d] = v
                if isinstance(r.get("currencyCode"), str):
                    ccys.add(r["currencyCode"])
    if not rows:
        return None
    years = sorted({d for v in rows.values() for d in v})
    # one value per fiscal year: dates within 20 days of each other are the same year-end (restatements)
    merged: list[str] = []
    for d in years:
        if merged and (datetime.fromisoformat(d) - datetime.fromisoformat(merged[-1])).days < 20:
            merged[-1] = d
        else:
            merged.append(d)

    def pick(series: dict[str, float], y: str) -> float | None:
        near = [k for k in series if abs((datetime.fromisoformat(k) - datetime.fromisoformat(y)).days) < 20]
        return series[max(near)] if near else None

    vals = {k: [pick(s, y) for y in merged] for k, s in rows.items()}
    return Annual(merged, vals, ccys.pop() if len(ccys) == 1 else None, "Yahoo Finance — demonstrações anuais",
                  consulted)


def parse_analysts(res: object, consulted: str) -> Analysts:
    if not isinstance(res, dict):
        raise SourceError("Yahoo (estimativas): resposta sem resultado")
    a = Analysts(fonte="Yahoo Finance — estimativas de analistas e agenda (terceiros)", consultado_em=consulted)
    for t in (res.get("earningsTrend") or {}).get("trend") or []:
        if not isinstance(t, dict):
            continue
        tr = t.get("epsTrend") or {}
        trend = {k: _raw(tr.get(k)) for k in ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo")}
        if t.get("period") == "0y":
            a.eps_ano = trend
            rv = t.get("epsRevisions") or {}
            up, down = _raw(rv.get("upLast30days")), _raw(rv.get("downLast30days"))
            a.revisoes_30d = {"alta": int(up) if up is not None else None, "baixa": int(down) if down is not None else None}
            ee = t.get("earningsEstimate") or {}
            n = _raw(ee.get("numberOfAnalysts"))
            a.n_analistas = int(n) if n is not None else None
            a.crescimento_esperado = _raw(t.get("growth"))
        elif t.get("period") == "+1y":
            a.eps_proximo_ano = trend
    for r in (res.get("recommendationTrend") or {}).get("trend") or []:
        if isinstance(r, dict) and r.get("period") == "0m":
            a.recomendacoes = {k: int(_raw(r.get(k)) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
    cal = res.get("calendarEvents") or {}
    dates = (cal.get("earnings") or {}).get("earningsDate") or []
    if isinstance(dates, list) and dates:
        a.proximo_resultado = _date(dates[0])
    a.data_com = _date(cal.get("exDividendDate"))
    a.pagamento_dividendo = _date(cal.get("dividendDate"))
    return a


def _cache_path(cache_dir: Path | None, kind: str, ticker: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / "fundamentos" / f"{kind}_{re.sub(r'[^A-Za-z0-9._-]', '_', ticker)}.json"


def _cached(path: Path | None, now: float) -> object | None:
    if path is None or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("dados") if isinstance(d, dict) and now - float(d.get("t", 0)) < TTL_S else None


def _store(path: Path | None, data: object, now: float) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"t": now, "dados": data}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass                                          # read-only disk: run without cache


def fetch_annual(client: YahooFundamentals, ticker: str, cache_dir: Path | None = None, now: float | None = None
                 ) -> Annual | None:
    t_now = time.time() if now is None else now
    path = _cache_path(cache_dir, "anual", ticker)
    raw = _cached(path, t_now)
    consulted = datetime.fromtimestamp(t_now, UTC).date().isoformat()
    if raw is None:
        p2 = int(t_now)
        p1 = p2 - YEARS_BACK * 366 * 86400
        types = ",".join("annual" + v for v in FIELDS.values())
        sym = urllib.parse.quote(ticker, safe="")
        raw = client.get_json(lambda crumb: TS_URL.format(sym=sym, types=types, p1=p1, p2=p2,
                                                          crumb=urllib.parse.quote(crumb, safe="")))
        _store(path, {"consultado_em": consulted, "resposta": raw}, t_now)
    else:
        consulted, raw = raw.get("consultado_em", consulted), raw.get("resposta")  # type: ignore[union-attr]
    return parse_timeseries(raw, consulted)


def fetch_analysts(client: YahooFundamentals, ticker: str, cache_dir: Path | None = None, now: float | None = None
                   ) -> Analysts:
    t_now = time.time() if now is None else now
    path = _cache_path(cache_dir, "analistas", ticker)
    raw = _cached(path, t_now)
    consulted = datetime.fromtimestamp(t_now, UTC).date().isoformat()
    if raw is None:
        sym = urllib.parse.quote(ticker, safe="")
        data = client.get_json(lambda crumb: QS_URL.format(sym=sym, mods=MODULES, crumb=urllib.parse.quote(crumb, safe="")))
        qs = data.get("quoteSummary") if isinstance(data, dict) else None
        res = (qs.get("result") or [None])[0] if isinstance(qs, dict) else None
        _store(path, {"consultado_em": consulted, "resposta": res}, t_now)
    else:
        consulted, res = raw.get("consultado_em", consulted), raw.get("resposta")  # type: ignore[union-attr]
    return parse_analysts(res, consulted)
