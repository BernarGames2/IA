"""Dashboard of the investor's portfolio: value today, evolution, returns against benchmarks, allocation, alerts and
movements.

Returns (stated in the result):

* time-weighted return (TWR, "rentabilidade da carteira"): daily chaining ``r_t = V_t / (V_{t-1} + F_t) - 1``, with
  ``F_t`` the net money put in on day t (purchases minus sales, at the typed price) — flows at the start of the day,
  so the first day of a purchase measures close / purchase price. It measures the investments, not the timing of
  the investor's deposits;
* money-weighted return (XIRR, "rentabilidade do seu dinheiro"): the annual rate that zeroes the investor's cash
  flows (purchases negative, sales and dividends positive, the final value positive);
* without movements, the evolution is HYPOTHETICAL: today's quantities held through the whole period (labelled).

Benchmarks over the same window: CDI accumulated from BCB SGS 12, BOVA11 (Ibovespa) and IVVB11 (S&P 500 in reais),
adjusted closes. Alerts are fixed rules with their numbers (per-asset cap of the profile, sector above 35%, foreign
currency above 50%, drawdown beyond the profile's reference, stale prices, assets without data).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from core.market_data.cash import CASH_TICKER
from core.market_data.yahoo_chart import FetchFn
from core.portfolio_analytics.data import (
    BENCHMARKS,
    CLASS_PT,
    DISCLAIMER,
    MAX_FFILL_DAYS,
    Book,
    Market,
    PortfolioRequest,
    asset_info,
    build_book,
    load_market,
    logo,
    norm,
    require_some_data,
    today,
)
from core.profiler import ProfilerSettings
from core.purpose.language import guarantee_phrases
from models.investor import RiskProfile

WINDOWS = ("1m", "3m", "6m", "ytd", "1a", "3a", "inicio")
WINDOW_PT = {"1m": "1 mês", "3m": "3 meses", "6m": "6 meses", "ytd": "no ano", "1a": "12 meses", "3a": "3 anos",
             "inicio": "desde o início"}
SECTOR_CAP = 0.35
FOREIGN_WARN = 0.50
STALE_DAYS = 7          # calendar days (~5 business days)
MAX_POINTS = 260        # points sent for the evolution chart


def window_start(code: str, end: pd.Timestamp, first: pd.Timestamp) -> pd.Timestamp:
    if code == "inicio":
        return first
    if code == "ytd":
        s = pd.Timestamp(end.year, 1, 1) - pd.Timedelta(days=1)
    else:
        s = end - pd.DateOffset(months={"1m": 1, "3m": 3, "6m": 6, "1a": 12, "3a": 36}[code])
    return max(pd.Timestamp(s), first)


def xirr(flows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """Annual rate r with sum cf / (1 + r)^(days/365) = 0; None without both signs or without a root."""
    flows = [(d, v) for d, v in flows if np.isfinite(v) and abs(v) > 1e-9]
    if not flows or not (any(v < 0 for _, v in flows) and any(v > 0 for _, v in flows)):
        return None
    d0 = min(d for d, _ in flows)
    t = np.array([(d - d0).days / 365.0 for d, _ in flows])
    v = np.array([x for _, x in flows])

    def npv(r: float) -> float:
        return float(np.sum(v / (1.0 + r) ** t))

    lo, hi = -0.9999, 100.0
    if npv(lo) * npv(hi) > 0:
        return None
    try:
        return float(brentq(npv, lo, hi, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return None


def _frame(market: Market, tickers: list[str], start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Prices in reais of ``tickers`` on the union of their dates, carried over holidays only."""
    cols = {t: market.brl[t] for t in tickers if market.has(t)}
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols).sort_index().ffill(limit=MAX_FFILL_DAYS)
    return df.loc[start:] if start is not None else df


def _cash_path(market: Market, value_today: float, index: pd.DatetimeIndex) -> pd.Series | None:
    """Today's cash, back in time discounted by the CDI actually paid (hypothesis: the cash was held all along)."""
    if market.cdi is None or market.cdi.empty:
        return None
    lvl = market.cdi.reindex(market.cdi.index.union(index)).ffill().reindex(index)
    return value_today * lvl / float(market.cdi.iloc[-1])


def valuation(book: Book, market: Market) -> tuple[pd.Series, pd.Series, bool, list[str]]:
    """Daily portfolio value in reais and net external flows. With movements: quantities from the trades; without:
    today's quantities held over the whole history (hypothetical)."""
    notes: list[str] = []
    held = [t for t in book.tickers if t != CASH_TICKER and market.has(t)]
    px = _frame(market, held)
    real = bool(book.from_movements and book.daily_qty is not None and not book.daily_qty.empty)
    if px.empty:
        idx = market.cdi.index if market.cdi is not None else pd.DatetimeIndex([])
        value = pd.Series(0.0, index=idx)
    elif real:
        q = book.daily_qty.reindex(columns=held).fillna(0.0)                       # type: ignore[union-attr]
        q = q.reindex(q.index.union(px.index)).ffill().fillna(0.0).reindex(px.index)
        first_trade = book.daily_qty.index.min()                                     # type: ignore[union-attr]
        px = px.loc[first_trade:]
        q = q.loc[first_trade:]
        missing = (q > 0) & px.isna()
        value = (q * px.fillna(0.0)).sum(axis=1)
        value[missing.any(axis=1)] = np.nan
    else:
        qty = pd.Series({t: book.quantity.get(t, 0.0) for t in held})
        complete = px.dropna()
        value = (complete * qty).sum(axis=1)
        if len(complete) < len(px):
            notes.append("a evolução começa quando todos os ativos têm cotação")
    if CASH_TICKER in book.quantity and book.quantity[CASH_TICKER] > 0:
        cash = _cash_path(market, book.quantity[CASH_TICKER], pd.DatetimeIndex(value.index))
        if cash is not None:
            value = value.add(cash, fill_value=0.0) if len(value) else cash
        else:
            notes.append("caixa (CAIXA_CDI) fora da evolução: CDI indisponível agora")
    value = value.dropna()
    flows = pd.Series(0.0, index=value.index)
    if real and book.flows is not None and len(book.flows):
        f = book.flows.reindex(columns=held).fillna(0.0).sum(axis=1)     # assets without data are out of both
        # a trade on a day without a portfolio value (holiday) counts on the next valued day
        pos = value.index.searchsorted(f.index)
        for p, amount in zip(pos, f.to_numpy(), strict=True):
            if p < len(flows):
                flows.iloc[p] += amount
    return value, flows, real, notes


def twr_index(value: pd.Series, flows: pd.Series) -> pd.Series:
    """Index (first valued day = 1) of the time-weighted return; days whose base is not positive (everything sold)
    add no return."""
    v = value.to_numpy()
    f = flows.reindex(value.index).fillna(0.0).to_numpy()
    out = np.ones(len(v))
    for i in range(1, len(v)):
        base = v[i - 1] + f[i]
        r = v[i] / base - 1.0 if base > 1e-9 else 0.0
        out[i] = out[i - 1] * (1.0 + r)
    return pd.Series(out, index=value.index)


def _bench_series(market: Market) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if market.cdi is not None and not market.cdi.empty:
        out["CDI"] = market.cdi
    for t, name in BENCHMARKS.items():
        if market.has(t):
            out[name] = market.brl[t]
    return out


def _ret(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float | None:
    sa, sb = s.loc[:a], s.loc[:b]
    if sa.empty or sb.empty or (a - sa.index[-1]).days > STALE_DAYS or (b - sb.index[-1]).days > STALE_DAYS:
        return None
    return float(sb.iloc[-1] / sa.iloc[-1] - 1.0)


def _on_or_before(index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    """The last valued day up to ``d`` (the window's base), or the first one."""
    i = int(index.searchsorted(d, side="right")) - 1
    return index[max(i, 0)]


def _downsample(idx: pd.DatetimeIndex) -> np.ndarray:
    n = len(idx)
    if n <= MAX_POINTS:
        return np.arange(n)
    keep = np.unique(np.linspace(0, n - 1, MAX_POINTS).round().astype(int))
    return keep


def _alerts(weights: pd.Series, info: dict[str, dict[str, str]], perfil: str | None, dd_now: float | None,
            stale: dict[str, str], missing: dict[str, str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if perfil:
        lim = ProfilerSettings().limits[RiskProfile(perfil)]
        for t, w in weights.items():
            if t != CASH_TICKER and w > lim.max_weight_per_asset + 1e-9:
                out.append({"nivel": "alta", "texto": f"{t} é {w:.0%} da carteira, acima do limite de "
                            f"{lim.max_weight_per_asset:.0%} por ativo do perfil {perfil}",
                            "regra": "peso por ativo ≤ limite do perfil"})
        if dd_now is not None and dd_now < -lim.max_drawdown_reference:
            out.append({"nivel": "alta", "texto": f"a carteira está {abs(dd_now):.0%} abaixo do seu pico, mais que a "
                        f"queda de referência de {lim.max_drawdown_reference:.0%} do perfil {perfil}",
                        "regra": "queda desde o pico ≤ referência do perfil"})
    elif len(weights):
        top = weights.drop(CASH_TICKER, errors="ignore")
        if len(top) and top.max() > 0.40:
            out.append({"nivel": "media", "texto": f"{top.idxmax()} é {top.max():.0%} da carteira: um único ativo "
                        "concentra boa parte do resultado", "regra": "peso por ativo ≤ 40% (sem perfil informado)"})
    by_sector = pd.Series({t: w for t, w in weights.items()}).groupby(
        pd.Series({t: info[t]["setor"] for t in weights.index})).sum()
    for sec, w in by_sector.items():
        if sec not in ("caixa", "não classificado", "") and w > SECTOR_CAP:
            out.append({"nivel": "media", "texto": f"setor {sec} soma {w:.0%} da carteira (acima de {SECTOR_CAP:.0%})",
                        "regra": f"peso por setor ≤ {SECTOR_CAP:.0%}"})
    foreign = float(sum(w for t, w in weights.items() if info[t]["moeda"] != "BRL"))
    if foreign > FOREIGN_WARN:
        out.append({"nivel": "info", "texto": f"{foreign:.0%} da carteira está em ativos cotados em dólar: o "
                    "resultado em reais depende também do câmbio", "regra": f"exposição cambial > {FOREIGN_WARN:.0%}"})
    for t, why in stale.items():
        out.append({"nivel": "media", "texto": f"{t}: {why}", "regra": "cotação com mais de 5 dias úteis"})
    for t, why in missing.items():
        out.append({"nivel": "alta", "texto": f"{t} sem dados: {why} (fora dos números)", "regra": "dados indisponíveis"})
    return out


def build_dashboard(req: PortfolioRequest, provider: object, fetch: FetchFn | None = None,
                    as_of: date | None = None) -> dict[str, object]:
    end = as_of or req.data_referencia or today()
    first_move = min((m.data for m in req.movimentos), default=None)
    period = "10y" if first_move is not None and (end - first_move).days > 5 * 365 - 30 else "5y"
    tickers = sorted({norm(p.ticker) for p in req.posicoes} | {norm(m.ticker) for m in req.movimentos})
    market = load_market(tickers, provider, fetch, period, end, extra=(*BENCHMARKS, "CDI"))
    book = build_book(req, market)
    require_some_data(book, market)
    missing = {t: market.missing[t] for t in book.tickers if t in market.missing}
    value, flows, real, notes = valuation(book, market)
    notes = [*book.notes, *notes, *market.notes]
    if value.empty:
        raise ValueError("sem cotações suficientes para avaliar a carteira no período")

    # ----- today: positions and weights
    rows, values = [], {}
    end_ts = pd.Timestamp(end)
    stale: dict[str, str] = {}
    for t in book.tickers:
        a = asset_info(t)
        q = float(book.quantity.get(t, 0.0))
        if q <= 0 or t in missing:
            continue
        if t == CASH_TICKER:
            values[t] = q
            rows.append({"ticker": t, "nome": "Caixa em reais (CDI)", "classe": CLASS_PT["caixa"], "setor": "caixa",
                         "moeda": "BRL", "quantidade": None, "preco": None, "preco_data": None, "preco_moeda": None,
                         "valor": q, "preco_medio": None, "resultado_preco": None, "logo": None})
            continue
        px_brl, px_nat = market.brl[t], market.native[t]
        last_d = px_brl.index[-1]
        if (end_ts - last_d).days > STALE_DAYS:
            stale[t] = f"última cotação em {last_d:%d/%m/%Y}"
        values[t] = q * float(px_brl.iloc[-1])
        pm = book.cost.get(t)
        rows.append({"ticker": t, "nome": a.name if a else t, "classe": CLASS_PT.get(a.macro_class, a.macro_class) if a else "não classificado",
                     "setor": a.sector if a else "não classificado", "moeda": market.currency.get(t, "?"), "quantidade": q,
                     "preco": float(px_nat.iloc[-1]), "preco_data": f"{px_nat.index[-1]:%Y-%m-%d}",
                     "preco_moeda": market.currency.get(t), "valor": values[t], "preco_medio": pm,
                     "resultado_preco": (float(px_nat.iloc[-1]) / pm - 1.0) if pm else None, "logo": logo(t)})
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("as posições informadas não têm valor com as cotações disponíveis")
    weights = pd.Series(values) / total
    for r in rows:
        r["peso"] = float(weights[r["ticker"]])
    rows.sort(key=lambda r: -float(r["valor"]))  # type: ignore[arg-type]
    info = {r["ticker"]: {"setor": str(r["setor"]), "moeda": str(r["moeda"]), "classe": str(r["classe"])} for r in rows}

    # ----- evolution and returns
    idx = twr_index(value, flows)
    benches = _bench_series(market)
    first = value.index[0]
    last = value.index[-1]
    rets: dict[str, float | None] = {}
    for code in WINDOWS:
        a = window_start(code, last, first)
        rets[code] = None if a < first or (code != "inicio" and window_start(code, last, pd.Timestamp.min) < first) \
            else _ret(idx, a, last)
    a = _on_or_before(idx.index, window_start(req.periodo, last, first))
    covered = req.periodo == "inicio" or window_start(req.periodo, last, pd.Timestamp.min) >= first
    win = value.loc[a:]
    w_idx = idx.loc[a:] / float(idx.loc[a])
    port_ret = float(idx.iloc[-1] / idx.loc[a] - 1.0)
    bench_rows = []
    curves: dict[str, list[float | None]] = {}
    keep = _downsample(pd.DatetimeIndex(w_idx.index))
    for name, s in benches.items():
        r = _ret(s, a, last)
        bench_rows.append({"nome": name, "retorno": r, "excesso": None if r is None else port_ret - r})
        base = s.loc[:a]
        aligned = s.reindex(s.index.union(w_idx.index)).ffill().reindex(w_idx.index)
        curves[name] = ([None if not np.isfinite(x) else round(float(x / base.iloc[-1] * 100.0), 3)
                         for x in aligned.to_numpy()[keep]] if not base.empty else [])
    mwr = None
    if real:
        cf: list[tuple[pd.Timestamp, float]] = []
        if a > first:
            cf.append((a, -float(value.loc[a])))
            after = flows.loc[a:].iloc[1:]
            divs = book.dividends.loc[a:].iloc[1:] if book.dividends is not None and len(book.dividends) else None
        else:
            after = flows.loc[a:]
            divs = book.dividends.loc[a:] if book.dividends is not None and len(book.dividends) else None
        cf += [(d, -x) for d, x in after.items() if abs(x) > 1e-9]
        if divs is not None:
            cf += [(d, float(x)) for d, x in divs.items()]
        cf.append((last, float(value.iloc[-1])))
        mwr = xirr(cf)
    wealth = idx.to_numpy()
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    result = None
    if real:
        valued = [t for t in book.tickers if t != CASH_TICKER and t not in missing]
        invested = float(book.flows.reindex(columns=valued).fillna(0.0).to_numpy().sum()) if book.flows is not None else 0.0
        divs = float(book.dividends.sum()) if book.dividends is not None and len(book.dividends) else 0.0
        result = {"total_investido_liquido": invested, "proventos": divs, "resultado": total - invested + divs,
                  "resultado_pct": (total - invested + divs) / invested if invested > 0 else None,
                  "nota": "resultado = patrimônio de hoje − (compras − vendas) + proventos registrados"}
    alloc: dict[str, dict[str, float]] = {"ativo": {t: float(w) for t, w in weights.sort_values(ascending=False).items()}}
    for dim in ("classe", "setor", "moeda"):
        g = pd.Series({t: info[t][dim] for t in weights.index})
        alloc[dim] = {k: float(v) for k, v in weights.groupby(g).sum().sort_values(ascending=False).items()}
    alerts = _alerts(weights, info, req.perfil, float(dd[-1]) if len(dd) else None, stale, missing)
    hyp = ("hipótese: a evolução usa as quantidades de hoje mantidas no período inteiro (você não registrou "
           "movimentações), não o seu histórico real") if not real else ""
    warns = [w for w in [hyp, *notes] if w]
    warns.append("cotações ajustadas por proventos: o retorno já inclui dividendos; proventos registrados aparecem "
                 "como renda recebida e não são somados de novo")
    if not covered:
        warns.append(f"o período pedido ({WINDOW_PT[req.periodo]}) começa antes dos dados: mostrado desde {first:%d/%m/%Y}")
    out = {
        "data_referencia": f"{last:%Y-%m-%d}", "moeda": "BRL", "modo": "movimentacoes" if real else "hipotetico",
        "patrimonio": total, "posicoes": rows, "resultado": result,
        "periodo": {"codigo": req.periodo, "rotulo": WINDOW_PT[req.periodo], "inicio": f"{w_idx.index[0]:%Y-%m-%d}",
                    "fim": f"{last:%Y-%m-%d}", "retorno_twr": port_ret,
                    "retorno_mwr_anual": mwr, "benchmarks": bench_rows},
        "retornos": {k: v for k, v in rets.items()}, "rotulos_periodos": WINDOW_PT,
        "evolucao": {"datas": [f"{d:%Y-%m-%d}" for d in w_idx.index[keep]],
                     "carteira": [round(float(x) * 100.0, 3) for x in w_idx.to_numpy()[keep]],
                     "valor": [round(float(x), 2) for x in win.to_numpy()[keep]], "benchmarks": curves,
                     "hipotese": not real},
        "drawdown": {"maximo": float(dd.min()) if len(dd) else None, "atual": float(dd[-1]) if len(dd) else None},
        "alocacao": alloc, "alertas": alerts, "movimentacoes": book.trades[-200:][::-1], "sem_dados": missing,
        "fontes": market.sources, "avisos": warns, "aviso": DISCLAIMER,
    }
    bad = guarantee_phrases(" ".join(str(x) for x in [*warns, *[a["texto"] for a in alerts]]))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out
