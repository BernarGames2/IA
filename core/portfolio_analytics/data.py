"""Inputs and market data for the investor's portfolio: request models, ticker normalisation, prices converted
to reais, and the holdings and cash flows rebuilt from the movements.

Conventions (stated to the investor in the results):

* Prices are ADJUSTED closes (dividends and splits folded into the price history), so a return computed from them
  already includes the dividends; dividends typed as movements are shown as income received and are NOT added to
  the returns again.
* Assets quoted in dollars are converted with the USDBRL=X close of the same date (the last one available up to
  ``MAX_FFILL_DAYS`` days before; never a later rate).
* A price is carried forward over holidays only (``MAX_FFILL_DAYS`` business days); a longer gap leaves the date
  without a value instead of inventing one.
* Cash in reais (``CAIXA_CDI``) has no movements: its value is the amount typed today and, back in time, that
  amount discounted by the CDI actually paid (BCB SGS 12) — a hypothesis stated in the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.market_data.base import DataUnavailableError, RawSeries, last_per_date
from core.market_data.cash import CASH_TICKER, fetch_cdi_cash
from core.market_data.logos import logo_path
from core.market_data.yahoo_chart import FetchFn
from core.purpose.catalog import REAL_CATALOG, CatalogAsset, by_ticker
from core.purpose.engine import norm_ticker

FX_TICKER = "USDBRL=X"
BENCHMARKS = {"BOVA11.SA": "Ibovespa (BOVA11)", "IVVB11.SA": "S&P 500 em reais (IVVB11)"}
RATES_BR, RATES_US = "IRFM11.SA", "IEF"
MAX_FFILL_DAYS = 5
CATALOG = by_ticker(REAL_CATALOG)
CLASS_PT = {"acoes_br": "Ações (Brasil)", "acoes_eua": "Ações (EUA)", "fii": "Fundos imobiliários", "etf_br": "ETFs na B3",
            "etf_eua": "ETFs nos EUA", "renda_fixa_br": "Renda fixa (Brasil)", "ouro": "Ouro", "cripto": "Cripto",
            "caixa": "Caixa (CDI)"}
DISCLAIMER = ("Leitura quantitativa para pesquisa, com cotações com atraso. Não é recomendação de compra ou venda, não "
              "executa ordens e não oferece garantia de resultado; estimativas de risco não são limites de perda.")


def today() -> date:
    return datetime.now(UTC).date()


class Position(BaseModel):
    """One holding: quantity (preferred) or value in reais when the quantity is unknown."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(..., min_length=1, max_length=20)
    quantidade: float | None = Field(None, gt=0, le=1e12)
    valor: float | None = Field(None, gt=0, le=1e12)
    preco_medio: float | None = Field(None, gt=0, le=1e9)

    @model_validator(mode="after")
    def _sized(self) -> Position:
        if self.quantidade is None and self.valor is None:
            raise ValueError(f"{self.ticker}: informe a quantidade ou o valor da posição")
        return self


class Movement(BaseModel):
    """A buy, a sell (quantity and, optionally, the price in the asset's currency) or a dividend (amount in reais)."""

    model_config = ConfigDict(extra="forbid")

    data: date
    ticker: str = Field(..., min_length=1, max_length=20)
    tipo: Literal["compra", "venda", "dividendo"]
    quantidade: float | None = Field(None, gt=0, le=1e12)
    preco: float | None = Field(None, gt=0, le=1e9)
    valor: float | None = Field(None, gt=0, le=1e12)

    @model_validator(mode="after")
    def _complete(self) -> Movement:
        if self.tipo in ("compra", "venda") and self.quantidade is None:
            raise ValueError(f"{self.tipo} de {self.ticker} em {self.data}: informe a quantidade")
        if self.tipo == "dividendo" and self.valor is None:
            raise ValueError(f"provento de {self.ticker} em {self.data}: informe o valor recebido em reais")
        return self


class PortfolioRequest(BaseModel):
    """What the page sends: the positions, the optional movements and the reading options."""

    model_config = ConfigDict(extra="forbid")

    posicoes: list[Position] = Field(default_factory=list, max_length=60)
    movimentos: list[Movement] = Field(default_factory=list, max_length=2000)
    periodo: Literal["1m", "3m", "6m", "ytd", "1a", "3a", "inicio"] = "1a"
    perfil: Literal["conservador", "moderado", "arrojado"] | None = None
    confianca: Literal[0.95, 0.99] = 0.95
    horizonte_var_dias: Literal[1, 10] = 1
    data_referencia: date | None = None          # "today" (tests); default: the current UTC date

    @model_validator(mode="after")
    def _usable(self) -> PortfolioRequest:
        if not self.posicoes and not self.movimentos:
            raise ValueError("informe ao menos uma posição (código e quantidade ou valor) ou uma movimentação")
        ref = self.data_referencia or today()
        late = sorted({str(m.data) for m in self.movimentos if m.data > ref})
        if late:
            raise ValueError("movimentação com data futura: " + ", ".join(late[:5]))
        return self

    def key(self) -> str:
        return "carteira|" + self.model_dump_json()


def norm(t: str) -> str:
    t = t.strip().upper()
    return CASH_TICKER if t in (CASH_TICKER, "CAIXA", "CDI") else norm_ticker(t)


def asset_info(t: str) -> CatalogAsset | None:
    return CATALOG.get(t)


@dataclass
class Market:
    """Prices of the requested symbols in reais (``brl``) and in their own currency (``native``)."""

    brl: dict[str, pd.Series]
    native: dict[str, pd.Series]
    volume: dict[str, pd.Series | None]
    currency: dict[str, str]
    cdi: pd.Series | None
    missing: dict[str, str]
    sources: list[dict[str, str]]
    notes: list[str] = field(default_factory=list)

    def has(self, t: str) -> bool:
        return t in self.brl and len(self.brl[t]) > 0


def _clean(s: pd.Series) -> pd.Series:
    s = last_per_date(s.dropna()).sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize() if s.index.tz is not None \
        else pd.DatetimeIndex(s.index).normalize()
    s = last_per_date(s)
    return s[s > 0]


def load_market(tickers: list[str], provider: object, fetch: FetchFn | None, period: str, end: date,
                extra: tuple[str, ...] = ()) -> Market:
    """Download ``tickers`` (+ ``extra`` reference series) and USDBRL=X; convert dollar assets to reais; load the
    CDI when cash is held or used as a benchmark. Failures are recorded per symbol, never filled."""
    wanted = sorted({t for t in tickers if t != CASH_TICKER} | set(extra) | {FX_TICKER})
    got: dict[str, RawSeries] = provider.fetch(wanted, period, "1d") if wanted else {}  # type: ignore[attr-defined]
    failures = dict(getattr(provider, "failures", {}) or {})
    missing: dict[str, str] = {}
    notes: list[str] = []
    native: dict[str, pd.Series] = {}
    volume: dict[str, pd.Series | None] = {}
    currency: dict[str, str] = {}
    for t in wanted:
        rs = got.get(t)
        if rs is None or rs.close is None or rs.close.dropna().empty:
            missing[t] = failures.get(t, "sem cotações na fonte")
            continue
        native[t] = _clean(rs.close).loc[:pd.Timestamp(end)]          # never a price after the reference date
        if native[t].empty:
            missing[t] = "cotações inválidas (zeradas ou vazias)"
            del native[t]
            continue
        currency[t] = (rs.currency or (CATALOG[t].currency if t in CATALOG else "")).upper() or "?"
        volume[t] = _clean(rs.volume).reindex(native[t].index) if rs.volume is not None else None
    fx = native.get(FX_TICKER)
    brl: dict[str, pd.Series] = {}
    for t, px in native.items():
        if t == FX_TICKER:
            continue
        ccy = currency[t]
        if ccy == "BRL":
            brl[t] = px
        elif ccy == "USD":
            if fx is None:
                missing[t] = "sem a cotação do dólar (USDBRL=X) para converter em reais"
                continue
            conv = fx.reindex(fx.index.union(px.index)).ffill(limit=MAX_FFILL_DAYS).reindex(px.index)
            brl[t] = (px * conv).dropna()
        else:
            missing[t] = f"cotado em {ccy}: conversão para reais não disponível"
    cdi = None
    if CASH_TICKER in tickers or "CDI" in extra:
        start = date(end.year - (10 if period == "10y" else 5), end.month, 1)
        try:
            cdi = _clean(fetch_cdi_cash(fetch, start=start, end=end).close).loc[:pd.Timestamp(end)]
        except Exception as e:  # noqa: BLE001 - any failure of the BCB source is reported, never filled
            missing[CASH_TICKER] = f"CDI (BCB SGS 12) indisponível: {e}"
    sources = [{"dado": "cotações diárias (fechamento ajustado por proventos e desdobramentos)",
                "fonte": getattr(provider, "name", "provedor de cotações"),
                "data": max((s.index[-1] for s in native.values()), default=pd.Timestamp(end)).strftime("%Y-%m-%d")}]
    if fx is not None:
        sources.append({"dado": "câmbio USD/BRL", "fonte": "USDBRL=X (mesmo provedor)", "data": fx.index[-1].strftime("%Y-%m-%d")})
    if cdi is not None:
        sources.append({"dado": "CDI diário", "fonte": "Banco Central do Brasil, SGS 12", "data": cdi.index[-1].strftime("%Y-%m-%d")})
    return Market(brl, native, volume, currency, cdi, missing, sources, notes)


@dataclass
class Book:
    """The investor's holdings: current quantities/values and, with movements, the daily quantities and flows."""

    tickers: list[str]                        # held now, normalised, without data problems resolved yet
    quantity: dict[str, float]                # current quantity per ticker (cash: value in reais)
    cost: dict[str, float]                    # average price per ticker (asset currency), when known
    from_movements: bool
    daily_qty: pd.DataFrame | None            # dates x tickers, quantity at each trade date (step function)
    flows: pd.DataFrame | None                # net money put in, in reais, per trade date and ticker (buys +, sells -)
    dividends: pd.Series | None               # dividends in reais per date
    trades: list[dict[str, object]]
    notes: list[str] = field(default_factory=list)


def _price_on(px: pd.Series | None, d: pd.Timestamp) -> float | None:
    """Close on ``d`` or the last one up to MAX_FFILL_DAYS business days before it; never a later price."""
    if px is None or px.empty:
        return None
    s = px.loc[:d]
    if s.empty or (d - s.index[-1]).days > MAX_FFILL_DAYS + 2:
        return None
    return float(s.iloc[-1])


def build_book(req: PortfolioRequest, market: Market | None = None) -> Book:
    """Current holdings from the positions; with movements, the daily quantities and the cash flows. A sale larger
    than the quantity held at that date is refused (ValueError)."""
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    for p in req.posicoes:
        t = norm(p.ticker)
        if t == CASH_TICKER:
            qty[t] = qty.get(t, 0.0) + float(p.valor if p.valor is not None else p.quantidade)  # type: ignore[arg-type]
            continue
        if p.quantidade is not None:
            qty[t] = qty.get(t, 0.0) + p.quantidade
        elif market is not None and market.has(t):
            last = float(market.brl[t].iloc[-1])
            qty[t] = qty.get(t, 0.0) + float(p.valor) / last  # type: ignore[arg-type]
        else:
            qty[t] = qty.get(t, 0.0) + 0.0
        if p.preco_medio is not None:
            cost[t] = p.preco_medio
    notes: list[str] = []
    if not req.movimentos:
        return Book(sorted(qty), qty, cost, False, None, None, None, [], notes)
    moves = sorted(enumerate(req.movimentos), key=lambda im: (im[1].data, im[0]))
    held: dict[str, float] = {}
    rows: list[dict[str, float]] = []
    dates: list[pd.Timestamp] = []
    flows: dict[tuple[pd.Timestamp, str], float] = {}
    divs: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    for _, m in moves:
        t, d = norm(m.ticker), pd.Timestamp(m.data)
        if m.tipo == "dividendo":
            divs[d] = divs.get(d, 0.0) + float(m.valor)  # type: ignore[arg-type]
            trades.append({"data": str(m.data), "ticker": t, "tipo": m.tipo, "quantidade": None, "preco": None,
                           "valor_reais": float(m.valor), "nota": "provento recebido"})  # type: ignore[arg-type]
            continue
        q = float(m.quantidade)  # type: ignore[arg-type]
        if m.tipo == "venda" and q > held.get(t, 0.0) + 1e-9:
            raise ValueError(f"venda de {q:g} {t} em {m.data} maior que a quantidade que você tinha "
                             f"({held.get(t, 0.0):g}): confira as movimentações")
        held[t] = held.get(t, 0.0) + (q if m.tipo == "compra" else -q)
        px_native = market.native.get(t) if market is not None else None
        price, note = m.preco, ""
        if price is None:
            price = _price_on(px_native, d)
            note = "preço não informado: usado o fechamento do dia"
        if price is None and market is not None and market.has(t):
            raise ValueError(f"informe o preço da {m.tipo} de {t} em {m.data}: a fonte não tem cotação desse dia")
        rate = 1.0
        if market is not None and market.currency.get(t, "BRL") == "USD":
            rate = _price_on(market.native.get(FX_TICKER), d) or float("nan")
            if not np.isfinite(rate) and market.has(t):
                raise ValueError(f"sem câmbio USD/BRL em {m.data} para converter a {m.tipo} de {t}")
        value = q * price * rate if price is not None else float("nan")
        flows[(d, t)] = flows.get((d, t), 0.0) + (value if m.tipo == "compra" else -value)
        trades.append({"data": str(m.data), "ticker": t, "tipo": m.tipo, "quantidade": q, "preco": price,
                       "valor_reais": None if not np.isfinite(value) else value, "nota": note})
        dates.append(d)
        rows.append(dict(held))
    daily = pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).fillna(0.0)
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    from_moves = {t: float(v) for t, v in held.items() if v > 1e-12}
    if req.posicoes:
        diff = [t for t in set(from_moves) | {t for t in qty if t != CASH_TICKER}
                if abs(from_moves.get(t, 0.0) - qty.get(t, 0.0)) > 1e-6 * max(1.0, qty.get(t, 0.0))]
        if diff:
            notes.append("as movimentações não somam as posições informadas para " + ", ".join(sorted(diff)[:8])
                         + ": o painel usa as movimentações para a evolução e as posições para o valor atual")
        current = qty
    else:
        current = from_moves
    fl = pd.Series(flows, dtype=float)
    flow_df = (fl.unstack(fill_value=0.0).sort_index() if len(fl) else pd.DataFrame(dtype=float))
    return Book(sorted(set(current) | set(daily.columns)), current, cost, True, daily, flow_df,
                pd.Series(divs, dtype=float).sort_index(), trades, notes)


def require_some_data(book: Book, market: Market) -> None:
    usable = [t for t in book.tickers if t == CASH_TICKER and market.cdi is not None or market.has(t)]
    if not usable:
        raise DataUnavailableError("nenhum ativo da carteira tem cotações disponíveis agora: "
                                   + "; ".join(f"{t}: {market.missing.get(t, 'sem dados')}" for t in book.tickers[:8]))


def logo(t: str) -> str | None:
    return None if t == CASH_TICKER else logo_path(t)
