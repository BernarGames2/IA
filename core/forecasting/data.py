"""Price, benchmark and risk-free series for the forecasts (real data; failures reported, never filled).

* asset and benchmark (BOVA11.SA for reais, SPY for dollars): adjusted daily closes from the provider;
* risk-free as an ACCUMULATION INDEX L(t), so the rate known on day t is (L_t / L_{t-1})^252 - 1 and the money a
  risk-free deposit made between t0 and t1 is L(t1) / L(t0) - 1: for reais the CDI (BCB SGS 12, already an index),
  for dollars the 13-week T-bill (^IRX, annual % yield) accumulated day by day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from core.market_data.base import DataUnavailableError, last_per_date
from core.market_data.cash import fetch_cdi_cash
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import norm_ticker

BENCH = {"BRL": ("BOVA11.SA", "Ibovespa (BOVA11)"), "USD": ("SPY", "S&P 500 (SPY)")}
RF_NAME = {"BRL": "CDI", "USD": "T-bill de 13 semanas"}
P = 252


def _clean(s: pd.Series, end: date) -> pd.Series:
    s = last_per_date(s.dropna()).sort_index()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    s = last_per_date(s)
    return s[s > 0].loc[:pd.Timestamp(end)]


def rf_index_from_annual_pct(rate_pct: pd.Series) -> pd.Series:
    """Accumulation index from an annual yield in percent quoted daily (^IRX): each quoted day earns
    (1 + rate)^(1/252) - 1, so the rate read back by ``rf_rate_at`` is the quoted one."""
    r = rate_pct.astype(float) / 100.0
    return ((1.0 + r) ** (1.0 / P)).cumprod()


def rf_rate_at(idx: pd.Series | None, t: pd.Timestamp) -> float | None:
    """Annual rate known on day t (from the last two index values up to t)."""
    if idx is None:
        return None
    s = idx.loc[:t]
    if len(s) < 2:
        return None
    return float((s.iloc[-1] / s.iloc[-2]) ** P - 1.0)


def rf_growth(idx: pd.Series | None, t0: pd.Timestamp, t1: pd.Timestamp) -> float | None:
    if idx is None:
        return None
    a, b = idx.loc[:t0], idx.loc[:t1]
    if a.empty or b.empty or (t1 - b.index[-1]).days > 7 or (t0 - a.index[-1]).days > 7:
        return None
    return float(b.iloc[-1] / a.iloc[-1] - 1.0)


@dataclass
class Series3:
    ticker: str
    currency: str
    px: pd.Series
    bench: pd.Series | None
    bench_name: str | None
    rf_index: pd.Series | None
    rf_name: str
    sources: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def bench_ticker(self) -> str | None:
        return BENCH[self.currency][0] if self.bench is not None and self.currency in BENCH else None


def load(ticker: str, provider: object, fetch: FetchFn | None, period: str = "10y", end: date | None = None) -> Series3:
    end = end or datetime.now(UTC).date()
    t = norm_ticker(ticker)
    got = provider.fetch([t], period, "1d")  # type: ignore[attr-defined]
    if t not in got or got[t].close is None or got[t].close.dropna().empty:
        why = getattr(provider, "failures", {}).get(t, "sem cotações na fonte")
        raise DataUnavailableError(f"{t}: {why}")
    rs = got[t]
    ccy = (rs.currency or "").upper() or ("BRL" if t.endswith(".SA") else "USD")
    px = _clean(rs.close, end)
    notes: list[str] = []
    sources = [{"dado": f"{t}: fechamentos diários ajustados", "fonte": getattr(provider, "name", "provedor"),
                "data": f"{px.index[-1]:%Y-%m-%d}"}]
    bench = bench_name = None
    if ccy in BENCH and BENCH[ccy][0] != t:
        bt, bench_name = BENCH[ccy]
        b = provider.fetch([bt], period, "1d").get(bt)  # type: ignore[attr-defined]
        if b is not None and b.close is not None and not b.close.dropna().empty:
            bench = _clean(b.close, end)
            sources.append({"dado": f"referência: {bench_name}", "fonte": getattr(provider, "name", "provedor"),
                            "data": f"{bench.index[-1]:%Y-%m-%d}"})
        else:
            bench_name = None
            notes.append("índice de referência indisponível: sem comparação com o mercado nem beta")
    rf_index = None
    try:
        if ccy == "BRL":
            # the BCB serves at most 10 years of a daily series per request
            start = end - timedelta(days=3650 if period == "10y" else 1830)
            rf_index = _clean(fetch_cdi_cash(fetch, start=start, end=end).close, end)
            sources.append({"dado": "CDI diário", "fonte": "Banco Central do Brasil, SGS 12", "data": f"{rf_index.index[-1]:%Y-%m-%d}"})
        elif ccy == "USD":
            irx = provider.fetch(["^IRX"], period, "1d").get("^IRX")  # type: ignore[attr-defined]
            if irx is not None and irx.close is not None and not irx.close.dropna().empty:
                rf_index = rf_index_from_annual_pct(_clean(irx.close, end))
                sources.append({"dado": "T-bill de 13 semanas (^IRX)", "fonte": getattr(provider, "name", "provedor"),
                                "data": f"{rf_index.index[-1]:%Y-%m-%d}"})
    except Exception as e:  # noqa: BLE001 - any failure of the rate source is reported, never filled
        notes.append(f"taxa livre de risco histórica indisponível ({e}): sem eventos 'superar a renda fixa'")
        rf_index = None
    if rf_index is None and not any("taxa livre" in n for n in notes):
        notes.append("taxa livre de risco histórica indisponível: sem eventos 'superar a renda fixa'")
    return Series3(t, ccy, px, bench, bench_name, rf_index, RF_NAME.get(ccy, "renda fixa"), sources, notes)


def daily_returns(px: pd.Series) -> np.ndarray:
    return px.pct_change(fill_method=None).dropna().to_numpy()
