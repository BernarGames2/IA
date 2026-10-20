"""Shared test doubles for the market (no network): a price provider, a structured synthetic market in
which asset families differ in drift and volatility (so the optimiser has signal and profiles can
differ), and an HTTP function for the exchange listings and the Banco Central series (Selic, CDI)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.market_data.base import RawSeries
from tests.test_real_data import _real_symbols as real_symbols
from tests.test_universe import _http as listings_http

FAMILY_PARAMS = {  # (annual drift, annual volatility) by symbol family
    "fi": (0.10, 0.04), "eq_br": (0.12, 0.28), "eq_us": (0.11, 0.22), "fii": (0.09, 0.15),
    "gold": (0.07, 0.15), "crypto": (0.30, 0.70), "fx": (0.03, 0.12), "irx": (0.0, 0.0)}
FIXED_INCOME = {"IMAB11.SA", "B5P211.SA", "IRFM11.SA", "AGG", "SHY", "TIP", "BND", "IEF", "TLT", "LQD", "HYG"}
EQUITY_ETF_BR = {"BOVA11.SA", "IVVB11.SA", "SMAL11.SA", "DIVO11.SA", "NASD11.SA"}
END = "2026-10-16"
N_DAYS = 1300


def family(s: str) -> str:
    if s == "^IRX":
        return "irx"
    if s.endswith("=X"):
        return "fx"
    if s in FIXED_INCOME:
        return "fi"
    if s.endswith("11.SA") and s not in EQUITY_ETF_BR:
        return "fii"
    if s in ("GLD", "GOLD11.SA", "SLV"):
        return "gold"
    if s.endswith("-USD") or s == "HASH11.SA":
        return "crypto"
    return "eq_br" if s.endswith(".SA") else "eq_us"


def structured_market(seed_offset: int = 0, n: int = N_DAYS, end: str = END,
                      extra: dict[str, str] | None = None, drift_shift: dict[str, float] | None = None
                      ) -> dict[str, RawSeries]:
    """Prices for the curated catalogue (+ ``extra`` symbol -> currency), one seed per symbol;
    ``seed_offset`` gives a different market history, ``drift_shift`` moves one symbol's drift."""
    idx = pd.bdate_range(end=end, periods=n)
    syms = dict(real_symbols())
    syms.update(extra or {})
    out = {}
    for i, (sym, ccy) in enumerate(sorted(syms.items())):
        mu, vol = FAMILY_PARAMS[family(sym)]
        mu += (drift_shift or {}).get(sym, 0.0)
        rng = np.random.default_rng(10_000 + 97 * seed_offset + i)
        if sym == "^IRX":
            x = pd.Series(np.full(n, 4.0), index=idx)
        else:
            d = vol / np.sqrt(252)
            x = pd.Series((5.0 if sym.endswith("=X") else 50.0)
                          * np.exp(np.cumsum(rng.normal(mu / 252 - d * d / 2, d, n))), index=idx)
        out[sym] = RawSeries(sym, x.rename(sym), pd.Series(np.full(n, 1e6), index=idx), "fake_test_double",
                             "adjclose", "teste", ccy, "1d", ingested_at_utc=f"{end}T21:00:00+00:00")
    return out


class Prov:
    """Minimal MarketDataProvider double; ``fail`` lists symbols that come back without data."""

    name = "fake"

    def __init__(self, series: dict[str, RawSeries], fail: set[str] | None = None) -> None:
        self.series = series
        self.fail = set(fail or ())
        self.failures: dict[str, str] = {}
        self.calls = 0

    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        self.calls += 1
        for s in symbols:
            if s in self.fail or s not in self.series:
                self.failures[s] = "HTTP 404 (teste)"
        return {s: self.series[s] for s in symbols if s in self.series and s not in self.fail}


def cdi_json(end: str = END, n: int = N_DAYS, daily_pct: float = 0.04) -> bytes:
    idx = pd.bdate_range(end=end, periods=n)
    return json.dumps([{"data": d.strftime("%d/%m/%Y"), "valor": f"{daily_pct:.6f}"} for d in idx]).encode()


def http(url: str, timeout: float) -> tuple[int, bytes]:
    """Listings (B3, Nasdaq) + Selic (SGS 432) + CDI (SGS 12)."""
    if "bcdata.sgs.12" in url:
        return 200, cdi_json()
    return listings_http(url, timeout)


def http_without_cdi(url: str, timeout: float) -> tuple[int, bytes]:
    if "bcdata.sgs.12" in url:
        return 503, b""
    return listings_http(url, timeout)
