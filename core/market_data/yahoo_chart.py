"""Yahoo Finance *chart* endpoint (v8) using only the standard library (no ``yfinance``).

Used by the purpose engine and the web app (keeps the serverless bundle small). Same guarantees
as the yfinance connector:

* price field: ``adjclose`` (adjusted by Yahoo for splits AND dividends); falls back to
  ``close`` only when the response has no adjusted series, and says so in ``adjusted_for``;
* currency and exchange time zone come from the response metadata; bar timestamps are
  converted to the exchange's local calendar date;
* request pacing, timeouts, retries with exponential backoff for network errors, HTTP 429
  and 5xx (alternating the two Yahoo hosts); HTTP 404 is a definitive "ticker not found";
* optional file cache with expiry; a failed symbol is reported in ``failures`` and is never
  replaced by synthetic data.

Yahoo's endpoint is unofficial and its terms restrict redistribution: use it for personal
research, with conservative pacing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from core.market_data.base import DownloadError, EmptyDataError, RawSeries
from core.market_data.cache import SeriesCache
from core.market_data.yfinance_provider import RateLimiter
from utils.logging_config import get_logger

log = get_logger("market_data.yahoo_chart")

HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36 qpi-research")
_SYMBOL_OK = re.compile(r"^[A-Za-z0-9.^=\-]{1,20}$")
_RETRY_STATUS = {429, 500, 502, 503, 504}

FetchFn = Callable[[str, float], tuple[int, bytes]]     # (url, timeout_s) -> (http_status, body)


class NetworkDownloadError(DownloadError):
    """The provider could not be reached at all (DNS, proxy, connection refused, timeout)."""


def urllib_fetch(url: str, timeout_s: float) -> tuple[int, bytes]:
    """HTTP GET with the standard library. Network failures raise ``OSError``/``URLError``."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except OSError:
            body = b""
        return int(e.code), body


def chart_url(symbol: str, period: str, interval: str, host: str = HOSTS[0]) -> str:
    if not _SYMBOL_OK.match(symbol):
        raise DownloadError(f"símbolo inválido: {symbol!r}")
    q = urllib.parse.urlencode({"range": period, "interval": interval, "includeAdjustedClose": "true",
                                "events": "div,split"})
    return f"https://{host}/v8/finance/chart/{urllib.parse.quote(symbol)}?{q}"


def parse_chart(symbol: str, payload: dict, provider: str = "yahoo_chart_v8") -> RawSeries:
    """Convert a chart-API JSON payload into a :class:`RawSeries` (no value is invented:
    null prices are dropped and left to the quality layer)."""
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        err = chart.get("error") or {}
        raise EmptyDataError(f"{symbol}: {err.get('description') or err.get('code') or 'resposta sem dados'}")
    r = results[0]
    meta = r.get("meta") or {}
    ts = r.get("timestamp") or []
    ind = r.get("indicators") or {}
    adj = ((ind.get("adjclose") or [{}])[0] or {}).get("adjclose")
    quote = (ind.get("quote") or [{}])[0] or {}
    raw_close = adj if adj is not None else quote.get("close")
    if not ts or raw_close is None or len(raw_close) != len(ts):
        raise EmptyDataError(f"{symbol}: série de preços vazia ou inconsistente")
    tz = meta.get("exchangeTimezoneName") or "UTC"
    idx = (pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True)
           .tz_convert(tz).tz_localize(None).normalize())
    close = pd.Series(pd.to_numeric(pd.Series(raw_close), errors="coerce").to_numpy(float), index=idx, name=symbol)
    vol_raw = quote.get("volume")
    volume = None
    if vol_raw is not None and len(vol_raw) == len(ts):
        volume = pd.Series(pd.to_numeric(pd.Series(vol_raw), errors="coerce").to_numpy(float), index=idx, name=symbol)
    keep = close.notna()
    close = close[keep]
    volume = volume[keep] if volume is not None else None
    if close.empty:
        raise EmptyDataError(f"{symbol}: todos os preços são nulos")
    notes = [f"bolsa {meta.get('exchangeName', '?')} ({tz}); instrumento {meta.get('instrumentType', '?')}"]
    if adj is None:
        notes.append("sem série ajustada na resposta: usado o fechamento ajustado só por desdobramentos")
    return RawSeries(symbol=symbol, close=close, volume=volume, provider=provider,
                     price_field="adjclose" if adj is not None else "close",
                     adjusted_for="desdobramentos e dividendos (adjclose do Yahoo)" if adj is not None
                     else "somente desdobramentos (close do Yahoo)",
                     currency=str(meta.get("currency") or "unknown").upper(), frequency="1d",
                     ingested_at_utc=datetime.now(UTC).isoformat(), notes=notes)


class YahooChartProvider:
    """MarketDataProvider over the chart endpoint. ``fetch_fn`` is injectable for tests."""

    name = "yahoo_chart_v8"

    def __init__(self, cache: SeriesCache | None = None, timeout_s: float = 20.0, max_retries: int = 3,
                 min_interval_s: float = 0.35, fetch_fn: FetchFn | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_consecutive_network_failures: int = 3) -> None:
        self.cache = cache
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._fetch = fetch_fn or urllib_fetch
        self._sleep = sleep
        self.limiter = RateLimiter(min_interval_s, sleep=sleep)
        self.failures: dict[str, str] = {}
        self.n_requests = 0
        # circuit breaker: when the host is unreachable, stop after a few symbols instead of
        # spending minutes in backoff for every remaining ticker
        self.max_consecutive_network_failures = max_consecutive_network_failures
        self._net_fail_streak = 0

    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        out: dict[str, RawSeries] = {}
        for s in symbols:
            if self.cache is not None:
                hit = self.cache.get(self.name, s, period, interval)
                if hit is not None:
                    out[s] = hit
                    continue
            if self._net_fail_streak >= self.max_consecutive_network_failures:
                self.failures[s] = ("não tentado: provedor inacessível (falhas de rede em "
                                    f"{self._net_fail_streak} tickers seguidos)")
                continue
            try:
                rs = self._download(s, period, interval)
            except NetworkDownloadError as e:
                self._net_fail_streak += 1
                self.failures[s] = str(e)
                log.warning("download failed (network)", extra={"symbol": s, "error": str(e)})
                continue
            except (DownloadError, EmptyDataError) as e:
                self._net_fail_streak = 0
                self.failures[s] = str(e)
                log.warning("download failed", extra={"symbol": s, "error": str(e)})
                continue
            self._net_fail_streak = 0
            out[s] = rs
            if self.cache is not None:
                try:
                    self.cache.put(rs, period, interval)
                except OSError as e:                      # read-only file system etc.: keep going
                    log.warning("cache write failed", extra={"symbol": s, "error": str(e)})
        return out

    def _download(self, symbol: str, period: str, interval: str) -> RawSeries:
        last = "sem tentativa"
        net_errors = 0
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
            self.limiter.wait()
            url = chart_url(symbol, period, interval, HOSTS[attempt % len(HOSTS)])
            self.n_requests += 1
            try:
                status, body = self._fetch(url, self.timeout_s)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"erro de rede: {e}"
                net_errors += 1
                if net_errors >= 2:              # unreachable host: one quick retry is enough
                    raise NetworkDownloadError(f"{symbol}: {last}") from e
                continue
            if status == 200:
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    last = f"resposta inválida: {e}"
                    continue
                return parse_chart(symbol, payload, self.name)
            if status == 404:
                raise DownloadError(f"{symbol}: ticker não encontrado no provedor (HTTP 404)")
            last = f"HTTP {status}"
            if status not in _RETRY_STATUS:
                raise DownloadError(f"{symbol}: {last}")
        raise DownloadError(f"{symbol}: falhou após {self.max_retries + 1} tentativas ({last})")
