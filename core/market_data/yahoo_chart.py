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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


class ThrottledError(DownloadError):
    """The provider kept answering 429/5xx (rate limiting or outage) until the retries ran out."""


class YahooChartProvider:
    """MarketDataProvider over the chart endpoint. ``fetch_fn`` is injectable for tests.

    Speed and robustness for interactive use (a web request must not hang for minutes):
    * the first missing symbols are fetched one at a time until one succeeds (a cheap probe);
      the rest are then downloaded in parallel (``max_workers``) under a shared rate limiter;
    * a circuit breaker stops after ``max_consecutive_failures`` symbols in a row failed for
      network or throttling reasons — the provider is unreachable or rate limiting us;
    * an optional ``time_budget_s`` (for the provider's whole lifetime) bounds the total
      download time; symbols not started in time are reported as such;
    * ``progress`` receives one short message per symbol.
    """

    name = "yahoo_chart_v8"

    def __init__(self, cache: SeriesCache | None = None, timeout_s: float = 20.0, max_retries: int = 3,
                 min_interval_s: float = 0.15, fetch_fn: FetchFn | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_consecutive_failures: int = 3,
                 max_workers: int = 6, time_budget_s: float | None = None,
                 progress: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cache = cache
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._fetch = fetch_fn or urllib_fetch
        self._sleep = sleep
        self.limiter = RateLimiter(min_interval_s, sleep=sleep)
        self.failures: dict[str, str] = {}
        self.n_requests = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.max_workers = max(1, int(max_workers))
        self.time_budget_s = time_budget_s
        self.progress = progress
        self._clock = clock
        self._deadline: float | None = None
        self._streak = 0
        self._lock = threading.Lock()
        self._limit_lock = threading.Lock()

    # -- public -------------------------------------------------------------------------
    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        out: dict[str, RawSeries] = {}
        total = len(symbols)
        state = {"done": 0}

        def report(sym: str, ok: bool) -> None:
            with self._lock:
                state["done"] += 1
                done = state["done"]
            if self.progress is not None:
                self.progress(f"cotações {done}/{total}: {sym} {'ok' if ok else 'sem dados'}")

        pending = []
        for s in symbols:
            hit = self.cache.get(self.name, s, period, interval) if self.cache is not None else None
            if hit is not None:
                out[s] = hit
                report(s, True)
            else:
                pending.append(s)
        if not pending:
            return out
        if self.time_budget_s is not None and self._deadline is None:
            self._deadline = self._clock() + self.time_budget_s
        i = 0
        while i < len(pending) and not self._circuit_open():       # probe until one success
            sym = pending[i]
            i += 1
            if self._one(sym, period, interval, out, report):
                break
        rest = pending[i:]
        if rest and self.max_workers > 1 and not self._circuit_open():
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(rest))) as ex:
                list(ex.map(lambda sym: self._one(sym, period, interval, out, report), rest))
        else:
            for sym in rest:
                self._one(sym, period, interval, out, report)
        return out

    # -- internals ----------------------------------------------------------------------
    def _circuit_open(self) -> bool:
        return self._streak >= self.max_consecutive_failures

    def _remaining(self) -> float | None:
        return None if self._deadline is None else self._deadline - self._clock()

    def _one(self, sym: str, period: str, interval: str, out: dict[str, RawSeries],
             report: Callable[[str, bool], None]) -> bool:
        if self._circuit_open():
            self.failures[sym] = (f"não tentado: provedor inacessível ou limitando requisições ({self._streak} "
                                  "tickers seguidos falharam)")
            report(sym, False)
            return False
        rem = self._remaining()
        if rem is not None and rem <= 0:
            self.failures[sym] = "não tentado: tempo de download esgotado"
            report(sym, False)
            return False
        try:
            rs = self._download(sym, period, interval)
        except (NetworkDownloadError, ThrottledError) as e:
            with self._lock:
                self._streak += 1
            self.failures[sym] = str(e)
            log.warning("download failed (provider)", extra={"symbol": sym, "error": str(e)})
            report(sym, False)
            return False
        except (DownloadError, EmptyDataError) as e:           # definitive for this symbol only
            with self._lock:
                self._streak = 0
            self.failures[sym] = str(e)
            log.warning("download failed", extra={"symbol": sym, "error": str(e)})
            report(sym, False)
            return False
        with self._lock:
            self._streak = 0
        out[sym] = rs
        if self.cache is not None:
            try:
                self.cache.put(rs, period, interval)
            except OSError as e:                              # read-only file system etc.: keep going
                log.warning("cache write failed", extra={"symbol": sym, "error": str(e)})
        report(sym, True)
        return True

    def _download(self, symbol: str, period: str, interval: str) -> RawSeries:
        last = "sem tentativa"
        net_errors = 0
        throttled = False
        for attempt in range(self.max_retries + 1):
            if attempt:
                rem = self._remaining()
                wait = min(2.0 ** (attempt - 1), 8.0)
                if rem is not None and rem <= wait:
                    raise ThrottledError(f"{symbol}: tempo de download esgotado ({last})")
                self._sleep(wait)
            with self._limit_lock:
                self.limiter.wait()
            url = chart_url(symbol, period, interval, HOSTS[attempt % len(HOSTS)])
            with self._lock:
                self.n_requests += 1
            rem = self._remaining()
            timeout = self.timeout_s if rem is None else max(1.0, min(self.timeout_s, rem))
            try:
                status, body = self._fetch(url, timeout)
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
                    throttled = False
                    continue
                return parse_chart(symbol, payload, self.name)
            if status == 404:
                raise DownloadError(f"{symbol}: ticker não encontrado no provedor (HTTP 404)")
            if status in (401, 403):              # the provider refuses this client, not just this ticker
                raise ThrottledError(f"{symbol}: provedor recusou o acesso (HTTP {status})")
            last = f"HTTP {status}"
            if status not in _RETRY_STATUS:
                raise DownloadError(f"{symbol}: {last}")
            throttled = True
        err = ThrottledError if throttled else DownloadError
        raise err(f"{symbol}: falhou após {self.max_retries + 1} tentativas ({last})")
