"""Yahoo Finance connector via ``yfinance`` (initial connector, not a universal source).

* Uses ``auto_adjust=True`` and the ``Close`` column only, i.e. prices adjusted for
  splits and dividends by Yahoo. ``Adj Close`` is never mixed in, which avoids
  double-adjusting dividends.
* Rate limiting (minimum interval between requests), retries with exponential
  backoff and a request timeout are applied.
* A symbol that fails to download is reported as a *download failure*; it is never
  replaced by synthetic data. Gaps inside a valid series are market closures or
  missing observations and are handled by the quality layer.
* Respect Yahoo's terms of use: this connector is for personal research; the
  default request pacing is deliberately conservative.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from core.market_data.base import DownloadError, EmptyDataError, RawSeries
from core.market_data.cache import SeriesCache
from core.market_data.universe import meta_for
from utils.logging_config import get_logger

log = get_logger("market_data.yfinance")

_NETWORK_ERROR = re.compile(r"ConnectionError|Timeout|timed out|CONNECT tunnel|curl: \(\d+\)|Max retries|"
                            r"Name or service not known|ProxyError|Connection reset|SSLError|Too Many Requests|429",
                            re.IGNORECASE)

DownloadFn = Callable[..., pd.DataFrame]


class RateLimiter:
    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delta = now - self._last
            if delta < self.min_interval_s:
                self._sleep(self.min_interval_s - delta)
        self._last = self._clock()


def normalize_download(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a yfinance download into per-symbol frames with lower-case columns.

    Handles (field, ticker) and (ticker, field) MultiIndex layouts and flat
    single-ticker frames. Only ``close`` and ``volume`` are kept.
    """
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    fields = {"open", "high", "low", "close", "adj close", "volume", "dividends", "stock splits"}
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = {str(x).lower() for x in df.columns.get_level_values(0)}
        field_level = 0 if lvl0 & fields else 1
        ticker_level = 1 - field_level
        for sym in symbols:
            if sym not in set(df.columns.get_level_values(ticker_level)):
                continue
            sub = df.xs(sym, axis=1, level=ticker_level).copy()
            sub.columns = [str(c).lower() for c in sub.columns]
            out[sym] = sub
    else:
        if len(symbols) != 1:
            raise DownloadError("flat columns returned for a multi-symbol request")
        sub = df.copy()
        sub.columns = [str(c).lower() for c in sub.columns]
        out[symbols[0]] = sub
    cleaned: dict[str, pd.DataFrame] = {}
    for sym, sub in out.items():
        if "close" not in sub.columns:
            continue
        keep = sub[[c for c in ("close", "volume") if c in sub.columns]]
        keep = keep.dropna(how="all")
        if keep["close"].notna().sum() == 0:
            continue
        idx = pd.DatetimeIndex(keep.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        keep.index = idx.normalize()
        cleaned[sym] = keep
    return cleaned


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, cache: SeriesCache | None = None, timeout_s: float = 30.0,
                 max_retries: int = 3, min_interval_s: float = 1.0, batch_size: int = 10,
                 download_fn: DownloadFn | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cache = cache
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._sleep = sleep
        self.limiter = RateLimiter(min_interval_s, sleep=sleep)
        self._download = download_fn or self._default_download
        self.failures: dict[str, str] = {}

    @staticmethod
    def _default_download(tickers: list[str], period: str, interval: str,
                          timeout: float) -> pd.DataFrame:
        """Call yfinance and turn its *logged* network failures into exceptions.

        ``yf.download`` does not raise on connection errors: it logs them and returns
        an empty frame. Without this, retries/backoff would never run and an outage
        would be misreported as "no observations" (an invalid ticker).
        """
        import yfinance as yf  # imported lazily: offline mode never needs it

        class _Capture(logging.Handler):
            def __init__(self) -> None:
                super().__init__(logging.DEBUG)
                self.messages: list[str] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.messages.append(record.getMessage())

        cap = _Capture()
        yf_logger = logging.getLogger("yfinance")
        yf_logger.addHandler(cap)
        try:
            df = yf.download(tickers=tickers, period=period, interval=interval,
                             auto_adjust=True, actions=False, progress=False, threads=False,
                             group_by="column", timeout=timeout)
        finally:
            yf_logger.removeHandler(cap)
        net = [m for m in cap.messages if _NETWORK_ERROR.search(m)]
        got = normalize_download(df, tickers) if df is not None else {}
        if net and len(got) < len(tickers):
            raise ConnectionError("network failure reported by yfinance: " + " | ".join(net[:2])[:500])
        return df

    def _download_with_retry(self, tickers: list[str], period: str, interval: str) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                return self._download(tickers, period, interval, self.timeout_s)
            except (OSError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
                last_err = e
                log.warning("download attempt failed", extra={"tickers": tickers,
                                                              "attempt": attempt, "error": str(e)})
                if attempt < self.max_retries:
                    self._sleep(2.0 ** attempt)
        raise DownloadError(f"download failed after {self.max_retries + 1} attempts: {last_err}")

    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        """Fetch symbols (cache first, then batched download, then per-symbol retry)."""
        self.failures = {}
        result: dict[str, RawSeries] = {}
        pending: list[str] = []
        for s in symbols:
            cached = self.cache.get(self.name, s, period, interval) if self.cache else None
            if cached is not None:
                result[s] = cached
            else:
                pending.append(s)

        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            try:
                frames = normalize_download(self._download_with_retry(batch, period, interval), batch)
            except DownloadError as e:
                frames = {}
                for s in batch:
                    self.failures[s] = str(e)
            missing = [s for s in batch if s not in frames and s not in self.failures]
            for s in missing:  # individual retry distinguishes bad ticker from batch failure
                try:
                    single = normalize_download(self._download_with_retry([s], period, interval), [s])
                    if s in single:
                        frames[s] = single[s]
                    else:
                        raise EmptyDataError(f"{s}: provider returned no observations")
                except (DownloadError, EmptyDataError) as e:
                    self.failures[s] = str(e)
            now = datetime.now(timezone.utc).isoformat()
            for s, f in frames.items():
                rs = RawSeries(
                    symbol=s, close=f["close"].rename(s),
                    volume=f["volume"].rename(s) if "volume" in f.columns else None,
                    provider=self.name, price_field="Close (auto_adjust=True)",
                    adjusted_for="splits+dividends (Yahoo adjustment)",
                    currency=meta_for(s).currency, frequency=interval, ingested_at_utc=now,
                )
                if self.cache:
                    self.cache.put(rs, period, interval)
                result[s] = rs
        return result
