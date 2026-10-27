"""Latest quotes — price of the last trade, change from the previous session's close, time of the last trade —
from the metadata of the Yahoo chart endpoint (one-day chart, 5-minute bars). Used by the market screen and by the
"Ação do dia".

* Delayed data, not real time: every quote carries the time of its last trade and, when the source reports it, the
  delay of the exchange's data (``exchangeDataDelayedBy``); B3 data on free sources is usually 15 minutes late.
* Change of the day = last price / previous session's close − 1 (``chartPreviousClose`` of a one-day chart); out of
  trading hours it is the change of the last session, said by the date of the last trade.
* Market open or closed from the exchange's regular trading period in the same metadata.
* The 5-minute closes of the day are returned for a small chart.
* A symbol the source does not answer is reported in ``failures`` and never filled.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from core.market_data.base import EmptyDataError
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider


@dataclass
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float | None
    change: float | None
    change_pct: float | None
    currency: str
    time_utc: str                 # time of the last trade, ISO
    date_local: str               # exchange's calendar date of the last trade
    time_local: str               # HH:MM, exchange's clock
    exchange: str
    market_open: bool | None
    delay_min: int | None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    intraday: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _num(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_quote(symbol: str, payload: dict, now: datetime | None = None) -> Quote:
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        err = chart.get("error") or {}
        raise EmptyDataError(f"{symbol}: {err.get('description') or err.get('code') or 'resposta sem dados'}")
    r = results[0]
    meta = r.get("meta") or {}
    quote = ((r.get("indicators") or {}).get("quote") or [{}])[0] or {}
    closes = [v for v in (_num(x) for x in (quote.get("close") or [])) if v is not None and v > 0]
    price = _num(meta.get("regularMarketPrice")) or (closes[-1] if closes else None)
    if price is None or price <= 0:
        raise EmptyDataError(f"{symbol}: sem preço na resposta")
    prev = _num(meta.get("chartPreviousClose")) or _num(meta.get("previousClose"))
    prev = prev if prev is not None and prev > 0 else None
    t = _num(meta.get("regularMarketTime"))
    stamps = r.get("timestamp") or []
    if t is None and stamps:
        t = _num(stamps[-1])
    when = datetime.fromtimestamp(t, UTC) if t is not None else None
    offset = _num(meta.get("gmtoffset"))
    if when is not None and offset is not None:
        local = when + timedelta(seconds=offset)
    elif when is not None:
        local = pd.Timestamp(when).tz_convert(meta.get("exchangeTimezoneName") or "UTC").to_pydatetime()
    else:
        local = None
    period = ((meta.get("currentTradingPeriod") or {}).get("regular") or {})
    start, end = _num(period.get("start")), _num(period.get("end"))
    now_ts = (now or datetime.now(UTC)).timestamp()
    is_open = (start <= now_ts < end) if start is not None and end is not None else None
    delay = meta.get("exchangeDataDelayedBy")
    return Quote(symbol=symbol, name=str(meta.get("longName") or meta.get("shortName") or symbol), price=price,
                 previous_close=prev, change=(price - prev) if prev else None,
                 change_pct=(price / prev - 1.0) if prev else None,
                 currency=str(meta.get("currency") or "").upper(),
                 time_utc=when.isoformat(timespec="seconds") if when else "",
                 date_local=local.strftime("%Y-%m-%d") if local else "",
                 time_local=local.strftime("%H:%M") if local else "",
                 exchange=str(meta.get("fullExchangeName") or meta.get("exchangeName") or ""),
                 market_open=is_open, delay_min=int(delay) if isinstance(delay, int | float) else None,
                 day_high=_num(meta.get("regularMarketDayHigh")), day_low=_num(meta.get("regularMarketDayLow")),
                 volume=_num(meta.get("regularMarketVolume")), intraday=closes[-120:])


class QuoteProvider(YahooChartProvider):
    """The chart provider's retries, pacing, parallel downloads, circuit breaker and time budget, returning
    :class:`Quote` objects (never cached on disk: a quote is only useful fresh)."""

    name = "yahoo_quote"

    def __init__(self, fetch_fn: FetchFn | None = None, time_budget_s: float | None = None,
                 progress: Callable[[str], None] | None = None, max_workers: int = 8,
                 now: Callable[[], datetime] | None = None, **kw: object) -> None:
        super().__init__(cache=None, fetch_fn=fetch_fn, time_budget_s=time_budget_s, progress=progress,
                         max_workers=max_workers, **kw)  # type: ignore[arg-type]
        self._now = now

    def _download(self, symbol: str, period: str, interval: str) -> Quote:  # type: ignore[override]
        return parse_quote(symbol, self._get_json(symbol, period, interval), self._now() if self._now else None)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return self.fetch(list(dict.fromkeys(symbols)), "1d", "5m")  # type: ignore[return-value]


QuotesFn = Callable[[list[str]], tuple[dict[str, Quote], dict[str, str]]]

_MEMORY: dict[str, tuple[float, Quote]] = {}
_MEMORY_LOCK = threading.Lock()


def remembered(get: QuotesFn, ttl_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> QuotesFn:
    """``get`` with a short memory shared by every request of this process: a quote fetched less than ``ttl_s`` ago
    is reused (screens refreshed by many people do not multiply the requests to the source)."""
    def run(symbols: list[str]) -> tuple[dict[str, Quote], dict[str, str]]:
        now = clock()
        hit: dict[str, Quote] = {}
        miss: list[str] = []
        with _MEMORY_LOCK:
            for s in dict.fromkeys(symbols):
                v = _MEMORY.get(s)
                if v is not None and now - v[0] < ttl_s:
                    hit[s] = v[1]
                else:
                    miss.append(s)
        got, failures = get(miss) if miss else ({}, {})
        with _MEMORY_LOCK:
            for s, q in got.items():
                _MEMORY[s] = (now, q)
        return {**hit, **got}, failures
    return run


def live_quotes(fetch: FetchFn | None = None, budget_s: float | None = 30.0,
                progress: Callable[[str], None] | None = None, **kw: object) -> QuotesFn:
    """A function symbols -> (quotes, failures) over the real source (``fetch`` = HTTP function, None = urllib;
    ``kw`` goes to :class:`QuoteProvider`, e.g. ``min_interval_s`` or ``now`` in tests)."""
    def get(symbols: list[str]) -> tuple[dict[str, Quote], dict[str, str]]:
        prov = QuoteProvider(fetch_fn=fetch, time_budget_s=budget_s, progress=progress, **kw)  # type: ignore[arg-type]
        got = prov.quotes(symbols)
        return got, {s: why for s, why in prov.failures.items() if s not in got}
    return get
