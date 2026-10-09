"""Provider interface and errors for market data acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


class MarketDataError(RuntimeError):
    """Base class for market-data errors."""


class DownloadError(MarketDataError):
    """The provider could not be reached or returned an error (NOT a market closure)."""


class EmptyDataError(MarketDataError):
    """The provider answered but returned no observations for the symbol."""


class DataUnavailableError(MarketDataError):
    """Not enough valid series remain to run the requested analysis."""


@dataclass
class RawSeries:
    """One symbol as returned by a provider, before quality checks/alignment."""

    symbol: str
    close: pd.Series                   # adjusted close as defined by ``price_field``
    volume: pd.Series | None
    provider: str
    price_field: str
    adjusted_for: str
    currency: str
    frequency: str
    cache_hit: bool = False
    ingested_at_utc: str = ""
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        """Fetch symbols. Must raise or omit failed symbols; never fabricate data."""
        ...
