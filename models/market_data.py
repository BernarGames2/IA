"""Market-data containers with provenance and quality metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd


class DataOrigin(str, Enum):
    REAL = "real"            # downloaded from an external provider
    SYNTHETIC = "synthetic"  # deterministic generator, never market data
    CACHE = "cache"          # real data served from local cache
    USER_FILE = "user_file"  # CSV supplied by the user


class QualityFlag(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True)
class AssetMeta:
    """Static descriptive metadata. Source must be stated (config, provider...)."""

    symbol: str
    asset_class: str = "unknown"
    sector: str = "unknown"
    country: str = "unknown"
    currency: str = "unknown"
    market: str = "unknown"
    calendar: str = "unknown"   # B3, NYSE, 24x7
    metadata_source: str = "static_config"


@dataclass(frozen=True)
class SeriesProvenance:
    symbol: str
    provider: str
    origin: DataOrigin
    price_field: str              # e.g. "Close (auto_adjust=True)"
    adjusted_for: str             # e.g. "splits+dividends" / "none" / "synthetic"
    currency: str
    unit: str
    frequency: str
    first_observation: str | None
    last_observation: str | None
    n_observations: int
    ingested_at_utc: str
    published_at_utc: str | None = None
    cache_hit: bool = False
    notes: tuple[str, ...] = ()


@dataclass
class QualityIssue:
    symbol: str
    check: str
    severity: QualityFlag
    detail: str
    count: int = 0


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    per_symbol_flag: dict[str, QualityFlag] = field(default_factory=dict)
    excluded_symbols: dict[str, str] = field(default_factory=dict)

    def add(self, issue: QualityIssue) -> None:
        self.issues.append(issue)
        prev = self.per_symbol_flag.get(issue.symbol, QualityFlag.OK)
        order = {QualityFlag.OK: 0, QualityFlag.WARNING: 1, QualityFlag.FAIL: 2}
        if order[issue.severity] > order[prev]:
            self.per_symbol_flag[issue.symbol] = issue.severity
        else:
            self.per_symbol_flag.setdefault(issue.symbol, prev)

    @property
    def overall(self) -> QualityFlag:
        flags = list(self.per_symbol_flag.values())
        if any(f == QualityFlag.FAIL for f in flags):
            return QualityFlag.FAIL
        if any(f == QualityFlag.WARNING for f in flags):
            return QualityFlag.WARNING
        return QualityFlag.OK

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ | {"severity": i.severity.value} for i in self.issues],
                            columns=["symbol", "check", "severity", "detail", "count"])


@dataclass
class PriceDataset:
    """Aligned price panel plus provenance. ``prices`` columns are symbols."""

    prices: pd.DataFrame
    raw_prices: dict[str, pd.Series]
    provenance: dict[str, SeriesProvenance]
    quality: QualityReport
    meta: dict[str, AssetMeta]
    origin: DataOrigin
    alignment: str
    volumes: pd.DataFrame | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_synthetic(self) -> bool:
        return self.origin == DataOrigin.SYNTHETIC

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    def content_hash(self) -> str:
        """SHA-256 of the aligned price panel (values + index + columns)."""
        h = hashlib.sha256()
        h.update(",".join(map(str, self.prices.columns)).encode())
        h.update(",".join(self.prices.index.astype(str)).encode())
        h.update(self.prices.to_numpy(dtype="float64").tobytes())
        return h.hexdigest()
