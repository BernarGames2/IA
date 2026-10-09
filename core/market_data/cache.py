"""File cache with expiry for downloaded series (CSV + JSON metadata sidecar)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from core.market_data.base import RawSeries


class SeriesCache:
    """Cache keyed by (provider, symbol, period, interval). TTL in hours."""

    def __init__(self, root: Path, ttl_hours: float = 24.0) -> None:
        self.root = Path(root)
        self.ttl_s = ttl_hours * 3600.0

    def _key(self, provider: str, symbol: str, period: str, interval: str) -> str:
        raw = f"{provider}|{symbol}|{period}|{interval}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.csv", self.root / f"{key}.json"

    def get(self, provider: str, symbol: str, period: str, interval: str) -> RawSeries | None:
        key = self._key(provider, symbol, period, interval)
        csv_p, meta_p = self._paths(key)
        if not (csv_p.exists() and meta_p.exists()):
            return None
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if time.time() - float(meta["stored_at_epoch"]) > self.ttl_s:
            return None
        df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
        if hashlib.sha256(csv_p.read_bytes()).hexdigest() != meta.get("sha256"):
            return None  # corrupted / tampered cache entry: ignore it
        vol = df["volume"] if "volume" in df.columns else None
        return RawSeries(
            symbol=symbol, close=df["close"].rename(symbol),
            volume=None if vol is None else vol.rename(symbol),
            provider=meta["provider"], price_field=meta["price_field"],
            adjusted_for=meta["adjusted_for"], currency=meta["currency"],
            frequency=meta["frequency"], cache_hit=True,
            ingested_at_utc=meta["ingested_at_utc"],
            notes=list(meta.get("notes", [])) + ["served from local cache"],
        )

    def put(self, s: RawSeries, period: str, interval: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        key = self._key(s.provider, s.symbol, period, interval)
        csv_p, meta_p = self._paths(key)
        df = pd.DataFrame({"close": s.close})
        if s.volume is not None:
            df["volume"] = s.volume
        df.to_csv(csv_p)
        meta = {
            "provider": s.provider, "symbol": s.symbol, "period": period, "interval": interval,
            "price_field": s.price_field, "adjusted_for": s.adjusted_for, "currency": s.currency,
            "frequency": s.frequency, "ingested_at_utc": s.ingested_at_utc,
            "stored_at_epoch": time.time(), "notes": s.notes,
            "sha256": hashlib.sha256(csv_p.read_bytes()).hexdigest(),
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return csv_p
