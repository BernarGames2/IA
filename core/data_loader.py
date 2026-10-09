"""Market Intelligence Engine facade: acquire, validate, convert and align prices."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.market_data.base import DataUnavailableError, MarketDataProvider, RawSeries
from core.market_data.quality import QualitySettings, align_prices, clean_series, staleness_check
from core.market_data.universe import meta_for
from models.market_data import (
    AssetMeta,
    DataOrigin,
    PriceDataset,
    QualityFlag,
    QualityIssue,
    QualityReport,
    SeriesProvenance,
)
from utils.logging_config import get_logger

log = get_logger("data_loader")


def fx_symbol(ccy: str, base: str) -> str:
    """Yahoo FX symbol giving units of ``base`` per 1 unit of ``ccy`` (e.g. USDBRL=X)."""
    return f"{ccy}{base}=X"


def convert_currency(close: pd.Series, fx: pd.Series, max_fx_staleness_days: int = 3) -> tuple[pd.Series, int]:
    """Convert a price series with an FX series observed on/before each date.

    FX is carried forward at most ``max_fx_staleness_days`` *calendar* days (FX and
    exchange calendars differ). Asset prices are never forward-filled; dates without
    a recent FX quote become missing. Returns (converted, n_dates_without_fx).
    """
    fx = fx.sort_index()
    aligned = fx.reindex(close.index, method="ffill",
                         tolerance=pd.Timedelta(days=max_fx_staleness_days))
    converted = close * aligned
    return converted, int(aligned.isna().sum())


def load_price_dataset(
    symbols: list[str],
    provider: MarketDataProvider,
    *,
    period: str = "5y",
    interval: str = "1d",
    base_currency: str = "BRL",
    meta: Mapping[str, AssetMeta] | None = None,
    quality: QualitySettings | None = None,
    origin: DataOrigin = DataOrigin.REAL,
) -> PriceDataset:
    """Fetch, quality-check, convert to base currency and align a price panel.

    Raises :class:`DataUnavailableError` if fewer than two usable series remain.
    Failed symbols are excluded and documented — never replaced by synthetic data.
    """
    qs = quality or QualitySettings()
    report = QualityReport()
    raw = provider.fetch(symbols, period, interval)
    failures: dict[str, str] = dict(getattr(provider, "failures", {}) or {})
    for s in symbols:
        if s not in raw:
            reason = failures.get(s, "not returned by provider")
            report.add(QualityIssue(s, "download_failure", QualityFlag.FAIL, reason))
            report.excluded_symbols[s] = f"download failure: {reason}"

    metas = {s: (meta or {}).get(s) or meta_for(s) for s in symbols}
    fx_cache: dict[str, RawSeries | None] = {}
    cleaned: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    prov: dict[str, SeriesProvenance] = {}
    for s, rs in raw.items():
        m = metas[s]
        x = clean_series(rs, m, report, qs)
        if x is None:
            continue
        notes = list(rs.notes)
        ccy = rs.currency if rs.currency != "unknown" else m.currency
        if ccy == "unknown":
            report.add(QualityIssue(s, "unknown_currency", QualityFlag.WARNING,
                                    "currency unknown; assumed to be base currency"))
            notes.append(f"currency unknown, ASSUMED {base_currency}")
            ccy = base_currency
        if ccy != base_currency:
            fx_sym = fx_symbol(ccy, base_currency)
            if fx_sym not in fx_cache:
                fx_raw = provider.fetch([fx_sym], period, interval)
                fx_cache[fx_sym] = fx_raw.get(fx_sym)
            fx_rs = fx_cache[fx_sym]
            if fx_rs is None or fx_rs.close.dropna().empty:
                report.add(QualityIssue(s, "fx_unavailable", QualityFlag.FAIL,
                                        f"{fx_sym} unavailable; cannot express in {base_currency}"))
                report.excluded_symbols[s] = f"FX {fx_sym} unavailable"
                continue
            x, n_nofx = convert_currency(x, fx_rs.close.dropna())
            if n_nofx:
                report.add(QualityIssue(s, "fx_gaps", QualityFlag.WARNING,
                                        f"{n_nofx} dates without FX quote within 3 days dropped", n_nofx))
            x = x.dropna()
            notes.append(f"converted {ccy}->{base_currency} with {fx_sym} (provider {fx_rs.provider})")
            ccy = base_currency
        cleaned[s] = x
        if rs.volume is not None:
            volumes[s] = rs.volume.reindex(x.index)
        prov[s] = SeriesProvenance(
            symbol=s, provider=rs.provider,
            origin=DataOrigin.CACHE if rs.cache_hit else origin,
            price_field=rs.price_field, adjusted_for=rs.adjusted_for, currency=ccy,
            unit=f"{ccy} per share/unit", frequency=rs.frequency,
            first_observation=str(x.index.min().date()), last_observation=str(x.index.max().date()),
            n_observations=int(len(x)), ingested_at_utc=rs.ingested_at_utc,
            cache_hit=rs.cache_hit, notes=tuple(notes),
        )

    staleness_check(cleaned, report, qs)
    if len(cleaned) < 2:
        raise DataUnavailableError(
            f"only {len(cleaned)} usable series after quality checks; excluded: "
            f"{report.excluded_symbols}")
    prices, desc, dropped = align_prices(cleaned)
    if len(prices) < qs.min_history:
        raise DataUnavailableError(
            f"aligned panel has {len(prices)} common dates < minimum {qs.min_history} "
            "(calendars/histories do not overlap enough)")
    if dropped:
        report.add(QualityIssue("*", "calendar_alignment", QualityFlag.OK, desc, dropped))
    vol_df = pd.DataFrame(volumes).reindex(prices.index) if volumes else None
    log.info("dataset loaded", extra={"symbols": list(prices.columns), "rows": len(prices),
                                      "origin": origin.value})
    return PriceDataset(prices=prices, raw_prices=cleaned, provenance=prov, quality=report,
                        meta={s: metas[s] for s in prices.columns}, origin=origin,
                        alignment=desc, volumes=vol_df)


def load_synthetic_dataset(seed: int = 42, n_days: int = 1260,
                           quality: QualitySettings | None = None) -> tuple[PriceDataset, object]:
    """Offline mode: deterministic synthetic dataset (clearly labelled SYNTHETIC)."""
    from core.market_data.synthetic import SyntheticProvider, SyntheticSpec

    prov = SyntheticProvider(SyntheticSpec(seed=seed, n_days=n_days))
    syms = list(prov.result.series.keys())
    ds = load_price_dataset(syms, prov, base_currency="BRL", meta=prov.result.meta,
                            quality=quality, origin=DataOrigin.SYNTHETIC)
    return ds, prov.result
