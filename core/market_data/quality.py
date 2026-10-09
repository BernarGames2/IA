"""Data-quality checks. Problems are reported; returns are never fabricated.

Checks per raw series: empty, unsorted index, duplicated timestamps, non-positive
or non-finite prices, missing-value fraction, frozen (repeated) prices, suspicious
jumps (possible unadjusted corporate events), weekend observations for exchange
calendars, staleness versus the rest of the universe, and minimum history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.market_data.base import RawSeries
from models.market_data import AssetMeta, QualityFlag, QualityIssue, QualityReport


@dataclass(frozen=True)
class QualitySettings:
    min_history: int = 252
    max_missing_fraction: float = 0.10
    frozen_price_run: int = 5
    suspicious_return_abs: float = 0.40
    stale_business_days: int = 5


def _longest_constant_run(x: pd.Series) -> int:
    v = x.dropna().to_numpy()
    if v.size < 2:
        return 0
    best = run = 1
    for a, b in zip(v[:-1], v[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def clean_series(s: RawSeries, meta: AssetMeta, report: QualityReport,
                 qs: QualitySettings) -> pd.Series | None:
    """Return a cleaned close series (or ``None`` if the symbol must be excluded).

    Cleaning only *removes* invalid observations (sets them to NaN / drops
    duplicates); it never interpolates or forward-fills prices.
    """
    sym = s.symbol
    x = s.close.copy()
    if x is None or x.dropna().empty:
        report.add(QualityIssue(sym, "empty", QualityFlag.FAIL, "no observations"))
        report.excluded_symbols[sym] = "empty series"
        return None
    if not x.index.is_monotonic_increasing:
        report.add(QualityIssue(sym, "unsorted_index", QualityFlag.WARNING, "index sorted"))
        x = x.sort_index()
    dup = x.index.duplicated(keep="last")
    if dup.any():
        report.add(QualityIssue(sym, "duplicate_timestamps", QualityFlag.WARNING,
                                "duplicated timestamps; last value kept", int(dup.sum())))
        x = x[~dup]
    x = x.astype(float)
    bad = ~np.isfinite(x.to_numpy()) | (x.to_numpy() <= 0)
    bad &= x.notna().to_numpy()
    if bad.any():
        report.add(QualityIssue(sym, "invalid_prices", QualityFlag.WARNING,
                                "non-positive or non-finite prices set to missing", int(bad.sum())))
        x[bad] = np.nan
    miss_frac = float(x.isna().mean())
    if miss_frac > 0:
        sev = QualityFlag.FAIL if miss_frac > qs.max_missing_fraction else QualityFlag.WARNING
        report.add(QualityIssue(sym, "missing_values", sev,
                                f"{miss_frac:.2%} missing (not filled)", int(x.isna().sum())))
        if sev == QualityFlag.FAIL:
            report.excluded_symbols[sym] = f"missing fraction {miss_frac:.2%}"
            return None
    x = x.dropna()
    run = _longest_constant_run(x)
    if run >= qs.frozen_price_run:
        report.add(QualityIssue(sym, "frozen_prices", QualityFlag.WARNING,
                                f"longest run of identical prices = {run}", run))
    r = x.pct_change(fill_method=None).dropna()
    jumps = r[np.abs(r) > qs.suspicious_return_abs]
    if len(jumps) > 0 and meta.asset_class != "crypto":
        report.add(QualityIssue(sym, "suspicious_returns", QualityFlag.WARNING,
                                f"|return| > {qs.suspicious_return_abs:.0%} on "
                                f"{', '.join(d.strftime('%Y-%m-%d') for d in jumps.index[:5])} "
                                "(possible unadjusted corporate event)", int(len(jumps))))
    if meta.calendar in ("B3", "NYSE", "business_days"):
        wk = int((x.index.dayofweek >= 5).sum())
        if wk:
            report.add(QualityIssue(sym, "weekend_observations", QualityFlag.WARNING,
                                    f"{wk} weekend observations for an exchange calendar", wk))
    if len(x) < qs.min_history:
        report.add(QualityIssue(sym, "insufficient_history", QualityFlag.FAIL,
                                f"{len(x)} observations < minimum {qs.min_history}", len(x)))
        report.excluded_symbols[sym] = f"history {len(x)} < {qs.min_history}"
        return None
    report.per_symbol_flag.setdefault(sym, QualityFlag.OK)
    return x


def staleness_check(cleaned: dict[str, pd.Series], report: QualityReport,
                    qs: QualitySettings) -> None:
    """Flag series whose last observation lags the universe (delisting/suspension?)."""
    if not cleaned:
        return
    last = max(s.index.max() for s in cleaned.values())
    for sym, s in cleaned.items():
        lag = len(pd.bdate_range(s.index.max(), last)) - 1
        if lag > qs.stale_business_days:
            report.add(QualityIssue(sym, "stale_series", QualityFlag.WARNING,
                                    f"last observation {s.index.max().date()} lags universe by "
                                    f"{lag} business days (suspension/delisting?)", lag))


def align_prices(cleaned: dict[str, pd.Series]) -> tuple[pd.DataFrame, str, int]:
    """Inner-join prices on common dates (no forward fill).

    Aligning *prices* (not returns) means a Friday->Monday return for a 24x7 asset
    correctly compounds the weekend. Returns (panel, description, dates_dropped).
    """
    panel = pd.concat(cleaned, axis=1, join="outer", sort=True).sort_index()
    total = len(panel)
    aligned = panel.dropna(how="any")
    desc = ("inner join of price dates across all symbols (no forward-fill); "
            f"{total - len(aligned)} of {total} dates dropped")
    return aligned, desc, total - len(aligned)
