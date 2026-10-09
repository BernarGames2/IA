import json

import numpy as np
import pandas as pd
import pytest

from core.data_loader import convert_currency, load_price_dataset, load_synthetic_dataset
from core.market_data.base import DataUnavailableError, DownloadError, RawSeries
from core.market_data.cache import SeriesCache
from core.market_data.quality import QualitySettings, align_prices, clean_series
from core.market_data.synthetic import SyntheticSpec, generate_synthetic_market
from core.market_data.yfinance_provider import RateLimiter, YFinanceProvider, normalize_download
from models.market_data import AssetMeta, DataOrigin, QualityFlag, QualityReport

QS = QualitySettings(min_history=50)


def raw(sym, values, idx=None, ccy="BRL"):
    idx = idx if idx is not None else pd.bdate_range("2022-01-03", periods=len(values))
    return RawSeries(sym, pd.Series(values, index=idx, dtype=float), None, "test", "close", "none", ccy, "1d")


class FakeProvider:
    name = "fake"

    def __init__(self, series, failures=None):
        self.series = series
        self.failures = failures or {}

    def fetch(self, symbols, period, interval):
        return {s: self.series[s] for s in symbols if s in self.series}


def walk(n, seed):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))


def test_empty_series_excluded():
    rep = QualityReport()
    assert clean_series(raw("X", []), AssetMeta("X"), rep, QS) is None
    assert "X" in rep.excluded_symbols and rep.per_symbol_flag["X"] == QualityFlag.FAIL


def test_duplicates_invalid_prices_frozen_and_jumps_flagged():
    idx = pd.bdate_range("2022-01-03", periods=100)
    v = walk(100, 1)
    v[10] = -5.0                     # invalid
    v[20:27] = v[19]                 # frozen run of 8
    v[60:] *= 2.0                    # unadjusted split-like jump
    s = pd.Series(v, index=idx)
    s = pd.concat([s, s.iloc[[5]]]).sort_index()  # duplicate timestamp
    rs = RawSeries("X", s, None, "t", "close", "none", "BRL", "1d")
    rep = QualityReport()
    out = clean_series(rs, AssetMeta("X", calendar="B3"), rep, QS)
    checks = {i.check for i in rep.issues}
    assert {"duplicate_timestamps", "invalid_prices", "frozen_prices", "suspicious_returns"} <= checks
    assert out is not None and (out > 0).all() and out.index.is_unique
    assert len(out) == 99  # invalid price removed, NOT interpolated


def test_insufficient_history_and_missing_fraction_fail():
    rep = QualityReport()
    assert clean_series(raw("S", walk(30, 2)), AssetMeta("S"), rep, QS) is None
    v = walk(100, 3)
    v[::3] = np.nan
    rep2 = QualityReport()
    assert clean_series(raw("M", v), AssetMeta("M"), rep2, QS) is None
    assert "missing" in rep2.excluded_symbols["M"]


def test_different_calendars_align_on_prices_without_ffill():
    b3 = pd.bdate_range("2022-01-03", periods=120)
    crypto = pd.date_range("2022-01-01", periods=170, freq="D")
    a = pd.Series(walk(120, 4), index=b3)
    c = pd.Series(walk(170, 5), index=crypto)
    panel, desc, dropped = align_prices({"A": a, "C": c})
    assert dropped > 0 and "no forward-fill" in desc
    assert (panel.index.dayofweek < 5).all()
    # Monday return of crypto compounds the weekend (price-based alignment)
    mon = panel.index[panel.index.dayofweek == 0][1]
    prev = panel.index[panel.index.get_loc(mon) - 1]
    assert panel.loc[mon, "C"] / panel.loc[prev, "C"] == pytest.approx(c[mon] / c[prev])


def test_download_failure_not_replaced_by_synthetic():
    prov = FakeProvider({"A": raw("A", walk(100, 6))}, failures={"BAD": "invalid ticker"})
    with pytest.raises(DataUnavailableError):
        load_price_dataset(["A", "BAD"], prov, quality=QS)


def test_partial_failure_excluded_and_documented():
    prov = FakeProvider({"A": raw("A", walk(100, 6)), "B": raw("B", walk(100, 7))},
                        failures={"BAD": "invalid ticker"})
    ds = load_price_dataset(["A", "B", "BAD"], prov, quality=QS)
    assert ds.symbols == ["A", "B"]
    assert "BAD" in ds.quality.excluded_symbols and ds.origin == DataOrigin.REAL
    assert ds.provenance["A"].provider == "test"


def test_fx_conversion_and_missing_fx():
    idx = pd.bdate_range("2022-01-03", periods=100)
    usd = RawSeries("U", pd.Series(walk(100, 8), index=idx), None, "t", "close", "none", "USD", "1d")
    fx = RawSeries("USDBRL=X", pd.Series(np.full(100, 5.0), index=idx), None, "t", "close", "none", "BRL", "1d")
    prov = FakeProvider({"U": usd, "B": raw("B", walk(100, 9)), "USDBRL=X": fx})
    ds = load_price_dataset(["U", "B"], prov, quality=QS, meta={"U": AssetMeta("U", currency="USD"),
                                                                 "B": AssetMeta("B", currency="BRL")})
    np.testing.assert_allclose(ds.prices["U"].to_numpy(), usd.close.to_numpy() * 5.0)
    assert ds.provenance["U"].currency == "BRL"
    prov2 = FakeProvider({"U": usd, "B": raw("B", walk(100, 9)), "C": raw("C", walk(100, 10))})
    ds2 = load_price_dataset(["U", "B", "C"], prov2, quality=QS,
                             meta={"U": AssetMeta("U", currency="USD"), "B": AssetMeta("B", currency="BRL"),
                                   "C": AssetMeta("C", currency="BRL")})
    assert "U" in ds2.quality.excluded_symbols


def test_convert_currency_staleness_limit():
    close = pd.Series([10.0, 10.0, 10.0], index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-20"]))
    fx = pd.Series([5.0], index=pd.to_datetime(["2022-01-03"]))
    conv, n = convert_currency(close, fx, 3)
    assert conv.iloc[1] == 50.0 and np.isnan(conv.iloc[2]) and n == 1


def test_normalize_download_layouts():
    idx = pd.bdate_range("2022-01-03", periods=3)
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [10, 20, 30]}, index=idx)
    assert list(normalize_download(flat, ["X"])["X"].columns) == ["close", "volume"]
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["X", "Y"]])
    mi = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4), index=idx, columns=cols)
    out = normalize_download(mi, ["X", "Y", "Z"])
    assert set(out) == {"X", "Y"}
    cols2 = pd.MultiIndex.from_product([["X"], ["Close", "Volume"]])
    mi2 = pd.DataFrame(np.ones((3, 2)), index=idx, columns=cols2)
    assert "X" in normalize_download(mi2, ["X"])
    empty_close = pd.DataFrame({"Close": [np.nan] * 3}, index=idx)
    assert normalize_download(empty_close, ["X"]) == {}


def test_retries_with_backoff_then_failure():
    calls, sleeps = [], []

    def boom(tickers, period, interval, timeout):
        calls.append(tickers)
        raise ConnectionError("network down")

    p = YFinanceProvider(download_fn=boom, max_retries=2, min_interval_s=0, sleep=sleeps.append)
    res = p.fetch(["A"], "1y", "1d")
    assert res == {} and "A" in p.failures
    # a failed batch request is not hammered again symbol by symbol (network down)
    assert len(calls) == 3
    assert [s for s in sleeps if s >= 1] == [1.0, 2.0]


def test_invalid_ticker_vs_valid_in_batch(tmp_path):
    idx = pd.bdate_range("2022-01-03", periods=5)

    def dl(tickers, period, interval, timeout):
        cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
        df = pd.DataFrame(np.nan, index=idx, columns=cols)
        if "GOOD" in tickers:
            df[("Close", "GOOD")] = np.arange(1, 6, dtype=float)
            df[("Volume", "GOOD")] = 100.0
        return df

    cache = SeriesCache(tmp_path, ttl_hours=1)
    p = YFinanceProvider(cache=cache, download_fn=dl, min_interval_s=0, sleep=lambda s: None)
    res = p.fetch(["GOOD", "BAD"], "1y", "1d")
    assert set(res) == {"GOOD"} and "no observations" in p.failures["BAD"]
    p2 = YFinanceProvider(cache=cache, download_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
                          min_interval_s=0)
    again = p2.fetch(["GOOD"], "1y", "1d")
    assert again["GOOD"].cache_hit  # served from cache, no network call


def test_cache_expiry_and_tamper_detection(tmp_path):
    c = SeriesCache(tmp_path, ttl_hours=1)
    s = raw("A", [1.0, 2.0, 3.0])
    c.put(s, "1y", "1d")
    assert c.get("test", "A", "1y", "1d") is not None
    meta = next(tmp_path.glob("*.json"))
    d = json.loads(meta.read_text())
    d["stored_at_epoch"] -= 7200
    meta.write_text(json.dumps(d))
    assert c.get("test", "A", "1y", "1d") is None  # expired
    c.put(s, "1y", "1d")
    csv = next(tmp_path.glob("*.csv"))
    csv.write_text(csv.read_text().replace("2.0", "9.0"))
    assert c.get("test", "A", "1y", "1d") is None  # tampered


def test_rate_limiter_waits():
    t = [0.0]
    slept = []
    rl = RateLimiter(1.0, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    rl.wait()
    t[0] += 0.3
    rl.wait()
    assert slept == [pytest.approx(0.7)]


def test_synthetic_deterministic_and_labelled():
    a = generate_synthetic_market(SyntheticSpec(seed=7, n_days=300))
    b = generate_synthetic_market(SyntheticSpec(seed=7, n_days=300))
    c = generate_synthetic_market(SyntheticSpec(seed=8, n_days=300))
    s = "SINT_ACOES_BR"
    pd.testing.assert_series_equal(a.series[s].close, b.series[s].close)
    assert not a.series[s].close.equals(c.series[s].close)
    assert all(k.startswith("SINT_") for k in a.series)
    ds, _ = load_synthetic_dataset(seed=7, n_days=300, quality=QualitySettings(min_history=252))
    assert ds.is_synthetic and ds.origin == DataOrigin.SYNTHETIC
    assert all(p.origin == DataOrigin.SYNTHETIC for p in ds.provenance.values())
    assert len(ds.content_hash()) == 64
