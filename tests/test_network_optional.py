"""Opt-in online test (real Yahoo Finance data). Skipped unless QPI_NETWORK_TESTS=1.

It was NOT executed in the development environment, whose network policy
blocks Yahoo Finance (HTTP 403 from the proxy).
"""

import os

import pytest

pytestmark = pytest.mark.network


@pytest.mark.skipif(os.environ.get("QPI_NETWORK_TESTS") != "1", reason="online test disabled (QPI_NETWORK_TESTS!=1)")
def test_yfinance_download_real(tmp_path):
    from core.data_loader import load_price_dataset
    from core.market_data.cache import SeriesCache
    from core.market_data.quality import QualitySettings
    from core.market_data.yfinance_provider import YFinanceProvider
    from models.market_data import DataOrigin

    prov = YFinanceProvider(cache=SeriesCache(tmp_path), min_interval_s=1.0)
    ds = load_price_dataset(["SPY", "AGG", "GLD"], prov, period="2y", base_currency="USD",
                            quality=QualitySettings(min_history=252))
    assert ds.origin == DataOrigin.REAL and not ds.is_synthetic
    assert all(p.provider == "yfinance" for p in ds.provenance.values())
