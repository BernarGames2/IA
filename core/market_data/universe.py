"""Static, documented asset metadata for demo/test symbols.

Metadata here comes from configuration (``metadata_source='static_config'``), not
from a provider, and only covers the example symbols named in the specification.
These are technical examples, not recommendations.
"""

from __future__ import annotations

from models.market_data import AssetMeta

STATIC_META: dict[str, AssetMeta] = {
    "PETR4.SA": AssetMeta("PETR4.SA", "equity", "energy", "BR", "BRL", "B3", "B3"),
    "VALE3.SA": AssetMeta("VALE3.SA", "equity", "materials", "BR", "BRL", "B3", "B3"),
    "ITUB4.SA": AssetMeta("ITUB4.SA", "equity", "financials", "BR", "BRL", "B3", "B3"),
    "BOVA11.SA": AssetMeta("BOVA11.SA", "equity", "broad_market", "BR", "BRL", "B3", "B3"),
    "IVVB11.SA": AssetMeta("IVVB11.SA", "equity", "broad_market", "US", "BRL", "B3", "B3"),
    "KNRI11.SA": AssetMeta("KNRI11.SA", "real_estate", "reit", "BR", "BRL", "B3", "B3"),
    "AAPL": AssetMeta("AAPL", "equity", "technology", "US", "USD", "NASDAQ", "NYSE"),
    "MSFT": AssetMeta("MSFT", "equity", "technology", "US", "USD", "NASDAQ", "NYSE"),
    "SPY": AssetMeta("SPY", "equity", "broad_market", "US", "USD", "NYSEARCA", "NYSE"),
    "AGG": AssetMeta("AGG", "fixed_income", "aggregate_bonds", "US", "USD", "NYSEARCA", "NYSE"),
    "GLD": AssetMeta("GLD", "commodity", "gold", "US", "USD", "NYSEARCA", "NYSE"),
    "BTC-USD": AssetMeta("BTC-USD", "crypto", "crypto", "GLOBAL", "USD", "CRYPTO", "24x7"),
}


def meta_for(symbol: str) -> AssetMeta:
    """Return static metadata or an explicit 'unknown' record (never guessed)."""
    if symbol in STATIC_META:
        return STATIC_META[symbol]
    ccy = "BRL" if symbol.endswith(".SA") else "unknown"
    return AssetMeta(symbol, currency=ccy, metadata_source="unknown")
