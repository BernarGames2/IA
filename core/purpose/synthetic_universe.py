"""Offline demonstration universe for the purpose engine: individual SYNTHETIC tickers in BRL
and USD plus a synthetic USD/BRL FX series. USD assets go through the same FX conversion
as real data. Correlations come from an explicit 5-factor structure (documented below)."""

from __future__ import annotations

import numpy as np

from core.data_loader import load_price_dataset
from core.market_data.quality import QualitySettings
from core.market_data.synthetic import SyntheticAsset, SyntheticProvider, SyntheticSpec
from core.purpose.catalog import SYNTHETIC_CATALOG, CatalogAsset
from models.market_data import AssetMeta, DataOrigin, PriceDataset

FX = "USDBRL=X"
# factors: BR equity, global equity, BR rates (bond prices), USD/BRL, commodities
_LOAD = {
    "SINT_ENERGIA3": (0.55, 0.0, 0.25, 0.0, 0.0),
    "SINT_BANCO4": (0.75, 0.0, 0.20, 0.0, 0.0),
    "SINT_PETRO4": (0.60, 0.0, 0.0, 0.10, 0.40),
    "SINT_SAUDE3": (0.55, 0.0, 0.10, 0.0, 0.0),
    "SINT_EDUCA3": (0.60, 0.0, 0.20, 0.0, 0.0),
    "SINT_VAREJO3": (0.65, 0.0, 0.25, 0.0, 0.0),
    "SINT_LOGI11": (0.35, 0.0, 0.45, 0.0, 0.0),
    "SINT_IPCA": (0.15, 0.0, 0.80, 0.0, 0.0),
    "SINT_SELIC": (0.0, 0.0, 0.05, 0.0, 0.0),
    "SINT_TECH_US": (0.0, 0.80, 0.0, 0.0, 0.0),
    "SINT_SAUDE_US": (0.0, 0.60, 0.0, 0.0, 0.0),
    "SINT_CLEAN_US": (0.0, 0.65, 0.0, 0.0, 0.20),
    "SINT_TREAS_US": (0.0, -0.20, 0.0, 0.0, 0.0),
    "SINT_OURO_US": (0.0, 0.05, 0.0, 0.0, 0.40),
    FX: (-0.30, 0.0, 0.0, 0.85, 0.0),
}
# (annual continuous drift, calm volatility, regime sensitive) in the NATIVE currency
_PARAMS = {
    "SINT_ENERGIA3": (0.11, 0.24, True), "SINT_BANCO4": (0.12, 0.27, True), "SINT_PETRO4": (0.12, 0.33, True),
    "SINT_SAUDE3": (0.10, 0.26, True), "SINT_EDUCA3": (0.09, 0.38, True), "SINT_VAREJO3": (0.08, 0.40, True),
    "SINT_LOGI11": (0.09, 0.14, True), "SINT_IPCA": (0.105, 0.06, False), "SINT_SELIC": (0.10, 0.006, False),
    "SINT_TECH_US": (0.10, 0.25, True), "SINT_SAUDE_US": (0.07, 0.16, True), "SINT_CLEAN_US": (0.07, 0.30, True),
    "SINT_TREAS_US": (0.04, 0.06, False), "SINT_OURO_US": (0.05, 0.15, True), FX: (0.03, 0.15, False),
}


def asset_meta(a: CatalogAsset, synthetic: bool) -> AssetMeta:
    return AssetMeta(a.ticker, a.asset_class, a.sector, a.country, a.currency,
                     "SYNTHETIC" if synthetic else a.macro_class, "business_days" if synthetic else
                     ("24x7" if a.asset_class == "crypto" else ("B3" if a.ticker.endswith(".SA") else "NYSE")),
                     "synthetic_generator" if synthetic else "catalog_static")


def build_synthetic_universe(seed: int = 42, n_days: int = 1260,
                             catalog: tuple[CatalogAsset, ...] = SYNTHETIC_CATALOG
                             ) -> tuple[PriceDataset, object]:
    symbols = [a.ticker for a in catalog] + [FX]
    b = np.array([_LOAD[s] for s in symbols])
    corr = b @ b.T
    np.fill_diagonal(corr, 1.0)
    assets = []
    for s in symbols:
        mu, sig, sens = _PARAMS[s]
        if s == FX:
            meta = AssetMeta(FX, "fx", "cambio", "GLOBAL", "BRL", "SYNTHETIC", "business_days", "synthetic_generator")
            assets.append(SyntheticAsset(s, mu, sig, meta, adv=1e9, start_price=5.0, regime_sensitive=sens))
        else:
            a = next(x for x in catalog if x.ticker == s)
            assets.append(SyntheticAsset(s, mu, sig, asset_meta(a, True), regime_sensitive=sens))
    spec = SyntheticSpec(assets=tuple(assets), calm_corr=corr, n_days=n_days, seed=seed)
    prov = SyntheticProvider(spec)
    meta = {a.ticker: asset_meta(a, True) for a in catalog}
    ds = load_price_dataset([a.ticker for a in catalog], prov, base_currency="BRL", meta=meta,
                            quality=QualitySettings(min_history=252), origin=DataOrigin.SYNTHETIC)
    return ds, prov.result
