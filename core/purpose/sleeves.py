"""Step 3 — currency sleeves: metrics of each sleeve in its OWN currency with its OWN rf.

A sleeve groups the eligible assets by native currency. Expected return, volatility
and Sharpe are computed on native-currency returns with the risk-free rate of that
currency (never a single rf, e.g. Selic, for USD assets). The aggregate portfolio is
then expressed in the objective currency (FX-converted returns) with that currency's rf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.metrics import rf_per_period
from models.market_data import PriceDataset


@dataclass
class Sleeve:
    currency: str
    assets: list[str]
    weight_in_portfolio: float
    rf: float
    rf_source: str
    expected_return_native: float | None
    volatility_native: float | None
    sharpe_native: float | None
    return_in_base_hist: float | None


def currency_sleeves(ds: PriceDataset, weights: pd.Series, rf_by_ccy: dict[str, float],
                     rf_source: dict[str, str], base_ccy: str, periods: int = 252) -> list[Sleeve]:
    out: list[Sleeve] = []
    ccys = sorted({ds.native_currency.get(s, base_ccy) for s in weights.index})
    for ccy in ccys:
        members = [s for s in weights.index if ds.native_currency.get(s, base_ccy) == ccy]
        wv = weights.loc[members]
        tot = float(wv.sum())
        if tot <= 1e-12:
            out.append(Sleeve(ccy, members, 0.0, rf_by_ccy[ccy], rf_source[ccy], None, None, None, None))
            continue
        wn = wv / tot
        native = pd.DataFrame({s: ds.native_prices[s] for s in members}).pct_change(fill_method=None).dropna()
        base = ds.prices[members].pct_change(fill_method=None).dropna()
        rn = native.to_numpy() @ wn.to_numpy()
        rb = base.to_numpy() @ wn.to_numpy()
        er = float(rn.mean() * periods)
        vol = float(rn.std(ddof=1) * np.sqrt(periods))
        ex = rn - rf_per_period(rf_by_ccy[ccy], periods)
        sh = float(ex.mean() / ex.std(ddof=1) * np.sqrt(periods)) if ex.std(ddof=1) > 0 else None
        out.append(Sleeve(ccy, members, tot, rf_by_ccy[ccy], rf_source[ccy], er, vol, sh,
                          float(rb.mean() * periods)))
    return out


def native_asset_stats(ds: PriceDataset, rf_by_ccy: dict[str, float], base_ccy: str,
                       periods: int = 252) -> pd.DataFrame:
    """Per-asset statistics in the asset's NATIVE currency, with that currency's rf (step 3,
    computed BEFORE the optimisation). Historical [H] sample estimates, not forecasts."""
    rows = []
    for s in ds.symbols:
        ccy = ds.native_currency.get(s, base_ccy)
        r = ds.native_prices.get(s, ds.prices[s]).pct_change(fill_method=None).dropna().to_numpy()
        ex = r - rf_per_period(rf_by_ccy[ccy], periods)
        sd = ex.std(ddof=1)
        rows.append({"ticker": s, "moeda": ccy, "rf_moeda": rf_by_ccy[ccy],
                     "retorno_medio_anual_nativo": float(r.mean() * periods),
                     "volatilidade_nativa": float(r.std(ddof=1) * np.sqrt(periods)),
                     "sharpe_nativo": float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else None,
                     "n_obs": len(r)})
    return pd.DataFrame(rows).set_index("ticker")
