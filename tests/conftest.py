import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synth():
    from core.data_loader import load_synthetic_dataset

    ds, truth = load_synthetic_dataset(seed=42)
    return ds, truth


@pytest.fixture(scope="session")
def rets(synth):
    from core.metrics import simple_returns

    return simple_returns(synth[0].prices)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def make_prices(n=300, n_assets=3, seed=0, start="2020-01-01", vol=0.01):
    r = np.random.default_rng(seed).normal(0.0003, vol, size=(n, n_assets))
    p = 100 * np.exp(np.cumsum(r, axis=0))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(p, index=idx, columns=[f"A{i}" for i in range(n_assets)])
