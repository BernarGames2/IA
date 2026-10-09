"""Profiling and scalability benchmark (offline, synthetic data).

Measures: per-stage wall time of a full run (cProfile top functions), Monte
Carlo time and peak memory vs number of paths and batch size, and
reproducibility (identical results for identical seed). Writes CSVs to
outputs/tables/ and prints a summary. Usage:

    python scripts/profile_run.py [--quick]
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AppConfig  # noqa: E402
from core.data_loader import load_synthetic_dataset  # noqa: E402
from core.metrics import log_returns  # noqa: E402
from core.portfolio_engine import run_analysis  # noqa: E402
from core.profiler import demo_profile  # noqa: E402
from core.simulation.engine import SimulationSettings, simulate_portfolio  # noqa: E402
from core.simulation.models import GBMModel  # noqa: E402


def mc_scaling(quick: bool) -> pd.DataFrame:
    ds, _ = load_synthetic_dataset()
    gbm = GBMModel.from_log_returns(log_returns(ds.prices))
    w = np.full(len(ds.symbols), 1 / len(ds.symbols))
    rows = []
    grid = [(5000, 5000), (20000, 5000), (20000, 1000)] if quick else \
        [(5000, 5000), (20000, 5000), (20000, 1000), (20000, 20000), (50000, 5000)]
    for n, b in grid:
        tracemalloc.start()
        t0 = time.perf_counter()
        r = simulate_portfolio(gbm, w, SimulationSettings(n_paths=n, n_steps=252, batch_size=b, seed=1))
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"n_paths": n, "batch_size": b, "seconds": round(dt, 2), "peak_mb": round(peak / 2 ** 20, 1),
                     "median_terminal": r.summary["terminal_wealth_percentiles"]["p50"],
                     "mc_se_mean": r.summary["terminal_wealth_mean_mc_se"]})
    return pd.DataFrame(rows)


def reproducibility(tmp: Path) -> dict[str, object]:
    digests = []
    for i in range(2):
        cfg = AppConfig(offline=True, run_heavy_analyses=False, n_simulations=2000, make_charts=False,
                        output_dir=tmp / f"r{i}")
        res = run_analysis(cfg, demo_profile())
        payload = json.dumps({"w": res.selected.weights.round(12).to_dict(),
                              "mc": res.mc_main.summary["terminal_wealth_percentiles"]}, sort_keys=True)
        digests.append(hashlib.sha256(payload.encode()).hexdigest())
    return {"identical": digests[0] == digests[1], "digest": digests[0][:16]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(offline=True, make_charts=False, output_dir=ROOT / "outputs" / "profiling",
                    n_simulations=5000 if a.quick else 20000, run_heavy_analyses=not a.quick)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    res = run_analysis(cfg, demo_profile())
    pr.disable()
    total = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    stages = pd.DataFrame({"seconds": res.timings}).sort_values("seconds", ascending=False)
    stages.to_csv(out / "profile_stages.csv")
    (out / "profile_cprofile.txt").write_text(s.getvalue(), encoding="utf-8")
    print(f"Execução completa: {total:.1f}s")
    print(stages.to_string())
    mc = mc_scaling(a.quick)
    mc.to_csv(out / "benchmark_mc_scaling.csv", index=False)
    print(mc.to_string(index=False))
    rep = reproducibility(ROOT / "outputs" / "profiling")
    print("Reprodutibilidade:", rep)


if __name__ == "__main__":
    main()
