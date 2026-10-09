"""Writes the run's Markdown report, CSV tables, JSON results/manifest and PNG charts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from config import APP_VERSION
from core.reporting import charts as ch
from core.reporting.report_md import build_report


def to_jsonable(o: object) -> object:  # noqa: C901
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="split", date_format="iso", default_handler=str))
    if isinstance(o, pd.Series):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [to_jsonable(x) for x in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(asdict(o))
    if hasattr(o, "model_dump"):
        return to_jsonable(o.model_dump())  # type: ignore[attr-defined]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _versions() -> dict[str, str]:
    import matplotlib
    import pydantic
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "pydantic": pydantic.__version__,
            "platform": platform.platform()}


def write_outputs(res) -> dict[str, str]:  # noqa: ANN001
    cfg = res.config
    rid = res.run_id
    ds = res.dataset
    src = ("gerador sintético determinístico (seed %d)" % cfg.seed) if ds.is_synthetic else \
        "Yahoo Finance via yfinance (preços ajustados)"
    charts: dict[str, str] = {}
    tables: dict[str, str] = {}
    cdir, tdir, rdir = cfg.charts_dir, cfg.tables_dir, cfg.reports_dir
    syn = ds.is_synthetic

    def rel(p: Path) -> str:
        return str(Path("..") / p.parent.name / p.name)

    if cfg.make_charts:
        astats = pd.DataFrame({"volatility": np.sqrt(np.diag(res.cov.cov.to_numpy())), "expected_return": res.mu},
                              index=res.mu.index)
        pts = {}
        for k, lab in (("min_variance", "Mínima variância"), ("max_sharpe", "Máximo Sharpe"),
                       ("equal_weight", "Pesos iguais (viável)"), (res.selected_name, f"Selecionada ({res.selected_name})")):
            c = res.candidates.get(k)
            if c is not None and c.success and np.isfinite(c.volatility):
                pts[lab] = (c.volatility, c.expected_return)
        charts["frontier"] = rel(ch.efficient_frontier_chart(res.frontier, pts, astats, cdir / f"{rid}_frontier.png",
                                                             src + "; estimativas " + res.cov.method, syn))
        mc = res.mc_profile or res.mc_main
        if mc is not None:
            charts["mc_fan"] = rel(ch.mc_fan_chart(mc.percentiles, mc.representative_paths, mc.time_index,
                                                   cdir / f"{rid}_mc_fan.png", src + f"; modelo {mc.model['model']}", syn))
            charts["mc_hist"] = rel(ch.mc_terminal_hist(mc.terminal_wealth, mc.settings["initial_capital"],
                                                        mc.summary["net_invested"], cdir / f"{rid}_mc_terminal.png",
                                                        src + f"; {mc.summary['n_paths']} trajetórias", syn))
        corr = res.returns.corr()
        charts["correlation"] = rel(ch.correlation_heatmap(corr, cdir / f"{rid}_correlation.png", src, syn))
        wnames = [k for k in ("equal_weight", "min_variance", "risk_parity", "max_sharpe", res.selected_name)
                  if k in res.candidates and res.candidates[k].success]
        wnames = list(dict.fromkeys(wnames))
        wdf = pd.DataFrame({k: res.candidates[k].weights for k in wnames})
        cls = pd.Series({s: ds.meta[s].asset_class for s in wdf.index})
        bycls = wdf.groupby(cls).sum()
        charts["weights"] = rel(ch.weights_chart(wdf, bycls, cdir / f"{rid}_weights.png", src, syn))
        st = res.sections.get("stress")
        if isinstance(st, pd.DataFrame):
            charts["stress"] = rel(ch.stress_chart(st, cdir / f"{rid}_stress.png", src + "; choques hipotéticos", syn))
        if res.backtest is not None:
            eq = pd.DataFrame({k: (1 + v.net_returns).cumprod() for k, v in res.backtest.strategies.items()})
            dd = eq / eq.cummax() - 1
            keep = [c for c in ("equal_weight", "min_variance", "risk_parity", "max_sharpe", "robust_mean_variance",
                                "benchmark", res.selected_name) if c in eq.columns]
            keep = list(dict.fromkeys(keep))
            charts["backtest"] = rel(ch.backtest_chart(eq[keep], dd[keep], cdir / f"{rid}_backtest.png",
                                                       src + "; líquido de custos", syn))

    def save(name: str, df: pd.DataFrame, index: bool = True) -> None:
        p = tdir / f"{rid}_{name}.csv"
        df.to_csv(p, index=index)
        tables[name] = str(p)

    save("asset_stats", res.asset_stats)
    save("frontier", res.frontier, index=False)
    save("candidates", pd.DataFrame({k: c.summary() for k, c in res.candidates.items()}).T.drop(columns=["weights"]))
    save("candidate_weights", pd.DataFrame({k: c.weights for k, c in res.candidates.items()}))
    save("risk_contributions", res.risk_contrib)
    save("var_es", res.var_table.drop(columns=["params"]), index=False)
    save("data_quality", ds.quality.to_frame(), index=False)
    for key in ("stress", "historical_stress", "parametric_stress", "liquidity", "mc_model_comparison", "asset_roles"):
        v = res.sections.get(key)
        if isinstance(v, pd.DataFrame):
            save(key, v.drop(columns=[c for c in ("contributions", "shocks", "assumptions") if c in v.columns]))
    if res.mc_main is not None:
        save("mc_percentiles", res.mc_main.percentiles)
        save("mc_convergence", res.mc_main.convergence)
    if res.mc_profile is not None:
        save("mc_profile_percentiles", res.mc_profile.percentiles)
    if res.backtest is not None:
        save("backtest_validation", res.backtest.metrics_validation)
        save("backtest_test", res.backtest.metrics_test)
        save("backtest_returns", pd.DataFrame({k: v.net_returns for k, v in res.backtest.strategies.items()}))
    if res.validation is not None:
        save("validation", res.validation.to_frame(), index=False)

    manifest = {
        "run_id": rid, "app_version": APP_VERSION, "git_commit": _git_commit(), "started_at_utc": res.started_at_utc,
        "finished_at_utc": res.finished_at_utc, "timings_s": res.timings, "versions": _versions(),
        "config": json.loads(cfg.model_dump_json()), "seed": cfg.seed,
        "data": {"origin": ds.origin.value, "synthetic": ds.is_synthetic, "sha256_prices": ds.content_hash(),
                 "first_date": str(ds.prices.index.min().date()), "last_date": str(ds.prices.index.max().date()),
                 "n_obs": len(ds.prices), "symbols": ds.symbols,
                 "provenance": {s: to_jsonable(p) for s, p in ds.provenance.items()},
                 "excluded": ds.quality.excluded_symbols},
        "profile": to_jsonable(res.profile),
        "selected_portfolio": res.selected_name,
        "validation_overall": res.validation.overall.value if res.validation else None,
        "validated": res.validation.validated if res.validation else False,
        "warnings": res.warnings,
        "outputs": {"charts": charts, "tables": tables},
    }
    results = {
        "selection": res.selection,
        "candidates": {k: c.summary() for k, c in res.candidates.items()},
        "risk_contributions": res.risk_contrib,
        "var_es": res.var_table,
        "mc_main": res.mc_main.summary if res.mc_main else None,
        "mc_profile": res.mc_profile.summary if res.mc_profile else None,
        "mc_model": res.mc_main.model if res.mc_main else None,
        "stress": res.sections.get("stress"),
        "reverse_stress": res.sections.get("reverse_stress"),
        "evt": {str(k): {kk: vv for kk, vv in to_jsonable(v).items() if kk != "fit"} | {"fit": to_jsonable(v.fit)}
                for k, v in res.sections.get("evt", {}).items()},
        "validation": [c.as_dict() for c in res.validation.checks] if res.validation else [],
        "concentration": res.sections.get("concentration"),
    }
    jdir = rdir
    (jdir / f"{rid}_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False), "utf-8")
    (jdir / f"{rid}_results.json").write_text(json.dumps(to_jsonable(results), indent=2, ensure_ascii=False), "utf-8")
    md = build_report(res, charts, tables)
    md_path = rdir / f"{rid}_report.md"
    md_path.write_text(md, "utf-8")
    return {"report": str(md_path), "manifest": str(jdir / f"{rid}_manifest.json"),
            "results": str(jdir / f"{rid}_results.json"), **{f"chart:{k}": str(cdir / Path(v).name) for k, v in charts.items()},
            **{f"table:{k}": v for k, v in tables.items()}}
