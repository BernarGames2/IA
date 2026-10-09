"""Compact JSON summary of an :class:`AnalysisResult` for the web page.

Only already-computed values are exposed (nothing is recomputed here), so the
page shows exactly what the engines and the independent validator produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import APP_VERSION
from core.profiler import DISCLAIMER
from utils.serialization import to_jsonable


def _downsample_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _mc(mc, periods: int) -> dict[str, object] | None:  # noqa: ANN001
    if mc is None:
        return None
    pct = mc.percentiles
    idx = _downsample_idx(len(pct), 150)
    term = mc.terminal_wealth
    lo, hi = np.percentile(term, [0.5, 99.5])
    counts, edges = np.histogram(np.clip(term, lo, hi), bins=40)
    conv = mc.convergence.reset_index().to_dict(orient="records") if mc.convergence is not None else []
    return {
        "model": mc.model.get("model"),
        "summary": mc.summary,
        "fan": {"years": (mc.time_index[idx] / periods).tolist(),
                **{c: pct[c].to_numpy()[idx].tolist() for c in pct.columns}},
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist(), "clipped_pct": [0.5, 99.5]},
        "convergence": conv,
        "settings": {k: mc.settings[k] for k in ("n_paths", "n_steps", "seed", "rebalance_every",
                                                  "rebalance_cost_bps", "inflation_annual")},
    }


def build_summary(res, mode: str, elapsed_s: float) -> dict[str, object]:  # noqa: ANN001, C901
    ds = res.dataset
    S = res.sections
    cfg = res.config
    P = cfg.trading_days
    sel = res.selected
    rc = res.risk_contrib
    cov = res.cov.cov

    weights = [{"symbol": s, "asset_class": ds.meta[s].asset_class, "weight": float(sel.weights[s]),
                "rc_pct": float(rc.loc[s, "rc_pct"])} for s in sel.weights.index]
    candidates = [{"name": k, "success": c.success, "feasible": c.feasible,
                   "expected_return": c.expected_return, "volatility": c.volatility, "sharpe": c.sharpe,
                   "notes": (c.violations + c.warnings)[:2]} for k, c in res.candidates.items()]
    frontier = [{"vol": float(r["volatility"]), "ret": float(r["expected_return"]), "sharpe": r["sharpe"]}
                for _, r in res.frontier.iterrows()] if not res.frontier.empty else []
    marks = {}
    for key, label in (("min_variance", "Mínima variância"), ("max_sharpe", "Máximo Sharpe"),
                       (res.selected_name, "Selecionada")):
        c = res.candidates.get(key)
        if c is None or not c.success or not np.isfinite(c.volatility):
            continue
        same = next((k for k, m in marks.items() if m["name"] == key), None)
        if same is not None:   # selected portfolio is one of the marked ones: one marker, joint label
            marks[f"{same} (selecionada)"] = marks.pop(same)
            continue
        marks[label] = {"vol": c.volatility, "ret": c.expected_return, "name": key}
    a = res.asset_stats
    assets = [{"symbol": s, "asset_class": ds.meta[s].asset_class, "mu_used": float(res.mu[s]),
               "vol_est": float(np.sqrt(cov.loc[s, s])), "arith_mean_hist": float(a.loc[s, "arith_mean_ann"]),
               "cagr_hist": float(a.loc[s, "cagr"]), "vol_hist": float(a.loc[s, "vol_ann"]),
               "max_drawdown_hist": float(a.loc[s, "max_drawdown"])} for s in res.mu.index]
    corr = res.returns.corr()
    var_rows = [{"method": r["method"], "confidence": r["confidence"], "horizon": r["horizon"], "var": r["var"],
                 "es": r["es"], "n_obs": r["n_obs"], "warnings": list(r["warnings"])}
                for _, r in res.var_table.iterrows()]
    stress = []
    st = S.get("stress")
    if isinstance(st, pd.DataFrame):
        for name, r in st.iterrows():
            stress.append({"scenario": name, "description": r["description"], "portfolio_return": r["portfolio_return"],
                           "loss_money": r["loss_money"], "recovery_required": r["recovery_required"],
                           "worst_contributor": r["worst_contributor"], "baseline_return": r.get("baseline_return")})
    rev = S.get("reverse_stress") or {}
    reverse = None
    if rev:
        g = rev["gaussian"]
        reverse = {"loss_limit": g["loss_limit"], "horizon_days": rev["horizon_days"],
                   "mahalanobis_distance": g["mahalanobis_distance"], "prob_normal": g["prob_normal"],
                   "prob_student_t": g.get("prob_student_t"), "shock": g["shock"],
                   "empirical": rev["empirical"],
                   "scenario_multipliers": rev["scenario_multipliers"].reset_index().to_dict(orient="records")}
    evt = []
    for alpha, e in (S.get("evt") or {}).items():
        evt.append({"alpha": alpha, "refused": e.refused, "reason": e.refusal_reason, "var": e.var, "es": e.es,
                    "ci_var": e.ci_var, "empirical_var": e.empirical_var, "empirical_es": e.empirical_es,
                    "xi": e.fit.xi if e.fit else None, "n_exceed": e.fit.n_exceed if e.fit else None,
                    "warnings": e.warnings})
    v = res.validation
    validation = {"overall": v.overall.value if v else None, "validated": bool(v.validated) if v else False,
                  "checks": [{k: c.as_dict()[k] for k in ("name", "status", "critical", "justification")}
                             for c in v.checks] if v else []}
    backtest = None
    bt = res.backtest
    if bt is not None:
        cols = ["cagr", "volatility_ann", "sharpe", "max_drawdown", "turnover_annual", "cost_drag_annual"]
        eq = pd.DataFrame({k: (1 + x.net_returns).cumprod() for k, x in bt.strategies.items()})
        keep = list(dict.fromkeys([c for c in ("equal_weight", "min_variance", "risk_parity", "max_sharpe",
                                                "benchmark", res.selected_name) if c in eq.columns]))
        idx = _downsample_idx(len(eq), 200)
        backtest = {
            "validation": bt.metrics_validation[cols].reset_index(names="strategy").to_dict(orient="records"),
            "test": bt.metrics_test[cols].reset_index(names="strategy").to_dict(orient="records"),
            "equity": {"dates": [d.strftime("%Y-%m-%d") for d in eq.index[idx]],
                       **{k: eq[k].to_numpy()[idx].tolist() for k in keep}},
            "validation_end": str(bt.validation_end.date()), "oos_start": str(bt.oos_start.date()),
            "bias_warnings": bt.bias_warnings,
        }
    prof = res.profile
    es = S.get("estimator_selection", {})
    out = {
        "meta": {"run_id": res.run_id, "app_version": APP_VERSION, "mode": mode, "elapsed_s": round(elapsed_s, 2),
                 "synthetic": ds.is_synthetic, "data_start": str(ds.prices.index.min().date()),
                 "data_end": str(ds.prices.index.max().date()), "n_obs": len(ds.prices),
                 "symbols": ds.symbols, "seed": cfg.seed, "risk_free_rate": cfg.risk_free_rate,
                 "risk_free_is_fallback": cfg.rf_is_fallback, "inflation_assumed": cfg.inflation_annual,
                 "mu_method": cfg.expected_return_method.value, "cov_method": res.cov.method,
                 "data_hash": ds.content_hash()[:16], "timings_s": res.timings,
                 "disclaimer": DISCLAIMER},
        "warnings": res.warnings,
        "profile": {"final": prof.final_profile.value, "suggested_by_score": prof.score_band_suggestion.value,
                    "conflicts": prof.conflicts, "caps_applied": prof.caps_applied,
                    "missing_information": prof.missing_information,
                    "limits": prof.research_limits.model_dump(),
                    "capital": res.profile_input.capital,
                    "monthly_contribution": res.profile_input.monthly_contribution,
                    "horizon_years": res.profile_input.horizon_years},
        "feasibility": res.feasibility,
        "selection": {"selected": res.selected_name, "rule": res.selection.get("rule"),
                      "excluded": res.selection.get("excluded_reference_strategies", []),
                      "excluded_by_volatility_band": res.selection.get("excluded_by_volatility_band", [])},
        "selected": {"name": res.selected_name, "expected_return": sel.expected_return,
                     "volatility": sel.volatility, "sharpe": sel.sharpe, "weights": weights},
        "candidates": candidates,
        "frontier": {"points": frontier, "marks": marks},
        "assets": assets,
        "correlation": {"symbols": list(corr.columns), "matrix": corr.to_numpy().tolist()},
        "estimators": {"cov_used": res.cov.method, "cov_selection": es.get("cov_selection"),
                       "mu_used": cfg.expected_return_method.value, "mu_selection": es.get("mu_selection")},
        "var_es": var_rows,
        "monte_carlo": {"main": _mc(res.mc_main, P), "profile": _mc(res.mc_profile, P)},
        "stress": stress,
        "reverse_stress": reverse,
        "evt": evt,
        "concentration": S.get("concentration"),
        "validation": validation,
        "backtest": backtest,
    }
    return to_jsonable(out)  # type: ignore[return-value]
