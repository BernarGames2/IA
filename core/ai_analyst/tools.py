"""Validated tools for the AI Research Analyst.

Each tool has a pydantic input schema, resource limits and is logged (call id,
inputs, output digest, duration, status). Tools only compute with the analysis
context (already-loaded data and validated engines); they cannot fetch URLs,
run shell commands, read environment variables or place orders.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.extreme_risk.stress import default_scenarios, run_scenarios
from core.extreme_risk.var_es import historical_var_es
from core.metrics import (
    UndefinedMetricError,
    annualized_arithmetic_mean,
    annualized_volatility,
    max_drawdown,
    risk_contributions,
    sharpe_ratio,
)
from core.optimization import optimizers as opt
from core.optimization.constraints import verify_weights
from core.simulation.engine import SimulationSettings, simulate_portfolio
from core.simulation.models import GBMModel


class ToolError(RuntimeError):
    """A tool call was rejected (invalid input, limit exceeded, unknown tool)."""


@dataclass
class AnalysisContext:
    """Read-only inputs the tools may use (built from an AnalysisResult)."""

    returns: pd.DataFrame
    log_returns: pd.DataFrame
    mu: pd.Series
    cov: pd.DataFrame
    constraints: object
    meta: dict
    rf: float
    synthetic: bool
    data_period: str
    portfolios: dict[str, pd.Series]
    validation_rows: list[dict[str, object]]
    cov_method: str
    mu_method: str
    capital: float

    @classmethod
    def from_result(cls, res) -> "AnalysisContext":  # noqa: ANN001
        ds = res.dataset
        return cls(returns=res.returns, log_returns=res.log_returns, mu=res.mu, cov=res.cov.cov,
                   constraints=res.constraints, meta=ds.meta, rf=float(res.config.risk_free_rate),
                   synthetic=ds.is_synthetic,
                   data_period=f"{ds.prices.index.min().date()} a {ds.prices.index.max().date()}",
                   portfolios={**{k: c.weights for k, c in res.candidates.items() if c.success},
                               "selected": res.selected.weights},
                   validation_rows=[c.as_dict() for c in res.validation.checks] if res.validation else [],
                   cov_method=res.cov.method, mu_method=res.config.expected_return_method.value,
                   capital=res.profile_input.capital)


# ----------------------------------------------------------------- schemas
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetMetricsIn(_Strict):
    symbols: list[str] = Field(..., min_length=1, max_length=50)


class PortfolioIn(_Strict):
    portfolio: str | None = None
    weights: dict[str, float] | None = None

    @field_validator("weights")
    @classmethod
    def _w(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            if any((not np.isfinite(x)) or x < 0 for x in v.values()):
                raise ValueError("weights must be finite and non-negative (no short selling)")
            if abs(sum(v.values()) - 1) > 1e-6:
                raise ValueError("weights must sum to 1")
        return v


class OptimizeIn(_Strict):
    method: Literal["min_variance", "max_sharpe", "risk_parity", "max_diversification", "hrp"]


class SimulateIn(PortfolioIn):
    n_paths: int = Field(5000, ge=100, le=50_000)
    horizon_days: int = Field(252, ge=1, le=2520)
    seed: int = 42


class CompareIn(_Strict):
    portfolios: list[str] = Field(..., min_length=2, max_length=10)


class EmptyIn(_Strict):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: type[BaseModel]
    fn: Callable[[AnalysisContext, BaseModel], dict[str, object]]


@dataclass
class ToolCallRecord:
    call_id: str
    tool: str
    inputs: dict[str, object]
    status: str
    started_at_utc: str
    duration_s: float
    output: dict[str, object] = field(default_factory=dict)
    error: str = ""

    @property
    def output_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.output, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _resolve_weights(ctx: AnalysisContext, inp: PortfolioIn) -> tuple[str, pd.Series]:
    if inp.weights is not None:
        unknown = set(inp.weights) - set(ctx.returns.columns)
        if unknown:
            raise ToolError(f"ativos desconhecidos no contexto: {sorted(unknown)}")
        w = pd.Series(0.0, index=ctx.returns.columns)
        for k, v in inp.weights.items():
            w[k] = v
        return "pesos informados", w
    name = inp.portfolio or "selected"
    if name not in ctx.portfolios:
        raise ToolError(f"carteira '{name}' não existe; disponíveis: {sorted(ctx.portfolios)}")
    return name, ctx.portfolios[name]


def _metric(fn: Callable[[], float]) -> float | None:
    try:
        return float(fn())
    except UndefinedMetricError:
        return None


def t_asset_metrics(ctx: AnalysisContext, inp: AssetMetricsIn) -> dict[str, object]:
    out = {}
    for s in inp.symbols:
        if s not in ctx.returns.columns:
            raise ToolError(f"ativo '{s}' não está no universo analisado")
        x = ctx.returns[s].to_numpy()
        out[s] = {"arith_mean_ann": annualized_arithmetic_mean(x), "vol_ann": annualized_volatility(x),
                  "sharpe": _metric(lambda: sharpe_ratio(x, ctx.rf)), "max_drawdown": max_drawdown(x),
                  "asset_class": ctx.meta[s].asset_class}
    return {"kind": "historical", "period": ctx.data_period, "synthetic": ctx.synthetic, "assets": out}


def t_portfolio_metrics(ctx: AnalysisContext, inp: PortfolioIn) -> dict[str, object]:
    name, w = _resolve_weights(ctx, inp)
    wv = w.to_numpy()
    rc = risk_contributions(wv, ctx.cov.to_numpy(), list(w.index))
    pr = ctx.returns.to_numpy() @ wv
    v95 = historical_var_es(pr, 0.95, "1 dia")
    v99 = historical_var_es(pr, 0.99, "1 dia")
    viol = verify_weights(wv, ctx.constraints)  # type: ignore[arg-type]
    return {"kind": "estimate", "portfolio": name, "weights": w.round(6).to_dict(),
            "expected_return_ann": float(wv @ ctx.mu.to_numpy()),
            "volatility_ann": float(np.sqrt(wv @ ctx.cov.to_numpy() @ wv)),
            "sharpe_est": float((wv @ ctx.mu.to_numpy() - ctx.rf) / np.sqrt(wv @ ctx.cov.to_numpy() @ wv)),
            "risk_contrib_pct": rc["rc_pct"].round(6).to_dict(),
            "hist_var95_1d": v95.var, "hist_es95_1d": v95.es, "hist_var99_1d": v99.var, "hist_es99_1d": v99.es,
            "constraint_violations": viol, "mu_method": ctx.mu_method, "cov_method": ctx.cov_method,
            "rf": ctx.rf, "period": ctx.data_period, "synthetic": ctx.synthetic}


def t_optimize(ctx: AnalysisContext, inp: OptimizeIn) -> dict[str, object]:
    pc = ctx.constraints
    m = inp.method
    if m == "max_sharpe":
        r = opt.max_sharpe(ctx.mu, ctx.cov, pc, ctx.rf)  # type: ignore[arg-type]
    elif m == "min_variance":
        r = opt.min_variance(ctx.cov, pc, ctx.mu, ctx.rf)  # type: ignore[arg-type]
    elif m == "risk_parity":
        r = opt.risk_parity(ctx.cov, pc, ctx.mu, ctx.rf)  # type: ignore[arg-type]
    elif m == "max_diversification":
        r = opt.max_diversification(ctx.cov, pc, ctx.mu, ctx.rf)  # type: ignore[arg-type]
    else:
        r = opt.hrp(ctx.cov, pc, ctx.mu, ctx.rf)  # type: ignore[arg-type]
    return {"kind": "estimate", **r.summary()}


def t_stress(ctx: AnalysisContext, inp: PortfolioIn) -> dict[str, object]:
    name, w = _resolve_weights(ctx, inp)
    t = run_scenarios(w, ctx.meta, ctx.capital, default_scenarios())
    return {"kind": "hypothetical_scenario", "portfolio": name,
            "scenarios": {k: {"portfolio_return": float(r["portfolio_return"]), "loss_money": float(r["loss_money"]),
                              "worst_contributor": r["worst_contributor"], "description": r["description"]}
                          for k, r in t.iterrows()}}


def t_simulate(ctx: AnalysisContext, inp: SimulateIn) -> dict[str, object]:
    name, w = _resolve_weights(ctx, PortfolioIn(portfolio=inp.portfolio, weights=inp.weights))
    gbm = GBMModel.from_log_returns(ctx.log_returns)
    res = simulate_portfolio(gbm, w.to_numpy(), SimulationSettings(
        n_paths=inp.n_paths, n_steps=inp.horizon_days, seed=inp.seed, initial_capital=ctx.capital))
    sm = res.summary
    return {"kind": "simulation", "model": "gbm", "portfolio": name, "n_paths": inp.n_paths,
            "horizon_days": inp.horizon_days, "seed": inp.seed,
            "terminal_p5": sm["terminal_wealth_percentiles"]["p5"],
            "terminal_p50": sm["terminal_wealth_percentiles"]["p50"],
            "terminal_p95": sm["terminal_wealth_percentiles"]["p95"],
            "p_negative_return": sm["p_negative_return"], "p_negative_return_mc_se": sm["p_negative_return_mc_se"],
            "es95_horizon": sm["var_es_horizon"]["levels"]["0.95"]["es"]}


def t_compare(ctx: AnalysisContext, inp: CompareIn) -> dict[str, object]:
    rows = {}
    for p in inp.portfolios:
        r = t_portfolio_metrics(ctx, PortfolioIn(portfolio=p))
        rows[p] = {k: r[k] for k in ("expected_return_ann", "volatility_ann", "sharpe_est", "hist_es95_1d")}
    return {"kind": "estimate", "comparison": rows}


def t_validation(ctx: AnalysisContext, inp: EmptyIn) -> dict[str, object]:
    st = [r["status"] for r in ctx.validation_rows]
    overall = "FAIL" if any(r["status"] == "FAIL" and r["critical"] for r in ctx.validation_rows) else \
        ("WARNING" if any(s != "PASS" for s in st) else "PASS")
    return {"kind": "validation", "overall": overall,
            "fails": [r["name"] for r in ctx.validation_rows if r["status"] == "FAIL"],
            "warnings": [r["name"] + ": " + str(r["justification"]) for r in ctx.validation_rows
                         if r["status"] == "WARNING"]}


REGISTRY: dict[str, ToolSpec] = {
    "asset_metrics": ToolSpec("asset_metrics", "Métricas históricas por ativo", AssetMetricsIn, t_asset_metrics),
    "portfolio_metrics": ToolSpec("portfolio_metrics", "Risco/retorno estimados de uma carteira", PortfolioIn,
                                  t_portfolio_metrics),
    "optimize": ToolSpec("optimize", "Otimização com as restrições do perfil", OptimizeIn, t_optimize),
    "stress": ToolSpec("stress", "Cenários de estresse hipotéticos", PortfolioIn, t_stress),
    "simulate": ToolSpec("simulate", "Monte Carlo GBM com limites de recursos", SimulateIn, t_simulate),
    "compare_portfolios": ToolSpec("compare_portfolios", "Compara carteiras candidatas", CompareIn, t_compare),
    "validation_status": ToolSpec("validation_status", "Status da validação independente", EmptyIn, t_validation),
}


class ToolRunner:
    """Executes registered tools with schema validation, limits and an audit log."""

    def __init__(self, ctx: AnalysisContext, max_calls: int = 25) -> None:
        self.ctx = ctx
        self.max_calls = max_calls
        self.log: list[ToolCallRecord] = []

    def call(self, tool: str, **kwargs: object) -> ToolCallRecord:
        if len(self.log) >= self.max_calls:
            raise ToolError(f"limite de {self.max_calls} chamadas de ferramenta atingido")
        if tool not in REGISTRY:
            raise ToolError(f"ferramenta '{tool}' não registrada (permitidas: {sorted(REGISTRY)})")
        spec = REGISTRY[tool]
        rec = ToolCallRecord(call_id=f"T{len(self.log) + 1}-{uuid.uuid4().hex[:6]}", tool=tool, inputs=dict(kwargs),
                             status="error", started_at_utc=datetime.now(timezone.utc).isoformat(), duration_s=0.0)
        t0 = time.perf_counter()
        try:
            inp = spec.schema(**kwargs)
            rec.output = spec.fn(self.ctx, inp)
            rec.status = "ok"
        except (ValidationError, ToolError, ValueError) as e:
            rec.error = str(e)
        rec.duration_s = round(time.perf_counter() - t0, 4)
        self.log.append(rec)
        return rec
