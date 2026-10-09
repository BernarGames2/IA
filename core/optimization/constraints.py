"""Constraint construction, feasibility detection and nearest-feasible diagnosis.

Feasibility of the linear constraint set (budget, bounds, groups, turnover) is
checked with an LP *before* any optimisation. If infeasible, an elastic LP finds
the allocation that minimises total constraint violation while keeping the
budget and the no-short-selling rule hard, and reports which constraints conflict
and by how much. User preferences are never relaxed silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from models.investor import ResearchLimits
from models.market_data import AssetMeta
from models.portfolio import GroupConstraint, PortfolioConstraints
from utils.numerical import BOUND_TOL, WEIGHT_SUM_TOL


class InfeasibleConstraintsError(ValueError):
    """The constraint set admits no portfolio."""


@dataclass
class FeasibilityReport:
    feasible: bool
    conflicts: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    nearest_weights: np.ndarray | None = None
    violations: list[str] = field(default_factory=list)


def build_constraints(symbols: list[str], meta: Mapping[str, AssetMeta],
                      limits: ResearchLimits | None = None, *, max_weight: float | None = None,
                      min_weight: float = 0.0, excluded: list[str] | None = None,
                      excluded_classes: list[str] | None = None) -> PortfolioConstraints:
    """Translate research limits + config into linear constraints (long-only)."""
    n = len(symbols)
    ub_val = min(x for x in [1.0, max_weight, limits.max_weight_per_asset if limits else None]
                 if x is not None)
    lb = np.full(n, float(min_weight))
    ub = np.full(n, float(ub_val))
    excl = set(excluded or [])
    excl_cls = set(excluded_classes or [])
    for i, s in enumerate(symbols):
        if s in excl or meta[s].asset_class in excl_cls:
            lb[i] = 0.0
            ub[i] = 0.0
    groups: list[GroupConstraint] = []
    if limits is not None:
        classes = sorted({meta[s].asset_class for s in symbols})
        for c in classes:
            members = tuple(s for s in symbols if meta[s].asset_class == c)
            hi = limits.max_class_weight.get(c, 1.0)
            lo = limits.min_class_weight.get(c, 0.0)
            if hi < 1.0 or lo > 0.0:
                groups.append(GroupConstraint(f"class:{c}", members, lo, hi))
        for c, lo in limits.min_class_weight.items():
            if c not in classes and lo > 0:
                groups.append(GroupConstraint(f"class:{c}", (), lo, 1.0))
    return PortfolioConstraints(list(symbols), lb, ub, groups)


def _lp_matrices(pc: PortfolioConstraints) -> tuple[np.ndarray | None, np.ndarray | None]:
    a, lo, hi, _ = pc.group_matrix()
    if a.shape[0] == 0:
        return None, None
    return np.vstack([a, -a]), np.concatenate([hi, -lo])


def verify_weights(w: np.ndarray, pc: PortfolioConstraints, tol: float = BOUND_TOL) -> list[str]:
    """Independent check of every linear constraint. Returns human-readable violations."""
    w = np.asarray(w, float)
    v: list[str] = []
    if not np.all(np.isfinite(w)):
        return ["pesos não finitos"]
    if abs(w.sum() - 1.0) > WEIGHT_SUM_TOL:
        v.append(f"soma dos pesos = {w.sum():.6f} != 1")
    for i, s in enumerate(pc.symbols):
        if w[i] < pc.lower[i] - tol:
            v.append(f"{s}: peso {w[i]:.4f} < mínimo {pc.lower[i]:.4f}")
        if w[i] > pc.upper[i] + tol:
            v.append(f"{s}: peso {w[i]:.4f} > máximo {pc.upper[i]:.4f}")
    a, lo, hi, names = pc.group_matrix()
    for k in range(a.shape[0]):
        g = float(a[k] @ w)
        if g < lo[k] - tol:
            v.append(f"{names[k]}: exposição {g:.4f} < mínimo {lo[k]:.4f}")
        if g > hi[k] + tol:
            v.append(f"{names[k]}: exposição {g:.4f} > máximo {hi[k]:.4f}")
    if pc.previous_weights is not None and pc.max_turnover is not None:
        to = float(np.abs(w - pc.previous_weights).sum())
        if to > pc.max_turnover + tol:
            v.append(f"turnover {to:.4f} > limite {pc.max_turnover:.4f}")
    return v


def check_feasibility(pc: PortfolioConstraints) -> FeasibilityReport:
    """LP feasibility test; on failure, diagnose conflicts and the nearest allocation."""
    n = pc.n
    if np.any(pc.lower > pc.upper + 1e-12):
        bad = [s for s, lo, hi in zip(pc.symbols, pc.lower, pc.upper) if lo > hi]
        return FeasibilityReport(False, [f"limite mínimo > máximo para {bad}"],
                                 ["corrigir limites individuais"])
    a_ub, b_ub = _lp_matrices(pc)
    bounds = list(zip(pc.lower, pc.upper))
    if pc.previous_weights is not None and pc.max_turnover is not None:
        # variables [w, u]; u >= |w - w0|; sum u <= TO
        w0 = pc.previous_weights
        eye = np.eye(n)
        rows = [np.hstack([eye, -eye]), np.hstack([-eye, -eye]),
                np.hstack([np.zeros(n), np.ones(n)])[None, :]]
        rhs = [w0, -w0, [pc.max_turnover]]
        if a_ub is not None:
            rows.insert(0, np.hstack([a_ub, np.zeros((a_ub.shape[0], n))]))
            rhs.insert(0, b_ub)
        res = linprog(np.zeros(2 * n), A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                      A_eq=np.hstack([np.ones(n), np.zeros(n)])[None, :], b_eq=[1.0],
                      bounds=bounds + [(0, None)] * n, method="highs")
    else:
        res = linprog(np.zeros(n), A_ub=a_ub, b_ub=b_ub, A_eq=np.ones((1, n)), b_eq=[1.0],
                      bounds=bounds, method="highs")
    if res.status == 0:
        return FeasibilityReport(True)
    if pc.previous_weights is not None and pc.max_turnover is not None:
        rep = _diagnose_turnover(pc)
        if rep is not None:
            return rep
    return diagnose_infeasibility(pc)


def _diagnose_turnover(pc: PortfolioConstraints) -> FeasibilityReport | None:
    """If the set is feasible WITHOUT the turnover limit, the limit is the conflict: report the
    minimum turnover needed and the allocation that achieves it (turnover limit not relaxed)."""
    n = pc.n
    w0 = np.asarray(pc.previous_weights, float)
    a_ub, b_ub = _lp_matrices(pc)
    eye = np.eye(n)
    rows = [np.hstack([eye, -eye]), np.hstack([-eye, -eye])]
    rhs = [w0, -w0]
    if a_ub is not None:
        rows.insert(0, np.hstack([a_ub, np.zeros((a_ub.shape[0], n))]))
        rhs.insert(0, b_ub)
    res = linprog(np.concatenate([np.zeros(n), np.ones(n)]), A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  A_eq=np.hstack([np.ones(n), np.zeros(n)])[None, :], b_eq=[1.0],
                  bounds=list(zip(pc.lower, pc.upper)) + [(0, None)] * n, method="highs")
    if res.status != 0:
        return None   # infeasible even without turnover: generic diagnosis
    w = res.x[:n]
    need = float(np.abs(w - w0).sum())
    return FeasibilityReport(
        False, [f"limite de turnover {pc.max_turnover:.2%} é menor que o mínimo necessário ({need:.2%}) para "
                "atender às demais restrições a partir dos pesos atuais"],
        [f"elevar o limite de turnover para >= {need:.2%} ou ajustar a carteira em mais de um rebalanceamento"],
        w, verify_weights(w, pc))


def diagnose_infeasibility(pc: PortfolioConstraints) -> FeasibilityReport:
    """Elastic LP: minimise total violation with budget and w >= 0 kept hard."""
    n = pc.n
    a, lo, hi, names = pc.group_matrix()
    g = a.shape[0]
    # variables: w (n), s_up (n), s_lo (n), g_up (g), g_lo (g)
    nv = 3 * n + 2 * g
    c = np.concatenate([np.zeros(n), np.ones(2 * n + 2 * g)])
    rows, rhs = [], []
    for i in range(n):
        r = np.zeros(nv); r[i] = 1; r[n + i] = -1; rows.append(r); rhs.append(pc.upper[i])
        r = np.zeros(nv); r[i] = -1; r[2 * n + i] = -1; rows.append(r); rhs.append(-pc.lower[i])
    for k in range(g):
        r = np.zeros(nv); r[:n] = a[k]; r[3 * n + k] = -1; rows.append(r); rhs.append(hi[k])
        r = np.zeros(nv); r[:n] = -a[k]; r[3 * n + g + k] = -1; rows.append(r); rhs.append(-lo[k])
    a_eq = np.zeros((1, nv)); a_eq[0, :n] = 1
    w_lb = None if pc.allow_short else 0.0
    bounds = [(w_lb, None)] * n + [(0, None)] * (2 * n + 2 * g)
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=a_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    conflicts: list[str] = []
    suggestions: list[str] = []
    if pc.upper.sum() < 1 - 1e-12:
        conflicts.append(f"soma dos pesos máximos = {pc.upper.sum():.2%} < 100%")
        suggestions.append(f"elevar o peso máximo por ativo para >= {1 / max(1, np.sum(pc.upper > 0)):.2%} "
                           "ou ampliar o universo de ativos")
    if pc.lower.sum() > 1 + 1e-12:
        conflicts.append(f"soma dos pesos mínimos = {pc.lower.sum():.2%} > 100%")
        suggestions.append("reduzir pesos mínimos individuais")
    for k in range(g):
        cap = float(a[k] @ pc.upper)
        if lo[k] > cap + 1e-12:
            what = "nenhum ativo da classe no universo" if a[k].sum() == 0 else \
                f"capacidade máxima dos membros = {cap:.2%}"
            conflicts.append(f"{names[k]}: mínimo exigido {lo[k]:.2%}, mas {what}")
            suggestions.append(f"incluir ativos elegíveis para {names[k]} ou reduzir seu mínimo")
    if g:
        # mass that must go outside capped groups
        member = (a.sum(axis=0) > 0)
        cap_total = float(sum(min(hi[k], float(a[k] @ pc.upper)) for k in range(g))) + \
            float(pc.upper[~member].sum())
        if cap_total < 1 - 1e-12:
            conflicts.append(f"limites de classe + limites individuais permitem no máximo "
                             f"{cap_total:.2%} de alocação")
            suggestions.append("relaxar limites máximos de classe ou incluir ativos de outras classes")
    nearest = None
    violations: list[str] = []
    if res.status == 0:
        nearest = res.x[:n]
        violations = verify_weights(nearest, pc)
    if not conflicts:
        conflicts.append("combinação de restrições sem solução (detectada por LP)")
    return FeasibilityReport(False, conflicts, suggestions, nearest, violations)
