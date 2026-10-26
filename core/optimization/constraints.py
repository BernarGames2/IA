"""Constraint construction, feasibility detection and nearest-feasible diagnosis.

Feasibility of the linear constraint set (budget, bounds, groups, turnover) is
checked with an LP *before* any optimisation. If infeasible, an elastic LP finds
the allocation that minimises total constraint violation while keeping the
budget and the no-short-selling rule hard, and reports which constraints conflict
and by how much. User preferences are never relaxed silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

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


def semicontinuous_weights(pc: PortfolioConstraints, floor: np.ndarray, target: np.ndarray | None = None,
                           pool: Sequence[int] | None = None, avoid: Sequence[Sequence[int]] = (), min_held: int = 1,
                           time_limit_s: float = 5.0, max_held: int | None = None,
                           count_groups: Sequence[tuple[Sequence[int], int]] = (),
                           distinct: tuple[Sequence[Sequence[int]], int] | None = None) -> np.ndarray | None:
    """Weights within the bounds and group limits where each asset is either out (0) or held at no less than its
    ``floor`` (e.g. the smallest purchase as a weight): at least ``min_held`` held (and at most ``max_held``), only
    assets in ``pool`` (default all), never exactly one of the held sets in ``avoid``; the closest to ``target`` in sum
    of absolute differences when one is given. Counting rules (e.g. from a competition rulebook): ``count_groups``
    lists (members, minimum held among them) — "at least one ETF"; ``distinct`` = (member sets, minimum number of
    sets with at least one holding) — "at least five sectors". Mixed-integer LP (HiGHS), exact where a greedy choice
    of the held set can spend the money on large lots before a group minimum is met. Turnover limits are not
    modelled. None when no such weights exist (or none was found within the time limit)."""
    n = pc.n
    can = np.zeros(n, bool)
    can[list(range(n) if pool is None else pool)] = True
    lo = np.maximum(pc.lower, np.asarray(floor, float))
    up = np.where(can & (pc.upper > 0) & (lo <= pc.upper + 1e-12), pc.upper, 0.0)
    must = pc.lower > 1e-12
    if np.any(must & (up <= 0)):
        return None
    k = 2 if target is None else 3          # variables: w, z (held or not) and, with a target, d >= |w - target|
    sets = [np.asarray(list(g), int) for g in distinct[0]] if distinct else []
    m = len(sets)                            # one more binary per set of ``distinct``: the set has a holding
    width = k * n + m
    eye, zero = np.eye(n), np.zeros((n, n))
    pad = [] if target is None else [zero]
    yz = np.zeros((n, m))
    a, glo, ghi, _ = pc.group_matrix()
    z_row = np.concatenate([np.zeros(n), np.ones(n), np.zeros((k - 2) * n + m)])
    cons = [LinearConstraint(np.vstack([np.hstack([eye, -np.diag(up), *pad, yz]),                  # w <= up * z
                                        np.hstack([-eye, np.diag(np.minimum(lo, up)), *pad, yz])]),  # floor * z <= w
                             -np.inf, 0.0),
            LinearConstraint(np.concatenate([np.ones(n), np.zeros((k - 1) * n + m)])[None, :], 1.0, 1.0),
            LinearConstraint(z_row[None, :], min_held, np.inf if max_held is None else max_held)]
    if a.shape[0]:
        cons.append(LinearConstraint(np.hstack([a, np.zeros((a.shape[0], (k - 1) * n + m))]), glo, ghi))
    if target is not None:
        t = np.asarray(target, float)
        cons.append(LinearConstraint(np.vstack([np.hstack([eye, zero, -eye, yz]), np.hstack([-eye, zero, -eye, yz])]),
                                     -np.inf, np.concatenate([t, -t])))
    for members, least in count_groups:
        r = np.zeros(width)
        r[n + np.asarray(list(members), int)] = 1.0
        cons.append(LinearConstraint(r[None, :], least, np.inf))
    if m:
        rows = np.zeros((m, width))
        for j, g in enumerate(sets):
            rows[j, n + g] = -1.0
            rows[j, k * n + j] = 1.0         # a set counts only when one of its members is held
        cons.append(LinearConstraint(rows, -np.inf, 0.0))
        r = np.zeros(width)
        r[k * n:] = 1.0
        cons.append(LinearConstraint(r[None, :], distinct[1], np.inf))  # type: ignore[index]
    for h in avoid:                          # no-good cut: not exactly this held set again
        r = -z_row
        r[n + np.asarray(list(h), int)] = 1.0
        cons.append(LinearConstraint(r[None, :], -np.inf, len(h) - 1))
    integer = z_row.copy()
    integer[k * n:] = 1.0
    cost = np.zeros(width)
    if target is not None:
        cost[2 * n:3 * n] = 1.0
    res = milp(cost, constraints=cons, integrality=integer, options={"time_limit": time_limit_s},
               bounds=Bounds(np.concatenate([np.zeros(n), must.astype(float), np.zeros((k - 2) * n + m)]),
                             np.concatenate([up, (up > 0).astype(float), np.full((k - 2) * n, np.inf), np.ones(m)])))
    if res.x is None:
        return None
    w = np.where(res.x[n:2 * n] > 0.5, np.clip(res.x[:n], 0.0, None), 0.0)
    return w / w.sum() if w.sum() > 0 else None


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
