"""Portfolio Construction Engine: mean-variance family and risk-based allocations.

Every optimiser:
1. checks feasibility of the linear constraints first (LP);
2. solves with SLSQP (or an LP / convex reformulation where appropriate) from
   several deterministic starting points;
3. verifies the solution independently (budget, bounds, groups, turnover,
   finiteness) -- a solver "success" flag alone is never trusted;
4. reports dispersion across starting points as a sensitivity diagnostic.

Inputs ``mu`` (annualised arithmetic expected returns) and ``cov`` (annualised
covariance) must be aligned with ``constraints.symbols``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from core.optimization.constraints import (
    InfeasibleConstraintsError,
    check_feasibility,
    verify_weights,
)
from models.portfolio import OptimizationResult, PortfolioConstraints
from utils.numerical import cov_to_corr
from utils.validation import validate_covariance

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


@dataclass
class SolverSettings:
    n_starts: int = 6
    seed: int = 42
    ftol: float = 1e-12
    maxiter: int = 1000
    tol: float = 1e-6


def _stats(w: np.ndarray, mu: np.ndarray | None, cov: np.ndarray, rf: float
           ) -> tuple[float, float, float | None]:
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    ret = float(w @ mu) if mu is not None else float("nan")
    sharpe = (ret - rf) / vol if (mu is not None and vol > 1e-12) else None
    return ret, vol, sharpe


def _starts(pc: PortfolioConstraints, s: SolverSettings, extra: list[np.ndarray] | None = None
            ) -> list[np.ndarray]:
    rng = np.random.default_rng(s.seed)
    n = pc.n
    ub = np.where(pc.upper > 0, pc.upper, 0)
    base = ub / ub.sum() if ub.sum() > 0 else np.full(n, 1.0 / n)
    starts = [np.clip(np.full(n, 1.0 / n), pc.lower, pc.upper), base]
    starts += list(extra or [])
    while len(starts) < s.n_starts:
        d = rng.dirichlet(np.ones(n))
        starts.append(np.clip(d, pc.lower, pc.upper))
    return starts[: max(s.n_starts, len(extra or []) + 2)]


def _scipy_constraints(pc: PortfolioConstraints, n_aux: int) -> list[dict]:
    n = pc.n
    cons: list[dict] = [{"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0,
                         "jac": lambda x: np.concatenate([np.ones(n), np.zeros(n_aux)])}]
    a, lo, hi, _ = pc.group_matrix()
    for k in range(a.shape[0]):
        ak = np.concatenate([a[k], np.zeros(n_aux)])
        cons.append({"type": "ineq", "fun": lambda x, ak=ak, h=hi[k]: h - ak @ x,
                     "jac": lambda x, ak=ak: -ak})
        cons.append({"type": "ineq", "fun": lambda x, ak=ak, l=lo[k]: ak @ x - l,
                     "jac": lambda x, ak=ak: ak})
    if n_aux:
        w0 = pc.previous_weights
        assert w0 is not None
        eye = np.eye(n)
        # u_i - (w_i - w0_i) >= 0 ; u_i + (w_i - w0_i) >= 0
        cons.append({"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - w0),
                     "jac": lambda x: np.hstack([-eye, eye])})
        cons.append({"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - w0),
                     "jac": lambda x: np.hstack([eye, eye])})
        if pc.max_turnover is not None:
            cons.append({"type": "ineq", "fun": lambda x: pc.max_turnover - np.sum(x[n:]),
                         "jac": lambda x: np.concatenate([np.zeros(n), -np.ones(n)])})
    return cons


def solve(method: str, objective: Objective, gradient: Gradient | None,
          pc: PortfolioConstraints, mu: np.ndarray | None, cov: np.ndarray, rf: float,
          settings: SolverSettings | None = None, extra_constraints: list[dict] | None = None,
          turnover_penalty: float = 0.0, start_hint: list[np.ndarray] | None = None
          ) -> OptimizationResult:
    """Generic multi-start SLSQP with independent verification of the solution."""
    s = settings or SolverSettings()
    feas = check_feasibility(pc)
    if not feas.feasible:
        w = feas.nearest_weights
        ws = pd.Series(w if w is not None else np.full(pc.n, np.nan), index=pc.symbols)
        ret, vol, sh = _stats(w, mu, cov, rf) if w is not None else (np.nan, np.nan, None)
        return OptimizationResult(
            method, ws, False, "infeasible_constraints", ret, vol, sh, False,
            violations=feas.violations,
            warnings=["restrições inviáveis: retornada a alocação viável mais próxima "
                      "(mínima violação total), NÃO uma solução ótima"] + feas.conflicts,
            diagnostics={"conflicts": feas.conflicts, "suggested_relaxations": feas.suggestions})

    n = pc.n
    use_aux = pc.previous_weights is not None and (pc.max_turnover is not None or turnover_penalty > 0)
    n_aux = n if use_aux else 0

    def f(x: np.ndarray) -> float:
        val = objective(x[:n])
        if n_aux and turnover_penalty:
            val += turnover_penalty * float(np.sum(x[n:]))
        return val

    def g(x: np.ndarray) -> np.ndarray:
        gw = gradient(x[:n]) if gradient is not None else _num_grad(objective, x[:n])
        if n_aux:
            return np.concatenate([gw, np.full(n, turnover_penalty)])
        return gw

    bounds = list(zip(pc.lower, pc.upper)) + [(0.0, 2.0)] * n_aux
    cons = _scipy_constraints(pc, n_aux) + list(extra_constraints or [])
    sols: list[tuple[float, np.ndarray, str]] = []
    statuses: list[str] = []
    for x0w in _starts(pc, s, start_hint):
        x0 = np.concatenate([x0w, np.abs(x0w - pc.previous_weights)]) if n_aux else x0w
        r = minimize(f, x0, jac=g, bounds=bounds, constraints=cons, method="SLSQP",
                     options={"ftol": s.ftol, "maxiter": s.maxiter})
        statuses.append(r.message if isinstance(r.message, str) else str(r.message))
        w = r.x[:n]
        if not np.all(np.isfinite(w)):
            continue
        ok_extra = all(
            (c["fun"](r.x) >= -s.tol if c["type"] == "ineq" else abs(c["fun"](r.x)) <= s.tol)
            for c in extra_constraints or [])
        if r.success and not verify_weights(w, pc, tol=s.tol) and ok_extra:
            sols.append((float(r.fun), w, str(r.message)))
    if not sols:
        return OptimizationResult(method, pd.Series(np.full(n, np.nan), index=pc.symbols), False,
                                  "no_converged_feasible_solution", np.nan, np.nan, None, False,
                                  warnings=[f"nenhum ponto inicial convergiu para solução viável: {statuses}"],
                                  diagnostics={"solver_messages": statuses})
    sols.sort(key=lambda t: t[0])
    best_f, best_w, msg = sols[0]
    best_w = np.where(np.abs(best_w) < 1e-10, 0.0, best_w)
    best_w = best_w / best_w.sum()
    near = [w for fv, w, _ in sols if fv <= best_f + max(1e-9, 1e-6 * abs(best_f))]
    disp = max((float(np.abs(w - best_w).sum()) for w in near), default=0.0)
    warnings = []
    if disp > 0.05:
        warnings.append(f"soluções quase-ótimas diferem até {disp:.2%} (L1) entre pontos iniciais: "
                        "problema mal condicionado / ótimo pouco identificado")
    ret, vol, sh = _stats(best_w, mu, cov, rf)
    viol = verify_weights(best_w, pc, tol=s.tol)
    return OptimizationResult(
        method, pd.Series(best_w, index=pc.symbols), True, msg, ret, vol, sh, not viol,
        violations=viol, warnings=warnings,
        diagnostics={"n_starts": len(statuses), "n_converged": len(sols), "objective": best_f,
                     "start_dispersion_l1": disp})


def _num_grad(fun: Objective, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


def _prep(mu: pd.Series | np.ndarray | None, cov: pd.DataFrame | np.ndarray,
          pc: PortfolioConstraints) -> tuple[np.ndarray | None, np.ndarray]:
    if isinstance(cov, pd.DataFrame):
        cov = cov.loc[pc.symbols, pc.symbols].to_numpy()
    c = validate_covariance(np.asarray(cov, float), pc.n)
    m = None
    if mu is not None:
        m = (mu.loc[pc.symbols].to_numpy() if isinstance(mu, pd.Series) else np.asarray(mu, float))
        if not np.all(np.isfinite(m)):
            raise ValueError("expected returns contain NaN/inf")
    return m, c


# ----------------------------------------------------------------- optimisers
def min_variance(cov, pc: PortfolioConstraints, mu=None, rf: float = 0.0,
                 settings: SolverSettings | None = None, turnover_penalty: float = 0.0
                 ) -> OptimizationResult:
    m, c = _prep(mu, cov, pc)
    return solve("min_variance", lambda w: float(w @ c @ w), lambda w: 2 * c @ w,
                 pc, m, c, rf, settings, turnover_penalty=turnover_penalty)


def max_sharpe(mu, cov, pc: PortfolioConstraints, rf: float,
               settings: SolverSettings | None = None) -> OptimizationResult:
    m, c = _prep(mu, cov, pc)
    assert m is not None

    def f(w: np.ndarray) -> float:
        v = float(np.sqrt(max(w @ c @ w, 1e-18)))
        return -float(w @ m - rf) / v

    def g(w: np.ndarray) -> np.ndarray:
        v = float(np.sqrt(max(w @ c @ w, 1e-18)))
        ex = float(w @ m - rf)
        return -(m / v - ex * (c @ w) / v ** 3)

    hint = []
    if np.any(m > rf):
        y = np.linalg.lstsq(c, np.clip(m - rf, 0, None), rcond=None)[0]
        y = np.clip(y, 0, None)
        if y.sum() > 0:
            hint.append(np.clip(y / y.sum(), pc.lower, pc.upper))
    res = solve("max_sharpe", f, g, pc, m, c, rf, settings, start_hint=hint)
    if not np.any(m > rf):
        res.warnings.append("nenhum ativo tem retorno esperado acima da taxa livre de risco: "
                            "Sharpe máximo é negativo e a carteira não é significativa")
    return res


def target_return(mu, cov, pc: PortfolioConstraints, target: float, rf: float = 0.0,
                  settings: SolverSettings | None = None) -> OptimizationResult:
    m, c = _prep(mu, cov, pc)
    assert m is not None
    lo, hi = return_range(m, pc)
    if not (lo - 1e-9 <= target <= hi + 1e-9):
        return OptimizationResult(
            "target_return", pd.Series(np.full(pc.n, np.nan), index=pc.symbols), False,
            "target_out_of_range", np.nan, np.nan, None, False,
            warnings=[f"retorno-alvo {target:.2%} fora do intervalo atingível [{lo:.2%}, {hi:.2%}]"])
    eq = [{"type": "eq", "fun": lambda w: float(w[:pc.n] @ m) - target,
           "jac": lambda w: np.concatenate([m, np.zeros(len(w) - pc.n)])}]
    return solve("target_return", lambda w: float(w @ c @ w), lambda w: 2 * c @ w,
                 pc, m, c, rf, settings, extra_constraints=eq)


def target_volatility(mu, cov, pc: PortfolioConstraints, target_vol: float, rf: float = 0.0,
                      settings: SolverSettings | None = None) -> OptimizationResult:
    m, c = _prep(mu, cov, pc)
    assert m is not None
    gmv = min_variance(c, pc, m, rf, settings)
    if gmv.success and target_vol < gmv.volatility - 1e-9:
        return OptimizationResult(
            "target_volatility", pd.Series(np.full(pc.n, np.nan), index=pc.symbols), False,
            "target_below_min_vol", np.nan, np.nan, None, False,
            warnings=[f"volatilidade-alvo {target_vol:.2%} abaixo da mínima viável "
                      f"{gmv.volatility:.2%}"])
    ineq = [{"type": "ineq", "fun": lambda w: target_vol ** 2 - float(w[:pc.n] @ c @ w[:pc.n]),
             "jac": lambda w: np.concatenate([-2 * c @ w[:pc.n], np.zeros(len(w) - pc.n)])}]
    hint = [gmv.weights.to_numpy()] if gmv.success else None
    return solve("target_volatility", lambda w: -float(w @ m), lambda w: -m,
                 pc, m, c, rf, settings, extra_constraints=ineq, start_hint=hint)


def return_range(m: np.ndarray, pc: PortfolioConstraints) -> tuple[float, float]:
    """Min and max attainable expected return under the linear constraints (LP)."""
    a, lo, hi, _ = pc.group_matrix()
    a_ub = np.vstack([a, -a]) if a.shape[0] else None
    b_ub = np.concatenate([hi, -lo]) if a.shape[0] else None
    out = []
    for sign in (1.0, -1.0):
        r = linprog(sign * m, A_ub=a_ub, b_ub=b_ub, A_eq=np.ones((1, pc.n)), b_eq=[1.0],
                    bounds=list(zip(pc.lower, pc.upper)), method="highs")
        if r.status != 0:
            raise InfeasibleConstraintsError("constraints infeasible")
        out.append(float(m @ r.x))
    return out[0], out[1]


def equal_weight(pc: PortfolioConstraints, mu=None, cov=None, rf: float = 0.0) -> OptimizationResult:
    """1/N over assets with positive upper bound. Constraint violations are reported."""
    elig = pc.upper > 0
    w = np.where(elig, 1.0 / max(1, elig.sum()), 0.0)
    m, c = _prep(mu, cov, pc) if cov is not None else (None, np.eye(pc.n))
    ret, vol, sh = _stats(w, m, c, rf)
    viol = verify_weights(w, pc)
    return OptimizationResult("equal_weight", pd.Series(w, index=pc.symbols), True, "closed_form",
                              ret, vol, sh, not viol, violations=viol,
                              warnings=["pesos iguais violam restrições"] if viol else [])


def risk_parity(cov, pc: PortfolioConstraints, mu=None, rf: float = 0.0,
                budgets: np.ndarray | None = None, settings: SolverSettings | None = None
                ) -> OptimizationResult:
    """Equal (or budgeted) risk contribution.

    Long-only with only box bounds [0, 1]: Spinu (2013) convex problem
    ``min 0.5 y'Sy - sum b_i log y_i`` -> w = y / sum(y) (unique solution).
    Otherwise SLSQP on squared deviations of risk shares from the budgets.
    """
    m, c = _prep(mu, cov, pc)
    n = pc.n
    eligible = pc.upper > 0
    if budgets is None:
        b = np.where(eligible, 1.0 / max(1, eligible.sum()), 0.0)
    else:
        b = np.asarray(budgets, float) / np.sum(budgets)
    simple = (not pc.groups and np.all(pc.lower <= 0) and np.all(pc.upper >= 1 - 1e-12)
              and pc.previous_weights is None)
    if simple:
        def f(y: np.ndarray) -> float:
            return 0.5 * float(y @ c @ y) - float(b @ np.log(y))

        def g(y: np.ndarray) -> np.ndarray:
            return c @ y - b / y

        y0 = 1.0 / np.sqrt(np.diag(c))
        r = minimize(f, y0, jac=g, method="L-BFGS-B", bounds=[(1e-12, None)] * n,
                     options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
        w = r.x / r.x.sum()
        ret, vol, sh = _stats(w, m, c, rf)
        rc = w * (c @ w) / max(vol, 1e-18)
        dev = float(np.max(np.abs(rc / vol - b)))
        viol = verify_weights(w, pc)
        return OptimizationResult("risk_parity", pd.Series(w, index=pc.symbols), bool(r.success),
                                  "spinu_convex", ret, vol, sh, not viol, violations=viol,
                                  warnings=[] if dev < 1e-4 else [f"desvio máximo de orçamento de risco {dev:.2e}"],
                                  diagnostics={"max_budget_deviation": dev})

    scale = 1e2

    def f2(w: np.ndarray) -> float:
        v = float(w @ c @ w)
        share = w * (c @ w) / max(v, 1e-18)
        return float(np.sum((share - b) ** 2)) * scale

    def g2(w: np.ndarray) -> np.ndarray:
        a = c @ w
        v = max(float(w @ a), 1e-18)
        share = w * a / v
        jac = (np.diag(a) + w[:, None] * c) / v - np.outer(w * a, 2.0 * a) / v ** 2
        return jac.T @ (2.0 * (share - b)) * scale

    inv_vol = np.where(eligible, 1.0 / np.sqrt(np.diag(c)), 0.0)
    hint = [np.clip(inv_vol / inv_vol.sum(), pc.lower, pc.upper)] if inv_vol.sum() > 0 else None
    res = solve("risk_parity", f2, g2, pc, m, c, rf, settings, start_hint=hint)
    if res.success:
        w = res.weights.to_numpy()
        share = w * (c @ w) / float(w @ c @ w)
        res.diagnostics["max_budget_deviation"] = float(np.max(np.abs(share - b)))
        if res.diagnostics["max_budget_deviation"] > 1e-3:
            res.warnings.append("restrições impedem contribuições de risco iguais; solução é a mais "
                                "próxima viável em mínimos quadrados")
    return res


def max_diversification(cov, pc: PortfolioConstraints, mu=None, rf: float = 0.0,
                        settings: SolverSettings | None = None) -> OptimizationResult:
    """Maximise DR = w'sigma / sqrt(w'Sigma w) (Choueifaty & Coignard, 2008)."""
    m, c = _prep(mu, cov, pc)
    sd = np.sqrt(np.diag(c))

    def f(w: np.ndarray) -> float:
        return -float(w @ sd) / float(np.sqrt(max(w @ c @ w, 1e-18)))

    def g(w: np.ndarray) -> np.ndarray:
        v = float(np.sqrt(max(w @ c @ w, 1e-18)))
        return -(sd / v - float(w @ sd) * (c @ w) / v ** 3)

    res = solve("max_diversification", f, g, pc, m, c, rf, settings)
    if res.success:
        res.diagnostics["diversification_ratio"] = -float(res.diagnostics["objective"])
    return res


def hrp(cov, pc: PortfolioConstraints, mu=None, rf: float = 0.0,
        linkage_method: str = "single") -> OptimizationResult:
    """Hierarchical Risk Parity (López de Prado, 2016).

    Correlation distance ``d = sqrt((1-rho)/2)``, hierarchical clustering,
    quasi-diagonalisation and recursive bisection with inverse-variance weights.
    HRP does not natively support bounds/groups: violations are REPORTED, not
    silently clipped.
    """
    m, c = _prep(mu, cov, pc)
    corr, sd = cov_to_corr(c)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method=linkage_method)
    order = list(leaves_list(z))
    w = np.ones(pc.n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]

            def cvar(idx: list[int]) -> float:
                sub = c[np.ix_(idx, idx)]
                ivp = 1.0 / np.diag(sub)
                ivp /= ivp.sum()
                return float(ivp @ sub @ ivp)

            vl, vr = cvar(left), cvar(right)
            alpha = 1.0 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    w = w / w.sum()
    ret, vol, sh = _stats(w, m, c, rf)
    viol = verify_weights(w, pc)
    return OptimizationResult("hrp", pd.Series(w, index=pc.symbols), True, f"hrp_{linkage_method}",
                              ret, vol, sh, not viol, violations=viol,
                              warnings=["HRP não incorpora limites; violações reportadas"] if viol else [],
                              diagnostics={"order": [pc.symbols[i] for i in order]})


def min_cvar(scenario_returns: np.ndarray, pc: PortfolioConstraints, alpha: float = 0.95,
             mu=None, cov=None, rf: float = 0.0) -> OptimizationResult:
    """Minimum Expected Shortfall via the Rockafellar-Uryasev (2000) linear program.

    Scenarios are rows of simple returns (historical or simulated, one period).
    Variables: w (n), zeta (VaR), u (T). min zeta + 1/((1-a)T) sum u
    s.t. u_t >= -r_t'w - zeta, u_t >= 0, budget/bounds/groups.
    """
    r = np.asarray(scenario_returns, float)
    t, n = r.shape
    if n != pc.n:
        raise ValueError("scenario matrix columns must match constraints.symbols")
    feas = check_feasibility(pc)
    if not feas.feasible:
        return OptimizationResult("min_cvar", pd.Series(np.full(n, np.nan), index=pc.symbols), False,
                                  "infeasible_constraints", np.nan, np.nan, None, False,
                                  warnings=feas.conflicts)
    nv = n + 1 + t
    cvec = np.concatenate([np.zeros(n), [1.0], np.full(t, 1.0 / ((1 - alpha) * t))])
    a_ub = np.hstack([-r, -np.ones((t, 1)), -np.eye(t)])
    b_ub = np.zeros(t)
    a, lo, hi, _ = pc.group_matrix()
    if a.shape[0]:
        a_ub = np.vstack([a_ub, np.hstack([a, np.zeros((a.shape[0], 1 + t))]),
                          np.hstack([-a, np.zeros((a.shape[0], 1 + t))])])
        b_ub = np.concatenate([b_ub, hi, -lo])
    a_eq = np.zeros((1, nv)); a_eq[0, :n] = 1.0
    bounds = list(zip(pc.lower, pc.upper)) + [(None, None)] + [(0, None)] * t
    res = linprog(cvec, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status != 0:
        return OptimizationResult("min_cvar", pd.Series(np.full(n, np.nan), index=pc.symbols), False,
                                  f"lp_status_{res.status}", np.nan, np.nan, None, False,
                                  warnings=[str(res.message)])
    w = res.x[:n]
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    w = w / w.sum()
    m = c = None
    if cov is not None:
        m, c = _prep(mu, cov, pc)
    ret, vol, sh = _stats(w, m, c, rf) if c is not None else (np.nan, np.nan, None)
    viol = verify_weights(w, pc)
    return OptimizationResult("min_cvar", pd.Series(w, index=pc.symbols), True, "highs_lp",
                              ret, vol, sh, not viol, violations=viol,
                              diagnostics={"cvar": float(res.fun), "var": float(res.x[n]),
                                           "alpha": alpha, "n_scenarios": t})


def efficient_frontier(mu, cov, pc: PortfolioConstraints, rf: float, n_points: int = 30,
                       settings: SolverSettings | None = None) -> pd.DataFrame:
    """Efficient frontier with ONLY verified feasible, converged, non-dominated points."""
    m, c = _prep(mu, cov, pc)
    assert m is not None
    gmv = min_variance(c, pc, m, rf, settings)
    if not gmv.success:
        return pd.DataFrame()
    _, r_max = return_range(m, pc)
    rows = []
    for tgt in np.linspace(gmv.expected_return, r_max, n_points):
        res = gmv if np.isclose(tgt, gmv.expected_return) else target_return(m, c, pc, float(tgt), rf, settings)
        if not (res.success and res.feasible):
            continue
        rows.append({"target_return": float(tgt), "expected_return": res.expected_return,
                     "volatility": res.volatility, "sharpe": res.sharpe,
                     **{f"w_{k}": float(v) for k, v in res.weights.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("expected_return").reset_index(drop=True)
    # point i is dominated if a point with higher return has strictly lower volatility
    vols = df["volatility"].to_numpy()
    suffix_min = np.minimum.accumulate(vols[::-1])[::-1]
    keep = np.ones(len(df), bool)
    keep[:-1] = ~(suffix_min[1:] < vols[:-1] - 1e-7)
    return df[keep].reset_index(drop=True)
