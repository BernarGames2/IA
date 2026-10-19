"""Step 1 — required CAGR: the annual rate that takes capital (+ periodic contributions)
to the target value in the horizon. It is computed BEFORE any optimisation and is the
yardstick for every later result.

Future value with contributions at the END of each period (m per year, n = floor(m*T)):

    FV(r) = C (1+r)^T + sum_{k=1..n} a (1+r)^(T - k/m)

FV is strictly increasing in r for C > 0 and a >= 0, so the root is unique; it is
found with Brent's method on r in (-99%, +1000%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

PERIODS_PER_YEAR = {"mensal": 12, "trimestral": 4, "semestral": 2, "anual": 1, "nenhum": 0}


@dataclass(frozen=True)
class RequiredCagr:
    rate: float | None
    status: str                 # ok | atingivel_sem_retorno | inatingivel
    total_contributed: float
    fv_at_zero_return: float
    explanation: str


def future_value(rate: float, capital: float, contribution: float, per_year: int, years: float) -> float:
    fv = capital * (1.0 + rate) ** years
    if per_year > 0 and contribution > 0:
        n = int(np.floor(per_year * years + 1e-9))
        k = np.arange(1, n + 1)
        fv += float(np.sum(contribution * (1.0 + rate) ** (years - k / per_year)))
    return float(fv)


def required_cagr(capital: float, contribution: float, periodicity: str, target: float,
                  years: float) -> RequiredCagr:
    if capital < 0 or target <= 0 or years <= 0:
        raise ValueError("capital must be >= 0; target and horizon must be positive")
    if contribution < 0:
        raise ValueError("contribution must be >= 0")
    if periodicity not in PERIODS_PER_YEAR:
        raise ValueError(f"periodicity must be one of {list(PERIODS_PER_YEAR)}")
    m = PERIODS_PER_YEAR[periodicity]
    n = int(np.floor(m * years + 1e-9)) if m else 0
    if capital == 0 and (contribution <= 0 or n == 0):
        raise ValueError("without initial capital at least one contribution must fall inside the horizon")
    total = capital + (contribution * n if m else 0.0)
    fv0 = future_value(0.0, capital, contribution, m, years)

    def g(r: float) -> float:
        return future_value(r, capital, contribution, m, years) - target

    lo, hi = -0.99, 10.0
    if g(hi) < 0:
        return RequiredCagr(None, "inatingivel", total, fv0,
                            "o objetivo exigiria retorno anual acima de 1000% no horizonte: inatingível na prática")
    if g(lo) > 0:
        return RequiredCagr(lo, "atingivel_sem_retorno", total, fv0,
                            "o objetivo é atingido mesmo perdendo quase todo o capital investido")
    r = float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500))
    status = "atingivel_sem_retorno" if r <= 0 else "ok"
    expl = (f"CAGR necessário de {r:.2%} a.a. para levar o capital inicial e os aportes ({total:,.0f} aportados no "
            f"total) a {target:,.0f} em {years:g} anos" if capital > 0 else
            f"CAGR necessário de {r:.2%} a.a. para levar só os aportes (sem capital inicial; {total:,.0f} aportados "
            f"no total) a {target:,.0f} em {years:g} anos") + (
            "; o objetivo não exige retorno positivo" if r <= 0 else "")
    return RequiredCagr(r, status, total, fv0, expl.replace(",", "."))
