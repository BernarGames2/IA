"""What changed from a previous portfolio, why, and what did not.

* Materiality: a weight change below 1 percentage point is reported as "mantido" (new prices move
  every estimate a little; that alone does not make a "new" portfolio). A position counts as held
  from 0.5%.
* Fingerprint: hash of the tickers and weights rounded to 0.1%. Two runs show the same fingerprint
  exactly when the displayed portfolio is the same — novelty is never manufactured.
* Stability rule, for the previous RECOMMENDATION (not the investor's real holdings, which are handled
  by the cost-aware optimisation and the purchase plan): when the previous portfolio still meets every
  current limit, it is kept unless the new portfolio's estimated utility gain [E] exceeds its standard
  error plus the cost of the change. The standard error is approximated by TE / sqrt(years of data),
  TE = annual volatility of the difference between the two portfolios (the error of a mean estimated
  from that many years of returns). Re-estimating with a few more days of prices is not, by itself, a
  reason to trade.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MATERIAL = 0.01
PRESENT = 0.005


def fingerprint(w: pd.Series | dict[str, float]) -> str:
    items = sorted((str(t), round(float(x), 3)) for t, x in dict(w).items() if float(x) >= PRESENT)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:10]


@dataclass
class Change:
    ticker: str
    before: float
    after: float
    kind: str          # entrou | saiu | aumentou | reduziu | mantido
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"ticker": self.ticker, "antes": self.before, "depois": self.after, "tipo": self.kind,
                "motivo": self.reason}


@dataclass
class ChangeReport:
    reference: str                      # "recomendação anterior" | "carteira atual"
    changes: list[Change]
    turnover: float                     # sum |after - before| / 2: share of the money that changes place
    fingerprint_before: str
    fingerprint_after: str
    stability: dict[str, object] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def same(self) -> bool:
        return self.fingerprint_before == self.fingerprint_after or all(c.kind == "mantido" for c in self.changes)

    def as_dict(self) -> dict[str, object]:
        return {"referencia": self.reference, "mudancas": [c.as_dict() for c in self.changes],
                "giro": self.turnover, "impressao_digital_antes": self.fingerprint_before,
                "impressao_digital_depois": self.fingerprint_after, "sem_mudanca_material": self.same,
                "regra_de_estabilidade": self.stability, "notas": self.notes}


def compare(before: dict[str, float], after: pd.Series, why: dict[str, str], reference: str) -> ChangeReport:
    """``why``: reason per ticker for whatever happened to it (absence, weight, entry)."""
    b = {t: float(x) for t, x in before.items() if float(x) > 0}
    tot = sum(b.values())
    b = {t: x / tot for t, x in b.items()} if tot > 0 else {}
    a = {str(t): float(x) for t, x in after.items() if float(x) >= PRESENT}
    out = []
    for t in sorted(set(b) | set(a), key=lambda t: -(a.get(t, 0.0) + b.get(t, 0.0))):
        x0, x1 = b.get(t, 0.0), a.get(t, 0.0)
        if x0 < PRESENT and x1 >= PRESENT:
            kind = "entrou"
        elif x0 >= PRESENT and x1 < PRESENT:
            kind = "saiu"
        elif abs(x1 - x0) < MATERIAL:
            kind = "mantido"
        else:
            kind = "aumentou" if x1 > x0 else "reduziu"
        out.append(Change(t, x0, x1, kind, why.get(t, "") if kind != "mantido" else
                          f"variação de {abs(x1 - x0) * 100:.1f} p.p., abaixo de {MATERIAL * 100:.0f} p.p.: não é "
                          "uma mudança material"))
    turnover = 0.5 * sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in set(a) | set(b))
    return ChangeReport(reference, out, turnover, fingerprint(b), fingerprint(after))


def stability_test(before: pd.Series, after: pd.Series, mu: pd.Series, cov: pd.DataFrame, risk_aversion: float,
                   years: float, cost_per_turnover: float) -> dict[str, object]:
    """Is the new portfolio better than the previous one by more than estimation error + trading cost?
    Both are valued with the SAME current estimates; ``before`` must be restricted to analysed assets."""
    idx = list(after.index)
    w0 = before.reindex(idx).fillna(0.0).to_numpy()
    w1 = after.reindex(idx).fillna(0.0).to_numpy()
    m = mu.loc[idx].to_numpy()
    c = cov.loc[idx, idx].to_numpy()

    def u(w: np.ndarray) -> float:
        return float(w @ m - 0.5 * risk_aversion * (w @ c @ w))

    d = w1 - w0
    te = float(np.sqrt(max(d @ c @ d, 0.0)))
    se = float(te / np.sqrt(max(years, 1e-9)))
    cost = float(cost_per_turnover * np.abs(d).sum())
    gain = u(w1) - u(w0)
    keep = bool(gain <= se + cost)
    return {"ganho_utilidade_E": gain, "erro_padrao_aprox": se, "custo_da_troca": cost, "manter_anterior": keep,
            "explicacao": (f"a nova carteira melhora a utilidade esperada [E] em {gain * 100:.2f} p.p. ao ano, "
                           + ("menos que" if keep else "mais que")
                           + f" a margem de erro da estimativa ({se * 100:.2f} p.p.) somada ao custo de trocar "
                             f"({cost * 100:.2f} p.p.): "
                           + ("a recomendação anterior foi mantida." if keep else "a mudança foi adotada."))}
