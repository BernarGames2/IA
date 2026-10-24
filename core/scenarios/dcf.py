"""Business futures simulator: what the company's equity is worth under explicit hypotheses — and what today's price
already implies. Every input is either a published number [H] or a hypothesis the investor can change [hipótese];
the output is an estimate [E] that inherits all of their uncertainty.

Model (in the currency of the statements, nominal, ``YEARS`` explicit years + perpetuity):

* revenue grows at ``crescimento`` a year for 5 years, then at ``crescimento_perpetuo`` forever;
* cash flow to shareholders = revenue × free-cash-flow margin, the margin moving in a straight line from today's
  (median of the published years) to ``margem_fcl`` by year 5. The free cash flow published by the source is operating
  cash flow minus investment, after interest — a flow to shareholders, so it is discounted at the cost of equity and
  debt is not subtracted again;
* banks and insurers have no meaningful free cash flow: cash to shareholders = net income × ``payout``, net income
  growing at ``crescimento``;
* discount rate = cost of equity ``custo_capital`` (default: risk-free rate of the currency + beta × 4%, as in the
  forecasts); the perpetuity needs ``custo_capital`` at least 1 p.p. above ``crescimento_perpetuo``;
* value per share = (Σ flows discounted + perpetuity discounted) / diluted shares.

Three scenarios (pessimista / base / otimista) take each driver's low / base / high (the pessimistic one with the
HIGH cost of capital). The Monte Carlo [S] draws each driver from a triangular distribution (low, base, high),
independently, and reports the distribution of the value per share and the share of draws above today's price.
Reverse DCF: the constant 5-year growth that makes the base-case value equal to today's price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

YEARS = 5
N_SIM = 20_000
MIN_SPREAD = 0.01
DRIVERS = ("crescimento", "margem_fcl", "custo_capital", "crescimento_perpetuo")
DRIVER_PT = {"crescimento": "crescimento da receita (5 anos)", "margem_fcl": "margem de caixa livre no ano 5",
             "custo_capital": "custo de capital (retorno exigido)", "crescimento_perpetuo": "crescimento depois do ano 5",
             "payout": "parcela do lucro distribuída"}
BOUNDS = {"crescimento": (-0.5, 1.0), "margem_fcl": (-0.5, 0.8), "custo_capital": (0.03, 0.40),
          "crescimento_perpetuo": (-0.02, 0.08), "payout": (0.0, 1.0)}


@dataclass(frozen=True)
class Base:
    """Published starting point [H]."""
    receita: float                 # last fiscal year (or net income for financials)
    margem_atual: float | None     # today's free-cash-flow margin (None for financials)
    acoes: float                   # diluted shares
    preco: float                   # last close, same currency
    financeira: bool = False


def _check(p: dict[str, tuple[float, float, float]]) -> None:
    for k, v in p.items():
        lo_b, hi_b = BOUNDS[k]
        if len(v) != 3 or not all(np.isfinite(x) for x in v):
            raise ValueError(f"{DRIVER_PT[k]}: informe pessimista, base e otimista")
        if not (lo_b <= min(v) and max(v) <= hi_b):
            raise ValueError(f"{DRIVER_PT[k]}: valores entre {lo_b:.0%} e {hi_b:.0%}")
        if not v[0] <= v[1] <= v[2] and k != "custo_capital":
            raise ValueError(f"{DRIVER_PT[k]}: o pessimista não pode ser maior que a base nem a base maior que o otimista")
        if k == "custo_capital" and not v[0] <= v[1] <= v[2]:
            raise ValueError(f"{DRIVER_PT[k]}: informe do menor para o maior (o pessimista usa o maior)")
    if p["custo_capital"][0] - p["crescimento_perpetuo"][2] < MIN_SPREAD - 1e-12:
        raise ValueError("o custo de capital precisa ficar ao menos 1 ponto percentual acima do crescimento perpétuo "
                         "em todos os cenários (senão o valor da perpetuidade não existe)")


def value_per_share(b: Base, g: np.ndarray | float, m: np.ndarray | float, ke: np.ndarray | float,
                    g_inf: np.ndarray | float, payout: np.ndarray | float = 1.0) -> np.ndarray:
    """Vectorised: every argument may be an array of draws."""
    g, m, ke, g_inf, payout = (np.asarray(x, float) for x in (g, m, ke, g_inf, payout))
    t = np.arange(1, YEARS + 1).reshape((-1,) + (1,) * g.ndim)
    level = b.receita * (1.0 + g) ** t
    if b.financeira:
        flows = level * payout
        last = level[-1] * (1.0 + g_inf) * payout
    else:
        m0 = b.margem_atual if b.margem_atual is not None else m
        margin = m0 + (m - m0) * t / YEARS
        flows = level * margin
        last = level[-1] * (1.0 + g_inf) * m
    disc = (1.0 + ke) ** t
    pv = (flows / disc).sum(axis=0)
    tv = last / (ke - g_inf) / disc[-1]
    return (pv + tv) / b.acoes


def projection(b: Base, g: float, m: float, ke: float, g_inf: float, payout: float = 1.0) -> list[dict[str, float]]:
    rows = []
    level = b.receita
    for t in range(1, YEARS + 1):
        level *= 1.0 + g
        margin = payout if b.financeira else (b.margem_atual if b.margem_atual is not None else m) + (
            m - (b.margem_atual if b.margem_atual is not None else m)) * t / YEARS
        flow = level * margin
        rows.append({"ano": t, "base": level, "margem": margin, "fluxo": flow, "valor_presente": flow / (1 + ke) ** t})
    return rows


def implied_growth(b: Base, m: float, ke: float, g_inf: float, payout: float = 1.0) -> float | None:
    """Constant 5-year growth that makes the value per share equal today's price (None when outside -50%..+100%)."""
    f = lambda g: float(value_per_share(b, g, m, ke, g_inf, payout)) - b.preco
    lo, hi = BOUNDS["crescimento"]
    try:
        if f(lo) * f(hi) > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-6))
    except (ValueError, ZeroDivisionError):
        return None


def simulate(b: Base, p: dict[str, tuple[float, float, float]], payout: tuple[float, float, float] | None = None,
             n: int = N_SIM, seed: int = 7) -> dict[str, object]:
    _check(p)
    if b.financeira and payout is None:
        raise ValueError("bancos e seguradoras: informe a parcela do lucro distribuída (payout)")
    rng = np.random.default_rng(seed)

    def tri(v: tuple[float, float, float]) -> np.ndarray:
        lo, mode, hi = v
        return np.full(n, mode) if hi - lo < 1e-12 else rng.triangular(lo, mode, hi, n)

    g, m, ke, gi = (tri(p[k]) for k in DRIVERS)
    po = tri(payout) if payout else np.ones(n)
    gi = np.minimum(gi, ke - MIN_SPREAD)
    vals = value_per_share(b, g, m, ke, gi, po)
    base_args = {k: p[k][1] for k in DRIVERS}
    pb = payout[1] if payout else 1.0
    scen = {
        "pessimista": value_per_share(b, p["crescimento"][0], p["margem_fcl"][0], p["custo_capital"][2],
                                      min(p["crescimento_perpetuo"][0], p["custo_capital"][2] - MIN_SPREAD), payout[0] if payout else 1.0),
        "base": value_per_share(b, base_args["crescimento"], base_args["margem_fcl"], base_args["custo_capital"],
                                base_args["crescimento_perpetuo"], pb),
        "otimista": value_per_share(b, p["crescimento"][2], p["margem_fcl"][2], p["custo_capital"][0],
                                    min(p["crescimento_perpetuo"][2], p["custo_capital"][0] - MIN_SPREAD), payout[2] if payout else 1.0),
    }
    tornado = []
    for k in (*DRIVERS, *(("payout",) if payout else ())):
        if k == "margem_fcl" and b.financeira:
            continue
        rng_v = payout if k == "payout" else p[k]
        out = []
        for v in (rng_v[0], rng_v[2]):
            args = {**base_args, k: v} if k != "payout" else dict(base_args)
            po_v = v if k == "payout" else pb
            gi_v = min(args["crescimento_perpetuo"], args["custo_capital"] - MIN_SPREAD)
            out.append(float(value_per_share(b, args["crescimento"], args["margem_fcl"], args["custo_capital"], gi_v, po_v)))
        tornado.append({"premissa": DRIVER_PT[k], "chave": k, "com_valor_baixo": out[0], "com_valor_alto": out[1],
                        "amplitude": abs(out[1] - out[0])})
    tornado.sort(key=lambda x: -x["amplitude"])
    qs = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
    lo_h, hi_h = np.percentile(vals, [1, 99])
    hist, edges = np.histogram(np.clip(vals, lo_h, hi_h), bins=30, range=(lo_h, hi_h))
    base_v = float(scen["base"])
    return {
        "valor_por_acao": {"pessimista": float(scen["pessimista"]), "base": base_v, "otimista": float(scen["otimista"])},
        "preco": b.preco,
        "potencial": {k: float(v) / b.preco - 1.0 for k, v in scen.items()},
        "simulacao": {"n": n, "p_acima_do_preco": float(np.mean(vals > b.preco)),
                      "erro_padrao": float(np.sqrt(np.mean(vals > b.preco) * (1 - np.mean(vals > b.preco)) / n)),
                      "percentis": {f"p{q}": float(v) for q, v in zip((5, 10, 25, 50, 75, 90, 95), qs, strict=True)},
                      "histograma": {"bordas": edges.tolist(), "contagens": hist.tolist()}},
        "sensibilidade": tornado,
        "crescimento_implicito": implied_growth(b, base_args["margem_fcl"], base_args["custo_capital"],
                                                base_args["crescimento_perpetuo"], pb),
        "projecao_base": projection(b, base_args["crescimento"], base_args["margem_fcl"], base_args["custo_capital"],
                                    base_args["crescimento_perpetuo"], pb),
        "parcela_perpetuidade": _tv_share(b, base_args, pb),
    }


def _tv_share(b: Base, a: dict[str, float], payout: float) -> float | None:
    total = float(value_per_share(b, a["crescimento"], a["margem_fcl"], a["custo_capital"], a["crescimento_perpetuo"], payout))
    rows = projection(b, a["crescimento"], a["margem_fcl"], a["custo_capital"], a["crescimento_perpetuo"], payout)
    pv = sum(r["valor_presente"] for r in rows) / b.acoes
    return float(1.0 - pv / total) if total > 0 else None
