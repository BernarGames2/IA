"""From target weights to what can actually be bought.

Lot rules (documented; brokers differ — the investor can change the US rule):
* B3 shares and units: 1 share (fractional market — the ticker with an "F", e.g. PETR4F);
* B3 FIIs and ETFs: 1 unit;
* US stocks and ETFs: 1 whole share, or fractions of 0.0001 when the investor's broker allows them;
* crypto: 0.00000001; daily-liquidity cash (CDI): any amount (R$ 0.01).
Prices are the last close in the data, in the investor's currency (converted at that day's exchange
rate when the asset trades in another currency): delayed data, not a real-time quote.

Three steps, each reported:
1. ``executable_universe`` — before optimising, an asset whose smallest sensible position (one lot or
   the minimum position, whichever is larger) exceeds its maximum weight times the amount cannot be
   held without breaking the limits: it is left out with the reason. The engine then optimises with the
   limits intact and drops positions that end up below their minimum executable weight, re-optimising
   (so the "viable alternative" still respects the profile).
2. ``plan`` — largest remainder: whole lots each target value affords, then one more lot for the
   position furthest below its target while money remains; what is left goes to the divisible cash
   instrument when there is one, otherwise it is reported as uninvested cash.
3. ``contribution_plan`` — each periodic contribution fills the largest gaps to the target first, with
   at least the minimum position per purchase, then rounds like ``plan`` (one more purchase for the
   position furthest below its gap when at least half of that purchase is missing); positions not bought
   this time come in later contributions. A contribution is never confused with total wealth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from core.purpose.catalog import CatalogAsset
from core.purpose.personalization import traits

DEFAULT_MIN_POSITION = 100.0     # in the investor's currency


@dataclass
class ExecutionPlan:
    amount: float
    rows: list[dict[str, object]]
    cash_left: float
    skipped: dict[str, str] = field(default_factory=dict)
    tracking_l1: float = 0.0          # sum |executed weight - target weight| (+ uninvested share)
    viable: bool = True
    notes: list[str] = field(default_factory=list)

    def executed_weights(self) -> pd.Series:
        tot = sum(float(r["valor"]) for r in self.rows) + self.cash_left
        return pd.Series({str(r["ticker"]): float(r["valor"]) / tot for r in self.rows}) if tot > 0 else pd.Series()

    def as_dict(self) -> dict[str, object]:
        return {"valor": self.amount, "posicoes": self.rows, "caixa_nao_investido": self.cash_left,
                "fora_desta_compra": self.skipped, "distancia_ao_alvo": self.tracking_l1, "viavel": self.viable,
                "notas": self.notes}


def lot_size(a: CatalogAsset, us_fractional: bool = False) -> tuple[float, str]:
    tr = traits(a)
    if tr.cash:
        return 0.01, "qualquer valor (aplicação com liquidez diária)"
    if tr.crypto:
        return 1e-8, "frações de 0,00000001"
    if tr.market == "B3":
        if a.macro_class in ("fii", "etf_br", "renda_fixa_br", "ouro", "cripto"):
            return 1.0, "lote de 1 cota"
        if a.ticker.endswith(".SA"):
            return 1.0, f"lote de 1 ação no mercado fracionário (código {a.ticker.removesuffix('.SA')}F)"
        return 1.0, "lote de 1 ação"
    return (1e-4, "frações de ação (sua corretora permite)") if us_fractional else (1.0, "ação inteira")


def unit_price(a: CatalogAsset, price: float) -> float:
    """Cash is measured in money, everything else in units of the last price."""
    return 1.0 if traits(a).cash else float(price)


def min_weight(a: CatalogAsset, price: float, amount: float, min_position: float, us_fractional: bool) -> float:
    """Smallest weight at which the asset can be held as a sensible position with ``amount``."""
    lot, _ = lot_size(a, us_fractional)
    smallest = max(min_position, unit_price(a, price) * lot) if not traits(a).cash else min_position
    return smallest / amount if amount > 0 else math.inf


def executable_universe(assets: dict[str, CatalogAsset], prices: pd.Series, upper: dict[str, float], amount: float,
                        min_position: float, us_fractional: bool) -> tuple[list[str], dict[str, str]]:
    keep, out = [], {}
    for s, a in assets.items():
        mw = min_weight(a, float(prices[s]), amount, min_position, us_fractional)
        cap = upper.get(s, 1.0)
        if mw > cap + 1e-12:
            lot, _ = lot_size(a, us_fractional)
            out[s] = (f"1 {'ação' if a.macro_class in ('acoes_br', 'acoes_eua') else 'cota'} custa "
                      f"{unit_price(a, float(prices[s])) * lot:,.2f}: acima do peso máximo de {cap:.0%} sobre "
                      f"{amount:,.0f}").replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            keep.append(s)
    return keep, out


def _fill(targets: pd.Series, units: dict[str, float], steps: dict[str, float], money: float,
          order: list[str] | None = None) -> tuple[dict[str, float], dict[str, float], float]:
    """Largest remainder on whole lots. Returns quantities, values and money left."""
    qty = {s: math.floor(targets[s] / (units[s] * steps[s]) + 1e-9) * steps[s] for s in targets.index}
    val = {s: qty[s] * units[s] for s in targets.index}
    left = money - sum(val.values())
    while True:
        best, gap = None, 0.0
        for s in order or list(targets.index):
            cost = units[s] * steps[s]
            short = targets[s] - val[s]
            if cost <= left + 1e-9 and short >= 0.5 * cost and short > gap:
                best, gap = s, short
        if best is None:
            break
        qty[best] += steps[best]
        val[best] += units[best] * steps[best]
        left -= units[best] * steps[best]
    return qty, val, left


def plan(weights: pd.Series, prices: pd.Series, assets: dict[str, CatalogAsset], amount: float,
         us_fractional: bool = False) -> ExecutionPlan:
    """Quantities for an amount invested all at once in the target weights (already executable)."""
    w = weights[weights > 1e-9].astype(float)
    w = w / w.sum()
    if amount <= 0 or w.empty:
        return ExecutionPlan(amount, [], max(amount, 0.0), viable=False, notes=["sem valor para investir"])
    units = {s: unit_price(assets[s], float(prices[s])) for s in w.index}
    steps = {s: lot_size(assets[s], us_fractional)[0] for s in w.index}
    rules = {s: lot_size(assets[s], us_fractional)[1] for s in w.index}
    qty, val, left = _fill(w * amount, units, steps, amount)
    sink = next((s for s in w.index if traits(assets[s]).cash), None)
    if sink is not None and left > 0.005:
        add = math.floor(left / steps[sink]) * steps[sink]
        qty[sink] += add
        val[sink] += add
        left -= add
    zero = {s: "o valor-alvo não compra um lote inteiro" for s in w.index if qty[s] <= 0}
    total = sum(val.values()) + left
    rows = [{"ticker": s, "quantidade": round(qty[s], 8), "preco": units[s], "valor": round(val[s], 2),
             "peso_alvo": float(w[s]), "peso_executado": val[s] / total, "regra_de_lote": rules[s]}
            for s in sorted(w.index, key=lambda x: -val[x]) if qty[s] > 0]
    tracking = sum(abs(val[s] / total - float(w[s])) for s in w.index) + (left / total if total else 0.0)
    notes = []
    if left > 0.01:
        notes.append(f"sobra não investida: {left:,.2f} (frações de lote)".replace(",", "X").replace(".", ",")
                     .replace("X", "."))
    return ExecutionPlan(amount, rows, round(left, 2), zero, round(tracking, 4), not zero, notes)


def contribution_plan(weights: pd.Series, current_values: dict[str, float], prices: pd.Series,
                      assets: dict[str, CatalogAsset], amount: float, min_position: float = DEFAULT_MIN_POSITION,
                      us_fractional: bool = False) -> ExecutionPlan:
    """What to buy with ONE contribution: the largest gaps to the target first, each purchase of at least
    ``min_position`` (or one lot); the rest of the portfolio is completed by the next contributions."""
    w = weights[weights > 1e-9].astype(float)
    w = w / w.sum()
    if amount <= 0 or w.empty:
        return ExecutionPlan(amount, [], max(amount, 0.0), viable=False, notes=["sem valor para investir"])
    held = sum(current_values.get(s, 0.0) for s in w.index)
    gaps = pd.Series({s: (held + amount) * float(w[s]) - current_values.get(s, 0.0) for s in w.index})
    gaps = gaps[gaps > 0].sort_values(ascending=False)
    units = {s: unit_price(assets[s], float(prices[s])) for s in w.index}
    steps = {s: lot_size(assets[s], us_fractional)[0] for s in w.index}
    rules = {s: lot_size(assets[s], us_fractional)[1] for s in w.index}
    left = amount
    buy: dict[str, tuple[float, float]] = {}
    later: dict[str, str] = {}
    for s in gaps.index:
        smallest = max(min_position, units[s] * steps[s]) if not traits(assets[s]).cash else min(min_position, left)
        budget = min(float(gaps[s]), left)
        if budget + 1e-9 < smallest:
            later[s] = "entra nos próximos aportes (falta menos que uma compra mínima ou o dinheiro deste aporte acabou)"
            continue
        q = math.floor(max(budget, smallest) / (units[s] * steps[s]) + 1e-9) * steps[s]
        cost = q * units[s]
        if q <= 0 or cost > left + 1e-9:
            later[s] = "entra nos próximos aportes (1 lote custa mais que o que sobrou deste aporte)"
            continue
        buy[s] = (q, cost)
        left -= cost
    while True:   # rounding, as in ``plan``: one more purchase for the position furthest below its gap
        best, best_short, best_q = None, 0.0, 0.0
        for s in gaps.index:
            if traits(assets[s]).cash:
                continue
            lot = units[s] * steps[s]
            # a first purchase still has to reach the minimum position; later ones are one lot
            q = steps[s] if s in buy else math.ceil(max(min_position, lot) / lot - 1e-9) * steps[s]
            cost = q * units[s]
            short = float(gaps[s]) - buy.get(s, (0.0, 0.0))[1]
            if cost <= left + 1e-9 and short >= 0.5 * cost and short > best_short:
                best, best_short, best_q = s, short, q
        if best is None:
            break
        q0, c0 = buy.get(best, (0.0, 0.0))
        buy[best] = (q0 + best_q, c0 + best_q * units[best])
        later.pop(best, None)
        left -= best_q * units[best]
    sink = next((s for s in w.index if traits(assets[s]).cash), None)
    if sink is not None and left > 0.005:
        q0, c0 = buy.get(sink, (0.0, 0.0))
        add = math.floor(left / steps[sink]) * steps[sink]
        buy[sink] = (q0 + add, c0 + add)
        later.pop(sink, None)
        left -= add
    total = held + amount
    rows = [{"ticker": s, "quantidade": round(q, 8), "preco": units[s], "valor": round(c, 2),
             "peso_alvo": float(w[s]), "peso_apos_compra": (current_values.get(s, 0.0) + c) / total,
             "regra_de_lote": rules[s]} for s, (q, c) in sorted(buy.items(), key=lambda kv: -kv[1][1])]
    notes = [f"{len(later)} ativo(s) da carteira-alvo entram nos próximos aportes."] if later else []
    if left > 0.01:
        notes.append(f"sobra deste aporte: {left:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    return ExecutionPlan(amount, rows, round(left, 2), later, 0.0, True, notes)
