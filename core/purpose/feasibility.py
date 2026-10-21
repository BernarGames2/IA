"""From target weights to what can actually be bought.

Lot rules (documented; brokers differ — the investor can change the US rule):
* B3 shares and units: 1 share (fractional market — the ticker with an "F", e.g. PETR4F);
* B3 FIIs and ETFs: 1 unit;
* US stocks and ETFs: 1 whole share, or fractions of 0.0001 when the investor's broker allows them;
* crypto bought directly (BTC-USD): 0.00000001 — a crypto ETF listed on the B3 trades in whole units;
* daily-liquidity cash (the CDI model): any amount (R$ 0.01). Only that instrument is "money": exchange-
  listed cash-like ETFs (LFTS11, SHY) trade in units at their price like any ETF.
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
3. ``contribution_plan`` — each periodic contribution goes to the positions furthest below the target,
   levelling their gaps (water filling: the largest gaps are reduced together, so new money is not all put
   into one position when holdings are far from the target), with at least the minimum position per
   purchase (a position whose share would be smaller passes it to the others), then rounds like ``plan``
   (one more purchase for the position furthest below its gap when at least half of that purchase is
   missing); positions not bought this time come in later contributions. What is left goes to the cash instrument, and cash held ABOVE
   its target (left over from earlier contributions, or already held) can complete a purchase: a lot that
   costs more than one contribution is bought once enough has been set aside (the plan says how much is
   redeemed from cash). The gaps are measured on everything the investor holds, inside the target or not.
   A contribution is never confused with total wealth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.purpose.catalog import CatalogAsset
from core.purpose.personalization import traits

DEFAULT_MIN_POSITION = 100.0     # in the investor's currency


def _money(x: float, ccy: str = "") -> str:
    v = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{ {'BRL': 'R$', 'USD': 'US$', 'EUR': '€'}.get(ccy, ccy)} {v}".strip()


def is_money(a: CatalogAsset) -> bool:
    """Daily-liquidity cash in the investor's currency (the CDI model, or its synthetic stand-in): applied and
    redeemed in any amount and measured in money. Exchange-listed cash-like ETFs are not money."""
    return traits(a).market == "CAIXA"


@dataclass
class ExecutionPlan:
    amount: float
    rows: list[dict[str, object]]
    cash_left: float
    skipped: dict[str, str] = field(default_factory=dict)
    tracking_l1: float = 0.0          # sum |executed weight - target weight| (+ uninvested share)
    viable: bool = True
    notes: list[str] = field(default_factory=list)
    redeemed: float = 0.0             # taken from cash held above its target to complete the purchases
    redeemed_from: str = ""

    def holdings_after(self, current: dict[str, float]) -> dict[str, float]:
        """What the investor holds after this plan: purchases added, the cash redemption subtracted."""
        out = dict(current)
        for r in self.rows:
            out[str(r["ticker"])] = out.get(str(r["ticker"]), 0.0) + float(r["valor"])
        if self.redeemed > 0:
            out[self.redeemed_from] = out.get(self.redeemed_from, 0.0) - self.redeemed
        return out

    def executed_weights(self) -> pd.Series:
        tot = sum(float(r["valor"]) for r in self.rows) + self.cash_left
        return pd.Series({str(r["ticker"]): float(r["valor"]) / tot for r in self.rows}) if tot > 0 else pd.Series()

    def as_dict(self) -> dict[str, object]:
        return {"valor": self.amount, "posicoes": self.rows, "caixa_nao_investido": self.cash_left,
                "fora_desta_compra": self.skipped, "distancia_ao_alvo": self.tracking_l1, "viavel": self.viable,
                "notas": self.notes, "resgate_do_caixa": self.redeemed, "resgate_de": self.redeemed_from}


def lot_size(a: CatalogAsset, us_fractional: bool = False) -> tuple[float, str]:
    tr = traits(a)
    if is_money(a):
        return 0.01, "qualquer valor (aplicação com liquidez diária)"
    if tr.market == "B3":           # listed on the B3 (including crypto, gold and cash-like ETFs): whole units
        if a.macro_class in ("fii", "etf_br", "renda_fixa_br", "ouro", "cripto"):
            return 1.0, "lote de 1 cota"
        if a.ticker.endswith(".SA"):
            return 1.0, f"lote de 1 ação no mercado fracionário (código {a.ticker.removesuffix('.SA')}F)"
        return 1.0, "lote de 1 ação"
    if tr.crypto:                   # crypto bought directly
        return 1e-8, "frações de 0,00000001"
    return (1e-4, "frações de ação (sua corretora permite)") if us_fractional else (1.0, "ação inteira")


def unit_price(a: CatalogAsset, price: float) -> float:
    """Money is measured in money, everything else in units of the last price."""
    return 1.0 if is_money(a) else float(price)


def min_weight(a: CatalogAsset, price: float, amount: float, min_position: float, us_fractional: bool) -> float:
    """Smallest weight at which the asset can be held as a sensible position with ``amount``."""
    lot, _ = lot_size(a, us_fractional)
    smallest = min_position if is_money(a) else max(min_position, unit_price(a, price) * lot)
    return smallest / amount if amount > 0 else math.inf


def executable_universe(assets: dict[str, CatalogAsset], prices: pd.Series, upper: dict[str, float], amount: float,
                        min_position: float, us_fractional: bool, ccy: str = "") -> tuple[list[str], dict[str, str]]:
    keep, out = [], {}
    for s, a in assets.items():
        mw = min_weight(a, float(prices[s]), amount, min_position, us_fractional)
        cap = upper.get(s, 1.0)
        if mw > cap + 1e-12:
            lot, _ = lot_size(a, us_fractional)
            lot_cost = unit_price(a, float(prices[s])) * lot
            room = f"do peso máximo de {cap:.0%} sobre {_money(amount, ccy)} ({_money(cap * amount, ccy)})"
            out[s] = (f"1 {'ação' if a.macro_class in ('acoes_br', 'acoes_eua') else 'cota'} custa "
                      f"{_money(lot_cost, ccy)}: acima {room}" if lot_cost > cap * amount + 1e-9 else
                      f"a compra mínima de {_money(min_position, ccy)} passa {room}")
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
    sink = next((s for s in w.index if is_money(assets[s])), None)
    if sink is not None and left > 0.005:
        add = math.floor(left / steps[sink]) * steps[sink]
        qty[sink] += add
        val[sink] += add
        left -= add
    zero = {s: "o valor-alvo não compra um lote inteiro" for s in w.index if qty[s] <= 0}
    total = sum(val.values()) + left
    rows = [{"ticker": s, "quantidade": round(qty[s], 8), "preco": units[s], "valor": round(val[s], 2),
             "peso_alvo": float(w[s]), "peso_executado": val[s] / total, "regra_de_lote": rules[s],
             "em_dinheiro": is_money(assets[s])}
            for s in sorted(w.index, key=lambda x: -val[x]) if qty[s] > 0]
    tracking = sum(abs(val[s] / total - float(w[s])) for s in w.index) + (left / total if total else 0.0)
    notes = []
    if left > 0.01:
        notes.append(f"sobra não investida: {_money(left)} (frações de lote)")
    return ExecutionPlan(amount, rows, round(left, 2), zero, round(tracking, 4), not zero, notes)


def _level(gaps: pd.Series, money: float) -> pd.Series:
    """Water filling: budgets b_i = max(0, gap_i - level) adding up to ``money`` (every gap in full when the
    money covers them all): the positions furthest below the target are raised together."""
    g = gaps.sort_values(ascending=False)
    if g.empty or money >= float(g.sum()) - 1e-9:
        return g.copy()
    v = g.to_numpy(dtype=float)
    c = np.cumsum(v)
    level = 0.0
    for k in range(1, len(v) + 1):
        level = (c[k - 1] - money) / k
        if k == len(v) or level >= v[k]:
            break
    return (g - level).clip(lower=0.0)


def contribution_plan(weights: pd.Series, current_values: dict[str, float], prices: pd.Series,
                      assets: dict[str, CatalogAsset], amount: float, min_position: float = DEFAULT_MIN_POSITION,
                      us_fractional: bool = False, more_coming: bool = True) -> ExecutionPlan:
    """What to buy with ONE contribution (or one amount of new money): the positions furthest below the
    target, levelling their gaps, each purchase of at least ``min_position`` (or one lot); cash held above its
    target completes purchases the contribution alone cannot make. ``current_values``: everything the investor holds, inside the
    target or not; ``more_coming``: whether later contributions will complete the portfolio (wording only)."""
    w = weights[weights > 1e-9].astype(float)
    w = w / w.sum()
    sink = next((s for s in w.index if is_money(assets[s])), None)
    held = sum(float(v) for v in current_values.values())
    total = held + amount
    reserve = max(0.0, current_values.get(sink, 0.0) - total * float(w[sink])) if sink is not None else 0.0
    if w.empty or (amount <= 0 and reserve <= 0.005):
        return ExecutionPlan(amount, [], max(amount, 0.0), viable=False, notes=["sem valor para investir"])
    later_txt = "entra nos próximos aportes" if more_coming else "não coube neste valor"
    gaps = pd.Series({s: total * float(w[s]) - current_values.get(s, 0.0) for s in w.index})
    gaps = gaps[gaps > 0].sort_values(ascending=False)
    units = {s: unit_price(assets[s], float(prices[s])) for s in w.index}
    steps = {s: lot_size(assets[s], us_fractional)[0] for s in w.index}
    rules = {s: lot_size(assets[s], us_fractional)[1] for s in w.index}
    left = amount + reserve
    buy: dict[str, tuple[float, float]] = {}
    later: dict[str, str] = {}

    def smallest(s: str) -> float:
        return min_position if s == sink else max(min_position, units[s] * steps[s])

    # a position whose levelled share would be smaller than one sensible purchase passes it to the others
    active = gaps.copy()
    while True:
        budget = _level(active, left)
        small = [s for s in budget.index if 0.0 < budget[s] < smallest(s) - 1e-9]
        if not small:
            break
        s_ = min(small, key=lambda x: budget[x])
        later[s_] = f"{later_txt} (a parte dele agora seria menor que uma compra mínima de {_money(smallest(s_))})"
        active = active.drop(s_)
    for s in budget.sort_values(ascending=False).index:
        if budget[s] <= 1e-9:
            later.setdefault(s, f"{later_txt} (o dinheiro foi antes para as posições mais abaixo do alvo)")
            continue
        q = math.floor(budget[s] / (units[s] * steps[s]) + 1e-9) * steps[s]
        cost = q * units[s]
        if q <= 0 or cost > left + 1e-9:
            later[s] = f"{later_txt} (1 lote custa {_money(units[s] * steps[s])}, mais que o dinheiro que sobrou)"
            continue
        buy[s] = (q, cost)
        left -= cost
    while True:   # rounding, as in ``plan``: one more purchase for the position furthest below its gap
        best, best_short, best_q = None, 0.0, 0.0
        for s in gaps.index:
            if s == sink:
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
    redeemed = 0.0
    if sink is not None:
        # what is left goes back to cash; the cash above target that was used is a redemption
        _, c0 = buy.pop(sink, (0.0, 0.0))
        net = c0 + left - reserve
        left = 0.0
        if net > 0.005:
            add = math.floor(net / steps[sink] + 1e-9) * steps[sink]
            buy[sink] = (add, add)
            left = net - add
            later.pop(sink, None)
        elif net < -0.005:
            redeemed = -net
    rows = [{"ticker": s, "quantidade": round(q, 8), "preco": units[s], "valor": round(c, 2),
             "peso_alvo": float(w[s]), "peso_apos_compra": (current_values.get(s, 0.0) + c) / total,
             "regra_de_lote": rules[s], "em_dinheiro": is_money(assets[s])}
            for s, (q, c) in sorted(buy.items(), key=lambda kv: -kv[1][1])]
    notes = ([f"{len(later)} ativo(s) da carteira-alvo {'entram nos próximos aportes' if more_coming else 'não couberam neste valor'}."]
             if later else [])
    if redeemed > 0:
        notes.append(f"inclui {_money(redeemed)} resgatados do caixa, que está acima do alvo (dinheiro guardado de "
                     "aportes anteriores ou já aplicado), para completar as compras")
    if left > 0.01:
        notes.append(f"sobra deste aporte: {_money(left)}")
    return ExecutionPlan(amount, rows, round(left, 2), later, 0.0, True, notes, round(redeemed, 2),
                         sink if redeemed > 0 and sink is not None else "")
