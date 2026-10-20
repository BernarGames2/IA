"""Stage B of the universe pipeline: measure every downloaded asset and pick the candidates the
optimiser will weigh.

Metrics are historical [H], over each asset's own history up to today (the decision date), in the
investor's currency — one recent listing does not shorten the history used for every other asset:
* annual volatility, maximum drawdown and 12-1 month momentum (12-month return without the last month);
* Sharpe ratio shrunk toward the mean of its class: SR_s = m + k (SR - m), k = tau^2 / (tau^2 + se^2),
  se = 1/sqrt(years), tau = 0.30 — a few good years do not make an asset look great;
* liquidity: average daily traded value over 63 sessions (cash instruments: daily liquidity).
Every metric becomes a z-score WITHIN the asset's class (fixed income is compared with fixed income)
and the profile sets their weights (PROFILE_WEIGHTS; "renda" adds the income flag). Then:
1. duplicates: two instruments of the same class whose daily returns correlate >= 0.95 carry the same
   exposure (two S&P 500 ETFs, an ETF in reais and its dollar twin) — the better-scored stays;
2. correlation groups (average linkage, correlation >= 0.70): at most a few members of each group
   reach the optimiser, so the list is not many tickers for one risk;
3. coverage first: classes the investor's limits require (fixed income minimum, cash, income), then the
   investor's themes (up to 30%), then the best scores.
Every asset gets a decision and a plain-language reason. The backtest that follows compares
weighting methods on this universe; because the universe is chosen with data up to today, that
backtest is optimistic about what a past investor would have obtained (selection bias; disclosed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.purpose.catalog import CatalogAsset
from core.purpose.personalization import traits
from models.investor import RiskProfile

TAU_SHARPE = 0.30
DUPLICATE_CORR = 0.95
GROUP_CORR = 0.70
THEME_SHARE = 0.30
PROFILE_WEIGHTS = {   # weights of the within-class z-scores (vol enters with a minus sign)
    RiskProfile.CONSERVADOR: {"sharpe": 0.20, "vol": -0.35, "mdd": 0.30, "mom": 0.00, "liq": 0.15},
    RiskProfile.MODERADO: {"sharpe": 0.40, "vol": -0.20, "mdd": 0.20, "mom": 0.05, "liq": 0.15},
    RiskProfile.ARROJADO: {"sharpe": 0.45, "vol": -0.05, "mdd": 0.10, "mom": 0.20, "liq": 0.20},
}
INCOME_WEIGHT = 0.25


@dataclass
class ScreenResult:
    selected: list[str]
    table: pd.DataFrame                      # one row per measured asset, with decision and reason
    duplicates: dict[str, str] = field(default_factory=dict)        # dropped -> kept
    groups: dict[str, int] = field(default_factory=dict)            # ticker -> correlation group
    notes: list[str] = field(default_factory=list)

    def alternatives(self, ticker: str, n: int = 2) -> list[str]:
        """Best-scored assets of the same correlation group or class that were not selected."""
        t = self.table
        if ticker not in t.index:
            return []
        same = t[((t["grupo"] == t.at[ticker, "grupo"]) | (t["classe"] == t.at[ticker, "classe"]))
                 & ~t.index.isin(self.selected)]
        return list(same.sort_values("pontuacao", ascending=False).index[:n])


def _zscore(x: pd.Series, by: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=x.index)
    for idx in by.groupby(by).groups.values():
        v = x.loc[idx].astype(float)
        finite = v[np.isfinite(v)]
        if len(finite) < 2 or finite.std(ddof=0) <= 1e-12:
            continue
        z = (v - finite.mean()) / finite.std(ddof=0)
        out.loc[idx] = z.clip(-3, 3).fillna(0.0)
    return out


def daily_returns(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Each asset's returns on its own trading days, then side by side (a holiday in one market
    does not erase the other market's returns; a 2-day return follows a holiday)."""
    rets = {s: x.dropna().pct_change(fill_method=None).iloc[1:] for s, x in series.items()}
    return pd.concat(rets, axis=1, sort=True)


def measure(series: dict[str, pd.Series], volumes: pd.DataFrame | None, assets: dict[str, CatalogAsset], rf: float,
            periods: int = 252) -> pd.DataFrame:
    """Historical metrics [H] per asset, each over its own history (prices already in the investor's
    currency)."""
    rows = {}
    for s, x in series.items():
        x = x.dropna()
        r = x.pct_change(fill_method=None).iloc[1:]
        vol = float(r.std(ddof=1) * np.sqrt(periods))
        mean = float(r.mean() * periods)
        wealth = (1 + r).cumprod()
        liq = np.nan
        if traits(assets[s]).cash:
            liq = np.inf                                       # daily liquidity by construction
        elif volumes is not None and s in volumes.columns:
            v = (volumes[s] * x.reindex(volumes.index)).dropna()
            liq = float(v.iloc[-63:].mean()) if len(v) else np.nan
        rows[s] = {"retorno_anual_H": mean, "volatilidade_H": vol,
                   "queda_maxima_H": float((wealth / wealth.cummax() - 1).min()),
                   "sharpe_H": (mean - rf) / vol if vol > 1e-9 else 0.0,
                   "momento_12_1": float(x.iloc[-22] / x.iloc[-253] - 1) if len(x) > 253 else np.nan,
                   "liquidez_diaria": liq, "anos": len(r) / periods}
    m = pd.DataFrame.from_dict(rows, orient="index")
    m.insert(0, "classe", [assets[s].asset_class for s in m.index])
    m.insert(1, "mercado", [traits(assets[s]).market for s in m.index])
    m.insert(2, "setor", [assets[s].sector for s in m.index])
    m["renda"] = [traits(assets[s]).income for s in m.index]
    m_c = m["sharpe_H"].groupby(m["classe"]).transform("mean")
    k = TAU_SHARPE ** 2 / (TAU_SHARPE ** 2 + 1.0 / m["anos"].clip(lower=1e-9))
    m.insert(m.columns.get_loc("sharpe_H") + 1, "sharpe_encolhido", m_c + k * (m["sharpe_H"] - m_c))
    return m


def screen(series: dict[str, pd.Series], volumes: pd.DataFrame | None, assets: dict[str, CatalogAsset],
           profile: RiskProfile, objective: str, rf: float, cap: int,
           required: list[tuple[str, list[str], int]] | None = None, themed: set[str] | None = None,
           keep: dict[str, str] | None = None, periods: int = 252) -> ScreenResult:
    """``series``: each asset's price series in the investor's currency; ``required``: (label, members,
    minimum count) groups the investor's limits need; ``keep``: assets that must reach the optimiser
    regardless of score, with the reason (the investor's holdings and previous recommendation)."""
    m = measure(series, volumes, assets, rf, periods)
    w = dict(PROFILE_WEIGHTS[profile])
    liq = m["liquidez_diaria"].astype(float)
    finite = liq[np.isfinite(liq)]
    top = float(finite.max()) if len(finite) else 0.0
    liq_log = np.log1p(liq.where(~np.isinf(liq), top).fillna(0.0))      # cash = most liquid; unknown = 0
    z = pd.DataFrame({
        "sharpe": _zscore(m["sharpe_encolhido"], m["classe"]), "vol": _zscore(m["volatilidade_H"], m["classe"]),
        "mdd": _zscore(m["queda_maxima_H"], m["classe"]), "mom": _zscore(m["momento_12_1"].fillna(0.0), m["classe"]),
        "liq": _zscore(liq_log, m["classe"]) if len(finite) else pd.Series(0.0, index=m.index)})
    score = sum(w[k] * z[k] for k in w)
    if objective == "renda":
        score = score + INCOME_WEIGHT * m["renda"].astype(float) * 2.0
    m["pontuacao"] = score.round(4)
    corr = daily_returns(series).corr(min_periods=126).reindex(index=m.index, columns=m.index).fillna(0.0)
    order = list(m.sort_values("pontuacao", ascending=False).index)
    decision: dict[str, tuple[str, str]] = {}
    keep = dict(keep or {})

    # 1. duplicates: same class, same daily behaviour
    dup: dict[str, str] = {}
    for i, a in enumerate(order):
        if a in dup or a in keep:
            continue
        for b in order[i + 1:]:
            if b in dup or b in keep or m.at[a, "classe"] != m.at[b, "classe"]:
                continue
            if corr.at[a, b] >= DUPLICATE_CORR:
                dup[b] = a
                decision[b] = ("descartado", (f"mesma exposição que {a} (correlação diária {corr.at[a, b]:.2f}); "
                                              f"{a} tem pontuação maior"))
    # 2. correlation groups
    names = list(m.index)
    if len(names) > 1:
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.loc[names, names].to_numpy()), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                       t=float(np.sqrt(0.5 * (1.0 - GROUP_CORR))), criterion="distance")
        group = {s: int(g) for s, g in zip(names, lab, strict=True)}
    else:
        group = {s: 1 for s in names}
    m["grupo"] = pd.Series(group)
    quota = max(2, cap // 5)
    chosen: list[str] = []
    per_group: dict[int, int] = {}

    def take(s: str, why: str, force: bool = False) -> bool:
        if s in chosen or s in dup or len(chosen) >= cap:
            return False
        g = group[s]
        if not force and per_group.get(g, 0) >= quota:
            return False
        chosen.append(s)
        per_group[g] = per_group.get(g, 0) + 1
        decision[s] = ("selecionado", why)
        return True

    for s in order:
        if s in keep:
            take(s, keep[s], force=True)
    for label, members, k in required or []:
        cands = [s for s in order if s in members and s not in dup]
        got = sum(1 for s in chosen if s in members)
        for s in cands:
            if got >= k:
                break
            if take(s, f"necessário para cumprir {label}; melhor pontuação disponível nesse grupo", force=True):
                got += 1
    themed = set(themed or ())
    if themed:
        q = max(1, int(THEME_SHARE * cap))
        for s in order:
            if q <= 0:
                break
            if s in themed and take(s, "ligado a um tema que você escreveu; pontuação entre as melhores do tema"):
                q -= 1
    for s in order:
        if len(chosen) >= cap:
            break
        rank = order.index(s) + 1
        take(s, f"pontuação {m.at[s, 'pontuacao']:+.2f} (posição {rank} de {len(order)}) para o perfil "
                f"{profile.value}")
    for s in order:
        if s not in decision:
            g = group[s]
            leaders = [x for x in chosen if group[x] == g]
            if leaders and per_group.get(g, 0) >= quota:
                why = (f"grupo de ativos que sobem e caem juntos já representado por {', '.join(leaders[:3])} "
                       "(pontuação maior)")
            else:
                why = (f"pontuação {m.at[s, 'pontuacao']:+.2f} abaixo das escolhidas (posição {order.index(s) + 1} de "
                       f"{len(order)}) com o limite de {cap} candidatos")
            decision[s] = ("descartado", why)
    m["decisao"] = pd.Series({s: d for s, (d, _) in decision.items()})
    m["motivo"] = pd.Series({s: why for s, (_, why) in decision.items()})
    notes = [(f"{len(m)} ativos medidos; {len(dup)} duplicatas de exposição; "
              f"{len(set(group.values()))} grupos de correlação; {len(chosen)} candidatos para o otimizador")]
    return ScreenResult(chosen, m.sort_values("pontuacao", ascending=False), dup, group, notes)
