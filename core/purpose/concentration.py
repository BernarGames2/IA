"""Diversification measured by risk, not by the number of tickers.

* effective number of positions: 1 / sum(w^2);
* effective number of independent bets = (diversification ratio)^2, DR = sum(w_i sigma_i) / sigma_p
  (Choueifaty, Froidure and Reynier 2013): 1 when everything moves together, N for N equal, independent
  positions — ten tickers that rise and fall together are close to one bet. (The principal-component
  version of Meucci 2009 is not used: with similar volatilities its components are not unique and the
  count becomes arbitrary.)
* overlap: pairs of held positions whose daily returns correlate >= 0.70. For an index ETF and a
  stock of the same market this is the visible part of the overlap: ETF compositions are not in the
  data sources, so the overlap is estimated from the co-movement and never claimed from holdings;
* hidden concentration: risk share of the largest correlation group, and exposure by currency,
  class, sector and market (weights) next to the share of the risk each one carries. The currency is
  the one the value follows (an S&P 500 ETF quoted in reais counts as dollars), not the quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from core.purpose.catalog import CatalogAsset
from core.purpose.personalization import traits

OVERLAP_CORR = 0.70
GROUP_RISK_WARN = 0.50


@dataclass
class ConcentrationReport:
    n_positions: int
    effective_n: float
    enb: float
    largest: tuple[str, float]
    risk_share: dict[str, float]
    overlaps: list[tuple[str, str, float]]
    groups: list[dict[str, object]]
    exposure: dict[str, dict[str, float]]
    risk_exposure: dict[str, dict[str, float]]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"n_posicoes": self.n_positions, "n_efetivo_pesos": round(self.effective_n, 2),
                "apostas_independentes_enb": round(self.enb, 2),
                "maior_posicao": {"ticker": self.largest[0], "peso": self.largest[1]},
                "participacao_no_risco": self.risk_share,
                "sobreposicoes": [{"a": a, "b": b, "correlacao": c} for a, b, c in self.overlaps],
                "grupos_que_andam_juntos": self.groups, "exposicao": self.exposure,
                "exposicao_ao_risco": self.risk_exposure, "alertas": self.warnings}


def effective_bets(w: np.ndarray, cov: np.ndarray) -> float:
    """(sum w_i sigma_i / portfolio sigma)^2: the number of independent, equally sized bets with the same
    diversification (Choueifaty et al. 2013)."""
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    port = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if port <= 0:
        return 1.0
    return float((w @ sd / port) ** 2)


def analyse(w: pd.Series, cov: pd.DataFrame, returns: pd.DataFrame, assets: dict[str, CatalogAsset]
            ) -> ConcentrationReport:
    """``w``: final weights; ``cov``: annual covariance; ``returns``: daily returns of the analysed assets."""
    held = w[w > 1e-9]
    names = list(held.index)
    wv = held.to_numpy() / held.sum()
    c = cov.loc[names, names].to_numpy()
    var = float(wv @ c @ wv)
    rc = pd.Series(wv * (c @ wv) / var if var > 0 else np.zeros(len(wv)), index=names)
    corr = returns[names].corr().fillna(0.0) if len(names) > 1 else pd.DataFrame([[1.0]], index=names, columns=names)
    overlaps = sorted(((a, b, round(float(corr.at[a, b]), 2)) for i, a in enumerate(names) for b in names[i + 1:]
                       if corr.at[a, b] >= OVERLAP_CORR), key=lambda t: -t[2])
    groups: list[dict[str, object]] = []
    if len(names) > 1:
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        lab = fcluster(linkage(squareform(dist, checks=False), "average"),
                       t=float(np.sqrt(0.5 * (1.0 - OVERLAP_CORR))), criterion="distance")
        for g in sorted(set(lab)):
            members = [s for s, x in zip(names, lab, strict=True) if x == g]
            if len(members) > 1:
                sub = corr.loc[members, members].to_numpy()
                avg = float(sub[np.triu_indices(len(members), 1)].mean())
                groups.append({"ativos": members, "peso": float(held[members].sum() / held.sum()),
                               "participacao_no_risco": float(rc[members].sum()), "correlacao_media": round(avg, 2)})
        groups.sort(key=lambda g: -float(g["participacao_no_risco"]))   # type: ignore[arg-type]

    def by(key) -> tuple[dict[str, float], dict[str, float]]:
        lab = pd.Series({s: key(s) for s in names})
        return ({k: round(float(v), 4) for k, v in (held / held.sum()).groupby(lab).sum().items()},
                {k: round(float(v), 4) for k, v in rc.groupby(lab).sum().items()})

    exposure, risk_exposure = {}, {}
    for dim, key in (("moeda", lambda s: traits(assets[s]).exposure), ("classe", lambda s: assets[s].macro_class),
                     ("setor", lambda s: assets[s].sector), ("mercado", lambda s: traits(assets[s]).market)):
        exposure[dim], risk_exposure[dim] = by(key)
    eff_n = float(1.0 / (wv ** 2).sum())
    enb = effective_bets(wv, c)
    warnings = []
    if groups and float(groups[0]["participacao_no_risco"]) > GROUP_RISK_WARN:   # type: ignore[arg-type]
        g = groups[0]
        warnings.append(f"{', '.join(g['ativos'])} sobem e caem juntos (correlação média {g['correlacao_media']:.2f}): "  # type: ignore[arg-type]
                        f"somam {g['participacao_no_risco']:.0%} do risco com {g['peso']:.0%} do dinheiro.")
    if len(names) >= 5 and enb < 0.4 * len(names):
        warnings.append(f"{len(names)} posições, mas o risco se comporta como cerca de {enb:.1f} apostas independentes: "
                        "mais tickers não significam, por si só, mais diversificação.")
    top = rc.idxmax() if len(rc) else ""
    if len(rc) and rc.max() > 0.40 and len(names) > 2:
        warnings.append(f"{top} sozinho responde por {rc.max():.0%} do risco da carteira (peso {held[top] / held.sum():.0%}).")
    for a, b, cr in overlaps:
        ta, tb = traits(assets[a]), traits(assets[b])
        etf = {assets[a].macro_class, assets[b].macro_class} & {"etf_br", "etf_eua"}
        if etf and ta.market == tb.market and assets[a].asset_class == assets[b].asset_class == "equity":
            warnings.append(f"{a} e {b} se sobrepõem (correlação {cr:.2f}): parte do que um compra o outro já tem. "
                            "A composição dos ETFs não está nas fontes; a sobreposição foi estimada pelo movimento "
                            "conjunto dos preços.")
    return ConcentrationReport(len(names), eff_n, enb, (str(held.idxmax()), float(held.max() / held.sum())),
                               {k: round(float(v), 4) for k, v in rc.sort_values(ascending=False).items()},
                               overlaps, groups, exposure, risk_exposure, warnings)
