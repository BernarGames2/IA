"""Steps 7-8 — thematic alignment score per asset and its combination with the quant weights.

Score (0..1) of asset i = share of the investor's MAPPABLE themes (weighted by number of
explicit mentions, capped at 3) that the asset's catalogue tags serve. Value themes with
no direct sector counterpart (família/legado, disciplina, superação, comunidade,
aposentadoria) are reported but not mapped to sectors — mapping them would be inference.

Combination: the quant portfolio w_q is never discarded. An alignment portfolio w_a
maximises the portfolio alignment s'w subject to ALL linear constraints, a tracking-error
budget around w_q and the profile volatility cap; the final weights are the convex
combination (1-a) w_q + a w_a (a = user slider, default 0.30). Convexity guarantees the
final portfolio satisfies every linear constraint, its volatility is <= max(vol(w_q),
vol(w_a)) and its tracking error vs w_q is <= a * TE budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.optimization.constraints import verify_weights
from core.optimization.optimizers import SolverSettings, solve
from core.purpose.catalog import CatalogAsset
from core.purpose.themes import ThemeExtraction
from models.portfolio import PortfolioConstraints

THEME_TO_TAGS: dict[str, tuple[str, ...]] = {
    "saude": ("saude",),
    "educacao": ("educacao",),
    "sustentabilidade": ("energia_limpa", "sustentabilidade", "saneamento"),
    "tecnologia": ("tecnologia", "inovacao"),
    "moradia": ("imoveis",),
    "renda": ("renda_recorrente",),
    "seguranca": ("preservacao_capital",),
}
THEME_LABEL = {"saude": "saúde", "educacao": "educação", "sustentabilidade": "sustentabilidade",
               "tecnologia": "tecnologia/inovação", "moradia": "moradia/imóveis", "renda": "renda recorrente",
               "seguranca": "segurança/preservação", "aposentadoria": "aposentadoria/independência",
               "familia_legado": "família/legado", "comunidade": "comunidade", "disciplina": "disciplina",
               "superacao": "superação"}


@dataclass
class AlignmentScore:
    ticker: str
    score: float | None
    matched: list[str]
    reason: str


@dataclass
class AlignmentResult:
    scores: dict[str, AlignmentScore]
    mapped_themes: list[str]
    unmapped_themes: list[str]
    computed: bool
    confidence: str
    notes: list[str] = field(default_factory=list)


def score_assets(extraction: ThemeExtraction, assets: list[CatalogAsset]) -> AlignmentResult:
    mapped = [t for t in extraction.themes if t.name in THEME_TO_TAGS]
    unmapped = [t.name for t in extraction.themes if t.name not in THEME_TO_TAGS]
    notes = list(extraction.notes)
    if unmapped:
        notes.append("temas de valor sem correspondência setorial direta (citados na tese, não usados no score): "
                     + ", ".join(THEME_LABEL.get(t, t) for t in unmapped))
    if not mapped:
        why = "motivo vazio" if extraction.empty else "nenhum tema setorial identificado no texto"
        scores = {a.ticker: AlignmentScore(a.ticker, None, [], f"score não calculado: {why}") for a in assets}
        return AlignmentResult(scores, [], unmapped, False, extraction.confidence, notes)
    weights = {t.name: float(min(t.mentions, 3)) for t in mapped}
    total = sum(weights.values())
    scores = {}
    for a in assets:
        hit = [t for t in weights if set(THEME_TO_TAGS[t]) & set(a.tags)]
        sc = sum(weights[t] for t in hit) / total
        if hit:
            ev = {t.name: ", ".join(f"\"{x}\"" for x in t.evidence[:2]) for t in mapped}
            reason = ("atende a " + ", ".join(f"{THEME_LABEL[t]} ({ev[t]})" for t in hit) +
                      f" pelo modelo de negócio: {a.business}")
        else:
            reason = "sem conexão com os temas declarados (" + ", ".join(THEME_LABEL[t] for t in weights) + ")"
        scores[a.ticker] = AlignmentScore(a.ticker, float(sc), [THEME_LABEL[t] for t in hit], reason)
    return AlignmentResult(scores, [t.name for t in mapped], unmapped, True, extraction.confidence, notes)


@dataclass
class CombinedPortfolio:
    quant: pd.Series
    alignment_target: pd.Series | None
    final: pd.Series
    alpha: float
    te_budget: float
    tracking_error: float
    vol_quant: float
    vol_final: float
    ret_quant: float
    ret_final: float
    alignment_quant: float | None
    alignment_final: float | None
    violations: list[str]
    notes: list[str] = field(default_factory=list)


def combine(w_q: pd.Series, alignment: AlignmentResult, mu: pd.Series, cov: pd.DataFrame,
            pc: PortfolioConstraints, alpha: float = 0.30, te_budget: float = 0.05,
            vol_cap: float | None = None) -> CombinedPortfolio:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha (peso do alinhamento) deve estar em [0, 1]")
    syms = pc.symbols
    wq = w_q.loc[syms].to_numpy()
    c = cov.loc[syms, syms].to_numpy()
    m = mu.loc[syms].to_numpy()
    vq = float(np.sqrt(wq @ c @ wq))
    notes: list[str] = []
    if not alignment.computed or alpha == 0.0:
        notes.append("alinhamento não aplicado (" + ("peso do alinhamento = 0" if alpha == 0 else
                                                      "score de alinhamento não calculado") + "): pesos = quantitativos")
        return CombinedPortfolio(w_q.loc[syms], None, w_q.loc[syms], alpha, te_budget, 0.0, vq, vq, float(wq @ m),
                                 float(wq @ m), None, None, verify_weights(wq, pc), notes)
    s = np.array([alignment.scores[t].score for t in syms], float)
    cap = max(vq, vol_cap) if vol_cap is not None else None
    extra = [{"type": "ineq", "fun": lambda w: te_budget ** 2 - float((w[:len(syms)] - wq) @ c @ (w[:len(syms)] - wq)),
              "jac": lambda w: -2 * c @ (w[:len(syms)] - wq)}]
    if cap is not None:
        extra.append({"type": "ineq", "fun": lambda w: cap ** 2 - float(w[:len(syms)] @ c @ w[:len(syms)]),
                      "jac": lambda w: -2 * c @ w[:len(syms)]})
    res = solve("alignment_target", lambda w: -float(s @ w), lambda w: -s, pc, m, c, 0.0,
                SolverSettings(n_starts=4), extra_constraints=extra, start_hint=[wq])
    if not res.success:
        notes.append(f"carteira de alinhamento não convergiu ({res.status}); pesos = quantitativos")
        return CombinedPortfolio(w_q.loc[syms], None, w_q.loc[syms], alpha, te_budget, 0.0, vq, vq, float(wq @ m),
                                 float(wq @ m), float(s @ wq), float(s @ wq), verify_weights(wq, pc), notes)
    wa = res.weights.to_numpy()
    wf = (1 - alpha) * wq + alpha * wa
    wf = np.where(np.abs(wf) < 1e-10, 0.0, wf)
    wf = wf / wf.sum()
    te = float(np.sqrt(max((wf - wq) @ c @ (wf - wq), 0.0)))
    vf = float(np.sqrt(wf @ c @ wf))
    viol = verify_weights(wf, pc)
    if te > alpha * te_budget + 1e-6:
        viol.append(f"tracking error {te:.4f} acima do orçamento {alpha * te_budget:.4f}")
    if cap is not None and vf > cap + 1e-6:
        viol.append(f"volatilidade {vf:.4f} acima do teto {cap:.4f}")
    return CombinedPortfolio(pd.Series(wq, index=syms), pd.Series(wa, index=syms), pd.Series(wf, index=syms),
                             alpha, te_budget, te, vq, vf, float(wq @ m), float(wf @ m), float(s @ wq),
                             float(s @ wf), viol, notes)
