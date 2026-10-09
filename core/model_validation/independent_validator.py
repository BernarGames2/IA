"""Independent Validation Engine.

Tries to REFUTE the results of the other engines. Critical numbers are
recomputed here with deliberately separate code paths (explicit loops / numpy
primitives, not ``core.metrics``). Each check yields PASS / WARNING / FAIL with a
justification and evidence; any critical FAIL blocks presenting the analysis as
validated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class Check:
    name: str
    status: Status
    critical: bool
    justification: str
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ValidationReport:
    checks: list[Check]

    @property
    def overall(self) -> Status:
        if any(c.status == Status.FAIL and c.critical for c in self.checks):
            return Status.FAIL
        if any(c.status != Status.PASS for c in self.checks):
            return Status.WARNING
        return Status.PASS

    @property
    def validated(self) -> bool:
        return self.overall != Status.FAIL

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in c.as_dict().items() if k != "evidence"} for c in self.checks])


# ---------------------------------------------------------------- recomputations
def _port_var_loop(w: np.ndarray, cov: np.ndarray) -> float:
    total = 0.0
    n = len(w)
    for i in range(n):
        for j in range(n):
            total += w[i] * w[j] * cov[i, j]
    return total


def check_data(prices: pd.DataFrame, synthetic: bool, quality_overall: str, currencies: dict[str, str],
               base_ccy: str, min_history: int) -> list[Check]:
    out = []
    arr = prices.to_numpy(dtype=float)
    ok = bool(np.all(np.isfinite(arr)) and np.all(arr > 0))
    out.append(Check("dados: preços finitos e positivos", Status.PASS if ok else Status.FAIL, True,
                     "todos os preços alinhados são finitos e > 0" if ok else "preços inválidos no painel",
                     {"n_rows": len(prices), "n_assets": prices.shape[1]}))
    out.append(Check("dados: índice cronológico único",
                     Status.PASS if prices.index.is_monotonic_increasing and prices.index.is_unique else Status.FAIL,
                     True, "datas ordenadas e sem duplicatas"))
    out.append(Check("dados: histórico mínimo", Status.PASS if len(prices) >= min_history else Status.FAIL, True,
                     f"{len(prices)} observações alinhadas vs mínimo {min_history}"))
    bad_ccy = {s: c for s, c in currencies.items() if c != base_ccy}
    out.append(Check("dados: unidades/moeda consistentes", Status.PASS if not bad_ccy else Status.FAIL, True,
                     f"todas as séries em {base_ccy}" if not bad_ccy else f"moedas divergentes: {bad_ccy}"))
    q = {"ok": Status.PASS, "warning": Status.WARNING, "fail": Status.WARNING}[quality_overall]
    out.append(Check("dados: relatório de qualidade", q, False,
                     f"status agregado de qualidade = {quality_overall} (símbolos com FAIL foram excluídos)"))
    out.append(Check("dados: origem", Status.WARNING if synthetic else Status.PASS, False,
                     "dados SINTÉTICOS: resultados demonstram a mecânica, não o mercado real" if synthetic
                     else "dados de provedor externo com proveniência registrada"))
    return out


def check_covariance(cov: pd.DataFrame) -> Check:
    c = cov.to_numpy()
    asym = float(np.max(np.abs(c - c.T)))
    eig = np.linalg.eigvalsh((c + c.T) / 2)
    cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    st = Status.PASS
    msg = "simétrica e PSD"
    if asym > 1e-10 or eig[0] < -1e-12:
        st, msg = Status.FAIL, "matriz não simétrica ou não PSD"
    elif cond > 1e6:
        st, msg = Status.WARNING, f"condicionamento elevado ({cond:.2e})"
    return Check("covariância: simetria/PSD/condicionamento", st, True, msg,
                 {"min_eig": float(eig[0]), "max_eig": float(eig[-1]), "condition": cond, "max_asym": asym})


def check_portfolio(name: str, w: pd.Series, cov: pd.DataFrame, lower: np.ndarray, upper: np.ndarray,
                    groups: list[tuple[str, list[str], float, float]], reported_vol: float | None,
                    rc_reported: pd.Series | None, tol: float = 1e-6) -> list[Check]:
    out = []
    wv = w.to_numpy(dtype=float)
    s = 0.0
    for x in wv:
        s += x
    viol = []
    for i, sym in enumerate(w.index):
        if wv[i] < lower[i] - tol or wv[i] > upper[i] + tol:
            viol.append(f"{sym}={wv[i]:.4f} fora de [{lower[i]:.2f},{upper[i]:.2f}]")
    for gname, members, lo, hi in groups:
        g = sum(float(w[m]) for m in members if m in w.index)
        if g < lo - tol or g > hi + tol:
            viol.append(f"{gname}={g:.4f} fora de [{lo:.2f},{hi:.2f}]")
    st = Status.PASS if abs(s - 1) < tol and np.all(np.isfinite(wv)) and not viol else Status.FAIL
    out.append(Check(f"{name}: soma, limites e grupos", st, True,
                     "restrições verificadas independentemente" if st == Status.PASS else "; ".join(viol) or f"soma={s}",
                     {"sum_weights": s, "violations": viol}))
    var = _port_var_loop(wv, cov.loc[w.index, w.index].to_numpy())
    vol = float(np.sqrt(max(var, 0.0)))
    if reported_vol is not None:
        ok = abs(vol - reported_vol) <= 1e-8 + 1e-6 * vol
        out.append(Check(f"{name}: volatilidade recalculada", Status.PASS if ok else Status.FAIL, True,
                         f"recalculada {vol:.6%} vs reportada {reported_vol:.6%}",
                         {"recomputed": vol, "reported": reported_vol}))
    if rc_reported is not None and vol > 0:
        c = cov.loc[w.index, w.index].to_numpy()
        rc = np.array([wv[i] * sum(c[i, j] * wv[j] for j in range(len(wv))) / vol for i in range(len(wv))])
        ok = abs(rc.sum() - vol) < 1e-9 and np.allclose(rc, rc_reported.loc[w.index].to_numpy(), atol=1e-9)
        out.append(Check(f"{name}: identidade sum(RC_i) = sigma_p", Status.PASS if ok else Status.FAIL, True,
                         f"sum(RC)={rc.sum():.8f}, sigma_p={vol:.8f}",
                         {"sum_rc": float(rc.sum()), "sigma_p": vol}))
    return out


def check_var_es(table: pd.DataFrame) -> Check:
    bad = []
    for _, r in table.iterrows():
        if r.get("var") is None or r.get("es") is None or pd.isna(r.get("var")):
            continue
        if not (np.isfinite(r["var"]) and np.isfinite(r["es"])) or r["es"] < r["var"] - 1e-12:
            bad.append(f"{r['method']}@{r['confidence']}")
        if not r.get("horizon") or not r.get("method"):
            bad.append("estimativa sem método/horizonte")
    return Check("VaR/ES: coerência (ES >= VaR, rótulos)", Status.PASS if not bad else Status.FAIL, True,
                 "todas as estimativas têm método, nível e horizonte; ES >= VaR" if not bad else f"falhas: {bad}")


def check_mc_reproducibility(run_fn, n_paths: int = 2000) -> Check:  # noqa: ANN001
    a = run_fn(seed=123, n_paths=n_paths)
    b = run_fn(seed=123, n_paths=n_paths)
    c = run_fn(seed=124, n_paths=n_paths)
    same = np.array_equal(a, b)
    diff = not np.array_equal(a, c)
    st = Status.PASS if same and diff else Status.FAIL
    return Check("Monte Carlo: reprodutibilidade por semente", st, True,
                 "mesma semente -> resultados idênticos; semente diferente -> resultados diferentes"
                 if st == Status.PASS else "simulação não reprodutível")


def check_mc_theory(mean_sim: float, se: float, mean_theory: float, label: str) -> Check:
    z = abs(mean_sim - mean_theory) / se if se > 0 else float("inf")
    st = Status.PASS if z < 4 else Status.FAIL
    return Check(f"Monte Carlo: caso teórico ({label})", st, True,
                 f"média simulada {mean_sim:.6f} vs teórica {mean_theory:.6f} (|z|={z:.2f}, limite 4)",
                 {"z": z})


def check_frontier(frontier: pd.DataFrame, max_sharpe: float | None, gmv_vol: float | None) -> Check:
    if frontier.empty:
        return Check("fronteira: pontos válidos", Status.FAIL, True, "fronteira vazia")
    issues = []
    if max_sharpe is not None and frontier["sharpe"].max() > max_sharpe + 1e-4:
        issues.append(f"ponto da fronteira com Sharpe {frontier['sharpe'].max():.4f} > máx. Sharpe {max_sharpe:.4f}")
    if gmv_vol is not None and frontier["volatility"].min() < gmv_vol - 1e-6:
        issues.append("ponto da fronteira com volatilidade abaixo da mínima variância")
    wcols = [c for c in frontier.columns if c.startswith("w_")]
    sums = frontier[wcols].sum(axis=1)
    if (sums.sub(1).abs() > 1e-6).any() or (frontier[wcols] < -1e-6).any().any():
        issues.append("pesos inválidos em pontos da fronteira")
    return Check("fronteira: consistência com mín. variância e máx. Sharpe",
                 Status.PASS if not issues else Status.FAIL, True,
                 "fronteira consistente" if not issues else "; ".join(issues), {"n_points": len(frontier)})


def check_concentration(w: pd.Series, max_allowed: float) -> Check:
    hhi = float((w ** 2).sum())
    eff = 1 / hhi
    st = Status.PASS if eff >= 3 else Status.WARNING
    return Check("concentração da carteira selecionada", st, False,
                 f"N efetivo = {eff:.2f}; peso máximo {w.max():.2%} (limite {max_allowed:.2%})",
                 {"effective_n": eff, "hhi": hhi})


def check_sensitivity(summary: pd.DataFrame, threshold: float = 0.10) -> Check:
    if summary is None or summary.empty:
        return Check("sensibilidade dos pesos", Status.WARNING, False, "análise de sensibilidade indisponível")
    m = float(summary["std"].max())
    st = Status.PASS if m < threshold else Status.WARNING
    return Check("sensibilidade dos pesos a erros em mu", st, False,
                 f"maior desvio-padrão de peso sob perturbação de mu = {m:.2%} (limite {threshold:.0%})",
                 {"max_weight_std": m})


def check_backtest(lookahead: dict[str, object], warnings: list[str]) -> list[Check]:
    out = [Check("backtest: invariância a dados futuros (look-ahead)",
                 Status.PASS if lookahead.get("passed") else Status.FAIL, True,
                 f"{lookahead.get('decisions_checked')} decisões inalteradas ao perturbar dados futuros; "
                 f"diferença máx. {lookahead.get('max_abs_weight_diff')}", lookahead)]
    for w in warnings:
        out.append(Check("backtest: viés potencial", Status.WARNING, False, w))
    return out


def check_complexity_gain(selection: dict[str, object]) -> Check:
    sel = selection.get("selected")
    vs = selection.get("vs_baseline_validation")
    if sel in ("equal_weight", None):
        return Check("complexidade vs baseline", Status.PASS, False,
                     "regra 1-EP manteve a estratégia simples (pesos iguais): complexidade adicional não "
                     "demonstrou ganho suficiente na validação")
    if isinstance(vs, dict) and "p_value" in vs:
        sig = vs["p_value"] < 0.10 and vs["dm_stat"] < 0
        return Check("complexidade vs baseline", Status.PASS if sig else Status.WARNING, False,
                     f"'{sel}' vs pesos iguais na validação: DM={vs['dm_stat']:.2f}, p={vs['p_value']:.3f}"
                     + ("" if sig else " — ganho NÃO significativo; preferência por simplicidade recomendada"),
                     vs)
    return Check("complexidade vs baseline", Status.WARNING, False, "teste contra baseline indisponível")


def check_risk_free(is_fallback: bool, rf: float) -> Check:
    return Check("taxa livre de risco", Status.WARNING if is_fallback else Status.PASS, False,
                 f"rf = {rf:.2%} é FALLBACK DE DEMONSTRAÇÃO (não é cotação)" if is_fallback
                 else f"rf = {rf:.2%} informada pelo usuário")


def check_profile_volatility(vol: float, band: tuple[float, float], name: str) -> Check:
    """Estimated volatility of the selected portfolio vs the profile's reference band."""
    lo, hi = band
    if lo <= vol <= hi:
        return Check("volatilidade vs faixa do perfil", Status.PASS, False,
                     f"{name}: volatilidade estimada {vol:.2%} dentro da faixa de referência {lo:.0%}–{hi:.0%}")
    return Check("volatilidade vs faixa do perfil", Status.WARNING, False,
                 f"{name}: volatilidade estimada {vol:.2%} FORA da faixa de referência {lo:.0%}–{hi:.0%} "
                 "(faixa é referência de pesquisa, não restrição do otimizador)", {"vol": vol, "band": band})
