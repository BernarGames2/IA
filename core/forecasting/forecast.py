"""Probabilistic forecast of one asset over a horizon, with explicit uncertainty — never a price target.

Facts [H]: the last 5 years of daily adjusted prices (mean, volatility, beta vs the benchmark). Estimate [E]: the
distribution of the horizon return under the 'Vale a pena?' model (``core/forecasting/model.py``): quantiles 5–95%,
scenarios pessimista = P10, base = P50, otimista = P90, and the probabilities of ending up, of beating the risk-free
rate, of losing more than 20% and of beating the benchmark. The last one is exact under the model: with
``ln(1+R_a)`` and ``ln(1+R_b)`` jointly normal (correlation of daily log returns), ``P(R_a > R_b) =
Phi((m_a - m_b) / sqrt(s_a^2 + s_b^2 - 2 rho s_a s_b))``; the benchmark's expected return uses the same shrinkage
with beta = 1.

Uncertainty is shown three ways: the standard error of the historical mean (and the weight it receives), a
sensitivity table (equity premium, prior dispersion tau, volatility, history only, no premium), and the model's own
historical calibration on this asset (``calibration.py``) for horizons up to 12 months — longer horizons cannot be
checked with 10 years of data and that is said.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import DEMO_RISK_FREE_FALLBACK
from core.forecasting import calibration
from core.forecasting.data import Series3, rf_rate_at
from core.forecasting.model import Fit, P, fit, prob_above, quantiles
from core.market_data.logos import logo_path
from core.purpose.catalog import REAL_CATALOG, by_ticker
from core.purpose.language import guarantee_phrases

LOOKBACK = 5 * P
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MODEL = "qpi-projecao-gbm-encolhimento-v1"
CATALOG = by_ticker(REAL_CATALOG)
DISCLAIMER = ("Distribuição de cenários de um modelo estatístico sobre o histórico de preços [E]: não é preço-alvo, não "
              "é recomendação e não prevê notícias. O passado pode não se repetir; a calibração histórica mostra quanto "
              "o modelo acertou as próprias probabilidades neste ativo.")


def _window(px: pd.Series) -> pd.Series:
    return px.iloc[-(LOOKBACK + 1):]


def _scen(m: float, s: float, rf_h: float) -> dict[str, float]:
    q = quantiles(m, s, (0.10, 0.50))
    return {"p_superar_rf": prob_above(m, s, rf_h), "mediana": q[0.50], "p_prejuizo": 1.0 - prob_above(m, s, 0.0), "p10": q[0.10]}


def _regime(r: np.ndarray) -> str | None:
    v = pd.Series(r).rolling(63).std().dropna()
    if v.empty:
        return None
    return "alta" if v.iloc[-1] > v.median() else "baixa"


def build(s3: Series3, horizon_years: float, rf_live: float | None = None, rf_source: str = "", erp: float = 0.04,
          tau: float = 0.10, with_calibration: bool = True) -> dict[str, object]:
    if not 0.25 <= horizon_years <= 10:
        raise ValueError("horizonte da previsão deve estar entre 3 meses e 10 anos")
    px = _window(s3.px)
    if len(px) < P + 1:
        raise ValueError(f"{s3.ticker}: menos de um ano de cotações; a previsão precisa de pelo menos 252 pregões")
    t_ref = px.index[-1]
    r = px.pct_change(fill_method=None).dropna()
    warns = list(s3.notes)
    rf, rf_src = rf_live, rf_source
    if rf is None:
        rf = rf_rate_at(s3.rf_index, t_ref)
        rf_src = f"{s3.rf_name} do último dia ({t_ref:%d/%m/%Y})" if rf is not None else ""
    if rf is None:
        rf = DEMO_RISK_FREE_FALLBACK.get(s3.currency, 0.10)
        rf_src = "FALLBACK DE DEMONSTRAÇÃO (não é cotação)"
        warns.append(f"taxa livre de risco indisponível: usado {rf:.2%} de demonstração — as probabilidades contra a renda "
                     "fixa ficam só ilustrativas")
    br = None
    if s3.bench is not None:
        b = s3.bench.reindex(px.index.union(s3.bench.index)).ffill(limit=5).reindex(px.index)
        br = b.pct_change(fill_method=None).reindex(r.index).to_numpy()
    f: Fit = fit(r.to_numpy(), rf, br, erp, tau)
    h = float(horizon_years)
    m, sd = f.log_params(h)
    rf_h = (1.0 + rf) ** h - 1.0
    qv = quantiles(m, sd, QS)
    probs = {"p_alta": prob_above(m, sd, 0.0), "p_superar_rf": prob_above(m, sd, rf_h),
             "p_queda_20": 1.0 - prob_above(m, sd, -0.20)}
    vs_bench = None
    if br is not None and s3.bench_name:
        ok = np.isfinite(br)
        fb = fit(br[ok], rf, br[ok], erp, tau)                  # beta of the benchmark to itself = 1
        mb, sb = fb.log_params(h)
        rho = float(np.corrcoef(np.log1p(r.to_numpy()[ok]), np.log1p(br[ok]))[0, 1])
        if np.isfinite(rho):
            den = np.sqrt(max(sd ** 2 + sb ** 2 - 2 * rho * sd * sb, 1e-12))
            probs["p_superar_referencia"] = float(norm.cdf((m - mb) / den))
            vs_bench = {"nome": s3.bench_name, "ticker": s3.bench_ticker, "retorno_esperado_referencia": fb.mu_e,
                        "correlacao": rho, "mediana_referencia": float(np.expm1(mb)), "m_log": mb, "s_log": sb}
    lo, hi = float(np.expm1(m + sd * norm.ppf(0.01))), float(np.expm1(m + sd * norm.ppf(0.99)))
    edges = np.linspace(lo, hi, 31)
    cdf = norm.cdf((np.log1p(edges) - m) / sd)
    hist = {"bordas": edges.tolist(), "probabilidades": np.diff(cdf).tolist()}
    sens = []
    def row(label: str, mu_e: float, vol_log: float) -> None:
        mu_c = P * np.log1p(mu_e / P)
        mm, ss = (mu_c - 0.5 * vol_log ** 2) * h, vol_log * np.sqrt(h)
        sens.append({"premissa": label, "retorno_esperado": mu_e, **_scen(mm, ss, rf_h)})
    row("modelo (premissas atuais)", f.mu_e, f.vol_log)
    for e2 in (0.02, 0.06):
        row(f"prêmio de risco de {e2:.0%} (em vez de {erp:.0%})", fit(r.to_numpy(), rf, br, e2, tau).mu_e, f.vol_log)
    for t2 in (0.05, 0.20):
        row(f"dispersão tau de {t2:.0%} (história pesa {'menos' if t2 < tau else 'mais'})", fit(r.to_numpy(), rf, br, erp, t2).mu_e, f.vol_log)
    row("volatilidade 25% menor", f.mu_e, f.vol_log * 0.75)
    row("volatilidade 25% maior", f.mu_e, f.vol_log * 1.25)
    row("só o histórico (sem encolhimento)", f.mu_hist, f.vol_log)
    row("sem prêmio de risco (rende a renda fixa em média)", rf, f.vol_log)
    calib = None
    if with_calibration:
        hs = round(min(h, 1.0) * P)
        rows = calibration.run(s3.px, hs, s3.bench, s3.rf_index, erp=erp, tau=tau) if s3.rf_index is not None else []
        calib = calibration.summarise(rows, hs) if rows else {"horizonte_pregoes": hs, "n": 0, "insuficiente": True,
                                                              "nota": "sem a taxa livre de risco histórica: calibração não calculada"}
        if h > 1.0:
            calib["nota_horizonte"] = (f"horizonte pedido de {h:g} anos: com 10 anos de dados não há janelas independentes "
                                       "suficientes para testá-lo; mostrada a calibração do mesmo modelo em 12 meses")
    a = CATALOG.get(s3.ticker)
    ref_price = float(px.iloc[-1])
    end_date = (t_ref + pd.Timedelta(days=round(h * 365.25))).date()
    q05 = qv[0.05]
    record = {
        "ativo": {"ticker": s3.ticker, "nome": a.name if a else s3.ticker, "classe": a.macro_class if a else "nao_classificado",
                  "moeda": s3.currency, "logo": logo_path(s3.ticker)},
        "preco_referencia": ref_price, "data_referencia": f"{t_ref:%Y-%m-%d}", "horizonte_dias": round(h * 365.25),
        "data_fim": f"{end_date:%Y-%m-%d}",
        "hipotese": (f"o retorno de {s3.ticker} em {h:g} ano(s) segue a distribuição do modelo: mediana {qv[0.50]:+.1%}, "
                     f"faixa de 80% de {qv[0.10]:+.1%} a {qv[0.90]:+.1%}; chance de superar a renda fixa {probs['p_superar_rf']:.0%}"),
        "distribuicao": {"m_log": m, "s_log": sd, "quantis": {f"p{round(q * 100):02d}": v for q, v in qv.items()}},
        "probabilidades": probs,
        "premissas": {"modelo": MODEL, "taxa_livre_de_risco": rf, "fonte_taxa": rf_src, "premio_de_risco": erp, "tau": tau,
                      "beta": f.beta, "peso_do_historico": f.lam, "retorno_esperado": f.mu_e, "volatilidade_log": f.vol_log,
                      "anos_de_dados": f.years, "referencia": s3.bench_name},
        "fontes": s3.sources,
        "riscos": [f"volatilidade histórica de {f.vol:.0%} ao ano: resultados longe da mediana são comuns",
                   "eventos da empresa, do setor e da economia não estão no modelo (só o histórico de preços)"],
        "invalidacao": [f"volatilidade realizada no período acima de {1.5 * f.vol_log:.0%} ao ano (1,5× a usada pelo modelo)",
                        f"fechamento abaixo de {ref_price * (1 + q05):.2f} {s3.currency} (o cenário dos 5% piores) antes do fim"],
        "regime_vol": _regime(r.to_numpy()),
    }
    out = {
        "ticker": s3.ticker, "nome": record["ativo"]["nome"], "moeda": s3.currency, "logo": record["ativo"]["logo"],
        "data_referencia": f"{t_ref:%Y-%m-%d}", "preco_referencia": ref_price, "horizonte_anos": h,
        "fatos_H": {"retorno_medio_anual": f.mu_hist, "volatilidade": f.vol, "beta": f.beta, "anos_de_dados": f.years,
                    "periodo": [f"{px.index[0]:%Y-%m-%d}", f"{t_ref:%Y-%m-%d}"]},
        "estimativas_E": {"retorno_esperado": f.mu_e, "erro_padrao_media": f.se, "peso_do_historico": f.lam,
                          "ic95_media_historica": [f.mu_hist - 1.96 * f.se, f.mu_hist + 1.96 * f.se], "prior": f.prior},
        "taxa_livre_de_risco": rf, "fonte_taxa": rf_src, "nome_taxa": s3.rf_name, "rf_no_horizonte": rf_h,
        "quantis": {f"p{round(q * 100):02d}": v for q, v in qv.items()},
        "cenarios": [{"cenario": "pessimista (P10)", "retorno": qv[0.10], "valor_de_100": 100 * (1 + qv[0.10])},
                     {"cenario": "base (P50)", "retorno": qv[0.50], "valor_de_100": 100 * (1 + qv[0.50])},
                     {"cenario": "otimista (P90)", "retorno": qv[0.90], "valor_de_100": 100 * (1 + qv[0.90])}],
        "probabilidades": probs, "referencia": vs_bench, "distribuicao": hist, "sensibilidade": sens,
        "calibracao": calib, "registro": record, "fontes": s3.sources, "avisos": warns, "aviso": DISCLAIMER,
    }
    texts = [record["hipotese"], *warns, *record["riscos"], *record["invalidacao"], DISCLAIMER,
             *(calib or {}).get("leitura", []), str((calib or {}).get("nota", "")), str((calib or {}).get("nota_horizonte", ""))]
    bad = guarantee_phrases(" ".join(texts))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out


def reference_date(out: dict[str, object]) -> date:
    return date.fromisoformat(str(out["data_referencia"]))
