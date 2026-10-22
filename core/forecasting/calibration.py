"""Historical calibration of the projection model on one asset: does it say "70%" when things happen ~70% of the
time?

Pseudo-out-of-sample protocol (no look-ahead, tested): at monthly origins t (every ``step`` sessions) from
``min_window`` sessions after the start until ``horizon`` sessions before the end, the model is fitted ONLY on the
prices up to t (trailing window of up to 5 years), with the risk-free rate known on day t; it states P(R > 0),
P(R > risk-free over the horizon) and the 10/50/90% quantiles of the horizon return R. The outcome is the realised
return from t to t + horizon (and the risk-free actually earned in that window).

Scores: Brier score (mean squared error of the probabilities) and its skill against the climatological base rate
known at t (the share of past windows, up to t, in which the event happened); log score; reliability table (bins
of predicted probability vs observed frequency); coverage of the 10–90% interval (80% if calibrated); PIT histogram
and a Kolmogorov–Smirnov test of uniformity; breakdown by volatility regime at t (trailing 63-session volatility
above or below its own median up to t).

Caveat stated in the result: consecutive origins overlap (a 12-month horizon sampled monthly shares 11 of 12 months
with the next one), so outcomes are correlated; the effective sample size is about n x step / horizon and tests
that assume independence (KS) are optimistic.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import kstest

from core.forecasting.data import rf_growth, rf_rate_at
from core.forecasting.model import P, fit, pit, prob_above, quantiles

WINDOW = 5 * P
MIN_WINDOW = 2 * P
STEP = 21
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EPS = 0.01


def _brier(p: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((p - o) ** 2))


def _logscore(p: np.ndarray, o: np.ndarray) -> float:
    q = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(o * np.log(q) + (1 - o) * np.log(1 - q))))


def reliability(p: np.ndarray, o: np.ndarray, bins: tuple[float, ...] = BINS) -> list[dict[str, float | int | None]]:
    out = []
    for lo, hi in pairwise(bins):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        out.append({"de": lo, "ate": hi, "n": int(m.sum()), "previsto_medio": float(p[m].mean()) if m.any() else None,
                    "observado": float(o[m].mean()) if m.any() else None})
    return out


def score(p: np.ndarray, o: np.ndarray, clim: np.ndarray | None) -> dict[str, object]:
    p, o = np.asarray(p, float), np.asarray(o, float)
    bs = _brier(p, o)
    out: dict[str, object] = {"n": len(p), "brier": bs, "log_score": _logscore(p, o), "frequencia_observada": float(o.mean()),
                              "previsto_medio": float(p.mean()), "confiabilidade": reliability(p, o)}
    if clim is not None and len(clim) == len(p):
        bc = _brier(np.asarray(clim, float), o)
        out["brier_climatologia"] = bc
        out["habilidade_brier"] = float(1.0 - bs / bc) if bc > 0 else None
    return out


def run(px: pd.Series, horizon: int, bench: pd.Series | None = None, rf_index: pd.Series | None = None,
        step: int = STEP, window: int = WINDOW, min_window: int = MIN_WINDOW, erp: float = 0.04, tau: float = 0.10
        ) -> list[dict[str, object]]:
    """One row per origin. ``px`` is the asset's daily adjusted close; every input to the forecast at origin i is
    sliced to positions <= i."""
    px = px.dropna()
    vals = px.to_numpy()
    idx = px.index
    b_al = bench.reindex(idx.union(bench.index)).ffill(limit=5).reindex(idx) if bench is not None else None
    rows = []
    h_years = horizon / P
    for i in range(min_window, len(px) - horizon, step):
        t = idx[i]
        lo = max(0, i - window)
        hist = vals[lo:i + 1]
        r = hist[1:] / hist[:-1] - 1.0
        br = None
        if b_al is not None:
            bh = b_al.to_numpy()[lo:i + 1]
            br = bh[1:] / bh[:-1] - 1.0
        rf_t = rf_rate_at(rf_index, t)
        if rf_t is None:                      # the prior needs the rate known on day t: no origin without it
            continue
        f = fit(r, rf_t, br, erp, tau)
        m, s = f.log_params(h_years)
        q = quantiles(m, s, (0.10, 0.50, 0.90))
        realised = float(vals[i + horizon] / vals[i] - 1.0)
        rf_exp = (1.0 + rf_t) ** h_years - 1.0
        rf_real = rf_growth(rf_index, t, idx[i + horizon])
        # climatology known at t: share of past horizon windows (inside the fit window) where the event happened
        past = hist[horizon:] / hist[:-horizon] - 1.0 if len(hist) > horizon else np.array([])
        clim_up = float(np.mean(past > 0)) if len(past) else 0.5
        clim_rf = None
        if rf_index is not None and len(past):
            dates = idx[lo:i + 1]
            past_rf = np.array([rf_growth(rf_index, dates[k], dates[k + horizon]) or np.nan for k in range(0, len(dates) - horizon, 5)])
            past_r = past[::5][:len(past_rf)]
            ok = np.isfinite(past_rf)
            clim_rf = float(np.mean(past_r[ok] > past_rf[ok])) if ok.any() else None
        vol63 = pd.Series(r).rolling(63).std().dropna()
        regime = None if vol63.empty else ("alta" if vol63.iloc[-1] > vol63.median() else "baixa")
        rows.append({"data": f"{t:%Y-%m-%d}", "fim": f"{idx[i + horizon]:%Y-%m-%d}", "p_alta": prob_above(m, s, 0.0),
                     "p_superar_rf": prob_above(m, s, rf_exp), "q10": q[0.10], "q50": q[0.50],
                     "q90": q[0.90], "realizado": realised, "rf_realizado": rf_real, "alta": realised > 0,
                     "superou_rf": None if rf_real is None else realised > rf_real, "pit": pit(m, s, realised),
                     "dentro_10_90": q[0.10] <= realised <= q[0.90], "clim_alta": clim_up, "clim_rf": clim_rf,
                     "regime_vol": regime, "retorno_esperado": f.mu_e})
    return rows


def _month(iso: object) -> str:
    y, m = str(iso).split("-")[:2]
    return f"{m}/{y}"


def reading(up: dict[str, object], beat: dict[str, object] | None, cov: float, above: float, below: float, n: int,
            n_eff: float, period: list[object], horizon: int) -> list[str]:
    """The calibration in plain words (rule-based; thresholds stated in the text)."""
    out = [(f"{n} previsões de {round(horizon / 21)} meses testadas entre {_month(period[0])} e {_month(period[1])}, "
            f"cada uma feita só com os dados até a sua data (≈ {n_eff:.0f} períodos independentes).")]
    out.append(f"Quando o modelo deu em média {float(up['previsto_medio']):.0%} de chance de subir, o ativo subiu em "  # type: ignore[arg-type]
               f"{float(up['frequencia_observada']):.0%} das vezes.")  # type: ignore[arg-type]
    if beat:
        out.append(f"Para superar a renda fixa: chance média prevista de {float(beat['previsto_medio']):.0%}; "  # type: ignore[arg-type]
                   f"aconteceu em {float(beat['frequencia_observada']):.0%} das vezes.")  # type: ignore[arg-type]
    if cov < 0.70 and above >= 2 * below:
        out.append(f"A faixa de 80% conteve o resultado em {cov:.0%} das vezes, e ele ficou ACIMA dela em {above:.0%}: o "
                   "centro da previsão ficou baixo demais neste período (o ativo subiu mais do que o modelo esperava).")
    elif cov < 0.70 and below >= 2 * above:
        out.append(f"A faixa de 80% conteve o resultado em {cov:.0%} das vezes, e ele ficou ABAIXO dela em {below:.0%}: o "
                   "centro da previsão ficou alto demais neste período.")
    elif cov < 0.70:
        out.append(f"A faixa de 80% conteve o resultado em {cov:.0%} das vezes: estreita demais — os resultados variaram "
                   "mais do que o modelo previa.")
    elif cov > 0.90:
        out.append(f"A faixa de 80% conteve o resultado em {cov:.0%} das vezes: larga demais — o modelo exagerou a "
                   "incerteza.")
    else:
        out.append(f"A faixa de 80% conteve o resultado em {cov:.0%} das vezes, perto do esperado.")
    sk = up.get("habilidade_brier")
    if isinstance(sk, float):
        if sk > 0.05:
            out.append(f"As probabilidades de subir foram melhores do que repetir a frequência do passado (habilidade "
                       f"de {sk:+.0%} no Brier).")
        elif sk < -0.05:
            out.append(f"As probabilidades de subir foram piores do que repetir a frequência do passado (habilidade "
                       f"de {sk:+.0%} no Brier).")
        else:
            out.append("As probabilidades de subir ficaram parecidas com repetir a frequência do passado.")
    if n_eff < 20:
        out.append("Com tão poucos períodos independentes, essas notas ainda oscilam muito: são um indício, não uma prova.")
    return out


def summarise(rows: list[dict[str, object]], horizon: int, step: int = STEP) -> dict[str, object]:
    if len(rows) < 10:
        return {"horizonte_pregoes": horizon, "n": len(rows), "insuficiente": True,
                "nota": "menos de 10 origens com resultado conhecido: histórico curto demais para avaliar a calibração"}
    df = pd.DataFrame(rows)
    up = score(df["p_alta"].to_numpy(), df["alta"].astype(float).to_numpy(), df["clim_alta"].to_numpy())
    rf_ok = df["p_superar_rf"].notna() & df["superou_rf"].notna()
    beat = None
    if rf_ok.sum() >= 10:
        sub = df[rf_ok]
        clim = sub["clim_rf"].to_numpy(dtype=float) if sub["clim_rf"].notna().all() else None
        beat = score(sub["p_superar_rf"].to_numpy(dtype=float), sub["superou_rf"].astype(float).to_numpy(), clim)
    pits = df["pit"].to_numpy(dtype=float)
    ks = kstest(pits, "uniform")
    hist, _ = np.histogram(pits, bins=10, range=(0.0, 1.0))
    by_regime = {}
    for reg, g in df.groupby("regime_vol"):
        item: dict[str, object] = {"n": len(g), "cobertura_10_90": float(g["dentro_10_90"].mean())}
        gg = g[g["p_superar_rf"].notna() & g["superou_rf"].notna()]
        if len(gg) >= 5:
            item["brier_superar_rf"] = _brier(gg["p_superar_rf"].to_numpy(dtype=float), gg["superou_rf"].astype(float).to_numpy())
        by_regime[str(reg)] = item
    n_eff = len(df) * min(1.0, step / horizon)
    above = float((df["realizado"] > df["q90"]).mean())
    below = float((df["realizado"] < df["q10"]).mean())
    keep = df.iloc[-120:]
    return {
        "horizonte_pregoes": horizon, "horizonte_meses": round(horizon / 21), "n": len(df), "n_efetivo": round(n_eff, 1),
        "periodo": [rows[0]["data"], rows[-1]["fim"]], "evento_alta": up, "evento_superar_rf": beat,
        "cobertura_10_90": float(df["dentro_10_90"].mean()), "cobertura_alvo": 0.80,
        "acima_da_faixa": above, "abaixo_da_faixa": below,
        "pit_histograma": hist.tolist(), "ks_estatistica": float(ks.statistic), "ks_p_valor": float(ks.pvalue),
        "por_regime": by_regime,
        "origens": keep[["data", "fim", "p_alta", "p_superar_rf", "q10", "q50", "q90", "realizado", "alta", "superou_rf"]].to_dict("records"),
        "nota": (f"origens a cada {step} pregões com janelas de {horizon} pregões: resultados sobrepostos e correlacionados "
                 f"(tamanho efetivo ≈ {n_eff:.0f}); o teste KS supõe independência e tende a ser otimista"),
        "leitura": reading(up, beat, float(df["dentro_10_90"].mean()), above, below, len(df), n_eff,
                           [rows[0]["data"], rows[-1]["fim"]], horizon),
    }
