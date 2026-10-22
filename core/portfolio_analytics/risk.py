"""Advanced risk report of the investor's CURRENT portfolio (today's weights), in reais, on real prices.

What is measured, with the rule of each item (all historical [H] unless marked [E] or hipótese):

* volatility: historical (sqrt(252) x daily standard deviation) and EWMA (lambda 0.94, today's level) [E];
* VaR and CVaR (expected shortfall) at 95% and 99%: historical (empirical quantile of the portfolio's daily returns),
  Gaussian, and EVT (peaks over threshold) when the tail has enough observations; the 10-day horizon uses
  overlapping 10-day returns for the historical method and sqrt(10) scaling for the Gaussian one (stated). None of
  them is a limit on losses;
* drawdown: the largest fall from a peak and today's distance from the peak;
* correlations in normal times and in STRESS days — days when the benchmark's trailing 20-day volatility (known on
  the day) is in its top quintile, or its daily return is in its worst 5% — with the average pairwise correlation;
* who carries the risk: Euler contributions w_i (Sigma w)_i / w'Sigma w (they add up to 100%), by asset, sector,
  class and currency; concentration (HHI, effective number of positions, independent bets);
* liquidity: days to sell each position trading 20% of its 60-session average daily traded value (close x volume);
* sensitivities (correlation measured, not causation): betas of the portfolio's daily returns to the Brazilian
  market (BOVA11), the S&P 500 in reais (IVVB11), the dollar (USDBRL=X) and interest-rate proxies (IRFM11 —
  fixed-rate government bonds: it falls when Brazilian rates rise; IEF for US rates when dollar assets are held),
  with Newey-West t-statistics and R^2;
* stress: historical windows replayed with the assets' actual returns (COVID-19 crash and the benchmark's worst 1-
  and 3-month windows in the data; only when the data cover at least 80% of the portfolio) and hypothetical shocks
  through the measured betas (labelled hipótese).

Assets with less than one year of prices stay out of the statistics (listed, with their weight).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from core.extreme_risk.evt import evt_analysis
from core.extreme_risk.var_es import historical_var_es, normal_var_es
from core.factors.models import factor_regression
from core.market_data.cash import CASH_TICKER
from core.market_data.yahoo_chart import FetchFn
from core.portfolio_analytics.data import (
    BENCHMARKS,
    CLASS_PT,
    DISCLAIMER,
    FX_TICKER,
    MAX_FFILL_DAYS,
    RATES_BR,
    RATES_US,
    Market,
    PortfolioRequest,
    asset_info,
    build_book,
    load_market,
    logo,
    norm,
    require_some_data,
    today,
)
from core.purpose.concentration import effective_bets
from core.purpose.language import guarantee_phrases
from utils.validation import InsufficientDataError

P = 252
LOOKBACK_YEARS = 5
MIN_OBS = 252
ADTV_DAYS = 60
PARTICIPATION = 0.20
STRESS_VOL_Q = 0.80
STRESS_RET_Q = 0.05
MIN_COVERAGE = 0.80
HISTORICAL_WINDOWS = (("Crise da COVID-19", "2020-02-19", "2020-03-23"),)
MAX_CORR_ASSETS = 15


def _returns(market: Market, tickers: list[str]) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        if t == CASH_TICKER:
            if market.cdi is not None:
                cols[t] = market.cdi
        elif market.has(t):
            cols[t] = market.brl[t]
    if not cols:
        return pd.DataFrame()
    px = pd.DataFrame(cols).sort_index().ffill(limit=MAX_FFILL_DAYS)
    return px.pct_change(fill_method=None)


def _beta(y: pd.Series, x: pd.Series, name: str, proxy: str, reading: str) -> dict[str, object] | None:
    try:
        r = factor_regression(y, x.rename("f").to_frame())
    except (InsufficientDataError, np.linalg.LinAlgError, ValueError):
        return None
    b, t = float(r["coef"]["f"]), float(r["t_hac"]["f"])
    return {"fator": name, "proxy": proxy, "beta": b, "t": t if np.isfinite(t) else None, "r2": float(r["r2"]),
            "n_obs": int(r["n_obs"]), "leitura": reading.format(b=b),
            "forca": "relação estatisticamente fraca (|t| < 2)" if not np.isfinite(t) or abs(t) < 2 else "relação estatisticamente clara (|t| ≥ 2)"}


def _avg_offdiag(c: pd.DataFrame) -> float | None:
    a = c.to_numpy()
    n = a.shape[0]
    if n < 2:
        return None
    vals = a[np.triu_indices(n, 1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else None


def _window_return(px: pd.DataFrame, w: pd.Series, start: pd.Timestamp, end: pd.Timestamp
                   ) -> tuple[float | None, float]:
    """Buy-and-hold return of today's weights over [start, end] with the assets that have prices at both ends;
    returns (return, weight covered)."""
    seg = px.loc[start - pd.Timedelta(days=7):end]
    if seg.empty:
        return None, 0.0
    a = seg.loc[:start].ffill().iloc[-1] if not seg.loc[:start].empty else None
    b = seg.ffill().iloc[-1]
    if a is None:
        return None, 0.0
    ok = [t for t in w.index if t in px.columns and np.isfinite(a.get(t, np.nan)) and np.isfinite(b.get(t, np.nan))
          and a[t] > 0]
    cov = float(w[ok].sum()) if ok else 0.0
    if cov < MIN_COVERAGE:
        return None, cov
    ww = w[ok] / cov
    return float((ww * (b[ok] / a[ok] - 1.0)).sum()), cov


def _worst_window(bench: pd.Series, days: int) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    s = bench.dropna()
    if len(s) <= days:
        return None
    ret = s / s.shift(days) - 1.0
    if ret.dropna().empty:
        return None
    end = ret.idxmin()
    i = s.index.get_loc(end)
    return s.index[i - days], end, float(ret.min())


def build_risk(req: PortfolioRequest, provider: object, fetch: FetchFn | None = None,
               as_of: date | None = None) -> dict[str, object]:
    end = as_of or req.data_referencia or today()
    tickers = sorted({norm(p.ticker) for p in req.posicoes} | {norm(m.ticker) for m in req.movimentos})
    refs = (*BENCHMARKS, RATES_BR, RATES_US)
    market = load_market(tickers, provider, fetch, "10y", end, extra=refs)
    book = build_book(req, market)
    require_some_data(book, market)
    missing = {t: market.missing[t] for t in book.tickers if t in market.missing}

    # ----- today's weights in reais
    values: dict[str, float] = {}
    for t in book.tickers:
        q = float(book.quantity.get(t, 0.0))
        if q <= 0 or t in missing:
            continue
        if t == CASH_TICKER:
            values[t] = q
        elif market.has(t):
            values[t] = q * float(market.brl[t].iloc[-1])
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("as posições informadas não têm valor com as cotações disponíveis")
    w_all = pd.Series(values) / total

    # ----- returns over the lookback; assets with less than a year are left out of the statistics
    rets_all = _returns(market, list(w_all.index))
    lookback_start = pd.Timestamp(end) - pd.DateOffset(years=LOOKBACK_YEARS)
    rets = rets_all.loc[lookback_start:]
    short = {t: f"só {int(rets[t].notna().sum())} pregões com cotação (mínimo {MIN_OBS})"
             for t in w_all.index if t not in rets.columns or rets[t].notna().sum() < MIN_OBS}
    used = [t for t in w_all.index if t not in short]
    if not used:
        raise ValueError("nenhum ativo da carteira tem um ano de cotações para medir o risco")
    w = w_all[used] / w_all[used].sum()
    r = rets[used].dropna()
    if len(r) < MIN_OBS:
        raise ValueError(f"os ativos têm poucos pregões em comum ({len(r)}); o risco precisa de pelo menos {MIN_OBS}")
    rp = (r * w).sum(axis=1)

    # ----- volatility, VaR/CVaR, drawdown
    vol_hist = float(rp.std(ddof=1) * np.sqrt(P))
    lam = 0.94
    ew = float(rp.pow(2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])
    vol_ewma = float(np.sqrt(ew * P))
    h = req.horizonte_var_dias
    var_rows = []
    for c in (0.95, 0.99):
        if h == 1:
            hist = historical_var_es(rp.to_numpy(), c, "1 dia")
            note = "retornos diários da carteira de hoje no histórico"
        else:
            agg = (1.0 + rp).rolling(h).apply(np.prod, raw=True).dropna() - 1.0
            hist = historical_var_es(agg.to_numpy(), c, f"{h} dias")
            note = f"retornos de {h} dias sobrepostos (observações correlacionadas: a incerteza é maior que parece)"
        var_rows.append({"metodo": "histórico", "confianca": c, "horizonte_dias": h, "var": hist.var, "cvar": hist.es,
                         "nota": note})
        norm_est = normal_var_es(float(rp.mean()) * h, float(rp.std(ddof=1)) * np.sqrt(h), c, f"{h} dias")
        var_rows.append({"metodo": "normal", "confianca": c, "horizonte_dias": h, "var": norm_est.var,
                         "cvar": norm_est.es, "nota": "supõe retornos normais (subestima caudas gordas)"
                         + ("; escala por raiz de 10 dias" if h > 1 else "")})
        if h == 1:
            evt = evt_analysis(-rp.to_numpy(), alpha=c, n_boot=100)
            var_rows.append({"metodo": "EVT (cauda)", "confianca": c, "horizonte_dias": 1,
                             "var": None if evt.refused else evt.var, "cvar": None if evt.refused else evt.es,
                             "nota": evt.refusal_reason if evt.refused else "teoria de valores extremos (cauda ajustada por GPD)"})
    for row in var_rows:
        row["var_reais"] = None if row["var"] is None else float(row["var"]) * total
        row["cvar_reais"] = None if row["cvar"] is None else float(row["cvar"]) * total
    wealth = (1.0 + rp).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough = dd.idxmin()
    dd_peak = wealth.loc[:trough].idxmax()

    # ----- correlations in normal and stress days (benchmark of the base currency: BOVA11)
    bench_t = "BOVA11.SA"
    corr_out = None
    names = list(w.sort_values(ascending=False).index[:MAX_CORR_ASSETS])
    stress_rule = (f"dias de estresse: volatilidade de 20 pregões do Ibovespa (BOVA11), conhecida no dia, entre os "
                   f"{1 - STRESS_VOL_Q:.0%} maiores, ou retorno diário entre os {STRESS_RET_Q:.0%} piores")
    if market.has(bench_t) and len(names) >= 2:
        b = market.brl[bench_t].pct_change(fill_method=None).reindex(r.index)
        bvol = b.rolling(20).std()
        stress = (bvol >= bvol.quantile(STRESS_VOL_Q)) | (b <= b.quantile(STRESS_RET_Q))
        normal = r.loc[~stress.fillna(False), names]
        st = r.loc[stress.fillna(False), names]
        if len(st) >= 30:
            cn, cs = normal.corr(), st.corr()
            corr_out = {"ativos": names, "normal": cn.round(3).to_numpy().tolist(), "estresse": cs.round(3).to_numpy().tolist(),
                        "media_normal": _avg_offdiag(cn), "media_estresse": _avg_offdiag(cs), "dias_estresse": len(st),
                        "dias_normais": len(normal), "regra": stress_rule}

    # ----- risk contributions and concentration
    cov = r.cov().to_numpy() * P
    wv = w.to_numpy()
    var_p = float(wv @ cov @ wv)
    rc = pd.Series(wv * (cov @ wv) / var_p if var_p > 0 else np.zeros(len(wv)), index=used)
    info = {}
    for t in used:
        a = asset_info(t)
        info[t] = {"classe": CLASS_PT["caixa"] if t == CASH_TICKER else (CLASS_PT.get(a.macro_class, a.macro_class) if a else "não classificado"),
                   "setor": "caixa" if t == CASH_TICKER else (a.sector if a else "não classificado"),
                   "moeda": "BRL" if t == CASH_TICKER else market.currency.get(t, "?")}
    contrib = [{"ticker": t, "nome": (asset_info(t).name if asset_info(t) else ("Caixa em reais (CDI)" if t == CASH_TICKER else t)),
                "peso": float(w[t]), "contribuicao": float(rc[t]), "logo": logo(t)}
               for t in rc.sort_values(ascending=False).index]
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for dim in ("setor", "classe", "moeda"):
        lab = pd.Series({t: info[t][dim] for t in used})
        groups[dim] = {k: {"peso": float(w.groupby(lab).sum()[k]), "risco": float(rc.groupby(lab).sum()[k])}
                       for k in w.groupby(lab).sum().sort_values(ascending=False).index}
    hhi = float((wv ** 2).sum())
    conc = {"hhi": hhi, "n_efetivo": 1.0 / hhi, "apostas_independentes": effective_bets(wv, cov),
            "maior_posicao": {"ticker": str(w.idxmax()), "peso": float(w.max())}, "grupos": groups}

    # ----- liquidity
    liq = []
    for t in used:
        if t == CASH_TICKER:
            liq.append({"ticker": t, "valor": values[t], "volume_medio_reais": None, "dias_para_vender": 0.0,
                        "nota": "liquidez diária (resgate no mesmo dia ou em D+1)"})
            continue
        vol = market.volume.get(t)
        if vol is None or vol.dropna().empty:
            liq.append({"ticker": t, "valor": values[t], "volume_medio_reais": None, "dias_para_vender": None,
                        "nota": "volume negociado não disponível na fonte: não medido"})
            continue
        traded = (market.brl[t] * vol.reindex(market.brl[t].index)).dropna().iloc[-ADTV_DAYS:]
        adtv = float(traded.mean()) if len(traded) else 0.0
        days = values[t] / (PARTICIPATION * adtv) if adtv > 0 else None
        liq.append({"ticker": t, "valor": values[t], "volume_medio_reais": adtv or None, "dias_para_vender": days,
                    "nota": f"vendendo até {PARTICIPATION:.0%} do volume médio diário dos últimos {ADTV_DAYS} pregões"})
    liq.sort(key=lambda x: -(x["dias_para_vender"] or 0.0))

    # ----- sensitivities (single-factor betas, HAC t)
    sens = []
    def factor(sym: str) -> pd.Series | None:
        if not market.has(sym) and sym != FX_TICKER:
            return None
        src = market.native[sym] if sym in (FX_TICKER, RATES_US) and sym in market.native else market.brl.get(sym)
        return None if src is None else src.pct_change(fill_method=None).reindex(rp.index)
    for sym, name, reading in (
            ("BOVA11.SA", "mercado brasileiro", "a carteira variou em média {b:.2f}% para cada 1% do Ibovespa"),
            ("IVVB11.SA", "S&P 500 em reais", "a carteira variou em média {b:.2f}% para cada 1% do S&P 500 em reais"),
            (FX_TICKER, "dólar", "a carteira variou em média {b:.2f}% para cada 1% de alta do dólar"),
            (RATES_BR, "juros no Brasil (IRFM11)", "a carteira variou em média {b:.2f}% para cada 1% do IRF-M; o IRF-M cai quando os juros sobem")):
        f = factor(sym)
        if f is not None:
            row = _beta(rp, f, name, sym, reading)
            if row:
                sens.append(row)
    if any(info[t]["moeda"] == "USD" for t in used):
        f = factor(RATES_US)
        if f is not None:
            row = _beta(rp, f, "juros nos EUA (IEF)", RATES_US,
                        "a carteira variou em média {b:.2f}% para cada 1% do IEF (Treasuries de 7–10 anos, em dólar)")
            if row:
                sens.append(row)
    by_proxy = {s["proxy"]: s for s in sens}

    # ----- stress: historical windows and hypothetical shocks
    px = pd.DataFrame({t: (market.cdi if t == CASH_TICKER else market.brl[t]) for t in used
                       if (t == CASH_TICKER and market.cdi is not None) or market.has(t)}).sort_index().ffill(limit=MAX_FFILL_DAYS)
    hist_stress = []
    windows = [(n, pd.Timestamp(a), pd.Timestamp(b)) for n, a, b in HISTORICAL_WINDOWS]
    if market.has(bench_t):
        for days, label in ((21, "pior mês do Ibovespa nos dados"), (63, "pior trimestre do Ibovespa nos dados")):
            ww = _worst_window(market.brl[bench_t], days)
            if ww:
                windows.append((f"{label} ({ww[2]:.0%})", ww[0], ww[1]))
    for name, a, b in windows:
        ret, coverage = _window_return(px, w, a, b)
        hist_stress.append({"nome": name, "inicio": f"{a:%Y-%m-%d}", "fim": f"{b:%Y-%m-%d}", "retorno": ret,
                            "perda_reais": None if ret is None else ret * total, "cobertura": coverage,
                            "nota": "retornos reais dos ativos no período, com os pesos de hoje" if ret is not None else
                            f"sem dados suficientes para o período (cobertura {coverage:.0%} da carteira, mínimo {MIN_COVERAGE:.0%})"})
    hypo = []
    for proxy, label, shock in (("BOVA11.SA", "Ações brasileiras caem 20%", -0.20), ("IVVB11.SA", "S&P 500 em reais cai 20%", -0.20),
                                (FX_TICKER, "Dólar sobe 15%", 0.15), (RATES_BR, "Juros sobem (IRF-M cai 5%)", -0.05)):
        s = by_proxy.get(proxy)
        if s:
            imp = float(s["beta"]) * shock
            hypo.append({"nome": label, "choque": shock, "impacto": imp, "impacto_reais": imp * total, "beta": s["beta"],
                         "base": "hipótese: impacto = beta medido × choque; relações passadas podem não se repetir"
                                 + ("; relação estatisticamente fraca" if s["t"] is None or abs(float(s["t"])) < 2 else "")})

    # ----- what explains the risk
    explain_cols = {}
    for sym in ("BOVA11.SA", FX_TICKER, RATES_BR):
        f = factor(sym)
        if f is not None:
            explain_cols[sym] = f
    r2_multi = None
    if explain_cols:
        try:
            r2_multi = float(factor_regression(rp, pd.DataFrame(explain_cols))["r2"])
        except (InsufficientDataError, np.linalg.LinAlgError, ValueError):
            r2_multi = None
    top = contrib[:3]
    text = ("Quem mais contribui para o risco: " + "; ".join(f"{c['ticker']} ({c['contribuicao']:.0%} do risco com "
                                                            f"{c['peso']:.0%} do dinheiro)" for c in top) + ".")
    if r2_multi is not None:
        text += (f" Cerca de {r2_multi:.0%} da variação diária da carteira acompanha o mercado brasileiro, o dólar e os "
                 "juros (R² de uma regressão linear: correlação, não causa); o restante é específico dos ativos.")
    warns = [f"{t}: {why} (fora das estatísticas; {w_all[t]:.0%} da carteira)" for t, why in short.items()]
    warns += [f"{t}: {why}" for t, why in missing.items()]
    warns.append("as estimativas de risco usam o passado: a perda futura pode ser maior do que elas indicam")
    out = {
        "data_referencia": f"{r.index[-1]:%Y-%m-%d}", "periodo_dados": [f"{r.index[0]:%Y-%m-%d}", f"{r.index[-1]:%Y-%m-%d}"],
        "n_obs": len(r), "patrimonio": total, "pesos": {t: float(x) for t, x in w.items()},
        "fora_da_analise": short, "peso_fora": float(w_all[list(short)].sum()) if short else 0.0, "sem_dados": missing,
        "volatilidade": {"historica": vol_hist, "ewma": vol_ewma},
        "var": var_rows,
        "drawdown": {"maximo": float(dd.min()), "atual": float(dd.iloc[-1]), "pico": f"{dd_peak:%Y-%m-%d}", "fundo": f"{trough:%Y-%m-%d}"},
        "correlacao": corr_out, "contribuicao_risco": contrib, "concentracao": conc, "liquidez": liq,
        "sensibilidades": sens, "estresse_historico": hist_stress, "estresse_hipotetico": hypo,
        "o_que_explica": {"maiores_contribuintes": top, "r2_fatores": r2_multi, "texto": text},
        "fontes": market.sources, "avisos": warns, "aviso": DISCLAIMER,
    }
    bad = guarantee_phrases(" ".join([text, *warns, *[str(x["nota"]) for x in var_rows]]))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out
