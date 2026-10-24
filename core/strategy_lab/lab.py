"""Strategy lab: test a simple, published investing idea on real prices — with the protections that keep a backtest
honest — and say plainly when there is no evidence.

Hypotheses (each a small family of variants; EVERY variant counts as a trial in the multiple-testing correction):

* momento — hold the k stocks with the highest return over the last L months excluding the most recent one
  (Jegadeesh & Titman, 1993); L in {6, 12}, k in {5, 10};
* baixa_volatilidade — the k stocks with the lowest volatility over the last year (Baker, Bradley & Wurgler, 2011);
  k in {5, 10}, plus every stock weighted by the inverse of its volatility;
* tendencia — each stock is held only while its price is above its N-day average, the rest stays in cash earning
  the risk-free rate (Faber, 2007); N in {100, 200};
* reversao — the k stocks that fell the most in the last month (Jegadeesh, 1990); k in {5, 10}.

Rules: monthly rebalance (21 sessions); each decision uses only the prices up to the previous close (walk-forward;
checked by perturbing the future); equal weights inside the selection; costs of ``custo_bps`` per unit of turnover
plus 5 bps of slippage.

Protocol: after one year of warm-up, the out-of-sample period is split in time: the first 60% picks the best variant
(validation Sharpe); the last 40% is a final test the choice never saw. Verdict, strict on purpose:

* "há evidência" — in the final test the chosen variant beats, after costs, both the equal-weight portfolio of the
  same stocks and the market index in Sharpe ratio; its deflated Sharpe ratio over the whole out-of-sample period,
  given all variants tried, is at least 0.95; and it beats equal weight in at least 2 of the 3 thirds of that period;
* "evidência fraca" — it beats both in the final test, but the deflated Sharpe or the consistency fails;
* "sem evidência" — otherwise.
Biases stated in the result: today's stocks applied to the past (survivorship), no taxes, no liquidity limits.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from core.backtesting.overfitting import dsr, psr
from core.backtesting.walk_forward import (
    BacktestSettings,
    lookahead_invariance_check,
    run_strategy,
)
from core.market_data.base import last_per_date
from core.metrics import performance_summary
from core.purpose.catalog import REAL_CATALOG
from core.purpose.engine import norm_ticker
from core.purpose.language import guarantee_phrases
from utils.br_format import num, pct
from utils.validation import InsufficientDataError

CASH, INDEX = "CAIXA", "INDICE"
LOOKBACK, REBAL = 252, 21
MIN_STOCKS = 8
INDEX_TICKER = {"BR": "BOVA11.SA", "US": "SPY"}
HYPOTHESES = {
    "momento": "Ações que mais subiram nos últimos meses continuam subindo? (momento)",
    "baixa_volatilidade": "Ações menos voláteis rendem melhor pelo risco que correm? (baixa volatilidade)",
    "tendencia": "Ficar fora das ações abaixo da média móvel reduz as quedas sem perder retorno? (tendência)",
    "reversao": "As que mais caíram no último mês voltam a subir? (reversão de curto prazo)",
}
DISCLAIMER = ("Teste histórico de uma ideia publicada, com custos e controles contra o excesso de ajuste: o passado pode "
              "não se repetir, o universo é o de hoje aplicado ao passado e nada aqui é recomendação ou garantia de "
              "resultado.")

Strategy = Callable[[pd.DataFrame, np.ndarray | None], np.ndarray]


def _equal(cols: list[str], chosen: list[str]) -> np.ndarray:
    w = np.array([1.0 if c in chosen else 0.0 for c in cols])
    return w / w.sum() if w.sum() > 0 else w


def variants(hyp: str, stocks: list[str], cols: list[str], has_cash: bool) -> dict[str, Strategy]:
    """Strategy functions on the training window (daily simple returns of every column)."""
    out: dict[str, Strategy] = {}

    def cum(train: pd.DataFrame, start: int, end: int) -> pd.Series:
        sub = train[stocks].iloc[start:end] if end else train[stocks].iloc[start:]
        return (1 + sub).prod() - 1

    if hyp == "momento":
        for months in (6, 12):
            for k in (5, 10):
                def f(train, prev, months=months, k=k):
                    score = cum(train, -21 * months - 21, -21)
                    return _equal(cols, list(score.nlargest(k).index))
                out[f"{months} meses, {k} ações"] = f
    elif hyp == "baixa_volatilidade":
        for k in (5, 10):
            def f(train, prev, k=k):
                return _equal(cols, list(train[stocks].std().nsmallest(k).index))
            out[f"{k} menos voláteis"] = f

        def inv(train, prev):
            iv = 1.0 / train[stocks].std().replace(0, np.nan)
            w = np.array([float(iv.get(c, 0.0)) if c in stocks else 0.0 for c in cols])
            w = np.nan_to_num(w)
            return w / w.sum()
        out["todas, peso inverso à volatilidade"] = inv
    elif hyp == "tendencia":
        if not has_cash:
            raise ValueError("a hipótese de tendência precisa do rendimento do caixa (taxa de juros histórica), "
                             "indisponível agora")
        for n in (100, 200):
            def f(train, prev, n=n):
                px = (1 + train[stocks]).cumprod()
                above = [s for s in stocks if px[s].iloc[-1] > px[s].iloc[-n:].mean()]
                w = np.array([1.0 / len(stocks) if c in above else 0.0 for c in cols])
                w[cols.index(CASH)] = 1.0 - w.sum()
                return w
            out[f"média de {n} dias"] = f
    elif hyp == "reversao":
        for k in (5, 10):
            def f(train, prev, k=k):
                return _equal(cols, list(cum(train, -21, 0).nsmallest(k).index))
            out[f"{k} que mais caíram no mês"] = f
    else:
        raise ValueError(f"hipótese desconhecida: {hyp}")
    return out


def build_returns(prices: dict[str, pd.Series], index_px: pd.Series | None, cash_index: pd.Series | None,
                  min_days: int = 1260) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Daily returns of the stocks with complete data over the longest common window (10 years when enough stocks
    have them, else 5); the cash and index columns when available. Returns (frame, stocks used, left out)."""
    clean = {}
    for t, s in prices.items():
        s = last_per_date(s.dropna()).sort_index()
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        clean[t] = s[s > 0]
    if index_px is None or index_px.dropna().empty:
        raise InsufficientDataError("índice de mercado indisponível")
    ix = last_per_date(index_px.dropna()).sort_index()
    ix.index = pd.DatetimeIndex(ix.index).tz_localize(None).normalize() if pd.DatetimeIndex(ix.index).tz is not None else pd.DatetimeIndex(ix.index).normalize()
    cal = ix.index
    aligned = pd.DataFrame({t: s.reindex(cal.union(s.index)).ffill(limit=5).reindex(cal) for t, s in clean.items()})
    window = None
    for days in (2520, min_days):
        tail = aligned.iloc[-(days + 1):]
        ok = [c for c in tail.columns if tail[c].notna().all()]
        if len(ok) >= MIN_STOCKS and len(tail) >= days:
            window = (days, ok)
            break
    if window is None:
        raise InsufficientDataError(f"menos de {MIN_STOCKS} ações com 5 anos completos de cotações")
    days, ok = window
    px = aligned[ok].iloc[-(days + 1):]
    frame = px.pct_change(fill_method=None).iloc[1:]
    frame[INDEX] = ix.reindex(px.index).pct_change(fill_method=None).iloc[1:]
    if cash_index is not None and not cash_index.dropna().empty:
        ci = last_per_date(cash_index.dropna()).sort_index()
        ci.index = pd.DatetimeIndex(ci.index).normalize()
        cr = ci.reindex(ci.index.union(px.index)).ffill(limit=5).reindex(px.index).pct_change(fill_method=None).iloc[1:]
        if cr.notna().mean() > 0.95:
            frame[CASH] = cr.fillna(0.0)
    frame = frame.dropna(subset=[INDEX])
    left = sorted(set(prices) - set(ok))
    return frame.fillna(0.0), ok, left


def _sharpe(x: pd.Series, rf: pd.Series | None) -> float | None:
    ex = x - (rf.reindex(x.index).fillna(0.0) if rf is not None else 0.0)
    sd = float(ex.std(ddof=1))
    return float(ex.mean() / sd * np.sqrt(252)) if sd > 0 else None


def run_lab(hyp: str, frame: pd.DataFrame, stocks: list[str], cost_bps: float = 15.0, rf_annual: float | None = None
            ) -> dict[str, object]:
    if not 0 <= cost_bps <= 200:
        raise ValueError("custo por operação entre 0 e 200 pontos-base")
    cols = list(frame.columns)
    has_cash = CASH in cols
    s = BacktestSettings(lookback=LOOKBACK, rebalance_every=REBAL, cost_bps=cost_bps, slippage_bps=5.0,
                         rf_annual=0.0, validation_frac=0.6)
    strat = variants(hyp, stocks, cols, has_cash)
    bts = {k: run_strategy(k, f, frame, s) for k, f in strat.items()}
    bts["igual"] = run_strategy("igual", lambda tr, pv: _equal(cols, stocks), frame, s)
    bts["indice"] = run_strategy("indice", lambda tr, pv: _equal(cols, [INDEX]), frame, s)
    rf = frame[CASH] if has_cash else None
    oos = bts["igual"].net_returns.index
    n_val = int(len(oos) * s.validation_frac)
    val_idx, test_idx = oos[:n_val], oos[n_val:]
    thirds = np.array_split(np.arange(len(oos)), 3)

    def seg(k: str, idx: pd.Index) -> dict[str, float | None]:
        x = bts[k].net_returns.loc[idx]
        m = performance_summary(x.to_numpy(), 0.0, 252)
        return {"sharpe": _sharpe(x, rf), "cagr": m["cagr"], "vol": m["volatility_ann"], "queda_maxima": m["max_drawdown"]}

    table = []
    for k in strat:
        v, t = seg(k, val_idx), seg(k, test_idx)
        to = bts[k].turnover
        table.append({"variante": k, "sharpe_validacao": v["sharpe"], "sharpe_teste": t["sharpe"], "cagr_teste": t["cagr"],
                      "queda_maxima_teste": t["queda_maxima"], "giro_anual": float(to.sum() / (len(to) / 252)),
                      "custo_anual": float(bts[k].costs.sum() / (len(to) / 252))})
    valid = [r for r in table if r["sharpe_validacao"] is not None]
    if not valid:
        raise ValueError("nenhuma variante teve retorno variável no período de validação")
    best = max(valid, key=lambda r: r["sharpe_validacao"])
    b = best["variante"]
    bench = {k: {"validacao": seg(k, val_idx), "teste": seg(k, test_idx)} for k in ("igual", "indice")}
    trial = [float((bts[k].net_returns.loc[oos] - (rf.loc[oos] if rf is not None else 0.0)).mean()
                   / max(1e-12, float((bts[k].net_returns.loc[oos] - (rf.loc[oos] if rf is not None else 0.0)).std(ddof=1))))
             for k in strat]
    ex_best = (bts[b].net_returns.loc[oos] - (rf.loc[oos] if rf is not None else 0.0)).to_numpy()
    try:
        d = dsr(ex_best, trial)
        dsr_v: float | None = float(d["dsr"])
        psr_v: float | None = float(psr(ex_best, 0.0))
    except InsufficientDataError:
        dsr_v = psr_v = None
    wins = 0
    third_rows = []
    for part in thirds:
        idx = oos[part]
        sb, se = _sharpe(bts[b].net_returns.loc[idx], rf), _sharpe(bts["igual"].net_returns.loc[idx], rf)
        won = sb is not None and se is not None and sb > se
        wins += int(won)
        third_rows.append({"inicio": f"{idx[0]:%Y-%m-%d}", "fim": f"{idx[-1]:%Y-%m-%d}", "sharpe": sb, "sharpe_igual": se, "venceu": won})
    tb, te, ti = best["sharpe_teste"], bench["igual"]["teste"]["sharpe"], bench["indice"]["teste"]["sharpe"]
    beats = tb is not None and te is not None and ti is not None and tb > te and tb > ti
    deflated_ok = dsr_v is not None and dsr_v >= 0.95
    if beats and deflated_ok and wins >= 2:
        verdict = "há evidência"
    elif beats:
        verdict = "evidência fraca"
    else:
        verdict = "sem evidência"
    look = lookahead_invariance_check(frame, strat[b], s, cut=len(frame) // 2)
    curves = {}
    for k, lab in ((b, "estratégia escolhida"), ("igual", "pesos iguais"), ("indice", "índice")):
        v = (1 + bts[k].net_returns.loc[oos]).cumprod() * 100
        step = max(1, len(v) // 260)
        curves[lab] = [float(x) for x in v.iloc[::step]]
    dates = [f"{d:%Y-%m-%d}" for d in oos[::max(1, len(oos) // 260)]]
    criteria = [
        {"criterio": "vence pesos iguais e o índice no teste final (Sharpe após custos)", "ok": beats,
         "detalhe": f"{num(tb, 2)} contra {num(te, 2)} e {num(ti, 2)}"},
        {"criterio": "Sharpe deflacionado ≥ 0,95 (considera as variantes testadas)", "ok": deflated_ok,
         "detalhe": f"{num(dsr_v, 2)} com {len(strat)} variantes"},
        {"criterio": "vence pesos iguais em pelo menos 2 dos 3 terços do período", "ok": wins >= 2, "detalhe": f"{wins} de 3"},
    ]
    reading = {"há evidência": "Os dados mostram evidência a favor da ideia neste universo e período, depois de custos e da "
                               "correção por testar várias variantes.",
               "evidência fraca": "A variante escolhida venceu no teste final, mas não passou em todos os controles: pode "
                                  "ter sido sorte ou excesso de ajuste.",
               "sem evidência": "Não há evidência de que a ideia supere as alternativas simples neste universo e período."}[verdict]
    out = {
        "hipotese": hyp, "pergunta": HYPOTHESES[hyp], "veredito": verdict, "leitura": reading, "criterios": criteria,
        "variante_escolhida": b, "variantes": table, "comparacao": bench, "tercos": third_rows,
        "dsr": dsr_v, "psr": psr_v, "n_variantes": len(strat),
        "periodo": {"inicio_fora_da_amostra": f"{oos[0]:%Y-%m-%d}", "fim_validacao": f"{val_idx[-1]:%Y-%m-%d}",
                    "inicio_teste": f"{test_idx[0]:%Y-%m-%d}", "fim": f"{oos[-1]:%Y-%m-%d}", "anos": round(len(frame) / 252, 1)},
        "curvas": {"datas": dates, "series": curves, "inicio_teste": f"{test_idx[0]:%Y-%m-%d}"},
        "sem_olhar_o_futuro": {"decisoes_conferidas": look["decisions_checked"], "passou": look["passed"]},
        "acoes": stocks, "custo_bps": cost_bps, "caixa": has_cash,
        "avisos": [("universo de hoje aplicado ao passado: empresas que deixaram de existir ou de ser grandes não entram "
                    "(viés de sobrevivência, favorável às estratégias)"),
                   (f"custos de {num(cost_bps, 0)} pontos-base por unidade de giro mais 5 de derrapagem; sem impostos e sem "
                    "limite de liquidez"),
                   f"Sharpe medido contra {'o caixa (taxa de juros)' if has_cash else 'zero (sem taxa de juros histórica)'}"],
        "aviso": DISCLAIMER,
    }
    if not has_cash:
        out["avisos"].append("sem a taxa de juros histórica: o Sharpe usa retorno sem desconto da renda fixa")  # type: ignore[union-attr]
    bad = guarantee_phrases(" ".join([reading, DISCLAIMER, *out["avisos"]]))  # type: ignore[arg-type]
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out


def universe(market: str, tickers: list[str] | None) -> list[str]:
    if tickers:
        out = sorted({norm_ticker(t) for t in tickers if t.strip()})
        if len(out) < MIN_STOCKS:
            raise ValueError(f"informe pelo menos {MIN_STOCKS} ações (são escolhidas k de cada vez)")
        return out
    cls = {"BR": "acoes_br", "US": "acoes_eua"}.get(market)
    if cls is None:
        raise ValueError("mercado: BR ou US")
    return [a.ticker for a in REAL_CATALOG if a.macro_class == cls]


def describe_rf(rf: float | None) -> str:
    return "n/d" if rf is None else pct(rf)
