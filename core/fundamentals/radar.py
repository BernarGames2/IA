"""Opportunity radar: scans the stocks of the catalogue (in the markets the investor chooses) and ranks them by an
explainable score. Every point of the score is a measured number with its source; each candidate comes with the
evidence against it, the value-trap check and the conditions that would undo its place. It ranks what the rules see
in published data — not a recommendation and not a forecast of returns.

Factors (z-scores among the scanned stocks, winsorised at ±3; a missing measure counts as 0 = neutral, and the
coverage is shown):

* valor: earnings yield (1 / P/L), free-cash-flow yield, dividend yield;
* qualidade: ROE, operating margin and net debt / EBITDA (not for banks and insurers), Piotroski F-score;
* crescimento: revenue growth over 3 years (statements) or in the last 12 months (snapshot), earnings growth;
* momento: return over 12 months excluding the last one, position vs the 200-day average;
* risco (less is better): volatility and maximum drawdown over 5 years.

The style sets the weights (``STYLES``). A stock that looks cheap loses 0.25 point per value-trap signal beyond the
first and 0.5 more when the trap risk is "alto". Stocks with less than half of the factors measured are listed apart.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.fundamentals.analysis import growth, piotroski, valuation, year_ratios
from core.fundamentals.sources import Annual, _cache_path, _cached, _store, fetch_annual
from core.fundamentals.value_trap import detect, price_stats
from core.market_data.base import last_per_date
from core.market_data.logos import logo_path
from core.purpose.catalog import REAL_CATALOG, SECTOR_LABEL, CatalogAsset
from core.purpose.language import guarantee_phrases
from core.stock_check.company import SourceError, YahooFundamentals, parse_summary
from core.stock_check.sectors import segment_for_class
from utils.br_format import num, pct

STYLES = {  # valor, qualidade, crescimento, momento, risco
    "equilibrado": (0.30, 0.30, 0.15, 0.15, 0.10),
    "valor": (0.50, 0.25, 0.05, 0.10, 0.10),
    "qualidade": (0.15, 0.50, 0.15, 0.10, 0.10),
    "crescimento": (0.10, 0.25, 0.45, 0.15, 0.05),
    "dividendos": (0.45, 0.30, 0.05, 0.05, 0.15),
    "momento": (0.05, 0.20, 0.15, 0.50, 0.10),
}
FACTORS = ("valor", "qualidade", "crescimento", "momento", "risco")
FACTOR_PT = {"valor": "valor (preço baixo pelo que entrega)", "qualidade": "qualidade", "crescimento": "crescimento",
             "momento": "momento do preço", "risco": "risco baixo"}
MARKETS = {"BR": "acoes_br", "US": "acoes_eua"}
MIN_COVERAGE = 0.5
DISCLAIMER = ("Ordenação por regras fixas sobre dados publicados (cotações e demonstrações): não é recomendação de "
              "compra ou venda, não é previsão de retorno e não oferece garantia de resultado. Cada candidato traz a "
              "evidência contra e o que desfaria a sua posição.")


@dataclass
class Item:
    asset: CatalogAsset
    px: pd.Series | None = None
    snap: dict[str, object] | None = None
    annual: Annual | None = None
    error: str | None = None


def _z(x: pd.Series) -> pd.Series:
    v = x.astype(float)
    ok = v.notna()
    if ok.sum() < 3 or float(v[ok].std(ddof=0)) == 0:
        return pd.Series(np.where(ok, 0.0, np.nan), index=x.index)
    z = (v - v[ok].mean()) / v[ok].std(ddof=0)
    return z.clip(-3, 3)


def _fetch_fundamentals(client: YahooFundamentals, ticker: str, cache_dir: Path | None, now: float
                        ) -> tuple[dict[str, object], Annual | None]:
    """Snapshot indicators and annual statements of one stock (runs in a worker; the caller assigns the result,
    so an answer that arrives after the time budget never changes a ranking already computed)."""
    path = _cache_path(cache_dir, "resumo", ticker)
    raw = _cached(path, now)
    if raw is None:
        raw = parse_summary(client.summary(ticker))[1]
        _store(path, raw, now)
    return raw, fetch_annual(client, ticker, cache_dir, now)  # type: ignore[return-value]


def measures(item: Item) -> dict[str, float | None]:
    """The measured inputs of every factor (None = not measured)."""
    a, snap = item.annual, item.snap or {}
    seg = segment_for_class(item.asset.macro_class, item.asset.asset_class, item.asset.business)
    fin = bool(seg and seg.financeiro)
    val = valuation(a, snap)
    r = year_ratios(a, -1) if a is not None else {}
    g = growth(a) if a is not None else {}
    fs = piotroski(a) if a is not None and not fin else None
    ps = price_stats(item.px) if item.px is not None and len(item.px) else {}
    rets = item.px.pct_change(fill_method=None).dropna() if item.px is not None else pd.Series(dtype=float)
    vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 60 else None
    dd = None
    if item.px is not None and len(item.px) > 60:
        w = item.px / item.px.cummax() - 1.0
        dd = float(w.min())
    return {
        "lucro_yield": val["lucro_yield"], "fcl_yield": None if fin else val["fcl_yield"],
        "dividend_yield": val["dividend_yield"], "roe": r.get("roe") if r else snap.get("roe"),  # type: ignore[dict-item]
        "margem_operacional": None if fin else (r.get("margem_operacional") if r else snap.get("margem_operacional")),  # type: ignore[dict-item]
        "divida_liquida_ebitda": None if fin else r.get("divida_liquida_ebitda"),
        "f_score": None if fs is None or fs["medidos"] < 6 else float(fs["pontos"]),  # type: ignore[arg-type]
        "receita_crescimento": g.get("receita_cagr_3a") if g.get("receita_cagr_3a") is not None else snap.get("crescimento_receita"),  # type: ignore[dict-item]
        "lucro_crescimento": snap.get("crescimento_lucro"),  # type: ignore[dict-item]
        "momento_12_1": ps.get("momento_12_1"), "acima_media_200": ps.get("acima_media_200"),
        "volatilidade": vol, "queda_maxima": dd, "pl": val["pl"],
    }


# measure -> (factor, sign): +1 = more is better, -1 = less is better
MEASURE_FACTOR = {"lucro_yield": ("valor", 1), "fcl_yield": ("valor", 1), "dividend_yield": ("valor", 1),
                  "roe": ("qualidade", 1), "margem_operacional": ("qualidade", 1), "divida_liquida_ebitda": ("qualidade", -1),
                  "f_score": ("qualidade", 1), "receita_crescimento": ("crescimento", 1), "lucro_crescimento": ("crescimento", 1),
                  "momento_12_1": ("momento", 1), "acima_media_200": ("momento", 1), "volatilidade": ("risco", -1),
                  "queda_maxima": ("risco", 1)}
MEASURE_PT = {"lucro_yield": "lucro sobre o preço", "fcl_yield": "caixa livre sobre o valor de mercado",
              "dividend_yield": "dividendos", "roe": "ROE", "margem_operacional": "margem operacional",
              "divida_liquida_ebitda": "dívida líquida/EBITDA", "f_score": "F-score", "receita_crescimento": "crescimento da receita",
              "lucro_crescimento": "crescimento do lucro", "momento_12_1": "retorno de 12 meses (sem o último)",
              "acima_media_200": "preço em relação à média de 200 dias", "volatilidade": "volatilidade",
              "queda_maxima": "maior queda em 5 anos"}


def _fmt_measure(k: str, v: float) -> str:
    if k in ("divida_liquida_ebitda",):
        return f"{num(v)}×"
    if k == "f_score":
        return f"{v:.0f} de 9"
    if k == "acima_media_200":
        return "acima" if v >= 0.5 else "abaixo"
    if k == "lucro_yield":
        return pct(v) + (f" (P/L {num(1 / v)})" if v > 0 else "")
    return pct(v, sign=k in ("momento_12_1", "receita_crescimento", "lucro_crescimento"))


def score(items: list[Item], style: str) -> pd.DataFrame:
    rows = {it.asset.ticker: measures(it) for it in items}
    m = pd.DataFrame(rows).T
    z = pd.DataFrame(index=m.index)
    for k, (_f, sign) in MEASURE_FACTOR.items():
        z[k] = _z(m[k].astype(float) * sign) if k in m else np.nan
    w = dict(zip(FACTORS, STYLES[style], strict=True))
    if style == "dividendos":                       # dividends weigh twice inside the value factor
        z["dividend_yield"] = z["dividend_yield"] * 2
    out = pd.DataFrame(index=m.index)
    for f in FACTORS:
        cols = [k for k, (ff, _) in MEASURE_FACTOR.items() if ff == f]
        sub = z[cols]
        out[f] = sub.mean(axis=1, skipna=True)
        out[f + "_medido"] = sub.notna().any(axis=1)
    out["cobertura"] = out[[f + "_medido" for f in FACTORS]].mean(axis=1)
    out["base"] = sum(out[f].fillna(0.0) * w[f] for f in FACTORS)
    return out.join(m, rsuffix="_m").join(z.add_prefix("z_"))


def scan(markets: list[str], style: str, provider: object, client: YahooFundamentals, cache_dir: Path | None,
         progress: Callable[[str], None] | None = None, budget_s: float = 35.0, top: int = 10,
         now: float | None = None) -> dict[str, object]:
    if style not in STYLES:
        raise ValueError(f"estilo desconhecido: {style}")
    classes = {MARKETS[m] for m in markets if m in MARKETS}
    if not classes:
        raise ValueError("escolha ao menos um mercado: BR ou US")
    assets = [a for a in REAL_CATALOG if a.macro_class in classes]
    t0 = time.monotonic()
    t_now = time.time() if now is None else now
    if progress:
        progress(f"baixando 5 anos de cotações de {len(assets)} ações…")
    got = provider.fetch([a.ticker for a in assets], "5y", "1d")  # type: ignore[attr-defined]
    items = []
    for a in assets:
        rs = got.get(a.ticker)
        px = None
        if rs is not None and rs.close is not None:
            s = last_per_date(rs.close.dropna()).sort_index()
            px = s[s > 0] if len(s) else None
        items.append(Item(a, px))
    if progress:
        progress(f"lendo demonstrações e indicadores de {len(items)} empresas (até {budget_s:.0f} s)…")
    pool = ThreadPoolExecutor(max_workers=8)
    futs = {pool.submit(_fetch_fundamentals, client, it.asset.ticker, cache_dir, t_now): it for it in items}
    done, pending = wait(futs, timeout=max(1.0, budget_s - (time.monotonic() - t0)))
    pool.shutdown(wait=False, cancel_futures=True)
    for f in done:
        try:
            futs[f].snap, futs[f].annual = f.result()
        except (SourceError, OSError, ValueError, KeyError, TypeError) as e:
            futs[f].error = str(e) or type(e).__name__
    for f in pending:
        futs[f].error = "tempo esgotado antes da resposta da fonte"
    timed_out = {futs[f].asset.ticker for f in pending}
    usable = [it for it in items if it.px is not None and len(it.px) > 252]
    if len(usable) < 5:
        raise ValueError("menos de 5 ações com cotações suficientes: tente de novo em alguns minutos")
    sc = score(usable, style)
    by_t = {it.asset.ticker: it for it in usable}
    results = []
    for tk, row in sc.iterrows():
        it = by_t[str(tk)]
        seg = segment_for_class(it.asset.macro_class, it.asset.asset_class, it.asset.business)
        fin = bool(seg and seg.financeiro)
        trap = detect(it.annual, it.snap or {}, None, price_stats(it.px), fin)  # type: ignore[arg-type]
        penalty = 0.0
        if trap["parece_barata"]:
            penalty = 0.25 * max(0, int(trap["acionados"]) - 1) + (0.5 if trap["nivel"] == "alto" else 0.0)  # type: ignore[call-overload]
        contrib = {f: (None if pd.isna(row[f]) else float(row[f]) * dict(zip(FACTORS, STYLES[style], strict=True))[f]) for f in FACTORS}
        ev, against = [], []
        for k in MEASURE_FACTOR:
            zk, v = row.get("z_" + k), row.get(k)
            if zk is None or pd.isna(zk) or v is None or pd.isna(v):
                continue
            text = f"{MEASURE_PT[k]}: {_fmt_measure(k, float(v))}"
            (ev if zk >= 0.5 else against if zk <= -0.5 else []).append((float(zk), text))
        ev.sort(key=lambda x: -x[0])
        against.sort(key=lambda x: x[0])
        hits = [f"{s['sinal']} ({s['valor']})" for s in trap["sinais"] if s["acionado"]]  # type: ignore[union-attr]
        top_f = max((f for f in FACTORS if contrib[f] is not None), key=lambda f: contrib[f], default=None)  # type: ignore[arg-type, return-value]
        undo = _undo(top_f, row, trap)
        results.append({
            "ticker": tk, "nome": it.asset.name, "logo": logo_path(str(tk)), "setor": SECTOR_LABEL.get(it.asset.sector, it.asset.sector),
            "classe": it.asset.macro_class, "pontuacao": float(row["base"]) - penalty, "penalidade_armadilha": penalty,
            "fatores": {f: {"z": None if pd.isna(row[f]) else float(row[f]), "contribuicao": contrib[f]} for f in FACTORS},
            "cobertura": float(row["cobertura"]), "evidencias": [t for _, t in ev[:3]],
            "contra": [t for _, t in against[:2]] + hits[:3],
            "armadilha": {"parece_barata": trap["parece_barata"], "nivel": trap["nivel"], "acionados": trap["acionados"],
                          "medidos": trap["medidos"], "criterios": trap["criterios_barata"]},
            "fator_principal": top_f, "desfaz_se": undo,
            "sem_fundamentos": it.annual is None and not it.snap, "erro_fonte": it.error,
        })
    ranked = sorted([r for r in results if r["cobertura"] >= MIN_COVERAGE], key=lambda r: -r["pontuacao"])
    for i, r in enumerate(ranked, 1):
        r["posicao"] = i
    thin = [r["ticker"] for r in results if r["cobertura"] < MIN_COVERAGE]
    n_fund = sum(1 for it in usable if it.annual is not None or it.snap)
    out = {
        "data": datetime.fromtimestamp(t_now, UTC).date().isoformat(),
        "estilo": style, "pesos": dict(zip(FACTORS, STYLES[style], strict=True)), "mercados": sorted(markets),
        "candidatos": ranked[:top], "analisados": len(usable), "com_fundamentos": n_fund,
        "sem_cotacoes": [a.ticker for a in assets if a.ticker not in by_t], "tempo_esgotado": sorted(timed_out),
        "dados_insuficientes": thin,
        "regras": ["fatores em notas z entre as ações analisadas (limitadas a ±3); medida ausente conta como neutra",
                   "pesos do estilo " + style + ": " + ", ".join(f"{FACTOR_PT[f]} {pct(w, 0)}" for f, w in
                                                                  zip(FACTORS, STYLES[style], strict=True)),
                   "ação que parece barata perde 0,25 ponto por sinal de armadilha além do primeiro, e mais 0,5 com risco alto",
                   f"ações com menos de {pct(MIN_COVERAGE, 0)} dos fatores medidos ficam fora da lista"],
        "fontes": ["Yahoo Finance — cotações ajustadas (5 anos)", "Yahoo Finance — indicadores e demonstrações anuais"],
        "aviso": DISCLAIMER,
    }
    bad = guarantee_phrases(" ".join([DISCLAIMER, *(" ".join(r["evidencias"] + r["contra"] + r["desfaz_se"]) for r in ranked)]))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out


def _undo(top: str | None, row: pd.Series, trap: dict[str, object]) -> list[str]:
    out = []
    g = lambda k: None if row.get(k) is None or pd.isna(row.get(k)) else float(row.get(k))
    if top == "valor" and g("pl") is not None and g("pl") > 0:  # type: ignore[operator]
        out.append(f"o P/L subir para mais de {num(g('pl') * 1.5)} (hoje {num(g('pl'))}) sem o lucro acompanhar")  # type: ignore[operator]
    if top == "valor" and g("dividend_yield"):
        out.append(f"os dividendos caírem para menos de {pct(g('dividend_yield') * 0.6)} ao ano (hoje {pct(g('dividend_yield'))})")  # type: ignore[operator]
    if top == "qualidade" and g("roe") is not None:
        out.append(f"o ROE cair abaixo de {pct(g('roe') - 0.05)} (hoje {pct(g('roe'))})")  # type: ignore[operator]
    if top == "crescimento" and g("receita_crescimento") is not None:
        out.append("a receita parar de crescer (variação anual abaixo de 0%)")
    if top == "momento":
        out.append("o preço fechar abaixo da média de 200 dias")
    if top == "risco" and g("volatilidade") is not None:
        out.append(f"a volatilidade passar de {pct(g('volatilidade') * 1.5, 0)} ao ano (hoje {pct(g('volatilidade'), 0)})")  # type: ignore[operator]
    if g("f_score") is not None:
        out.append("o F-score cair para 3 ou menos no próximo balanço anual")
    if trap["parece_barata"]:
        out.append("os sinais de armadilha chegarem a 4 ou mais")
    return out
