"""Ação do dia: with the money the investor has today, which single stock the fixed rules point to.

The answer is the result of rules applied to the price history, the same for every investor with the same
inputs and data — not a forecast of tomorrow's price and not a recommendation. Rules (shown in the result):

1. Universe: the stocks of the curated catalogue (large, liquid B3 and US companies, each with a documented
   business), in the markets the investor accepts. ETFs, funds and crypto are left out: a stock was asked for.
2. Data: about 5 years of adjusted daily prices in reais (US stocks converted at each day's exchange rate);
   a stock needs ``MIN_CANDIDATE_OBS`` prices (about 3 years) to be scored.
3. Score: the same score the portfolio engine uses to pick candidates (``core.universe.scoring``): z-scores of
   the Sharpe ratio shrunk toward the group mean, volatility, maximum drawdown, 12-1 momentum and liquidity,
   weighted by profile (a conservative profile weighs volatility and drawdown more; an aggressive one,
   momentum and Sharpe).
4. Risk ceiling for ONE stock (it carries all the risk of the money put in it): conservador — volatility up to
   25% a year and maximum drawdown no worse than -40%; moderado — volatility up to 35%; arrojado — no ceiling.
5. It must fit the money: one lot (B3: 1 share on the fractional market; US: 1 share, or fractions when the
   investor's broker allows them) at the last close.
6. Diversification with what the investor already has (optional): a stock already held is not suggested
   again, and the score is lowered by ``CORR_PENALTY`` x the correlation of its daily returns with the
   holdings (weighted by value) — a stock that moves with what is already owned adds little.
The pick is the best adjusted score; the next ones are shown as alternatives. With the same data the pick is
the same: it changes only when the prices (or the inputs) change.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEMO_RISK_FREE_FALLBACK
from core.market_data.cache import SeriesCache
from core.market_data.logos import logo_path
from core.market_data.quality import QualitySettings
from core.market_data.risk_free import fetch_live_rates
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider
from core.purpose.catalog import (
    REAL_CATALOG,
    SECTOR_LABEL,
    SYNTHETIC_CATALOG,
    CatalogAsset,
)
from core.purpose.feasibility import lot_size
from core.purpose.language import guarantee_phrases
from core.purpose.personalization import traits
from core.universe.scoring import (
    MIN_CANDIDATE_OBS,
    daily_returns,
    measure,
    score_components,
)
from models.investor import RiskProfile

MIN_OBS = 252
CORR_PENALTY = 0.5
CEILING = {"conservador": (0.25, -0.40), "moderado": (0.35, None), "arrojado": (None, None)}
STOCK_CLASSES = ("acoes_br", "acoes_eua")
DISCLAIMER = ("Resultado de regras fixas aplicadas ao histórico de preços: não é previsão do preço de amanhã, não é "
              "recomendação de investimento e não oferece garantia de resultado. Confira o preço na corretora antes de comprar.")


@dataclass
class DailyPick:
    amount: float
    profile: str
    origin: str                                  # "real" | "sintetico"
    as_of: str
    pick: dict[str, object] | None
    alternatives: list[dict[str, object]]
    left_out: dict[str, int]
    rules: list[str]
    warnings: list[str]
    sources: list[dict[str, str]]
    rf: float
    rf_source: str
    n_measured: int
    holdings_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        from utils.serialization import to_jsonable

        return to_jsonable({  # type: ignore[return-value]
            "valor": self.amount, "perfil": self.profile, "origem": self.origin, "data_referencia": self.as_of,
            "escolha": self.pick, "alternativas": self.alternatives, "fora": self.left_out, "regras": self.rules,
            "avisos": self.warnings, "fontes": self.sources, "taxa_livre_risco": self.rf,
            "fonte_taxa_livre_risco": self.rf_source, "acoes_medidas": self.n_measured,
            "carteira_considerada": self.holdings_used, "aviso": DISCLAIMER})


def _money(x: float) -> str:
    return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%".replace(".", ",")


def _load(assets: list[CatalogAsset], extra: list[str], offline: bool, provider: object | None,
          cache_dir: Path, budget_s: float | None) -> object:
    """Price dataset in reais: the synthetic demonstration, or real daily closes (Yahoo chart endpoint)."""
    from core.purpose.engine import subset_dataset
    from core.purpose.synthetic_universe import asset_meta, build_synthetic_universe

    if offline:
        ds, _ = build_synthetic_universe()
        return subset_dataset(ds, [a.ticker for a in assets] + extra)
    from core.data_loader import load_price_dataset

    prov = provider or YahooChartProvider(cache=SeriesCache(cache_dir, 4.0), time_budget_s=budget_s)
    meta = {a.ticker: asset_meta(a, False) for a in assets}
    return load_price_dataset([a.ticker for a in assets] + extra, prov, period="5y", interval="1d",  # type: ignore[arg-type]
                              base_currency="BRL", meta=meta, quality=QualitySettings(min_history=MIN_OBS))


def _strengths(s: str, m: pd.DataFrame, z: pd.DataFrame) -> list[str]:
    """Plain-language reasons from the metrics that pushed the score up most (historical facts [H])."""
    row, avg = m.loc[s], m.mean(numeric_only=True)
    text = {
        "sharpe": f"retorno acima da renda fixa por unidade de risco (Sharpe encolhido {_num(row['sharpe_encolhido'])}; "
                  f"média das ações medidas {_num(avg['sharpe_encolhido'])})",
        "vol": f"oscilação de {_pct(row['volatilidade_H'])} ao ano, abaixo da média das ações ({_pct(avg['volatilidade_H'])})",
        "mdd": f"maior queda de {_pct(row['queda_maxima_H'])} no período, menor que a média ({_pct(avg['queda_maxima_H'])})",
        "mom": f"alta de {_pct(row['momento_12_1'])} em 12 meses, sem contar o último mês",
        "liq": "negociação diária alta (fácil de comprar e vender)",
    }
    order = sorted(z.columns, key=lambda k: -float(z.at[s, k]))
    out = [text[k] for k in order if float(z.at[s, k]) > 0.25 and (k != "mom" or np.isfinite(row["momento_12_1"]))]
    return out[:3] or ["nenhuma métrica se destaca muito: é a melhor combinação entre as que cabem nas regras"]


def daily_pick(amount: float, profile: str = "moderado", markets: list[str] | None = None,
               holdings: dict[str, float] | None = None, us_fractional: bool = False, offline: bool = False,
               provider: object | None = None, rf_fetch: FetchFn | None = None, cache_dir: Path | None = None,
               progress: Callable[[str], None] | None = None, download_budget_s: float | None = None) -> DailyPick:
    if not 0 < amount <= 1e9:
        raise ValueError("informe quanto você tem hoje (um valor positivo)")
    if profile not in CEILING:
        raise ValueError(f"perfil deve ser um de {sorted(CEILING)}")
    from core.purpose.engine import norm_ticker

    mk = {"B3" if str(x).upper() in ("BR", "B3", "BRASIL") else "US" for x in (markets or ["BR", "US"])}
    catalog = SYNTHETIC_CATALOG if offline else REAL_CATALOG
    universe = [a for a in catalog if a.macro_class in STOCK_CLASSES and traits(a).market in mk]
    if not universe:
        raise ValueError("nenhuma ação no universo escolhido")
    held = {norm_ticker(t): float(v) for t, v in (holdings or {}).items() if t.strip() and float(v) > 0}
    cat = {a.ticker: a for a in catalog}
    extra = [t for t in held if t not in {a.ticker for a in universe}
             and (t in cat if offline else not t.startswith("SINT_"))]
    root = cache_dir or Path(".cache") / "market_data"
    if progress is not None:
        progress(f"baixando o histórico de {len(universe)} ações e a taxa Selic…")
    ds = _load(universe, extra, offline, provider, root, download_budget_s)
    series = {s: ds.raw_prices[s].dropna() for s in ds.symbols}                      # type: ignore[attr-defined]
    left_out = {"sem dados suficientes": len([a for a in universe if a.ticker not in series])}
    # risk-free rate in reais: live Selic > flagged demonstration fallback (never zero)
    warnings: list[str] = []
    live: dict = {}
    if not offline:
        live, _ = fetch_live_rates({"BRL"}, provider or YahooChartProvider(cache=SeriesCache(root, 4.0)), rf_fetch)  # type: ignore[arg-type]
    if "BRL" in live:
        rf, rf_src = live["BRL"].rate, f"{live['BRL'].source}, em {live['BRL'].as_of}"
    else:
        rf, rf_src = DEMO_RISK_FREE_FALLBACK["BRL"], "FALLBACK DE DEMONSTRAÇÃO (não é cotação)"
        warnings.append(f"Taxa Selic indisponível agora: usado {_pct(rf, 1)} de demonstração, que muda a relação "
                        "retorno/risco de todas as ações.")
    stocks = {s: x for s, x in series.items() if s in {a.ticker for a in universe}}
    long_enough = {s: x for s, x in stocks.items() if len(x) >= MIN_CANDIDATE_OBS}
    left_out["histórico curto (menos de ~3 anos)"] = len(stocks) - len(long_enough)
    if len(long_enough) < 2:
        raise ValueError("menos de duas ações com histórico suficiente para comparar agora")
    vols = ds.volumes.reindex(columns=list(long_enough)) if ds.volumes is not None else None  # type: ignore[attr-defined]
    m = measure(long_enough, vols, {s: cat[s] for s in long_enough}, rf)
    score, z = score_components(m, RiskProfile(profile))
    # diversification with what the investor already has
    hold_series = {t: series[t] for t in held if t in series}
    used = sorted(hold_series)
    corr = pd.Series(0.0, index=m.index)
    if hold_series:
        rets = daily_returns(hold_series)
        wts = pd.Series({t: held[t] for t in hold_series})
        port = rets.dropna().mul(wts / wts.sum(), axis=1).sum(axis=1)
        cand = daily_returns(long_enough)
        for s in m.index:
            pair = pd.concat([cand[s], port], axis=1).dropna()
            if len(pair) >= 126:
                corr[s] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    missing = sorted(set(held) - set(hold_series))
    if missing:
        warnings.append("Sem histórico para " + ", ".join(missing) + ": não entrou na conta de diversificação.")
    adjusted = score - CORR_PENALTY * corr.clip(lower=0.0)
    # price, lot and what the money buys
    vmax, dmin = CEILING[profile]
    rows = []
    for s in m.index:
        a = cat[s]
        price = float(long_enough[s].iloc[-1])
        lot, rule = lot_size(a, us_fractional)
        qty = math.floor(amount / (price * lot) + 1e-9) * lot
        reason = None
        if s in held:
            reason = "já na sua carteira"
        elif (vmax is not None and m.at[s, "volatilidade_H"] > vmax) or (dmin is not None and m.at[s, "queda_maxima_H"] < dmin):
            reason = f"acima do limite de risco do perfil {profile} para uma ação sozinha"
        elif qty < lot - 1e-12:
            reason = "o valor não compra 1 lote"
        rows.append({"ticker": s, "reason": reason, "price": price, "qty": qty, "rule": rule,
                     "adjusted": float(adjusted[s]), "score": float(score[s]), "corr": float(corr[s])})
    t = pd.DataFrame(rows).set_index("ticker").sort_values("adjusted", ascending=False)
    for why in ("já na sua carteira", f"acima do limite de risco do perfil {profile} para uma ação sozinha",
                "o valor não compra 1 lote"):
        left_out[why] = int((t["reason"] == why).sum())
    ok = t[t["reason"].isna()]
    rank = {s: i + 1 for i, s in enumerate(score.sort_values(ascending=False).index)}
    last_dates = {s: str(long_enough[s].index[-1].date()) for s in long_enough}
    as_of = max(last_dates.values())

    def entry(s: str, full: bool) -> dict[str, object]:
        a, r_ = cat[s], t.loc[s]
        qty, price = float(r_["qty"]), float(r_["price"])
        d: dict[str, object] = {
            "ticker": s, "nome": a.name, "logo": None if offline else logo_path(s),
            "mercado": "B3 (Brasil)" if traits(a).market == "B3" else "EUA", "setor": SECTOR_LABEL.get(a.sector, a.sector),
            "preco": price, "preco_data": last_dates[s], "quantidade": round(qty, 8), "custo": round(qty * price, 2),
            "sobra": round(amount - qty * price, 2), "regra_de_lote": r_["rule"], "pontuacao": round(float(r_["score"]), 4),
            "pontuacao_ajustada": round(float(r_["adjusted"]), 4), "posicao": rank[s], "de": len(score),
            "correlacao_com_sua_carteira": round(float(r_["corr"]), 2) if used else None,
            "motivo": _strengths(s, m, z)[0]}
        if full:
            row = m.loc[s]
            d.update({
                "negocio": a.business, "riscos": a.class_risk + "; uma ação sozinha concentra o risco: pode cair bem "
                                                              "mais que uma carteira diversificada",
                "por_que": _strengths(s, m, z),
                "fatos_H": {"retorno_medio_anual": float(row["retorno_anual_H"]), "volatilidade": float(row["volatilidade_H"]),
                            "queda_maxima": float(row["queda_maxima_H"]), "sharpe_encolhido": float(row["sharpe_encolhido"]),
                            "momento_12_1": None if not np.isfinite(row["momento_12_1"]) else float(row["momento_12_1"]),
                            "liquidez_diaria": None if not np.isfinite(row["liquidez_diaria"]) else float(row["liquidez_diaria"]),
                            "anos_de_dados": float(row["anos"])}})
        return d

    pick = entry(ok.index[0], True) if len(ok) else None
    alternatives = [entry(s, False) for s in ok.index[1:6]]
    if pick is None:
        cheapest = t[t["reason"].fillna("") == "o valor não compra 1 lote"].sort_values("price")
        if len(cheapest):
            c = cheapest.index[0]
            warnings.insert(0, f"Com {_money(amount)} não dá para comprar 1 ação de nenhuma das opções dentro das regras; "
                               f"a mais barata é {c} ({_money(float(cheapest.at[c, 'price']))}). Junte um pouco mais"
                               + ("" if us_fractional else " ou, se a sua corretora permitir, use frações de ações dos EUA")
                               + ".")
        else:
            warnings.insert(0, f"Nenhuma ação fica dentro do limite de risco do perfil {profile} para uma posição "
                               "isolada: para esse perfil, uma ação sozinha não é indicada. A seção Carteira com "
                               "propósito monta uma carteira diversificada para o mesmo valor.")
    if profile == "conservador" and pick is not None:
        warnings.append("Para o perfil conservador, mesmo a ação escolhida oscila muito mais que uma carteira "
                        "diversificada; considere usar o valor na seção Carteira com propósito.")
    warnings += [
        ("A escolha usa só o histórico; não prevê o preço de amanhã. Com os mesmos dados ela é a mesma: pode repetir em "
         "dias seguidos e muda quando os preços mudam."),
        ("Comprar uma ação diferente por dia não diversifica por si só, e compras pequenas pagam mais custo proporcional; "
         "para quem está começando, distribuir os aportes numa carteira costuma reduzir o risco."),
    ]
    if not used:
        warnings.append("Você não informou o que já tem: a escolha não considera a sua carteira (informe para que ela "
                        "evite o que você já possui e prefira o que diversifica).")
    rules = [
        (f"Universo: {len(universe)} ações do catálogo curado ({', '.join(sorted('Brasil' if x == 'B3' else 'EUA' for x in mk))}); "
         "ETFs, fundos e cripto ficam fora."),
        (f"Dados: ~5 anos de preços diários ajustados, em reais; cada ação precisa de {MIN_CANDIDATE_OBS} preços "
         "(~3 anos) para ser comparada."),
        (f"Pontuação do perfil {profile}: Sharpe encolhido, volatilidade, queda máxima, momento de 12 meses e liquidez, "
         "comparados entre as ações medidas (a mesma regra da carteira)."),
        "Limite de risco para uma ação sozinha: " + {"conservador": "volatilidade até 25% ao ano e queda máxima até 40%",
                                                    "moderado": "volatilidade até 35% ao ano",
                                                    "arrojado": "sem limite (o perfil aceita mais risco)"}[profile] + ".",
        "Precisa caber no valor: 1 lote (B3: 1 ação no fracionário; EUA: 1 ação"
        + (", ou frações)" if us_fractional else ")") + " ao último preço de fechamento.",
        (f"Diversificação: ação que você já tem não é sugerida; a pontuação cai {_num(CORR_PENALTY, 1)} × a correlação com a "
         "sua carteira."),
    ]
    sources = [{"dado": "preços", "fonte": "gerador sintético (demonstração)" if offline else
                "Yahoo Finance (fechamentos diários ajustados, com atraso; não é cotação em tempo real)",
                "data": as_of},
               {"dado": "taxa livre de risco (Selic)", "fonte": rf_src},
               {"dado": "lista de ações e setores", "fonte": "catálogo curado do app (revisado periodicamente)"}]
    out = DailyPick(amount, profile, "sintetico" if offline else "real", as_of, pick, alternatives, left_out, rules,
                    warnings, sources, rf, rf_src, len(score), used)
    text = " ".join([*(str(x) for x in (pick or {}).get("por_que", []) or []), *warnings, *rules, DISCLAIMER])
    hits = guarantee_phrases(text)
    if hits:                                       # defensive: every text above is fixed, so this should not happen
        raise RuntimeError(f"linguagem de garantia no resultado: {hits}")
    return out
