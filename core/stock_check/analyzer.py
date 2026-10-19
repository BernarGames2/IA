"""'Vale a pena?' — quantitative check of ONE asset and a projection of an investment in it.

What it is: a transparent, rules-based reading of the asset's PRICE HISTORY against the risk-free
rate of its own currency (Selic/CDI for BRL, T-bill for USD), plus a Monte Carlo projection of an
amount invested over the chosen horizon. Context shown next to it (never part of the verdict): what
the company does and the reasons it may or may not grow (``company.py``: catalogue, B3, Yahoo
Finance fundamentals when available, the history, the simulation and general segment knowledge).
What it is NOT: a recommendation, a price target or a news assessment.

Expected return used by the projection (shrinkage, every number is reported):

    prior  = rf + beta * ERP              ERP = assumed equity risk premium (default 4% a.a.)
    se     = vol / sqrt(years of data)    standard error of the historical mean
    lambda = tau^2 / (tau^2 + se^2)       tau = assumed dispersion of true expected returns (10% a.a.)
    mu_E   = prior + lambda * (mu_hist - prior)

The noisier the history, the less it counts: a few great years are not projected forward as is.

Verdict (fixed rules, identical for every asset; shown with the result):

    FAVORÁVEL      P(beat rf over the horizon) >= 60%  and  historical excess return t >= 1
    DESFAVORÁVEL   P(beat rf) <= 40%                    or   historical excess return t <= -1
    NEUTRO         otherwise — the history cannot separate the asset from the risk-free rate
    NÃO AVALIÁVEL  fewer than 252 valid observations

Fit with the investor profile (volatility, drawdown) is reported separately, never hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEMO_RISK_FREE_FALLBACK
from core.backtesting.overfitting import psr
from core.extreme_risk.var_es import empirical_var_es
from core.market_data.base import DataUnavailableError, RawSeries, last_per_date
from core.market_data.cache import SeriesCache
from core.market_data.quality import QualitySettings, clean_series
from core.market_data.risk_free import fetch_live_rates
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider
from core.metrics import max_drawdown, rf_per_period
from core.profiler import ProfilerSettings
from core.purpose.language import guarantee_phrases
from core.simulation.engine import (
    SimulationResult,
    SimulationSettings,
    simulate_portfolio,
)
from core.simulation.models import BlockBootstrapModel, GBMModel
from core.stock_check.company import (
    CompanyProfile,
    basic_profile,
    fundamentals_table,
    growth_case,
    load_company_profile,
)
from core.stock_check.resolver import Resolved, is_plain_code, resolve
from core.universe.listings import load_market_listings
from models.investor import RiskProfile
from models.market_data import AssetMeta, QualityReport
from utils.validation import InsufficientDataError

P = 252
MIN_OBS = 252
BENCHMARK = {"BRL": ("BOVA11.SA", "Ibovespa (BOVA11)"), "USD": ("SPY", "S&P 500 (SPY)")}
RF_NAME = {"BRL": "CDI/Selic", "USD": "T-bill (juros de curto prazo dos EUA)"}
DISCLAIMER = ("Análise quantitativa para pesquisa. Não é recomendação de compra ou venda e não é preço-alvo. O "
              "veredito usa só o histórico de preços; a descrição da empresa e os motivos para crescer ou não "
              "combinam dados públicos (B3 e Yahoo Finance, quando disponíveis), o histórico, a simulação e "
              "características gerais do setor, sem avaliar notícias. Projeções são simulações [S] com incerteza "
              "grande; resultados passados não se repetem necessariamente.")
PROFILE_WAIT_S = 15.0   # extra wait for B3/Yahoo after the analysis; the lookup started in parallel


@dataclass
class StockCheckResult:
    query: str
    resolved: Resolved
    origin: str                     # "real" | "sintetico"
    currency: str
    source: str
    first_date: str
    last_date: str
    last_price: float
    n_obs: int
    rf: float
    rf_source: str
    stats: dict[str, object]
    expected: dict[str, object]
    projection: dict[str, object]
    verdict: dict[str, object]
    profile_fit: dict[str, object]
    alerts: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)
    company: CompanyProfile | None = None
    growth: dict[str, list[dict[str, str]]] = field(default_factory=dict)


# ----------------------------------------------------------------------------- data
def _meta(r: Resolved, currency: str) -> AssetMeta:
    t = r.ticker
    cal = "24x7" if t.endswith("-USD") else ("B3" if t.endswith(".SA") else "NYSE")
    if t.startswith("SINT_"):
        cal = "business_days"
    cls = r.asset.asset_class if r.asset else "unknown"
    return AssetMeta(t, cls, r.asset.sector if r.asset else "unknown", r.asset.country if r.asset else "unknown",
                     currency, "catalog" if r.asset else "unknown", cal, "catalog_static" if r.asset else "provider")


def _load_real(r: Resolved, provider: object, cache_dir: Path | None
               ) -> tuple[RawSeries, RawSeries | None, str, str | None]:
    prov = provider or YahooChartProvider(cache=SeriesCache(cache_dir or Path(".cache") / "market_data", 12.0))
    got = prov.fetch([r.ticker], "5y", "1d")                    # type: ignore[attr-defined]
    if r.ticker not in got:
        why = str(getattr(prov, "failures", {}).get(r.ticker, f"{r.ticker}: sem dados"))
        raise DataUnavailableError(why if why.startswith(r.ticker) else f"{r.ticker}: {why}")
    rs = got[r.ticker]
    bench = BENCHMARK.get(rs.currency)
    b = None
    if bench and bench[0] != r.ticker:
        b = prov.fetch([bench[0]], "5y", "1d").get(bench[0])  # type: ignore[attr-defined]
    return rs, b, getattr(prov, "name", "provedor"), bench[1] if (bench and b is not None) else None


def _load_synthetic(r: Resolved, seed: int) -> tuple[RawSeries, RawSeries | None, str, str | None]:
    from core.purpose.catalog import SYNTHETIC_CATALOG
    from core.purpose.synthetic_universe import build_synthetic_universe

    ds, _ = build_synthetic_universe(seed=seed)
    if r.ticker not in ds.native_prices:
        raise ValueError(f"{r.ticker} não existe no universo sintético; use um código SINT_* "
                         "(ex.: SINT_ENERGIA3) ou o modo de dados reais")
    ccy = ds.native_currency[r.ticker]
    rs = RawSeries(r.ticker, ds.native_prices[r.ticker].dropna(), None, "synthetic_generator", "close",
                   "n/a (sintético)", ccy, "1d")
    # reference index: equal-weighted synthetic equities of the same currency (the asset itself excluded)
    peers = [a.ticker for a in SYNTHETIC_CATALOG if a.asset_class == "equity" and a.ticker != r.ticker
             and ds.native_currency.get(a.ticker) == ccy]
    bench = None
    if peers:
        ret = pd.DataFrame({t: ds.native_prices[t] for t in peers}).pct_change(fill_method=None).mean(axis=1)
        px = 100.0 * (1.0 + ret.dropna()).cumprod()
        bench = RawSeries(f"SINT_INDICE_{ccy}", px, None, "synthetic_generator", "close", "n/a", ccy, "1d")
    return rs, bench, "synthetic_generator", f"índice sintético ({ccy}, média das ações SINT)" if bench else None


# ----------------------------------------------------------------------------- analysis
def _stats(px: pd.Series, bench: pd.Series | None, rf: float, volume: pd.Series | None) -> dict[str, object]:
    r = px.pct_change(fill_method=None).dropna()
    lr = np.log(px).diff().dropna()
    n = len(r)
    years = n / P
    ex = r.to_numpy() - rf_per_period(rf, P)
    sd = float(ex.std(ddof=1))
    out: dict[str, object] = {
        "retorno_medio_anual": float(r.mean() * P),
        "cagr": float((px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1) if years > 0 else None,
        "volatilidade": float(r.std(ddof=1) * np.sqrt(P)),
        "vol_log": float(lr.std(ddof=1) * np.sqrt(P)),
        "queda_maxima": float(max_drawdown(r.to_numpy())),
        "excesso_anual": float(ex.mean() * P),
        "t_excesso": float(ex.mean() / sd * np.sqrt(n)) if sd > 0 else None,
        "sharpe_excesso": float(ex.mean() / sd * np.sqrt(P)) if sd > 0 else None,
        "anos_de_dados": years,
    }
    try:
        out["psr"] = psr(ex)
    except InsufficientDataError:
        out["psr"] = None
    v, e = empirical_var_es(-r.to_numpy(), 0.95)
    out["var95_diario"], out["es95_diario"] = v, e
    # share of rolling 12-month windows that beat the CURRENT rf (approximation: rf held constant)
    wealth = (1 + r).cumprod()
    if n > P:
        idx = np.arange(P, n, 21)
        w = wealth.to_numpy()
        win = w[idx] / w[idx - P] - 1
        out["janelas_12m_acima_rf"] = float(np.mean(win > rf))
        out["n_janelas_12m"] = len(win)
    out["retorno_12m"] = float(px.iloc[-1] / px.iloc[-P - 1] - 1) if len(px) > P else None
    out["retorno_6m"] = float(px.iloc[-1] / px.iloc[-127] - 1) if len(px) > 126 else None
    out["media_200d"] = float(px.iloc[-200:].mean()) if len(px) >= 200 else None
    out["acima_media_200d"] = bool(px.iloc[-1] > out["media_200d"]) if out["media_200d"] is not None else None
    out["beta"] = out["correlacao"] = None
    if bench is not None:
        br = bench.pct_change(fill_method=None)
        j = pd.concat([r, br], axis=1, join="inner").dropna()
        if len(j) >= 120:
            c = np.cov(j.iloc[:, 0], j.iloc[:, 1], ddof=1)
            out["beta"] = float(c[0, 1] / c[1, 1]) if c[1, 1] > 0 else None
            out["correlacao"] = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
    out["giro_diario_medio"] = None
    if volume is not None:
        tv = (volume.reindex(px.index) * px).iloc[-63:].dropna()
        if len(tv) and float(tv.mean()) > 0:
            out["giro_diario_medio"] = float(tv.mean())
    return out


def _expected(st: dict[str, object], rf: float, erp: float, tau: float) -> dict[str, object]:
    beta = st.get("beta")
    b = float(np.clip(beta, 0.0, 2.0)) if beta is not None else 1.0
    prior = rf + b * erp
    mu_h = float(st["retorno_medio_anual"])  # type: ignore[arg-type]
    se = float(st["volatilidade"]) / np.sqrt(max(float(st["anos_de_dados"]), 1e-9))  # type: ignore[arg-type]
    lam = tau ** 2 / (tau ** 2 + se ** 2)
    return {"prior": prior, "beta_usado": b, "beta_estimado": beta is not None, "erp": erp, "tau": tau,
            "erro_padrao_media": se, "peso_historico": lam, "retorno_historico": mu_h,
            "retorno_esperado": prior + lam * (mu_h - prior)}


def _gbm(mu_arith: float, vol_log: float) -> GBMModel:
    mu_c = P * np.log1p(mu_arith / P)            # E[1 + r_daily] = 1 + mu_arith / P
    return GBMModel(np.array([mu_c]), np.array([[vol_log ** 2]]), 1.0 / P)


def _sim(model: object, amount: float, years: float, n: int, seed: int) -> SimulationResult:
    ss = SimulationSettings(n_paths=n, n_steps=max(1, round(years * P)), seed=seed, initial_capital=amount,
                            rebalance_every=0, periods_per_year=P, record_every=21, batch_size=2000)
    return simulate_portfolio(model, np.array([1.0]), ss)  # type: ignore[arg-type]


def _verdict(p_beat: float, t: float | None) -> dict[str, object]:
    tt = t if t is not None else 0.0
    if p_beat >= 0.60 and tt >= 1.0:
        label, tone = "FAVORÁVEL", "good"
    elif p_beat <= 0.40 or tt <= -1.0:
        label, tone = "DESFAVORÁVEL", "crit"
    else:
        label, tone = "NEUTRO", "warn"
    return {"rotulo": label, "tom": tone}


def analyze_stock(query: str, horizon_years: float = 3.0, amount: float = 10_000.0,
                  profile: str = "moderado", offline: bool = False, rf: float | None = None,
                  n_paths: int = 5_000, seed: int = 42, erp: float = 0.04, tau: float = 0.10,
                  provider: object | None = None, rf_fetch: FetchFn | None = None,
                  cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                  download_budget_s: float | None = None) -> StockCheckResult:
    if not 0.25 <= horizon_years <= 30:
        raise ValueError("horizonte deve estar entre 0,25 e 30 anos")
    if not 0 < amount <= 1e10:
        raise ValueError("valor a investir deve ser positivo")
    try:
        res: Resolved | None = resolve(query)
        first_error = None
    except ValueError as e:
        res, first_error = None, e
    if not offline and (res is None or is_plain_code(res)):
        # not a known code or name: look it up among every company listed on the B3 and in the US
        if progress is not None:
            progress("procurando nas listagens completas da B3 e dos EUA…")
        listings = load_market_listings(cache_dir or Path(".cache") / "market_data", rf_fetch)
        if listings.available:
            res = resolve(query, listings)
    if res is None:
        raise first_error if first_error else ValueError("ação não reconhecida")
    if offline and not res.ticker.startswith("SINT_"):
        raise ValueError("no modo sintético use um código SINT_* (ex.: SINT_ENERGIA3, SINT_TECH_US)")
    if not offline and res.ticker.startswith("SINT_"):
        raise ValueError("códigos SINT_* são sintéticos: escolha 'Demonstração sintética'")
    # what the company does + fundamentals: looked up in parallel with the price download
    root = cache_dir or Path(".cache") / "market_data"
    pool = ThreadPoolExecutor(max_workers=1)
    company_future = pool.submit(load_company_profile, res, offline, rf_fetch, root)
    pool.shutdown(wait=False)
    if progress is not None:
        progress(f"baixando o histórico de {res.ticker} e do índice de referência…")
    if not offline and provider is None:
        provider = YahooChartProvider(cache=SeriesCache(cache_dir or Path(".cache") / "market_data", 12.0),
                                      time_budget_s=download_budget_s)
    rs, bench_rs, source, bench_name = (_load_synthetic(res, seed) if offline
                                        else _load_real(res, provider, cache_dir))
    ccy = rs.currency if rs.currency and rs.currency != "UNKNOWN" else "BRL"
    report = QualityReport()
    px = clean_series(rs, _meta(res, ccy), report, QualitySettings(min_history=MIN_OBS))
    notes = [f"{q.check}: {q.detail}" for q in report.issues if q.severity.value != "OK"]
    if px is None:
        why = report.excluded_symbols.get(res.ticker, "dados insuficientes")
        raise DataUnavailableError(f"{res.ticker}: {why} (mínimo {MIN_OBS} observações válidas)")
    px = px.dropna()
    alerts: list[str] = []
    # risk-free rate of the asset's OWN currency: user > live public source > flagged fallback
    if rf is not None:
        rf_val, rf_src = float(rf), "informada pelo usuário"
    else:
        live: dict = {}
        if not offline:
            live, _err = fetch_live_rates({ccy}, provider or YahooChartProvider(
                cache=SeriesCache(cache_dir or Path(".cache") / "market_data", 12.0)), rf_fetch)
        if ccy in live:
            rf_val, rf_src = live[ccy].rate, f"{live[ccy].source}, em {live[ccy].as_of}"
        elif ccy in DEMO_RISK_FREE_FALLBACK:
            rf_val, rf_src = DEMO_RISK_FREE_FALLBACK[ccy], "FALLBACK DE DEMONSTRAÇÃO (não é cotação)"
            alerts.append(f"Taxa livre de risco de {ccy} = {rf_val:.2%} é um valor de demonstração, não a taxa atual: "
                          "informe a taxa real para uma comparação correta.")
        else:
            raise ValueError(f"sem taxa livre de risco para {ccy}: informe-a (nunca é assumida igual a zero)")
    bench_px = None
    if bench_rs is not None:
        cal = "B3" if bench_rs.symbol.endswith(".SA") else ("business_days" if offline else "NYSE")
        b = clean_series(bench_rs, AssetMeta(bench_rs.symbol, "equity", "indice", "?", bench_rs.currency, "?",
                                             cal, "provider"), QualityReport(), QualitySettings(min_history=MIN_OBS))
        bench_px = b.dropna() if b is not None else None
    vol_s = last_per_date(rs.volume).reindex(px.index) if rs.volume is not None else None
    st = _stats(px, bench_px, rf_val, vol_s)
    exp = _expected(st, rf_val, erp, tau)

    # projection ------------------------------------------------------------
    if progress is not None:
        progress("simulando a projeção (Monte Carlo)…")
    main = _sim(_gbm(float(exp["retorno_esperado"]), float(st["vol_log"])), amount, horizon_years, n_paths, seed)
    T = main.settings["n_steps"] / P  # type: ignore[operator]
    rf_final = amount * (1 + rf_val) ** T
    tw = main.terminal_wealth
    p_beat = float(np.mean(tw >= rf_final))
    p_loss = float(np.mean(tw < amount))
    se = lambda p: float(np.sqrt(p * (1 - p) / len(tw)))
    pct = main.percentiles
    years_axis = (main.time_index / P).tolist()
    fan = {"years": years_axis, **{c: pct[c].tolist() for c in ("p5", "p25", "p50", "p75", "p95")},
           "rf_line": [amount * (1 + rf_val) ** t for t in years_axis]}
    scen = []
    for name, q in (("otimista (P90)", 90), ("base (P50)", 50), ("pessimista (P10)", 10)):
        v = float(np.percentile(tw, q))
        scen.append({"cenario": name, "valor_final": v, "retorno_anual": (v / amount) ** (1 / T) - 1,
                     "preco_equivalente": float(px.iloc[-1]) * v / amount})
    models = {
        "principal: retorno esperado encolhido": None,
        "se o passado se repetir (média histórica)": _gbm(float(st["retorno_medio_anual"]), float(st["vol_log"])),
        f"sem prêmio de risco (rende o {RF_NAME.get(ccy, 'rf')} em média)": _gbm(rf_val, float(st["vol_log"])),
        "bootstrap em blocos do histórico": BlockBootstrapModel(
            px.pct_change(fill_method=None).dropna().to_numpy()[:, None], 21),
    }
    sens = []
    for name, mdl in models.items():
        sim = main if mdl is None else _sim(mdl, amount, horizon_years, min(n_paths, 3000), seed + 1)
        t2 = sim.terminal_wealth
        sens.append({"modelo": name, "p_superar_rf": float(np.mean(t2 >= rf_final)),
                     "p_prejuizo": float(np.mean(t2 < amount)), "mediana_final": float(np.median(t2))})
    _, es_h = empirical_var_es(-(tw / amount - 1.0), 0.95)
    projection = {"horizonte_anos": T, "valor_investido": amount, "fan": fan, "cenarios": scen,
                  "p_superar_rf": p_beat, "p_superar_rf_ep": se(p_beat), "p_prejuizo": p_loss,
                  "p_prejuizo_ep": se(p_loss), "valor_rf_no_horizonte": rf_final,
                  "perda_media_5_piores_pct": es_h, "n_trajetorias": len(tw), "sensibilidade": sens,
                  "modelo": "GBM com o retorno esperado encolhido (ver metodologia) e a volatilidade histórica"}

    # verdict + profile fit -----------------------------------------------------
    v = _verdict(p_beat, st.get("t_excesso"))  # type: ignore[arg-type]
    rfn = RF_NAME.get(ccy, "taxa livre de risco")
    t = st.get("t_excesso")
    evid = ("indisponível" if t is None else "forte" if abs(t) >= 2 else "moderada" if abs(t) >= 1 else "fraca")
    if t is not None and t <= -1:
        hist = f"Evidência estatística de que o ativo rendeu MENOS que o {rfn} no histórico: {evid}."
    elif t is not None and t >= 1:
        hist = f"Evidência estatística de que o ativo rendeu mais que o {rfn} no histórico: {evid}."
    else:
        hist = f"No histórico, o ativo não se separa do {rfn} (evidência {evid})."
    v["manchete"] = (f"Segundo a simulação, {res.name} tem {p_beat:.0%} de chance de render mais que o {rfn} em "
                     f"{T:g} anos e {p_loss:.0%} de chance de terminar com prejuízo. {hist}")
    v["regras"] = ("FAVORÁVEL: chance de superar a taxa livre de risco ≥ 60% e retorno histórico acima dela com "
                   "t ≥ 1. DESFAVORÁVEL: chance ≤ 40% ou retorno histórico abaixo dela com t ≤ −1. NEUTRO: "
                   "demais casos (o histórico não separa o ativo da renda fixa).")
    lim = ProfilerSettings().limits[RiskProfile(profile)].model_copy()
    band = lim.target_volatility_band
    vol = float(st["volatilidade"])  # type: ignore[arg-type]
    dd = float(st["queda_maxima"])  # type: ignore[arg-type]
    fit_vol = vol <= band[1]
    fit_dd = abs(dd) <= lim.max_drawdown_reference
    profile_fit = {
        "perfil": profile, "faixa_vol_carteira": band, "queda_tolerada": lim.max_drawdown_reference,
        "peso_maximo_por_ativo": lim.max_weight_per_asset, "vol_compativel": fit_vol, "queda_compativel": fit_dd,
        "leitura": ("compatível com o perfil mesmo como posição isolada" if fit_vol and fit_dd else
                    f"risco acima do perfil {profile} como posição isolada (volatilidade {vol:.0%} vs até "
                    f"{band[1]:.0%} para a carteira; queda máxima histórica {dd:.0%} vs tolerância "
                    f"{lim.max_drawdown_reference:.0%}): se usar, que seja uma parte de uma carteira diversificada "
                    f"(no máximo {lim.max_weight_per_asset:.0%} por ativo nos limites deste perfil)"),
    }
    criteria = [
        {"criterio": f"Chance de superar o {rfn} em {T:g} anos (projeção principal)", "valor": p_beat,
         "formato": "pct", "leitura": "≥ 60% favorece · ≤ 40% desfavorece"},
        {"criterio": f"Retorno histórico acima do {rfn} (a.a.)", "valor": st["excesso_anual"], "formato": "pct",
         "leitura": (f"t = {t:.2f} — evidência {evid}" + (" de que rendeu menos" if t <= -1 else " de que rendeu "
                                                            "mais" if t >= 1 else "")) if t is not None
         else "indisponível"},
        {"criterio": "Sharpe histórico (excesso, a.a.)", "valor": st["sharpe_excesso"], "formato": "num",
         "leitura": "retorno extra por unidade de risco"},
        {"criterio": "Volatilidade anual", "valor": vol, "formato": "pct",
         "leitura": "compatível com o perfil" if fit_vol else "acima da faixa do perfil"},
        {"criterio": "Queda máxima histórica", "valor": dd, "formato": "pct",
         "leitura": "dentro da tolerância do perfil" if fit_dd else "acima da tolerância do perfil"},
        {"criterio": "Perda diária em dia ruim (VaR 95%)", "valor": st["var95_diario"], "formato": "pct",
         "leitura": "em 1 de cada 20 pregões a perda foi maior que isso"},
    ]
    if st.get("janelas_12m_acima_rf") is not None:
        criteria.append({"criterio": f"Janelas de 12 meses acima do {rfn}", "valor": st["janelas_12m_acima_rf"],
                         "formato": "pct", "leitura": "frequência histórica (com a taxa atual constante)"})
    if st.get("correlacao") is not None:
        bname = bench_name or "índice de referência"
        corr = float(st["correlacao"])  # type: ignore[arg-type]
        criteria.append({"criterio": f"Correlação com {bname}", "valor": corr, "formato": "num",
                         "leitura": "diversifica pouco" if corr > 0.7 else "diversifica parcialmente"
                         if corr > 0.3 else "diversifica bem"})
    if st.get("giro_diario_medio") is not None:
        g = float(st["giro_diario_medio"])  # type: ignore[arg-type]
        criteria.append({"criterio": "Giro financeiro diário médio (3 meses)", "valor": g, "formato": "money",
                         "leitura": "liquidez alta" if g >= 50e6 else "liquidez moderada" if g >= 5e6
                         else "liquidez BAIXA: cuidado ao comprar/vender"})
    if st.get("retorno_12m") is not None:
        criteria.append({"criterio": "Variação nos últimos 12 meses", "valor": st["retorno_12m"], "formato": "pct",
                         "leitura": "informativo (não entra no veredito)"})
    v["criterios"] = criteria

    if progress is not None and not company_future.done():
        progress("buscando o que a empresa faz e os indicadores (B3 e Yahoo Finance)…")
    try:
        company = company_future.result(timeout=PROFILE_WAIT_S)
    except FuturesTimeout:
        company = basic_profile(res, "B3/Yahoo Finance não responderam a tempo: perfil só com o catálogo")
    except Exception as e:  # noqa: BLE001 - the profile is context and never blocks the analysis
        company = basic_profile(res, f"falha ao montar o perfil da empresa: {type(e).__name__}: {e}")
    growth = growth_case(company, st, projection, rf_val, rfn, ccy,
                         f"{source}, {px.index[0].date()} a {px.index[-1].date()}", rf_src)
    if company.descricao is None:
        alerts.append(f"{res.ticker}: descrição do negócio indisponível nas fontes (catálogo, B3, Yahoo Finance); "
                      "nada é inventado.")
    if st.get("beta") is None:
        alerts.append("Beta indisponível (sem índice de referência com dados): prêmio esperado calculado com beta = 1.")
    elif bench_name:
        alerts.append(f"Beta {float(st['beta']):.2f} estimado contra {bench_name}.")
    if float(exp["peso_historico"]) < 0.5:
        alerts.append(f"O histórico pesa só {float(exp['peso_historico']):.0%} no retorno esperado: com "
                      f"{float(st['anos_de_dados']):.1f} anos de dados e volatilidade de {vol:.0%}, a média passada "
                      "é muito incerta (erro-padrão de " + f"{float(exp['erro_padrao_media']):.0%} a.a.).")
    spread = sens[1]["p_superar_rf"] - sens[2]["p_superar_rf"]
    alerts.append(f"Sensibilidade: a chance de superar o {rfn} vai de {sens[2]['p_superar_rf']:.0%} (sem prêmio) a "
                  f"{sens[1]['p_superar_rf']:.0%} (se o passado se repetir) — diferença de {abs(spread):.0%}.")
    if offline:
        alerts.insert(0, "DADOS SINTÉTICOS: o ativo não existe no mercado; demonstração do método.")
    out = StockCheckResult(query, res, "sintetico" if offline else "real", ccy, source,
                           str(px.index[0].date()), str(px.index[-1].date()), float(px.iloc[-1]), len(px),
                           rf_val, rf_src, st, exp, projection, v, profile_fit, alerts, notes, company, growth)
    hits = guarantee_phrases(render_markdown(out).replace(query, ""))
    if hits:
        out.alerts.append(f"ATENÇÃO: linguagem de garantia detectada no texto gerado: {hits}")
    return out


# ----------------------------------------------------------------------------- rendering
def _fmt(x: object, kind: str, ccy: str) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "indisponível"
    if kind == "pct":
        return f"{float(x):.1%}"
    if kind == "money":
        return f"{ccy} {float(x):,.0f}".replace(",", ".")
    return f"{float(x):.2f}"


def render_markdown(r: StockCheckResult) -> str:
    v, pj, e, st = r.verdict, r.projection, r.expected, r.stats
    c = r.currency
    L = [f"# Vale a pena? {r.resolved.name} ({r.resolved.ticker})", ""]
    if r.origin == "sintetico":
        L.append("> **DADOS SINTÉTICOS** — demonstração do método; o ativo não existe no mercado.\n")
    L.append(f"_{DISCLAIMER}_\n")
    if r.company is not None:
        L.extend(_company_md(r.company))
    L.append(f"## Veredito do modelo: **{v['rotulo']}**\n")
    L.append(str(v["manchete"]) + "\n")
    L.append(f"Regras: {v['regras']}\n")
    L.append(f"Perfil {r.profile_fit['perfil']}: {r.profile_fit['leitura']}.\n")
    if r.growth:
        L.extend(_growth_md(r.growth))
    if r.company is not None:
        rows = fundamentals_table(r.company, r.currency)
        L.append("## Indicadores da empresa\n")
        if rows:
            L.append(f"Fonte: {r.company.fundamentos_fonte}. Valores como publicados; nada é estimado no lugar.\n")
            L.append("| indicador | valor |\n|---|---|")
            L.extend(f"| {x['indicador']} | {x['valor']} |" for x in rows)
            L.append("")
        else:
            L.append("Indicadores fundamentalistas indisponíveis para este ativo (ver avisos do perfil acima).\n")
    L.append("## Critérios do veredito\n")
    L.append("| critério | resultado | leitura |\n|---|---|---|")
    for k in v["criterios"]:  # type: ignore[union-attr]
        L.append(f"| {k['criterio']} | {_fmt(k['valor'], k['formato'], c)} | {k['leitura']} |")
    L.append(f"\n## Projeção de {_fmt(pj['valor_investido'], 'money', c)} em {pj['horizonte_anos']:g} anos [S]\n")
    L.append(f"Chance de superar a taxa livre de risco: **{pj['p_superar_rf']:.0%}** (erro padrão "
             f"{pj['p_superar_rf_ep']:.1%}); chance de prejuízo: **{pj['p_prejuizo']:.0%}**; aplicado na taxa livre de "
             f"risco o valor seria {_fmt(pj['valor_rf_no_horizonte'], 'money', c)}. Média das 5% piores trajetórias: "
             f"{-float(pj['perda_media_5_piores_pct']):.0%} no período.\n")
    L.append("| cenário | valor final | retorno anual equivalente | preço equivalente |\n|---|---|---|---|")
    for s in pj["cenarios"]:  # type: ignore[union-attr]
        L.append(f"| {s['cenario']} | {_fmt(s['valor_final'], 'money', c)} | {s['retorno_anual']:.1%} | "
                 f"{c} {s['preco_equivalente']:.2f} |")
    L.append(f"\nPreço de referência: {c} {r.last_price:.2f} em {r.last_date}. O 'preço equivalente' só reescala o "
             "valor simulado; não é preço-alvo.\n")
    L.append("| modelo | chance de superar a taxa livre de risco | chance de prejuízo | mediana final |\n|---|---|---|---|")
    for s in pj["sensibilidade"]:  # type: ignore[union-attr]
        L.append(f"| {s['modelo']} | {s['p_superar_rf']:.0%} | {s['p_prejuizo']:.0%} | "
                 f"{_fmt(s['mediana_final'], 'money', c)} |")
    L.append("\n## Metodologia\n")
    L.append(f"Dados: {r.source}, {r.first_date} a {r.last_date} ({r.n_obs} pregões), moeda {c}. Taxa livre de risco "
             f"{r.rf:.2%} ({r.rf_source}).")
    L.append(f"Retorno esperado da projeção: {e['retorno_esperado']:.1%} a.a. = referência {e['prior']:.1%} "
             f"(taxa livre de risco + beta {e['beta_usado']:.2f} × prêmio de {e['erp']:.0%}) + "
             f"{e['peso_historico']:.0%} × (média histórica {e['retorno_historico']:.1%} − referência); o peso do "
             f"histórico cai quando a média é incerta (erro-padrão {e['erro_padrao_media']:.1%} a.a.).")
    L.append(f"Volatilidade da simulação: {st['vol_log']:.1%} a.a. (histórica). {pj['n_trajetorias']} trajetórias.")
    L.append("\n## Avisos\n")
    for a in r.alerts:
        L.append(f"- {a}")
    for q in r.quality_notes:
        L.append(f"- qualidade dos dados: {q}")
    return "\n".join(L) + "\n"


def _company_md(c: CompanyProfile) -> list[str]:
    L = ["## O que a empresa faz\n"]
    if c.descricao:
        L.append(f"**{c.nome}** — {c.descricao}. _(fonte: {c.descricao_fonte})_\n")
    else:
        L.append(f"**{c.nome}** — descrição do negócio indisponível nas fontes consultadas.\n")
    seg = c.segmento
    if seg is not None:
        L.append(f"- Segmento: {seg.rotulo} (identificado por: {c.segmento_fonte}).")
        L.append(f"- Como o segmento costuma ganhar dinheiro: {seg.como_ganha}")
    if c.atividade_b3 and c.atividade_b3 != c.descricao:
        L.append(f"- Atividade registrada na B3: {c.atividade_b3}.")
    if c.classificacao:
        L.append(f"- Classificação setorial: {c.classificacao} ({c.classificacao_fonte}).")
    extra = [x for x in (f"sede: {c.sede}" if c.sede else "",
                         f"funcionários: {c.funcionarios:,}".replace(",", ".") if c.funcionarios else "",
                         f"site: {c.site}" if c.site else "", f"razão social: {c.razao_social}" if c.razao_social
                         else "") if x]
    if extra:
        line = "; ".join(extra)
        L.append("- " + line[0].upper() + line[1:] + ".")
    if c.resumo_en:
        L.append(f"- Resumo publicado pela fonte (Yahoo Finance, em inglês): \"{c.resumo_en}\"")
    for u in c.indisponivel:
        L.append(f"- Indisponível: {u}.")
    L.append("")
    return L


def _growth_md(g: dict[str, list[dict[str, str]]]) -> list[str]:
    L: list[str] = []
    for key, title in (("pode_crescer", "Por que pode crescer"),
                       ("pode_nao_crescer_tanto", "Por que pode não crescer tanto")):
        L.append(f"## {title}\n")
        items = g.get(key) or []
        L.extend(f"- {x['texto']} _[{x['rotulo_base']}]_" for x in items)
        if not items:
            L.append("- Nenhum motivo identificado pelas regras com os dados disponíveis.")
        L.append("")
    sources = dict.fromkeys(f"{x['rotulo_base']}: {x['fonte']}" for k in g for x in g[k])
    L.append("Base de cada motivo: " + " · ".join(sources) + ". Os motivos são leituras de dados e de "
             "características do setor, não previsões nem recomendação.\n")
    return L


def to_dict(r: StockCheckResult) -> dict[str, object]:
    from utils.serialization import to_jsonable

    return to_jsonable({  # type: ignore[return-value]
        "consulta": r.query, "ticker": r.resolved.ticker, "nome": r.resolved.name,
        "negocio": r.company.descricao if r.company is not None else r.resolved.business,
        "empresa": None if r.company is None else {**r.company.to_dict(),
                                                    "indicadores": fundamentals_table(r.company, r.currency)},
        "por_que_pode_crescer": r.growth.get("pode_crescer", []),
        "por_que_pode_nao_crescer_tanto": r.growth.get("pode_nao_crescer_tanto", []),
        "interpretacao": r.resolved.how, "origem": r.origin, "moeda": r.currency, "fonte": r.source,
        "periodo": [r.first_date, r.last_date], "ultimo_preco": r.last_price, "n_obs": r.n_obs,
        "taxa_livre_risco": r.rf, "fonte_taxa_livre_risco": r.rf_source, "nome_taxa_livre_risco":
        RF_NAME.get(r.currency, "taxa livre de risco"), "estatisticas": r.stats, "retorno_esperado": r.expected,
        "projecao": r.projection, "veredito": r.verdict, "perfil": r.profile_fit, "alertas": r.alerts,
        "qualidade": r.quality_notes, "aviso": DISCLAIMER, "markdown": render_markdown(r),
    })
