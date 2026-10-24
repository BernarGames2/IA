"""Web entry point of the business futures simulator (routed by ``webapp/server.py``): real data only.

Defaults of the drivers come from the company's own published numbers and are labelled with their origin; every
range around them is a hypothesis the investor can change and send back.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from core.forecasting.data import load, rf_rate_at
from core.forecasting.model import fit
from core.fundamentals.analysis import dividends_paid, fcf, growth
from core.fundamentals.sources import Annual, fetch_annual
from core.market_data.logos import logo_path
from core.market_data.risk_free import fetch_live_rates
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import _default_provider
from core.purpose.language import guarantee_phrases
from core.scenarios.chain import SHOCKS, chain_effects
from core.scenarios.dcf import DRIVER_PT, DRIVERS, MIN_SPREAD, Base, simulate
from core.stock_check.company import SourceError, _yahoo_client, load_company_profile
from core.stock_check.resolver import resolve
from utils.br_format import num, pct

ERP = 0.04
G_INF = {"BRL": 0.04, "USD": 0.025}
DISCLAIMER = ("Simulação de valor sob premissas explícitas [E]: o resultado só é tão bom quanto as premissas, muda "
              "muito com o custo de capital e o crescimento de longo prazo, não é preço-alvo nem recomendação e não "
              "oferece garantia de resultado.")


class ScenarioRequest(BaseModel):
    ativo: str = Field(min_length=1, max_length=60)
    premissas: dict[str, list[float]] | None = None          # driver -> [pessimista, base, otimista]
    payout: list[float] | None = None

    @field_validator("premissas")
    @classmethod
    def _known(cls, v: dict[str, list[float]] | None) -> dict[str, list[float]] | None:
        if v is not None:
            bad = [k for k in v if k not in DRIVERS]
            if bad or any(len(x) != 3 for x in v.values()):
                raise ValueError("premissas: use crescimento, margem_fcl, custo_capital e crescimento_perpetuo, "
                                 "cada uma com [pessimista, base, otimista]")
        return v

    def key(self) -> str:
        return f"cenarios|{self.ativo.strip().upper()}|{sorted((self.premissas or {}).items())}|{self.payout}"


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def defaults(a: Annual, fin: bool, rf: float, beta: float | None, ccy: str
             ) -> tuple[dict[str, tuple[float, float, float]], dict[str, str], tuple[float, float, float] | None, str | None]:
    """(drivers, origin of each, payout range for financials, payout origin)."""
    gr = growth(a)
    origin: dict[str, str] = {}
    g_hist = gr.get("lucro_cagr_3a" if fin else "receita_cagr_3a")
    if g_hist is None:
        g_hist = gr.get("lucro_ultimo_ano" if fin else "receita_ultimo_ano")
        what = "do último ano"
    else:
        what = "média em 3 anos"
    if g_hist is None:
        g_base, origin["crescimento"] = 0.05, "sem histórico suficiente: 5% ao ano (hipótese)"
    else:
        g_base = _clip(g_hist, -0.10, 0.25)
        origin["crescimento"] = (f"{'lucro' if fin else 'receita'}: {pct(g_hist, sign=True)} ao ano ({what}, balanços)"
                                 + (f", limitado a {pct(g_base, sign=True)}" if g_base != g_hist else "")
                                 + "; faixa de ±5 p.p. (hipótese)")
    p = {"crescimento": (g_base - 0.05, g_base, g_base + 0.05)}
    pay = pay_origin = None
    if fin:
        p["margem_fcl"] = (0.0, 0.0, 0.0)
        ratios = [dividends_paid(a, i) / a.get("lucro_liquido", i) for i in range(len(a.anos))  # type: ignore[operator]
                  if dividends_paid(a, i) and a.get("lucro_liquido", i) and a.get("lucro_liquido", i) > 0]  # type: ignore[operator]
        po = _clip(float(np.median(ratios)), 0.1, 1.0) if ratios else 0.4
        pay = (max(0.0, po - 0.15), po, min(1.0, po + 0.15))
        pay_origin = (f"dividendos pagos ÷ lucro: mediana de {pct(po, 0)} nos balanços" if ratios
                      else "sem histórico de dividendos: 40% (hipótese)") + "; faixa de ±15 p.p. (hipótese)"
        origin["margem_fcl"] = "não se aplica a bancos e seguradoras (usa lucro × payout)"
    else:
        margins = [fcf(a, i) / a.get("receita", i) for i in range(len(a.anos))  # type: ignore[operator]
                   if fcf(a, i) is not None and a.get("receita", i)]
        if not margins:
            raise ValueError("sem fluxo de caixa livre publicado: informe a margem de caixa livre nas premissas")
        m_base = _clip(float(np.median(margins)), -0.2, 0.5)
        w = max(0.03, abs(m_base) * 0.3)
        p["margem_fcl"] = (m_base - w, m_base, m_base + w)
        origin["margem_fcl"] = (f"caixa livre ÷ receita: mediana de {pct(m_base)} em {len(margins)} anos (balanços); "
                                f"faixa de ±{pct(w)} (hipótese)")
    b = _clip(beta, 0.5, 2.0) if beta is not None else 1.0
    ke = rf + b * ERP
    p["custo_capital"] = (ke - 0.02, ke, ke + 0.02)
    origin["custo_capital"] = (f"renda fixa de {pct(rf)} + beta de {num(b, 2)} × prêmio de {pct(ERP, 0)} (mesmo modelo das "
                               "previsões); faixa de ±2 p.p. (hipótese)")
    gi = G_INF.get(ccy, 0.03)
    hi = min(gi + 0.015, p["custo_capital"][0] - MIN_SPREAD)
    p["crescimento_perpetuo"] = (min(gi - 0.015, hi), min(gi, hi), hi)
    origin["crescimento_perpetuo"] = (f"{pct(gi)} ao ano em {ccy} (inflação de longo prazo + crescimento real modesto; "
                                      "hipótese); faixa de ±1,5 p.p.")
    return p, origin, pay, pay_origin


def handle_scenarios(req: ScenarioRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                     cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                     budget_s: float | None = None) -> dict[str, object]:
    r = resolve(req.ativo)
    if r.ticker.startswith("SINT_") or r.ticker.endswith("-USD"):
        raise ValueError("o simulador precisa de uma empresa com balanços reais (não vale para SINT_* nem cripto)")
    prov = provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data", progress, budget_s)
    root = cache_dir or Path(".cache") / "market_data"
    client = _yahoo_client(fetch)
    if progress:
        progress(f"baixando balanços e cotações de {r.ticker}…")
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_px = pool.submit(load, r.ticker, prov, fetch, "5y", None)
        f_ann = pool.submit(fetch_annual, client, r.ticker, root)
        f_prof = pool.submit(load_company_profile, r, False, fetch, root)
    s3 = f_px.result()
    try:
        a = f_ann.result()
    except (SourceError, OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"balanços indisponíveis agora ({e}): o simulador não inventa números") from e
    if a is None or a.get("receita", -1) is None and a.get("lucro_liquido", -1) is None:
        raise ValueError("sem demonstrações publicadas para este ativo: o simulador não inventa números")
    profile = f_prof.result()
    seg = profile.segmento
    fin = bool(seg and seg.financeiro)
    if a.moeda and a.moeda != s3.currency:
        raise ValueError(f"o balanço está em {a.moeda} e a cotação em {s3.currency}: o simulador não converte moedas")
    live, _e = fetch_live_rates({s3.currency}, prov, fetch)  # type: ignore[arg-type]
    rf_q = live.get(s3.currency)
    warns: list[str] = []
    if rf_q is not None:
        rf, rf_src = rf_q.rate, f"{rf_q.source}, em {rf_q.as_of}"
    else:
        rf = rf_rate_at(s3.rf_index, s3.px.index[-1])
        if rf is None:
            raise ValueError("sem a taxa livre de risco da moeda: informe o custo de capital nas premissas")
        rf_src = f"{s3.rf_name} do último dia"
    px = s3.px.iloc[-1260:]
    rets = px.pct_change(fill_method=None).dropna().to_numpy()
    br = None
    if s3.bench is not None:
        br = s3.bench.reindex(px.index).ffill(limit=5).pct_change(fill_method=None).iloc[1:].to_numpy()
    beta = fit(rets, rf, br).beta
    p, origin, pay, pay_origin = defaults(a, fin, rf, beta, s3.currency)
    if req.premissas:
        for k, v in req.premissas.items():
            p[k] = (float(v[0]), float(v[1]), float(v[2]))
            origin[k] = "informada por você"
    if req.payout is not None:
        if len(req.payout) != 3:
            raise ValueError("payout: informe [pessimista, base, otimista]")
        pay, pay_origin = (float(req.payout[0]), float(req.payout[1]), float(req.payout[2])), "informada por você"
    shares = a.get("acoes", -1)
    shares_src = f"ações diluídas do ano fiscal de {a.anos[-1][:4]} (balanços)"
    price = float(px.iloc[-1])
    if not shares:
        mcap = profile.fundamentos.get("valor_mercado")
        if not isinstance(mcap, float):
            raise ValueError("sem o número de ações nos balanços nem o valor de mercado: valor por ação não calculável")
        shares, shares_src = mcap / price, "valor de mercado ÷ preço (indicador do Yahoo Finance)"
    level = a.get("lucro_liquido", -1) if fin else a.get("receita", -1)
    if level is None or level <= 0:
        raise ValueError("receita (ou lucro, para bancos) do último ano ausente ou não positiva: simulação sem base")
    m_now = None
    if not fin:
        f_last, rev = fcf(a, -1), a.get("receita", -1)
        m_now = f_last / rev if f_last is not None and rev else p["margem_fcl"][1]
    base = Base(float(level), m_now, float(shares), price, fin)
    if progress:
        progress("simulando os cenários…")
    res = simulate(base, p, pay)
    if res["parcela_perpetuidade"] is not None and res["parcela_perpetuidade"] > 0.75:
        warns.append(f"{pct(res['parcela_perpetuidade'], 0)} do valor vem depois do ano 5 (perpetuidade): o resultado "
                     "depende sobretudo das premissas de longo prazo")
    if m_now is not None and m_now < 0:
        warns.append("o caixa livre de hoje é negativo: o valor depende de a margem melhorar até o ano 5")
    if res["valor_por_acao"]["base"] <= 0:
        warns.append("valor negativo nas premissas base: o caixa gerado não cobre o retorno exigido")
    warns.append("o sorteio trata as premissas como independentes; os cenários pessimista e otimista combinam os extremos "
                 "de uma vez, por isso costumam ficar fora da faixa sorteada")
    out = {
        "ticker": r.ticker, "nome": profile.nome, "logo": logo_path(r.ticker), "moeda": s3.currency,
        "preco": price, "data_preco": f"{px.index[-1]:%Y-%m-%d}", "financeira": fin,
        "base": {"nivel": level, "nivel_nome": "lucro líquido" if fin else "receita", "ano_fiscal": a.anos[-1],
                 "margem_atual": m_now, "acoes": shares, "acoes_fonte": shares_src, "beta": beta},
        "premissas": {k: {"valores": list(p[k]), "origem": origin[k], "nome": DRIVER_PT[k]} for k in DRIVERS},
        "payout": None if pay is None else {"valores": list(pay), "origem": pay_origin, "nome": DRIVER_PT["payout"]},
        "taxa_livre_de_risco": rf, "fonte_taxa": rf_src, "resultado": res, "avisos": warns,
        "fontes": [f"{a.fonte} (consultado em {a.consultado_em})", *(x["dado"] + ": " + x["fonte"] for x in s3.sources)],
        "aviso": DISCLAIMER, "calculado_em": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    bad = guarantee_phrases(" ".join([*warns, *origin.values(), DISCLAIMER]))
    if bad:
        raise AssertionError(f"linguagem de garantia em texto gerado: {bad}")
    return out


class ChainRequest(BaseModel):
    choque: Literal[tuple(SHOCKS)]  # type: ignore[valid-type]
    tamanho: float | None = None
    mercado: Literal["BR", "US"] = "BR"
    ativos: list[str] | None = Field(None, max_length=40)

    def key(self) -> str:
        return f"cadeia|{self.choque}|{self.tamanho}|{self.mercado}|{sorted(self.ativos or [])}"


def handle_chain(req: ChainRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                 cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                 budget_s: float | None = None) -> dict[str, object]:
    prov = provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data", progress, budget_s)
    return chain_effects(req.choque, req.tamanho, req.mercado, prov, [t for t in (req.ativos or []) if t.strip()] or None, progress)
