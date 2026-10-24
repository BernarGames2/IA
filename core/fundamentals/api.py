"""Web entry points of the thesis and the opportunity radar (routed by ``webapp/server.py``): real data only."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.forecasting.data import load
from core.forecasting.forecast import build
from core.fundamentals.analysis import growth
from core.fundamentals.radar import STYLES, scan
from core.fundamentals.sources import fetch_analysts, fetch_annual
from core.fundamentals.thesis import build_thesis
from core.fundamentals.value_trap import detect, price_stats, trend_table
from core.market_data.risk_free import fetch_live_rates
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import _default_provider
from core.stock_check.company import SourceError, _yahoo_client, load_company_profile
from core.stock_check.resolver import resolve

SOURCE_ERRORS = (SourceError, OSError, ValueError, KeyError, TypeError)


class ThesisRequest(BaseModel):
    ativo: str = Field(min_length=1, max_length=60)
    horizonte_anos: float = Field(3.0, ge=0.25, le=10)

    def key(self) -> str:
        return f"tese|{self.ativo.strip().upper()}|{self.horizonte_anos:g}"


class RadarRequest(BaseModel):
    mercados: list[Literal["BR", "US"]] = Field(default_factory=lambda: ["BR"], min_length=1, max_length=2)
    estilo: Literal[tuple(STYLES)] = "equilibrado"  # type: ignore[valid-type]
    quantidade: int = Field(10, ge=3, le=20)

    def key(self) -> str:
        return f"radar|{','.join(sorted(set(self.mercados)))}|{self.estilo}|{self.quantidade}"


def _provider(provider: object | None, cache_dir: Path | None, progress: Callable[[str], None] | None,
              budget_s: float | None) -> object:
    return provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data",
                                                                   progress, budget_s)


def handle_thesis(req: ThesisRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                  cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                  budget_s: float | None = None, as_of: date | None = None) -> dict[str, object]:
    r = resolve(req.ativo)
    if r.ticker.startswith("SINT_"):
        raise ValueError("a tese usa só dados reais: códigos SINT_* são sintéticos")
    if r.ticker.endswith("-USD"):
        raise ValueError("criptoativos não têm empresa nem demonstrações: use a previsão probabilística")
    prov = _provider(provider, cache_dir, progress, budget_s)
    client = _yahoo_client(fetch)
    root = cache_dir or Path(".cache") / "market_data"
    if progress:
        progress(f"baixando cotações, demonstrações e estimativas de {r.ticker}…")
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_px = pool.submit(load, r.ticker, prov, fetch, "10y", as_of)
        f_prof = pool.submit(load_company_profile, r, False, fetch, root)
        f_ann = pool.submit(fetch_annual, client, r.ticker, root)
        f_an = pool.submit(fetch_analysts, client, r.ticker, root)
    s3 = f_px.result()                                    # no prices: DataUnavailableError -> 503
    profile = f_prof.result()
    missing = list(profile.indisponivel)
    annual = analysts = None
    try:
        annual = f_ann.result()
        if annual is None:
            missing.append("Yahoo Finance: nenhuma demonstração anual publicada para este ativo")
    except SOURCE_ERRORS as e:
        missing.append(f"demonstrações anuais indisponíveis ({e})")
    try:
        analysts = f_an.result()
    except SOURCE_ERRORS as e:
        missing.append(f"estimativas de analistas indisponíveis ({e})")
    live, _errors = fetch_live_rates({s3.currency}, prov, fetch)  # type: ignore[arg-type]
    q = live.get(s3.currency)
    if progress:
        progress("medindo fundamentos, sinais de armadilha e montando a tese…")
    fc = build(s3, req.horizonte_anos, q.rate if q else None, f"{q.source}, em {q.as_of}" if q else "")
    ps = price_stats(s3.px.iloc[-1260:])
    seg = profile.segmento
    fin = bool(seg and seg.financeiro)
    trap = detect(annual, profile.fundamentos, analysts, ps, fin)
    today = as_of or datetime.now(UTC).date()
    thesis = build_thesis(profile, annual, analysts, fc, trap, ps, today)
    cal = fc.get("calibracao") or {}
    return {
        "ticker": r.ticker, "nome": profile.nome, "logo": fc["logo"], "moeda": s3.currency,
        "data_referencia": fc["data_referencia"], "interpretacao": r.how,
        "empresa": {"descricao": profile.descricao, "descricao_fonte": profile.descricao_fonte,
                    "segmento": seg.rotulo if seg else None, "como_ganha": seg.como_ganha if seg else None,
                    "financeira": fin, "classificacao": profile.classificacao, "sede": profile.sede, "site": profile.site},
        "fundamentos": {"anos": trend_table(annual) if annual else [], "moeda": annual.moeda if annual else None,
                        "fonte": f"{annual.fonte} (consultado em {annual.consultado_em})" if annual else None,
                        "crescimento": growth(annual) if annual else {}, "multiplos": trap["multiplos"],
                        "piotroski": trap["piotroski"], "indicadores": profile.fundamentos,
                        "indicadores_fonte": profile.fundamentos_fonte},
        "analistas": asdict(analysts) if analysts else None,
        "armadilha": {k: v for k, v in trap.items() if k not in ("multiplos", "piotroski")},
        "tese": thesis,
        "previsao": {"horizonte_anos": fc["horizonte_anos"], "probabilidades": fc["probabilidades"], "quantis": fc["quantis"],
                     "nome_taxa": fc["nome_taxa"], "taxa_livre_de_risco": fc["taxa_livre_de_risco"],
                     "fonte_taxa": fc["fonte_taxa"], "referencia": fc["referencia"],
                     "calibracao": {k: cal.get(k) for k in ("n", "n_efetivo", "cobertura_10_90", "leitura", "insuficiente", "nota")}},
        "precos": {"ultimo": ps.get("ultimo"), "momento_12_1": ps.get("momento_12_1"),
                   "acima_media_200": ps.get("acima_media_200"), "queda_do_topo_3a": ps.get("queda_do_topo_3a")},
        "fontes": [*profile.fontes, *(x["dado"] + ": " + x["fonte"] for x in fc["fontes"])],  # type: ignore[union-attr]
        "indisponivel": missing, "avisos": fc["avisos"], "aviso": thesis["aviso"],
    }


def handle_radar(req: RadarRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                 cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                 budget_s: float | None = None) -> dict[str, object]:
    prov = _provider(provider, cache_dir, progress, budget_s)
    return scan(sorted(set(req.mercados)), req.estilo, prov, _yahoo_client(fetch), cache_dir or Path(".cache") / "market_data",
                progress, budget_s=35.0, top=req.quantidade)
