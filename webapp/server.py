"""Web API (FastAPI) for the Quant Portfolio Intelligence Engine — Vercel entrypoint.

Deployed as a Vercel Python Function (see ``pyproject.toml`` -> ``[tool.vercel]``).
* ``/api/analisar`` and ``/api/perguntar`` run on the deterministic synthetic market.
* ``/api/proposito`` (purpose-aligned portfolio) uses REAL prices by default: daily
  adjusted closes of individual tickers downloaded at request time from Yahoo Finance's
  chart endpoint (cached in /tmp), plus the Selic target (BCB) and the 13-week T-bill
  (^IRX) as risk-free rates. Only derived analytics are shown (no quote feed is
  redistributed). If real data is unavailable the request fails with a clear message;
  synthetic data is never substituted. ``fonte_dados="sintetico"`` runs the labelled demo.
Inputs are validated (pydantic, extra fields forbidden) and bounded so a request fits
the platform's time/memory limits. No order execution, no secrets.

Local run:  uvicorn webapp.server:app --reload   ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from config import APP_NAME, APP_VERSION, AppConfig
from core.ai_analyst.analyst import ResearchAnalyst
from core.ai_analyst.tools import AnalysisContext
from core.market_data.base import DataUnavailableError
from core.portfolio_engine import AnalysisResult, run_analysis
from core.profiler import demo_profile
from core.purpose.engine import PurposeRequest, PurposeResult, run_purpose
from core.purpose.report import render_markdown
from core.purpose.report import to_dict as purpose_to_dict
from core.stock_check.analyzer import analyze_stock
from core.stock_check.analyzer import to_dict as stock_to_dict
from models.investor import InvestorProfileInput, RiskProfile
from utils.serialization import to_jsonable
from webapp.summary import build_summary

log = logging.getLogger("qpi.web")

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_OUTPUT_DIR = Path(tempfile.gettempdir()) / "qpi-web"   # only /tmp is writable on serverless hosts
_MAX_CONCURRENT = 2
_CACHE_SIZE = 4


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["conservador", "moderado", "arrojado"] = "moderado"
    capital: float = Field(100_000.0, gt=0, le=1e9)
    monthly_contribution: float = Field(1_000.0, ge=0, le=1e7)
    horizon_years: float = Field(5.0, gt=0, le=30)
    simulations: int = Field(5_000, ge=500, le=20_000)
    mode: Literal["rapido", "completo"] = "rapido"
    seed: int = Field(42, ge=0, le=2**31 - 1)
    risk_free_rate: float | None = Field(None, ge=-0.05, le=1.0)

    def key(self) -> tuple[object, ...]:
        return tuple(self.model_dump().values())


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: AnalyzeRequest = Field(default_factory=AnalyzeRequest)
    question: str = Field(..., min_length=2, max_length=500)


class PurposeWebRequest(BaseModel):
    """Purpose-engine form, bounded for the platform limits. ``fonte_dados="real"`` downloads real
    prices (Yahoo Finance chart endpoint) for the catalogue of individual tickers; "sintetico" runs
    the labelled offline demonstration. A real-data failure is reported, never replaced."""

    model_config = ConfigDict(extra="forbid")

    fonte_dados: Literal["real", "sintetico"] = "real"

    capital_inicial: float = Field(100_000.0, gt=0, le=1e9)
    aporte_periodico: float = Field(1_000.0, ge=0, le=1e7)
    periodicidade: Literal["mensal", "trimestral", "semestral", "anual", "nenhum"] = "mensal"
    valor_alvo: float = Field(500_000.0, gt=0, le=1e11)
    horizonte_anos: float = Field(10.0, gt=0, le=30)
    perfil_risco: Literal["conservador", "moderado", "arrojado"] = "moderado"
    taxa_livre_risco: dict[Literal["BRL", "USD"], float] = Field(default_factory=dict)
    restricoes_governanca: dict[str, dict[str, str]] = Field(default_factory=dict, max_length=50)
    incluir_sob_revisao: bool = False
    motivo_investidor: str = Field("", max_length=1000)
    peso_alinhamento: float = Field(0.30, ge=0.0, le=1.0)
    n_simulacoes: int = Field(3_000, ge=500, le=10_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)

    def key(self) -> str:
        # real prices change every day: cached real results are reused only within the same UTC day
        day = datetime.now(UTC).date().isoformat() if self.fonte_dados == "real" else ""
        return self.model_dump_json() + day


class StockRequest(BaseModel):
    """'Vale a pena?' — one asset typed by the user (ticker or company name) + projection."""

    model_config = ConfigDict(extra="forbid")

    acao: str = Field(..., min_length=1, max_length=60)
    horizonte_anos: float = Field(3.0, ge=0.25, le=30)
    valor: float = Field(10_000.0, gt=0, le=1e9)
    perfil: Literal["conservador", "moderado", "arrojado"] = "moderado"
    fonte_dados: Literal["real", "sintetico"] = "real"
    taxa_livre_risco: float | None = Field(None, ge=-0.05, le=1.0)
    n_simulacoes: int = Field(5_000, ge=500, le=20_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)

    def key(self) -> str:
        day = datetime.now(UTC).date().isoformat() if self.fonte_dados == "real" else ""
        return "acao|" + self.model_dump_json() + day


app = FastAPI(title=f"{APP_NAME} — web", version=APP_VERSION, docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")

_slots = threading.BoundedSemaphore(_MAX_CONCURRENT)
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[object, ...], tuple[AnalysisResult, float]] = OrderedDict()
_purpose_cache: OrderedDict[str, dict[str, object]] = OrderedDict()


def _profile_input(p: AnalyzeRequest) -> InvestorProfileInput:
    base = demo_profile(RiskProfile(p.profile))
    return InvestorProfileInput(**{**base.model_dump(), "investor_id": f"web-{p.profile}",
                                   "capital": p.capital, "monthly_contribution": p.monthly_contribution,
                                   "horizon_years": p.horizon_years})


def _config(p: AnalyzeRequest) -> AppConfig:
    full = p.mode == "completo"
    return AppConfig(offline=True, profile=p.profile, n_simulations=p.simulations, seed=p.seed,
                     risk_free_rate=p.risk_free_rate, run_heavy_analyses=full, n_resamples=20 if full else 0,
                     make_charts=False, output_dir=_OUTPUT_DIR)


def _analysis(p: AnalyzeRequest) -> tuple[AnalysisResult, float]:
    """Run (or reuse) a deterministic analysis; identical inputs give identical results."""
    k = p.key()
    with _cache_lock:
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
    if not _slots.acquire(timeout=120):
        raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
    try:
        t0 = time.perf_counter()
        res = run_analysis(_config(p), _profile_input(p))
        out = (res, time.perf_counter() - t0)
    finally:
        _slots.release()
    with _cache_lock:
        _cache[k] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION,
            "data": "análise geral: sintética; carteira com propósito: real (Yahoo Finance) ou sintética"}


@app.post("/api/analisar")
def analisar(p: AnalyzeRequest) -> JSONResponse:
    try:
        res, elapsed = _analysis(p)
    except HTTPException:
        raise
    except (ValueError, ArithmeticError) as e:   # invalid combination of inputs -> client error
        raise HTTPException(400, f"análise não pôde ser executada: {e}") from e
    return JSONResponse(build_summary(res, p.mode, elapsed))


@app.post("/api/perguntar")
def perguntar(req: AskRequest) -> JSONResponse:
    try:
        res, _ = _analysis(req.params)
    except HTTPException:
        raise
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(400, f"análise não pôde ser executada: {e}") from e
    analyst = ResearchAnalyst(AnalysisContext.from_result(res))
    ans = analyst.ask(req.question)
    return JSONResponse(to_jsonable({
        "question": ans.question, "intents": ans.intents, "refusals": ans.refusals,
        "claims": [{"label": c.label, "value": c.value, "fmt": c.fmt, "kind": c.kind, "call_id": c.call_id,
                    "rendered": c.render()} for c in ans.claims],
        "limitations": ans.limitations, "verified": ans.verified,
        "tool_calls": [{"call_id": t.call_id, "tool": t.tool, "status": t.status, "error": t.error,
                        "digest": t.output_digest, "duration_s": t.duration_s} for t in ans.tool_calls],
        "engine": "regras determinísticas + ferramentas validadas (nenhum LLM externo)",
    }))


def _market_provider() -> object | None:
    """Real-price provider for the purpose engine (None = engine default: Yahoo chart + /tmp cache).
    Kept as a function so tests can substitute a fake provider."""
    return None


def _rf_fetch() -> object | None:
    """HTTP fetch used for the Selic (BCB) quote; None = standard library. Replaceable in tests."""
    return None


def _purpose_payload(res: PurposeResult, elapsed: float) -> dict[str, object]:
    out = purpose_to_dict(res)
    out["markdown"] = render_markdown(res)
    out["elapsed_s"] = round(elapsed, 2)
    out["run_id"] = res.analysis.run_id
    return out


@app.post("/api/proposito")
def proposito(p: PurposeWebRequest) -> JSONResponse:
    k = p.key()
    with _cache_lock:
        if k in _purpose_cache:
            _purpose_cache.move_to_end(k)
            return JSONResponse(_purpose_cache[k])
    offline = p.fonte_dados == "sintetico"
    try:
        req = PurposeRequest(**p.model_dump(exclude={"fonte_dados"}), offline=offline)
    except ValueError as e:
        raise HTTPException(400, f"entrada inválida: {e}") from e
    if not _slots.acquire(timeout=120):
        raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
    try:
        t0 = time.perf_counter()
        res = run_purpose(req, output_dir=_OUTPUT_DIR / "proposito", cache_dir=_OUTPUT_DIR / "cache",
                          provider=None if offline else _market_provider(), rf_fetch=_rf_fetch())
        payload = _purpose_payload(res, time.perf_counter() - t0)
    except DataUnavailableError as e:
        why = str(e)
        why = why if len(why) <= 240 else why[:240] + "…"
        raise HTTPException(503, f"dados reais indisponíveis agora ({why}). Nenhum dado sintético foi usado em "
                                 "substituição: tente de novo em alguns minutos ou escolha 'Demonstração "
                                 "sintética'.") from e
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(400, f"análise não pôde ser executada: {e}") from e
    finally:
        _slots.release()
    with _cache_lock:
        _purpose_cache[k] = payload
        while len(_purpose_cache) > _CACHE_SIZE:
            _purpose_cache.popitem(last=False)
    return JSONResponse(payload)


@app.post("/api/acao")
def acao(p: StockRequest) -> JSONResponse:
    k = p.key()
    with _cache_lock:
        if k in _purpose_cache:
            _purpose_cache.move_to_end(k)
            return JSONResponse(_purpose_cache[k])
    if not _slots.acquire(timeout=120):
        raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
    try:
        t0 = time.perf_counter()
        res = analyze_stock(p.acao, p.horizonte_anos, p.valor, p.perfil, offline=p.fonte_dados == "sintetico",
                            rf=p.taxa_livre_risco, n_paths=p.n_simulacoes, seed=p.seed,
                            provider=None if p.fonte_dados == "sintetico" else _market_provider(),
                            rf_fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache")
        payload = stock_to_dict(res)
        payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
    except DataUnavailableError as e:
        why = str(e)
        why = why if len(why) <= 240 else why[:240] + "…"
        raise HTTPException(503, f"dados indisponíveis agora ({why}). Nada foi estimado no lugar: confira o código "
                                 "da ação ou tente de novo em alguns minutos.") from e
    except (ValueError, ArithmeticError) as e:
        raise HTTPException(400, str(e)) from e
    finally:
        _slots.release()
    with _cache_lock:
        _purpose_cache[k] = payload
        while len(_purpose_cache) > _CACHE_SIZE:
            _purpose_cache.popitem(last=False)
    return JSONResponse(payload)
