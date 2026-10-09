"""Web API (FastAPI) for the Quant Portfolio Intelligence Engine — Vercel entrypoint.

Deployed as a Vercel Python Function (see ``pyproject.toml`` -> ``[tool.vercel]``).
The web version runs ONLY on deterministic synthetic data: redistributing
provider quotes on a public site is not allowed by Yahoo's terms, and the
offline mode is the validated path. Inputs are validated (pydantic, extra
fields forbidden) and bounded so a request fits the platform's time/memory
limits. No order execution, no external content, no secrets.

Local run:  uvicorn webapp.server:app --reload   ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from config import APP_NAME, APP_VERSION, AppConfig
from core.ai_analyst.analyst import ResearchAnalyst
from core.ai_analyst.tools import AnalysisContext
from core.portfolio_engine import AnalysisResult, run_analysis
from core.profiler import demo_profile
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


app = FastAPI(title=f"{APP_NAME} — web", version=APP_VERSION, docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")

_slots = threading.BoundedSemaphore(_MAX_CONCURRENT)
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[object, ...], tuple[AnalysisResult, float]] = OrderedDict()


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
    return {"status": "ok", "version": APP_VERSION, "data": "synthetic-only"}


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
