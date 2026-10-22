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

import hashlib
import json
import logging
import os
import queue
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import APP_NAME, APP_VERSION, AppConfig
from core.ai_analyst.analyst import ResearchAnalyst
from core.ai_analyst.tools import AnalysisContext
from core.forecasting.api import (
    EvaluateRequest,
    ForecastRequest,
    RegisterRequest,
    handle_evaluate,
    handle_forecast,
    handle_register,
)
from core.market_data.base import DataUnavailableError
from core.market_data.logos import clean_domain, domain_for, fetch_logo
from core.portfolio_analytics.api import handle_dashboard, handle_risk
from core.portfolio_analytics.data import PortfolioRequest
from core.portfolio_engine import AnalysisResult, run_analysis
from core.profiler import demo_profile
from core.purpose.engine import (
    AmountTooSmall,
    PurposeRequest,
    PurposeResult,
    run_purpose,
)
from core.purpose.report import render_markdown
from core.purpose.report import to_dict as purpose_to_dict
from core.stock_check.analyzer import analyze_stock
from core.stock_check.analyzer import to_dict as stock_to_dict
from core.stock_check.daily_pick import daily_pick
from models.investor import InvestorProfileInput, RiskProfile
from utils.serialization import to_jsonable
from webapp.summary import build_summary

log = logging.getLogger("qpi.web")

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
# the page's own static files (served from the app's address; a font changes rarely: cached for 30 days)
_ASSETS = {"fonts/inter-latin-wght-normal.woff2": "font/woff2"}
_ASSET_BYTES = {k: (Path(__file__).parent / "static" / k).read_bytes() for k in _ASSETS}
_OUTPUT_DIR = Path(tempfile.gettempdir()) / "qpi-web"   # only /tmp is writable on serverless hosts
_MAX_CONCURRENT = 2
_CACHE_SIZE = 4
_RESULT_TTL_S = 15 * 60       # a real-data result is reused for at most 15 minutes, then recomputed
_DOWNLOAD_BUDGET_S = 45.0     # total time allowed for downloading real quotes in one request
_PING_EVERY_S = 5.0           # keep-alive event interval of streamed responses


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

    capital_inicial: float = Field(100_000.0, ge=0, le=1e9)   # 0 = começa só com os aportes
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
    universo: Literal["completo", "curado"] = "completo"
    max_ativos: int = Field(40, ge=10, le=60)            # bounded so a request fits the platform time limit
    objetivo_tipo: Literal["crescimento", "renda", "preservacao"] | None = None
    tolerancia_perda: float | None = Field(None, gt=0, lt=1)
    necessidade_liquidez: float | None = Field(None, ge=0, le=1)
    mercados: list[Literal["BR", "US"]] | None = Field(None, min_length=1, max_length=2)
    permitir_cripto: bool = False
    ativos_excluidos: list[str] = Field(default_factory=list, max_length=100)
    setores_excluidos: list[str] = Field(default_factory=list, max_length=40)
    carteira_atual: dict[str, float] = Field(default_factory=dict, max_length=100)
    carteira_anterior: dict[str, float] = Field(default_factory=dict, max_length=100)
    carteira_anterior_parametros: str | None = Field(None, max_length=64)
    fracionario_eua: bool = False
    valor_minimo_posicao: float = Field(100.0, ge=0, le=1e6)

    def key(self) -> str:
        return self.model_dump_json()


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
        return "acao|" + self.model_dump_json()


class DailyPickRequest(BaseModel):
    """'Ação do dia' — how much the investor has today; which single stock the fixed rules point to."""

    model_config = ConfigDict(extra="forbid")

    valor: float = Field(..., gt=0, le=1e9)
    perfil: Literal["conservador", "moderado", "arrojado"] = "moderado"
    mercados: list[Literal["BR", "US"]] | None = Field(None, min_length=1, max_length=2)
    carteira_atual: dict[str, float] = Field(default_factory=dict, max_length=100)
    fracionario_eua: bool = False
    fonte_dados: Literal["real", "sintetico"] = "real"

    def key(self) -> str:
        return "dia|" + self.model_dump_json()


app = FastAPI(title=f"{APP_NAME} — web", version=APP_VERSION, docs_url="/api/docs", redoc_url=None,
              openapi_url="/api/openapi.json")

_slots = threading.BoundedSemaphore(_MAX_CONCURRENT)
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[object, ...], tuple[AnalysisResult, float]] = OrderedDict()
_purpose_cache: OrderedDict[str, tuple[float, dict[str, object]]] = OrderedDict()


def _cached(k: str, real: bool) -> dict[str, object] | None:
    """A stored result, if still valid: synthetic results never expire (same seed -> same numbers); a
    real-data result expires after _RESULT_TTL_S so new quotes lead to a new analysis. The copy says it
    came from the cache and when it was computed."""
    with _cache_lock:
        hit = _purpose_cache.get(k)
        if hit is None:
            return None
        t, payload = hit
        if real and time.time() - t > _RESULT_TTL_S:
            del _purpose_cache[k]
            return None
        _purpose_cache.move_to_end(k)
    return {**payload, "do_cache": True}


def _store(k: str, payload: dict[str, object]) -> None:
    with _cache_lock:
        _purpose_cache[k] = (time.time(), payload)
        while len(_purpose_cache) > _CACHE_SIZE:
            _purpose_cache.popitem(last=False)


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


@app.get("/static/{path:path}", include_in_schema=False)
def static_asset(path: str) -> Response:
    if path not in _ASSETS:
        raise HTTPException(404, "arquivo não encontrado")
    return Response(_ASSET_BYTES[path], media_type=_ASSETS[path],
                    headers={"Cache-Control": "public, max-age=2592000", "X-Content-Type-Options": "nosniff"})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION, "commit": _COMMIT,
            "data": "análise geral: sintética; carteira com propósito: real (Yahoo Finance) ou sintética"}


@app.post("/api/analisar")
def analisar(p: AnalyzeRequest, stream: bool = False) -> Response:
    def compute(progress: Callable[[str], None]) -> dict[str, object]:
        progress("calculando a análise " + ("completa (alguns minutos)…" if p.mode == "completo" else "rápida…"))
        try:
            res, elapsed = _analysis(p)
        except HTTPException:
            raise
        except (ValueError, ArithmeticError) as e:   # invalid inputs -> 400; library failure -> 500
            raise _analysis_error(e) from e
        return build_summary(res, p.mode, elapsed)

    return _respond(compute, stream)


@app.post("/api/perguntar")
def perguntar(req: AskRequest) -> JSONResponse:
    try:
        res, _ = _analysis(req.params)
    except HTTPException:
        raise
    except (ValueError, ArithmeticError) as e:
        raise _analysis_error(e) from e
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


def _logo_fetch() -> object | None:
    """HTTP fetch used for company logos; None = standard library. Replaceable in tests."""
    return None


def _logo_response(domain: str | None) -> Response:
    """The company's icon from this app's own address (the browser never calls the favicon service); 404 when
    there is none, so the page shows the ticker's initials."""
    img = fetch_logo(domain, _logo_fetch()) if domain else None  # type: ignore[arg-type]
    if img is None:
        return Response(status_code=404, headers={"Cache-Control": "public, max-age=3600, s-maxage=3600"})
    body, mime = img
    return Response(content=body, media_type=mime, headers={
        "Cache-Control": "public, max-age=604800, s-maxage=604800, stale-while-revalidate=86400",
        "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'"})


@app.get("/api/logo/site/{domain}")
def logo_site(domain: str) -> Response:
    """Logo by company domain (as the results give it)."""
    d = clean_domain(domain)
    return _logo_response(d if d == domain else None)


@app.get("/api/logo/ativo/{ticker}")
def logo_ticker(ticker: str) -> Response:
    """Logo by ticker, for the catalogue's companies (used before any result exists, e.g. the quick picks)."""
    return _logo_response(domain_for(ticker))


def _purpose_payload(res: PurposeResult, elapsed: float) -> dict[str, object]:
    out = purpose_to_dict(res)
    out["markdown"] = render_markdown(res)
    out["elapsed_s"] = round(elapsed, 2)
    out["run_id"] = res.analysis.run_id
    out["calculado_em_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    out["do_cache"] = False
    return out


@app.post("/api/proposito")
def proposito(p: PurposeWebRequest, stream: bool = False) -> Response:
    offline = p.fonte_dados == "sintetico"
    try:
        req = PurposeRequest(**p.model_dump(exclude={"fonte_dados"}), offline=offline)
    except ValidationError as e:
        msg = "; ".join(str(x.get("msg", "")).removeprefix("Value error, ") for x in e.errors())
        raise HTTPException(400, f"entrada inválida: {msg}") from e
    except ValueError as e:
        raise HTTPException(400, f"entrada inválida: {e}") from e

    def compute(progress: Callable[[str], None]) -> dict[str, object]:
        k = p.key()
        hit = _cached(k, not offline)
        if hit is not None:
            return hit
        if not _slots.acquire(timeout=120):
            raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
        try:
            t0 = time.perf_counter()
            res = run_purpose(req, output_dir=_OUTPUT_DIR / "proposito", cache_dir=_OUTPUT_DIR / "cache",
                              provider=None if offline else _market_provider(), rf_fetch=_rf_fetch(),
                              progress=progress, fast=True, download_budget_s=_DOWNLOAD_BUDGET_S)
            payload = _purpose_payload(res, time.perf_counter() - t0)
        except AmountTooSmall as e:
            raise HTTPException(400, str(e)) from e
        except DataUnavailableError as e:
            raise HTTPException(503, f"dados reais indisponíveis agora ({_short(e)}). Nenhum dado sintético foi "
                                     "usado em substituição: tente de novo em alguns minutos (as cotações já "
                                     "baixadas ficam guardadas) ou escolha 'Demonstração sintética'.") from e
        except (ValueError, ArithmeticError) as e:
            raise _analysis_error(e) from e
        finally:
            _slots.release()
        _store(k, payload)
        return payload

    return _respond(compute, stream)


@app.post("/api/acao")
def acao(p: StockRequest, stream: bool = False) -> Response:
    def compute(progress: Callable[[str], None]) -> dict[str, object]:
        k = p.key()
        hit = _cached(k, p.fonte_dados == "real")
        if hit is not None:
            return hit
        if not _slots.acquire(timeout=120):
            raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
        try:
            t0 = time.perf_counter()
            res = analyze_stock(p.acao, p.horizonte_anos, p.valor, p.perfil, offline=p.fonte_dados == "sintetico",
                                rf=p.taxa_livre_risco, n_paths=p.n_simulacoes, seed=p.seed,
                                provider=None if p.fonte_dados == "sintetico" else _market_provider(),
                                rf_fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
                                download_budget_s=_DOWNLOAD_BUDGET_S)
            payload = stock_to_dict(res)
            payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
            payload["calculado_em_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            payload["do_cache"] = False
        except DataUnavailableError as e:
            raise HTTPException(503, f"dados indisponíveis agora ({_short(e)}). Nada foi estimado no lugar: confira "
                                     "o código da ação ou tente de novo em alguns minutos.") from e
        except (ValueError, ArithmeticError) as e:
            if _raised_in_library(e) and not isinstance(e, ValidationError):
                raise _analysis_error(e) from e         # 500: a defect, not the user's input
            raise HTTPException(400, str(e)) from e
        finally:
            _slots.release()
        _store(k, payload)
        return payload

    return _respond(compute, stream)


@app.post("/api/acao-do-dia")
def acao_do_dia(p: DailyPickRequest, stream: bool = False) -> Response:
    if any(not (v >= 0) for v in p.carteira_atual.values()):
        raise HTTPException(400, "valores da carteira atual devem ser números maiores ou iguais a zero")

    def compute(progress: Callable[[str], None]) -> dict[str, object]:
        k = p.key()
        hit = _cached(k, p.fonte_dados == "real")
        if hit is not None:
            return hit
        if not _slots.acquire(timeout=120):
            raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
        try:
            t0 = time.perf_counter()
            res = daily_pick(p.valor, p.perfil, p.mercados, p.carteira_atual, p.fracionario_eua,
                             offline=p.fonte_dados == "sintetico",
                             provider=None if p.fonte_dados == "sintetico" else _market_provider(),
                             rf_fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
                             download_budget_s=_DOWNLOAD_BUDGET_S)
            payload = res.to_dict()
            payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
            payload["calculado_em_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            payload["do_cache"] = False
        except DataUnavailableError as e:
            raise HTTPException(503, f"dados indisponíveis agora ({_short(e)}). Nada foi estimado no lugar: tente de "
                                     "novo em alguns minutos ou escolha 'Demonstração sintética'.") from e
        except (ValueError, ArithmeticError) as e:
            if _raised_in_library(e) and not isinstance(e, ValidationError):
                raise _analysis_error(e) from e
            raise HTTPException(400, str(e)) from e
        finally:
            _slots.release()
        _store(k, payload)
        return payload

    return _respond(compute, stream)


def _engine(k: str, run: Callable[[Callable[[str], None]], dict[str, object]], stream: bool, real: bool = True,
            unavailable: str = "tente de novo em alguns minutos") -> Response:
    """Shared flow of the newer engines: cache (15 min for real data), one of the limited computation slots,
    input errors as 400, missing real data as 503 (nothing estimated in its place), defects as 500."""
    def compute(progress: Callable[[str], None]) -> dict[str, object]:
        hit = _cached(k, real)
        if hit is not None:
            return hit
        if not _slots.acquire(timeout=120):
            raise HTTPException(503, "servidor ocupado; tente novamente em instantes")
        try:
            t0 = time.perf_counter()
            payload = run(progress)
            payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
            payload["calculado_em_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            payload["do_cache"] = False
        except DataUnavailableError as e:
            raise HTTPException(503, f"dados indisponíveis agora ({_short(e)}). Nada foi estimado no lugar: "
                                     f"{unavailable}.") from e
        except (ValueError, ArithmeticError) as e:
            if _raised_in_library(e) and not isinstance(e, ValidationError):
                raise _analysis_error(e) from e
            raise HTTPException(400, str(e)) from e
        finally:
            _slots.release()
        _store(k, payload)
        return payload

    return _respond(compute, stream)


@app.post("/api/carteira/painel")
def carteira_painel(p: PortfolioRequest, stream: bool = False) -> Response:
    """The investor's portfolio: value, evolution, returns against CDI/Ibovespa/S&P 500, allocation, alerts."""
    return _engine(p.key() + "|painel", lambda progress: handle_dashboard(
        p, provider=_market_provider(), fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
        budget_s=_DOWNLOAD_BUDGET_S), stream, unavailable="confira os códigos ou tente de novo em alguns minutos")


@app.post("/api/carteira/risco")
def carteira_risco(p: PortfolioRequest, stream: bool = False) -> Response:
    """Advanced risk of today's portfolio: VaR/CVaR, drawdown, correlations in stress, contributions, liquidity,
    sensitivities, historical and hypothetical stress."""
    return _engine(p.key() + "|risco", lambda progress: handle_risk(
        p, provider=_market_provider(), fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
        budget_s=_DOWNLOAD_BUDGET_S), stream, unavailable="confira os códigos ou tente de novo em alguns minutos")


@app.post("/api/previsoes/prever")
def previsoes_prever(p: ForecastRequest, stream: bool = False) -> Response:
    """Probabilistic forecast of one asset (distribution, probabilities, sensitivity) and the model's own historical
    calibration on that asset; carries a sealed record that the investor may register."""
    return _engine(p.key(), lambda progress: handle_forecast(
        p, provider=_market_provider(), fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
        budget_s=_DOWNLOAD_BUDGET_S), stream, unavailable="confira o código ou tente de novo em alguns minutos")


@app.post("/api/previsoes/registrar")
def previsoes_registrar(p: RegisterRequest) -> Response:
    """Chains a sealed forecast to the investor's registry (kept in their browser); no data download."""
    try:
        return JSONResponse(to_jsonable(handle_register(p)))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/previsoes/avaliar")
def previsoes_avaliar(p: EvaluateRequest, stream: bool = False) -> Response:
    """Integrity of the registry and how each forecast turned out (pending until its horizon ends)."""
    k = "avaliar|" + hashlib.sha256(json.dumps(p.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    return _engine(k, lambda progress: handle_evaluate(
        p, provider=_market_provider(), fetch=_rf_fetch(), cache_dir=_OUTPUT_DIR / "cache", progress=progress,
        budget_s=_DOWNLOAD_BUDGET_S), stream, unavailable="tente de novo em alguns minutos")


_OWN_PACKAGES = {"core", "webapp", "models", "utils", "config", "main", "tests", "__main__"}
_COMMIT = (os.environ.get("VERCEL_GIT_COMMIT_SHA") or "")[:7] or "local"


def _raised_in_library(e: BaseException) -> bool:
    """True when the error surfaced inside a third-party library (pandas, numpy, scipy…): a defect of
    this service with valid inputs, not an invalid combination of inputs. Decided by the module
    name of the innermost frame, so it does not depend on where the packages are installed."""
    tb, last = e.__traceback__, None
    while tb is not None:
        last, tb = tb, tb.tb_next
    if last is None:
        return False
    module = str(last.tb_frame.f_globals.get("__name__", ""))
    return module.split(".")[0] not in _OWN_PACKAGES


def _analysis_error(e: Exception) -> HTTPException:
    if isinstance(e, ValidationError) or not _raised_in_library(e):
        return HTTPException(400, f"análise não pôde ser executada: {e}")
    log.error("internal failure while computing", exc_info=e)
    return HTTPException(500, "erro interno ao processar os dados: é uma falha do sistema, não dos valores que você "
                              f"informou (detalhe técnico: {_short(e, 160)}; versão {_COMMIT}). Tente de novo em "
                              "instantes; se o erro se repetir, ele precisa ser corrigido no código.")


def _short(e: Exception, n: int = 240) -> str:
    why = str(e)
    return why if len(why) <= n else why[:n] + "…"


def _line(obj: object) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n"


def _respond(compute: Callable[[Callable[[str], None]], dict[str, object]], stream: bool) -> Response:
    """Plain JSON, or (``?stream=1``) newline-delimited JSON events while the job runs:
    {"tipo":"progresso","msg":...} ... {"tipo":"ping"} every few seconds of silence, then exactly one
    {"tipo":"resultado","dados":{...}} or {"tipo":"erro","status":...,"detail":...}. The stream keeps
    the connection busy (mobile browsers and proxies drop idle requests) and shows progress."""
    if not stream:
        return JSONResponse(compute(lambda msg: None))
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def worker() -> None:
        try:
            events.put(("resultado", compute(lambda msg: events.put(("progresso", msg)))))
        except HTTPException as e:
            events.put(("erro", {"status": e.status_code, "detail": e.detail}))
        except Exception:  # the stream must always end with an answer
            log.exception("streamed job failed")
            events.put(("erro", {"status": 500, "detail": "erro interno inesperado; tente novamente"}))

    threading.Thread(target=worker, daemon=True).start()

    def gen() -> Iterator[str]:
        yield _line({"tipo": "progresso", "msg": "pedido recebido"})
        while True:
            try:
                kind, data = events.get(timeout=_PING_EVERY_S)
            except queue.Empty:
                yield _line({"tipo": "ping"})
                continue
            if kind == "progresso":
                yield _line({"tipo": "progresso", "msg": data})
            elif kind == "resultado":
                yield _line({"tipo": "resultado", "dados": data})
                return
            else:
                yield _line({"tipo": "erro", **data})  # type: ignore[dict-item]
                return

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
