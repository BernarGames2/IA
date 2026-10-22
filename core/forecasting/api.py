"""Web entry points of the forecasts (routed by ``webapp/server.py``): real data only, failures reported."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.forecasting.audit import (
    GENESIS,
    MAX_ENTRIES,
    audit_key,
    evaluate,
    is_retroactive,
    register,
    seal,
)
from core.forecasting.data import load
from core.forecasting.forecast import build
from core.market_data.risk_free import fetch_live_rates
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import _default_provider
from core.stock_check.resolver import resolve


class ForecastRequest(BaseModel):
    ativo: str = Field(min_length=1, max_length=60)
    horizonte_anos: float = Field(1.0, ge=0.25, le=10)
    premio_risco: float = Field(0.04, ge=0.0, le=0.10)
    tau: float = Field(0.10, ge=0.02, le=0.50)

    def key(self) -> str:
        return f"previsao|{self.ativo.strip().upper()}|{self.horizonte_anos:g}|{self.premio_risco:g}|{self.tau:g}"


class RegisterRequest(BaseModel):
    registro: dict[str, Any]
    hash_anterior: str = Field(GENESIS, min_length=64, max_length=64)


class EvaluateRequest(BaseModel):
    entradas: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_ENTRIES)
    custo_bps: float = Field(0.0, ge=0, le=500)


def _provider(provider: object | None, cache_dir: Path | None, progress: Callable[[str], None] | None,
              budget_s: float | None) -> object:
    return provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data",
                                                                   progress, budget_s)


def handle_forecast(req: ForecastRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                    cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                    budget_s: float | None = None, as_of: date | None = None) -> dict[str, object]:
    r = resolve(req.ativo)
    if r.ticker.startswith("SINT_"):
        raise ValueError("as previsões usam só dados reais: códigos SINT_* são sintéticos")
    prov = _provider(provider, cache_dir, progress, budget_s)
    if progress:
        progress(f"baixando 10 anos de cotações de {r.ticker}, do índice de referência e da renda fixa…")
    s3 = load(r.ticker, prov, fetch, "10y", as_of)
    live, _errors = fetch_live_rates({s3.currency}, prov, fetch)  # type: ignore[arg-type]
    q = live.get(s3.currency)
    if progress:
        progress("calculando a distribuição e testando o modelo no passado deste ativo…")
    out = build(s3, req.horizonte_anos, q.rate if q else None, f"{q.source}, em {q.as_of}" if q else "",
                req.premio_risco, req.tau)
    out["registro"] = seal(out["registro"], audit_key())  # type: ignore[arg-type]
    out["interpretacao"] = r.how
    return out


def handle_register(req: RegisterRequest) -> dict[str, object]:
    key = audit_key()
    entry = register(req.registro, req.hash_anterior, key)
    return {"entrada": entry, "assinado_pelo_servidor": key is not None,
            "retroativa": is_retroactive(entry["registro"], str(entry["registrado_em"]))}  # type: ignore[arg-type]


def handle_evaluate(req: EvaluateRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                    cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                    budget_s: float | None = None, as_of: date | None = None) -> dict[str, object]:
    if progress:
        progress("conferindo a cadeia do registro e baixando as cotações até hoje…")
    return evaluate(req.entradas, _provider(provider, cache_dir, progress, budget_s), fetch, as_of, req.custo_bps,
                    audit_key())
