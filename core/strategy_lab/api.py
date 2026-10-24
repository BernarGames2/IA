"""Web entry point of the strategy lab (routed by ``webapp/server.py``): real prices only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from core.forecasting.data import rf_index_from_annual_pct
from core.market_data.base import DataUnavailableError
from core.market_data.cash import fetch_cdi_cash
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import _default_provider
from core.strategy_lab.lab import (
    HYPOTHESES,
    INDEX_TICKER,
    build_returns,
    run_lab,
    universe,
)
from utils.validation import InsufficientDataError


class StrategyRequest(BaseModel):
    hipotese: Literal[tuple(HYPOTHESES)]  # type: ignore[valid-type]
    mercado: Literal["BR", "US"] = "BR"
    custo_bps: float = Field(15.0, ge=0, le=200)
    ativos: list[str] | None = Field(None, max_length=60)

    def key(self) -> str:
        return f"estrategia|{self.hipotese}|{self.mercado}|{self.custo_bps:g}|{sorted(self.ativos or [])}"


def handle_strategy(req: StrategyRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                    cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                    budget_s: float | None = None) -> dict[str, object]:
    names = universe(req.mercado, req.ativos)
    market = req.mercado
    if req.ativos:
        br = sum(t.endswith(".SA") for t in names)
        if 0 < br < len(names):
            raise ValueError("use ações de um só mercado (todas da B3 ou todas dos EUA): índice e juros são de uma moeda")
        market = "BR" if br else "US"
    prov = provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data", progress, budget_s)
    idx_t = INDEX_TICKER[market]
    if progress:
        progress(f"baixando até 10 anos de cotações de {len(names)} ações, do índice e dos juros…")
    extra = ["^IRX"] if market == "US" else []
    got = prov.fetch([*names, idx_t, *extra], "10y", "1d")  # type: ignore[attr-defined]
    if idx_t not in got or got[idx_t].close is None:
        why = getattr(prov, "failures", {}).get(idx_t, "sem cotações")
        raise DataUnavailableError(f"índice de mercado {idx_t} indisponível agora: {why}")
    prices = {t: got[t].close for t in names if t in got and got[t].close is not None}
    notes = []
    cash = None
    try:
        if market == "BR":
            end = datetime.now(UTC).date()
            cash = fetch_cdi_cash(fetch, start=end - timedelta(days=3650), end=end).close
        elif "^IRX" in got and got["^IRX"].close is not None:
            cash = rf_index_from_annual_pct(got["^IRX"].close.dropna())
    except Exception as e:  # noqa: BLE001 - reported; the trend hypothesis then refuses, the others measure vs zero
        notes.append(f"taxa de juros histórica indisponível ({e})")
    if progress:
        progress("rodando o teste fora da amostra de todas as variantes…")
    try:
        frame, used, left = build_returns(prices, got[idx_t].close, cash)
    except InsufficientDataError as e:
        raise ValueError(str(e)) from e
    out = run_lab(req.hipotese, frame, used, req.custo_bps)
    out["mercado"] = market
    out["indice"] = idx_t
    out["fora_por_falta_de_dados"] = left + sorted(set(names) - set(prices))
    out["avisos"] = [*notes, *out["avisos"]]  # type: ignore[misc]
    out["fontes"] = [f"{getattr(prov, 'name', 'provedor')}: cotações diárias ajustadas",
                     "Banco Central do Brasil, SGS 12 (CDI)" if market == "BR" else "Yahoo Finance ^IRX (T-bill)"]
    return out
