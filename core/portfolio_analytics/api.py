"""Web entry points of the investor's portfolio (routed by ``webapp/server.py``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.market_data.yahoo_chart import FetchFn
from core.portfolio_analytics.dashboard import build_dashboard
from core.portfolio_analytics.data import PortfolioRequest
from core.portfolio_analytics.risk import build_risk
from core.purpose.engine import _default_provider


def _provider(provider: object | None, cache_dir: Path | None, progress: Callable[[str], None] | None,
              budget_s: float | None) -> object:
    return provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data",
                                                                   progress, budget_s)


def handle_dashboard(req: PortfolioRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                     cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                     budget_s: float | None = None) -> dict[str, object]:
    if progress:
        progress("baixando cotações da carteira e dos índices de comparação…")
    return build_dashboard(req, _provider(provider, cache_dir, progress, budget_s), fetch)


def handle_risk(req: PortfolioRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
                budget_s: float | None = None) -> dict[str, object]:
    if progress:
        progress("baixando 10 anos de cotações para medir o risco…")
    return build_risk(req, _provider(provider, cache_dir, progress, budget_s), fetch)
