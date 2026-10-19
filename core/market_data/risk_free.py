"""Risk-free rates from public sources — never invented. Any failure raises, and the caller
falls back to an explicitly flagged demonstration value (never to zero).

* BRL: Banco Central do Brasil, SGS series 432 — meta da taxa Selic (% a.a.). The effective
  CDI usually sits ~0.1 p.p. below the target.
* USD: Yahoo ``^IRX`` — 13-week US Treasury bill yield (% a.a., discount basis), used as an
  approximation of the annual risk-free rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.market_data.base import DownloadError, EmptyDataError
from core.market_data.yahoo_chart import FetchFn, YahooChartProvider, urllib_fetch

BCB_SELIC_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"


@dataclass(frozen=True)
class RiskFreeQuote:
    currency: str
    rate: float
    source: str
    as_of: str


def _sane(rate: float, what: str) -> float:
    if not -0.05 <= rate <= 1.0:
        raise DownloadError(f"{what}: valor fora do intervalo plausível ({rate:.4f})")
    return rate


def fetch_selic(fetch: FetchFn | None = None, timeout_s: float = 10.0) -> RiskFreeQuote:
    try:
        status, body = (fetch or urllib_fetch)(BCB_SELIC_URL, timeout_s)
    except OSError as e:
        raise DownloadError(f"Selic (BCB): erro de rede: {e}") from e
    if status != 200:
        raise DownloadError(f"Selic (BCB): HTTP {status}")
    try:
        last = json.loads(body.decode("utf-8"))[-1]
        rate = float(str(last["valor"]).replace(",", ".")) / 100.0
        as_of = str(last["data"])
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError) as e:
        raise DownloadError(f"Selic (BCB): resposta inválida: {e}") from e
    return RiskFreeQuote("BRL", _sane(rate, "Selic"), "Banco Central do Brasil, SGS 432 (meta Selic)", as_of)


def fetch_us_tbill(provider: YahooChartProvider) -> RiskFreeQuote:
    """Last ^IRX close from any MarketDataProvider-like object."""
    got = provider.fetch(["^IRX"], "1mo", "1d").get("^IRX")
    if got is None:
        raise DownloadError(f"T-bill (^IRX): {getattr(provider, 'failures', {}).get('^IRX', 'indisponível')}")
    s = got.close.dropna()
    if s.empty:
        raise EmptyDataError("T-bill (^IRX): série vazia")
    return RiskFreeQuote("USD", _sane(float(s.iloc[-1]) / 100.0, "T-bill"),
                         "Yahoo Finance ^IRX (T-bill 13 semanas, base desconto; aproximação)",
                         str(s.index[-1].date()))


def fetch_live_rates(currencies: set[str], provider: YahooChartProvider,
                     fetch: FetchFn | None = None) -> tuple[dict[str, RiskFreeQuote], dict[str, str]]:
    """Fetch what is available for ``currencies``; returns (quotes, errors by currency)."""
    quotes: dict[str, RiskFreeQuote] = {}
    errors: dict[str, str] = {}
    for c in sorted(currencies):
        try:
            if c == "BRL":
                quotes[c] = fetch_selic(fetch)
            elif c == "USD":
                quotes[c] = fetch_us_tbill(provider)
            else:
                errors[c] = "sem fonte automática para esta moeda"
        except (DownloadError, EmptyDataError) as e:
            errors[c] = str(e)
    return quotes, errors
