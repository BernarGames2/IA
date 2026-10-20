"""Daily-liquidity cash in reais, modelled by the CDI (Banco Central do Brasil, SGS series 12).

It is not an exchange ticker: it stands for Tesouro Selic, daily-liquidity CDBs or DI funds, which
pay close to the CDI. The level series is 100 x the cumulative product of the published daily CDI
rates — real data, nothing simulated — and it is labelled as a model of those instruments wherever it
appears. When the source fails, the portfolio is built without it and the failure is reported.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime

import pandas as pd

from core.market_data.base import DownloadError, EmptyDataError, RawSeries
from core.market_data.yahoo_chart import FetchFn, urllib_fetch
from core.purpose.catalog import CatalogAsset

CASH_TICKER = "CAIXA_CDI"
BCB_CDI_URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               "&dataInicial={ini:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")
CASH_ASSET = CatalogAsset(
    CASH_TICKER, "Caixa pós-fixado (CDI)", "caixa", "fixed_income", "caixa", "BR", "BRL",
    "liquidez diária rendendo perto de 100% do CDI (Tesouro Selic, CDB de liquidez diária ou fundo DI); modelado "
    "pelo CDI diário publicado pelo Banco Central (SGS 12) — não é um ticker de bolsa",
    ("preservacao_capital", "renda_recorrente"),
    "risco do emissor (CDB/fundo) ou soberano (Tesouro); pode render menos que a inflação em alguns períodos")


def parse_cdi(body: bytes) -> pd.Series:
    rows = json.loads(body)
    if not isinstance(rows, list) or not rows:
        raise EmptyDataError("CDI (BCB SGS 12): resposta vazia ou em formato inesperado")
    dates, rates = [], []
    for r in rows:
        try:
            dates.append(pd.to_datetime(str(r["data"]), format="%d/%m/%Y"))
            rates.append(float(str(r["valor"]).replace(",", ".")))
        except (KeyError, TypeError, ValueError) as e:
            raise EmptyDataError(f"CDI (BCB SGS 12): linha inválida {r!r}") from e
    daily = pd.Series(rates, index=pd.DatetimeIndex(dates)).sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    return 100.0 * (1.0 + daily / 100.0).cumprod()


def fetch_cdi_cash(fetch: FetchFn | None = None, start: date | None = None, end: date | None = None,
                   timeout_s: float = 15.0) -> RawSeries:
    end = end or datetime.now(UTC).date()
    start = start or date(end.year - 5, end.month, 1)
    st, body = (fetch or urllib_fetch)(BCB_CDI_URL.format(ini=start, fim=end), timeout_s)
    if st != 200:
        raise DownloadError(f"CDI (BCB SGS 12): HTTP {st}")
    level = parse_cdi(body)
    return RawSeries(CASH_TICKER, level.rename(CASH_TICKER), None, "bcb_sgs_12", "indice_cdi_acumulado",
                     "juros diários do CDI acumulados", "BRL", "1d", ingested_at_utc=datetime.now(UTC).isoformat(),
                     notes=["caixa modelado pelo CDI diário (Banco Central, SGS 12); não é cotação de bolsa"])


class WithExtraSeries:
    """A provider for one request: serves series built elsewhere (the CDI cash), asks the wrapped
    provider for everything else and remembers what it got, so the candidates are re-read from
    exactly the series that were screened (same data within a request; no second download).
    Failures of the wrapped provider stay visible."""

    def __init__(self, base: object, extra: dict[str, RawSeries] | None = None) -> None:
        self.base = base
        self.extra = dict(extra or {})
        self.name = getattr(base, "name", "provedor")
        self._memo: dict[tuple[str, str, str], RawSeries] = {}

    @property
    def failures(self) -> dict[str, str]:
        return getattr(self.base, "failures", {})

    def fetch(self, symbols: list[str], period: str, interval: str) -> dict[str, RawSeries]:
        out = {s: self.extra[s] for s in symbols if s in self.extra}
        out.update({s: self._memo[(s, period, interval)] for s in symbols if (s, period, interval) in self._memo})
        rest = [s for s in symbols if s not in out]
        if rest:
            got = self.base.fetch(rest, period, interval)  # type: ignore[attr-defined]
            self._memo.update({(s, period, interval): rs for s, rs in got.items()})
            out.update(got)
        return out


CashLoader = Callable[[], RawSeries]
