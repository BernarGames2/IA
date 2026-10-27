"""What the market is doing today: the main indices, the biggest gains and drops, and the prices of the stocks
followed — the numbers only (the site does not attribute a move to a cause; news are not read here).

Universe (documented, from the exchanges' own lists, nothing picked by hand):
* Brazil — the ``n`` shares with the largest traded value in the B3's daily file of the last session available
  (the same listing the portfolio engine uses); when the file cannot be read, the shares of the curated catalogue.
* United States — the ``n`` largest companies by market value in the Nasdaq screener (NYSE, Nasdaq and NYSE
  American); when it cannot be read, the US shares of the curated catalogue.
Quotes come from ``core.market_data.quotes``: delayed (the time of the last trade is shown on every line), the change
of the day against the previous session's close. A stock the source does not answer is counted and left out.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.market_data.logos import logo_path
from core.market_data.quotes import Quote, QuotesFn
from core.market_data.yahoo_chart import FetchFn
from core.purpose.catalog import REAL_CATALOG
from core.universe.listings import load_market_listings

INDICES: tuple[tuple[str, str, str], ...] = (
    ("^BVSP", "Ibovespa", "pontos"), ("USDBRL=X", "Dólar", "R$ por US$"), ("^GSPC", "S&P 500", "pontos"),
    ("^IXIC", "Nasdaq", "pontos"), ("BZ=F", "Petróleo Brent", "US$ por barril"))
TOP = 10
FLAT = 0.0005          # |change| below 0.05%: unchanged
_CATALOG = {a.ticker: a for a in REAL_CATALOG}


class MarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mercado: Literal["BR", "US"] = "BR"
    quantidade: int = Field(60, ge=10, le=120, description="quantas ações acompanhar")

    def key(self) -> str:
        return f"mercado|{self.mercado}|{self.quantidade}"


def _clean_us_name(name: str) -> str:
    for tail in (" Common Stock", " Class A Common Stock", " Class B Common Stock", " Ordinary Shares",
                 " American Depositary Shares", " Common Shares"):
        if name.endswith(tail):
            return name[: -len(tail)]
    return name


def universe(mercado: str, n: int, cache_dir: Path | None, fetch: FetchFn | None
             ) -> tuple[list[tuple[str, str]], str]:
    """(ticker, name) of the stocks followed and where the list came from."""
    listings = load_market_listings(cache_dir, fetch)
    if mercado == "BR":
        rows = sorted((a for a in listings.b3 if a.kind == "acao" and a.size), key=lambda a: -float(a.size or 0))
        if rows:
            picked = [(a.ticker, _CATALOG[a.ticker].name if a.ticker in _CATALOG else a.name) for a in rows[:n]]
            return picked, (f"as {len(picked)} ações com maior volume financeiro no último pregão do arquivo da B3 "
                            f"({listings.sources.get('B3', 'B3')})")
        cat = [(a.ticker, a.name) for a in REAL_CATALOG if a.macro_class == "acoes_br"][:n]
        return cat, (f"{len(cat)} ações brasileiras do catálogo do site (arquivo diário da B3 indisponível agora"
                     + (f": {listings.errors['B3']}" if "B3" in listings.errors else "") + ")")
    rows = sorted((a for a in listings.us if a.kind == "acao" and a.size), key=lambda a: -float(a.size or 0))
    if rows:
        picked = [(a.ticker, _CATALOG[a.ticker].name if a.ticker in _CATALOG else _clean_us_name(a.name))
                  for a in rows[:n]]
        return picked, (f"as {len(picked)} maiores empresas por valor de mercado listadas nos EUA (screener da Nasdaq: "
                        "NYSE, Nasdaq e NYSE American)")
    cat = [(a.ticker, a.name) for a in REAL_CATALOG if a.macro_class == "acoes_eua"][:n]
    return cat, (f"{len(cat)} ações americanas do catálogo do site (listagem da Nasdaq indisponível agora"
                 + (f": {listings.errors['EUA']}" if "EUA" in listings.errors else "") + ")")


def _row(q: Quote, name: str | None = None) -> dict[str, object]:
    return {"ticker": q.symbol, "nome": name or q.name, "logo": logo_path(q.symbol), "preco": q.price,
            "fechamento_anterior": q.previous_close, "variacao": q.change, "variacao_pct": q.change_pct,
            "moeda": q.currency, "data": q.date_local, "hora": q.time_local, "hora_utc": q.time_utc,
            "aberto": q.market_open, "volume": q.volume,
            "volume_financeiro": (q.volume * q.price) if q.volume else None, "maxima": q.day_high,
            "minima": q.day_low}


def overview(req: MarketRequest, quotes: QuotesFn, *, cache_dir: Path | None = None, fetch: FetchFn | None = None,
             progress: Callable[[str], None] | None = None, now: datetime | None = None) -> dict[str, object]:
    from core.market_data.base import DataUnavailableError

    if progress:
        progress("lendo a lista de ações das bolsas…")
    stocks, origin = universe(req.mercado, req.quantidade, cache_dir, fetch)
    if progress:
        progress(f"buscando as cotações de {len(stocks) + len(INDICES)} ativos…")
    got, failures = quotes([s for s, _, _ in INDICES] + [t for t, _ in stocks])
    names = dict(stocks)
    rows = [_row(got[t], names[t]) for t, _ in stocks if t in got]
    if not rows:
        first = next(iter(failures.values()), "nenhuma cotação recebida")
        raise DataUnavailableError(f"cotações indisponíveis agora ({first})")
    moved = [r for r in rows if r["variacao_pct"] is not None]
    up = sorted((r for r in moved if r["variacao_pct"] > FLAT), key=lambda r: -float(r["variacao_pct"]))  # type: ignore[arg-type]
    down = sorted((r for r in moved if r["variacao_pct"] < -FLAT), key=lambda r: float(r["variacao_pct"]))  # type: ignore[arg-type]
    traded = sorted((r for r in rows if r["volume_financeiro"]), key=lambda r: -float(r["volume_financeiro"]))  # type: ignore[arg-type]
    indices = []
    for sym, label, unit in INDICES:
        if sym in got:
            indices.append({**_row(got[sym], label), "unidade": unit, "intradia": got[sym].intraday})
        else:
            indices.append({"ticker": sym, "nome": label, "unidade": unit, "erro": failures.get(sym, "sem resposta")})
    avg = sum(float(r["variacao_pct"]) for r in moved) / len(moved) if moved else None  # type: ignore[arg-type]
    delays = sorted({q.delay_min for q in got.values() if q.delay_min})
    open_now = [q.market_open for t, q in got.items() if t in names and q.market_open is not None]
    last = max((str(r["data"]) + " " + str(r["hora"]) for r in rows), default="")
    return {
        "mercado": req.mercado, "universo": origin, "acompanhadas": len(stocks), "com_cotacao": len(rows),
        "sem_cotacao": {t: failures.get(t, "sem resposta") for t, _ in stocks if t not in got},
        "indices": indices, "altas": up[:TOP], "baixas": down[:TOP], "mais_negociadas": traded[:TOP],
        "cotacoes": sorted(rows, key=lambda r: str(r["ticker"])),
        "amplitude": {"sobem": len(up), "caem": len(down), "estaveis": len(moved) - len(up) - len(down),
                      "variacao_media": avg},
        "pregao_aberto": (sum(open_now) > len(open_now) / 2) if open_now else None, "ultimo_negocio": last,
        "atraso_informado_min": delays,
        "consultado_em_utc": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "aviso": ("Cotações do Yahoo Finance com atraso (na B3, em geral 15 minutos): não é tempo real. Cada linha "
                  "mostra a hora do último negócio, e a variação é contra o fechamento do pregão anterior. O site "
                  "mostra os números, não o motivo dos movimentos: para entender uma alta ou queda, veja as notícias "
                  "do dia."),
    }
