"""On-demand follow-up of saved theses: re-reads the sources NOW and checks each invalidation condition against the
limit saved when the thesis was built. Nothing runs in the background — the check happens when the investor asks,
and the result says when it was made.

* Conditions on the annual statements (revenue, operating margin, net debt/EBITDA, dividends paid) refer to the NEXT
  fiscal year: they stay "aguardando o próximo balanço" until the source publishes a fiscal year later than the one
  the thesis used.
* The analysts' estimate (third-party) is compared with its limit at any time.
* The price condition uses the lowest close since the thesis was saved, until the end of its horizon.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from core.fundamentals.analysis import dividends_paid, year_ratios
from core.fundamentals.sources import fetch_analysts, fetch_annual
from core.market_data.base import last_per_date
from core.market_data.yahoo_chart import FetchFn
from core.purpose.engine import _default_provider, norm_ticker
from core.stock_check.company import SourceError, _yahoo_client
from utils.br_format import num, pct

STATEMENT_KEYS = ("receita", "margem_operacional", "divida_ebitda", "dividendos")
ERRORS = (SourceError, OSError, ValueError, KeyError, TypeError)


class Condition(BaseModel):
    chave: Literal["receita", "margem_operacional", "divida_ebitda", "lpa_estimado", "preco", "dividendos"]
    valor_limite: float
    direcao: Literal["abaixo", "acima"]
    condicao: str = Field("", max_length=200)
    ano_fiscal: str | None = Field(None, max_length=10)
    ate: date | None = None


class WatchedThesis(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    criada_em: date
    condicoes: list[Condition] = Field(max_length=10)


class MonitorRequest(BaseModel):
    teses: list[WatchedThesis] = Field(min_length=1, max_length=10)

    def key(self) -> str:
        return "monitor|" + "|".join(sorted(t.model_dump_json() for t in self.teses))


def _fmt(key: str, x: float | None) -> str:
    if x is None:
        return "n/d"
    if key == "margem_operacional":
        return pct(x)
    if key == "divida_ebitda":
        return f"{num(x)}×"
    if key in ("receita", "dividendos"):
        return f"{num(x / 1e9, 1)} bi" if abs(x) >= 1e9 else f"{num(x / 1e6, 0)} mi"
    return num(x, 2)


def check_one(item: WatchedThesis, provider: object, fetch: FetchFn | None, cache_dir: Path | None,
              today: date) -> dict[str, object]:
    tk = norm_ticker(item.ticker)
    keys = {c.chave for c in item.condicoes}
    out: dict[str, object] = {"ticker": tk, "criada_em": item.criada_em.isoformat(), "condicoes": [], "avisos": []}
    client = _yahoo_client(fetch)
    a = an = None
    if keys & set(STATEMENT_KEYS):
        try:
            a = fetch_annual(client, tk, cache_dir)
        except ERRORS as e:
            out["avisos"].append(f"demonstrações indisponíveis agora ({e})")  # type: ignore[union-attr]
    if "lpa_estimado" in keys:
        try:
            an = fetch_analysts(client, tk, cache_dir)
        except ERRORS as e:
            out["avisos"].append(f"estimativas indisponíveis agora ({e})")  # type: ignore[union-attr]
    got = provider.fetch([tk], "1y", "1d")  # type: ignore[attr-defined]
    px = None
    if tk in got and got[tk].close is not None:
        s = last_per_date(got[tk].close.dropna()).sort_index()
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        px = s[s > 0]
    if px is not None and len(px):
        since = px.loc[pd.Timestamp(item.criada_em):]
        out["preco_atual"] = float(px.iloc[-1])
        out["data_preco"] = f"{px.index[-1]:%Y-%m-%d}"
        if len(since) > 1:
            out["variacao_desde_criacao"] = float(px.iloc[-1] / since.iloc[0] - 1.0)
    else:
        out["avisos"].append("cotações indisponíveis agora")  # type: ignore[union-attr]
    for c in item.condicoes:
        status, value, note = "sem dados", None, ""
        if c.chave in STATEMENT_KEYS:
            if a is not None and a.anos:
                last = a.anos[-1]
                if c.ano_fiscal and last[:10] <= c.ano_fiscal[:10]:
                    status, note = "aguardando o próximo balanço", f"último ano fiscal publicado: {last[:4]}"
                else:
                    if c.chave == "receita":
                        value = a.get("receita", -1)
                    elif c.chave == "margem_operacional":
                        value = year_ratios(a, -1)["margem_operacional"]
                    elif c.chave == "divida_ebitda":
                        value = year_ratios(a, -1)["divida_liquida_ebitda"]
                    else:
                        value = dividends_paid(a, -1)
                    note = f"ano fiscal de {last[:4]}"
        elif c.chave == "lpa_estimado":
            value = an.eps_ano.get("current") if an is not None else None
        elif c.chave == "preco" and px is not None and len(px):
            end = pd.Timestamp(c.ate) if c.ate else px.index[-1]
            window = px.loc[pd.Timestamp(item.criada_em):end]
            if len(window):
                value = float(window.min())
                note = f"menor fechamento desde {item.criada_em:%d/%m/%Y}"
                if c.ate and today > c.ate:
                    note += " (prazo da tese encerrado)"
            else:
                status, note = "aguardando novas cotações", f"nenhum fechamento depois de {item.criada_em:%d/%m/%Y} ainda"
        if value is not None:
            hit = value < c.valor_limite if c.direcao == "abaixo" else value > c.valor_limite
            status = "acionada" if hit else "dentro do limite"
        out["condicoes"].append({"chave": c.chave, "condicao": c.condicao, "estado": status, "valor_atual": value,  # type: ignore[union-attr]
                                 "valor_limite": c.valor_limite, "atual": _fmt(c.chave, value),
                                 "limite": _fmt(c.chave, c.valor_limite), "nota": note})
    states = [x["estado"] for x in out["condicoes"]]  # type: ignore[union-attr]
    out["acionadas"] = states.count("acionada")
    out["leitura"] = ("Alguma condição de invalidação foi acionada: a tese, como foi escrita, deixou de valer."
                      if out["acionadas"] else "Nenhuma condição acionada até agora (as do balanço anual só podem ser "
                                               "conferidas quando um novo ano fiscal for publicado).")
    return out


def check(req: MonitorRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
          cache_dir: Path | None = None, progress: Callable[[str], None] | None = None,
          budget_s: float | None = None, today: date | None = None) -> dict[str, object]:
    prov = provider if provider is not None else _default_provider(cache_dir or Path(".cache") / "market_data", progress, budget_s)
    root = cache_dir or Path(".cache") / "market_data"
    day = today or datetime.now(UTC).date()
    items = []
    for i, t in enumerate(req.teses, 1):
        if progress:
            progress(f"conferindo {t.ticker} ({i} de {len(req.teses)})…")
        try:
            items.append(check_one(t, prov, fetch, root, day))
        except ERRORS as e:
            items.append({"ticker": t.ticker, "erro": str(e) or type(e).__name__, "condicoes": [], "acionadas": 0})
    return {"conferido_em": datetime.now(UTC).isoformat(timespec="seconds"), "teses": items,
            "acionadas": sum(int(x.get("acionadas", 0)) for x in items),
            "aviso": ("Conferência feita agora, a seu pedido: nada é acompanhado em segundo plano. Confira de novo quando "
                      "quiser, por exemplo depois de um novo balanço.")}
