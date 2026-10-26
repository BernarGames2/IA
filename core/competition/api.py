"""Web entry points of the competition page (routed by ``webapp/server.py``)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.competition.build import CompetitionRequest, build
from core.competition.reader import parse_tickers, read_case, read_rules
from core.market_data.yahoo_chart import FetchFn


class ReadRequest(BaseModel):
    """Texts pasted by the team: the client case, the rulebook and the approved-ETF list (any may be empty)."""

    model_config = ConfigDict(extra="forbid")

    desafio: str = Field("", max_length=30_000)
    regras: str = Field("", max_length=30_000)
    etfs: str = Field("", max_length=20_000)


def handle_read(req: ReadRequest, today: date | None = None) -> dict[str, object]:
    if not (req.desafio.strip() or req.regras.strip() or req.etfs.strip()):
        raise ValueError("cole o desafio do cliente, as regras ou a lista de ETFs para o site ler")
    return {"desafio": read_case(req.desafio, today).as_dict() if req.desafio.strip() else None,
            "regras": read_rules(req.regras, today).as_dict() if req.regras.strip() else None,
            "etfs": parse_tickers(req.etfs) if req.etfs.strip() else [],
            "aviso": ("Leitura por regras fixas (sem inteligência artificial): cada item mostra a frase de onde saiu. "
                      "Confira e corrija o formulário antes de montar a carteira — o que não foi encontrado fica em "
                      "branco, nada é adivinhado.")}


def handle_build(req: CompetitionRequest, *, provider: object | None = None, fetch: FetchFn | None = None,
                 cache_dir: Path | None = None, output_dir: Path | None = None,
                 progress: Callable[[str], None] | None = None, budget_s: float | None = None,
                 today: date | None = None) -> dict[str, object]:
    return build(req, provider=provider, rf_fetch=fetch, cache_dir=cache_dir, output_dir=output_dir,
                 progress=progress, budget_s=budget_s, today=today)
