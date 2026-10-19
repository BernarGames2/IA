"""Output-language guard: the purpose report must use probabilistic language, never
guarantees ("vai gerar", "certamente"). Every generated text passes through
:func:`guarantee_phrases`; any hit is reported as a FAIL alert by the engine."""

from __future__ import annotations

import re

from core.purpose.themes import _fold

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"\bvai (gerar|render|dar|multiplicar|garantir)\b",
    r"\bvao (gerar|render|dar)\b",
    r"\bcertamente\b",
    r"\bcom certeza\b",
    r"\bgarant(e|ido|ida|idos|idas|imos)\b",
    r"\bretorno certo\b",
    r"\bsem risco\b",
    r"\brisco zero\b",
    r"\bimpossivel perder\b",
)
_RX = [re.compile(p) for p in FORBIDDEN_PATTERNS]


def guarantee_phrases(text: str) -> list[str]:
    """Return the forbidden guarantee-style phrases found in ``text`` (accent-insensitive)."""
    folded = _fold(text)
    hits: list[str] = []
    for rx in _RX:
        for m in rx.finditer(folded):
            hits.append(text[m.start():m.end()])
    return hits
