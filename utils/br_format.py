"""Numbers written the Brazilian way in generated texts: decimal comma, dot for thousands."""

from __future__ import annotations


def pct(x: float | None, d: int = 1, sign: bool = False) -> str:
    if x is None:
        return "n/d"
    return f"{x * 100:{'+' if sign else ''}.{d}f}%".replace(".", ",")


def num(x: float | None, d: int = 1) -> str:
    if x is None:
        return "n/d"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def millions(x: float | None) -> str:
    """Amounts in millions (bi from a thousand millions), as reported."""
    if x is None:
        return "n/d"
    return f"{num(x / 1e9, 1)} bi" if abs(x) >= 1e9 else f"{num(x / 1e6, 0)} mi"
