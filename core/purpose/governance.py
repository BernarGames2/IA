"""Step 5 — governance filter. Flags come from a configuration file and/or the request
(``restricoes_governanca``) and MUST cite a source.

``sob_revisao`` assets are removed by default and kept only when the user explicitly
unlocks "incluir ativos sob revisão"; ``excluido`` assets are always removed. Entries
without ``source`` are rejected (no unsourced allegations about real companies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FLAGS_PATH = Path(__file__).resolve().parents[2] / "data" / "governance_flags.json"
VALID_STATUS = {"sob_revisao", "excluido"}


@dataclass
class GovernanceResult:
    kept: list[str]
    removed: dict[str, str] = field(default_factory=dict)          # ticker -> reason
    included_under_review: dict[str, str] = field(default_factory=dict)
    rejected_entries: dict[str, str] = field(default_factory=dict)  # malformed flags (ignored)


def _validate(raw: dict[str, object]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    ok, rejected = {}, {}
    for t, f in raw.items():
        if not isinstance(f, dict) or f.get("status") not in VALID_STATUS:
            rejected[t] = "status inválido"
        elif not str(f.get("source", "")).strip():
            rejected[t] = "sem fonte citada: ignorado"
        else:
            ok[t] = {k: str(v) for k, v in f.items()}
    return ok, rejected


def load_flags(path: Path | None = None) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    p = Path(path) if path else DEFAULT_FLAGS_PATH
    if not p.exists():
        return {}, {}
    return _validate(json.loads(p.read_text(encoding="utf-8")).get("flags", {}))


def apply_governance(tickers: list[str], include_under_review: bool = False,
                     path: Path | None = None, extra: dict[str, object] | None = None) -> GovernanceResult:
    """``extra``: flags supplied with the request (restricoes_governanca); same rules as the file,
    and they take precedence over it for the same ticker."""
    flags, rejected = load_flags(path)
    if extra:
        ok_x, rej_x = _validate(extra)
        flags.update(ok_x)
        rejected.update({t: f"{why} (entrada da requisição)" for t, why in rej_x.items()})
    res = GovernanceResult(kept=[], rejected_entries=rejected)
    for t in tickers:
        f = flags.get(t)
        if f is None:
            res.kept.append(t)
        elif f["status"] == "excluido":
            res.removed[t] = f"excluído por governança: {f.get('reason', '')} (fonte: {f['source']})"
        elif include_under_review:
            res.kept.append(t)
            res.included_under_review[t] = f"sob revisão, incluído por opção explícita: {f.get('reason', '')} " \
                                           f"(fonte: {f['source']})"
        else:
            res.removed[t] = f"sob revisão (excluído por padrão): {f.get('reason', '')} (fonte: {f['source']})"
    return res
