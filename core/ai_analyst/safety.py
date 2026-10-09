"""Safety layer for the AI Research Analyst.

* External content (web pages, documents, API payloads) is wrapped as
  :class:`UntrustedContent` -- DATA, never instructions. Instruction-like
  patterns (prompt injection) are detected and reported; nothing in external
  content can add tools, change permissions or trigger actions.
* Secrets (API keys, tokens, environment values) are redacted from anything the
  analyst outputs.
* There is no order-execution capability; requests for it are refused.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore (all |any )?(previous|prior|above) (instructions|rules)", "override_instructions"),
    (r"ignore (as |todas as )?(instruções|regras) (anteriores|acima)", "override_instructions"),
    (r"desconsidere (as |todas as )?(instruções|regras)", "override_instructions"),
    (r"(system prompt|prompt do sistema|you are now|você agora é)", "role_hijack"),
    (r"(execute|run|rode|execute o comando|executar)\s+(the\s+)?(command|comando|shell|script|código|code)", "command_execution"),
    (r"(rm\s+-rf|curl\s+http|wget\s+http|powershell|bash\s+-c|subprocess|os\.system)", "command_execution"),
    (r"(api[_ -]?key|secret|token|password|senha|credencia)", "secret_exfiltration"),
    (r"(env(ironment)? var|variáveis de ambiente|printenv|os\.environ)", "secret_exfiltration"),
    (r"(compre|venda|buy|sell|place (an )?order|envie (a |uma )?ordem|execute (a |uma )?ordem)", "order_execution"),
    (r"(new tool|nova ferramenta|grant (access|permission)|conceda (acesso|permiss))", "permission_escalation"),
)

ORDER_REQUEST = re.compile(r"\b(compr[ae]r?|vend[ae]r?|ordem|order|executar? (a |uma )?(compra|venda)|buy|sell)\b",
                           re.IGNORECASE)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
)


@dataclass
class UntrustedContent:
    """External content kept strictly as data, with provenance."""

    text: str
    source: str
    retrieved_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    injection_findings: list[str] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return bool(self.injection_findings)

    def as_quoted_data(self, max_chars: int = 2000) -> str:
        """Render for display: clearly delimited, truncated, never interpreted."""
        body = self.text[:max_chars].replace("```", "'''")
        return (f"[CONTEÚDO EXTERNO NÃO CONFIÁVEL — fonte: {self.source}; obtido em {self.retrieved_at_utc}]\n"
                f"```text\n{body}\n```")


def scan_injection(text: str) -> list[str]:
    found = []
    low = text.lower()
    for pat, label in INJECTION_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            found.append(label)
    return sorted(set(found))


def wrap_external(text: str, source: str) -> UntrustedContent:
    return UntrustedContent(text=text, source=source, injection_findings=scan_injection(text))


def redact_secrets(text: str) -> str:
    out = text
    for p in _SECRET_PATTERNS:
        out = p.sub("[REDACTED]", out)
    for k, v in os.environ.items():
        if v and len(v) >= 12 and any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            out = out.replace(v, "[REDACTED]")
    return out


def is_order_request(question: str) -> bool:
    return bool(ORDER_REQUEST.search(question))
