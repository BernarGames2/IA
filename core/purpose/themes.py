"""Step 6 — extract themes from the investor's free-text motive, WITHOUT inferring beyond it.

Deterministic lexicon matching (Portuguese, accent-insensitive):
* every extracted theme carries the exact phrases of the text that triggered it
  (``evidence`` are verbatim substrings, checked by tests);
* negated mentions ("não me importo com sustentabilidade", "nem saúde") are discarded
  and reported, not counted;
* the final "why" of the capital is the verbatim clause after "para/quero/sonho/objetivo";
* confidence is LOW for short or vague texts, so the alignment score is flagged.

No language model is used: nothing that was not written by the investor can appear.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# theme -> (kind, keywords). kind: "valor" (core value), "setor" (sector/cause) or "ambos".
LEXICON: dict[str, tuple[str, tuple[str, ...]]] = {
    "saude": ("ambos", ("saude", "saudavel", "hospital", "medic", "doenca", "tratamento", "bem-estar", "bem estar",
                        "cuidar da minha saude", "plano de saude", "remedio", "cancer", "oncolog")),
    "educacao": ("ambos", ("educacao", "estudo", "estudar", "faculdade", "universidade", "escola", "formacao",
                           "intercambio", "curso", "ensino")),
    "sustentabilidade": ("ambos", ("sustentab", "meio ambiente", "ambiental", "planeta", "clima", "climatic",
                                   "energia limpa", "renovav", "verde", "carbono", "natureza", "ecolog")),
    "tecnologia": ("setor", ("tecnologia", "inovacao", "inteligencia artificial", "startup", "digital")),
    "moradia": ("setor", ("casa propria", "imovel", "apartamento", "moradia", "minha casa")),
    "renda": ("valor", ("viver de renda", "renda passiva", "renda mensal", "dividendo", "aluguel")),
    "seguranca": ("valor", ("seguranca", "tranquilidade", "reserva", "emergencia", "estabilidade", "protecao",
                            "proteger", "nao perder", "sem sustos")),
    "aposentadoria": ("valor", ("aposentadoria", "aposentar", "independencia financeira", "liberdade financeira")),
    "familia_legado": ("valor", ("legado", "heranca", "filhos", "netos", "familia", "proxima geracao",
                                 "deixar para")),
    "comunidade": ("valor", ("comunidade", "social", "ajudar", "doacao", "doar", "filantrop", "igreja", "ong",
                             "voluntari")),
    "disciplina": ("valor", ("disciplina", "constancia", "regularidade", "habito", "todo mes", "consistencia")),
    "superacao": ("valor", ("superacao", "superar", "recomeco", "recomecar", "sair das dividas", "conquistar",
                            "vencer")),
}

NEGATIONS = ("nao", "nem", "sem", "nunca", "jamais")

WHY_PATTERN = re.compile(
    r"\b(objetivo|meta|sonho|para|pra|porque|quero|desejo)\b\s+(.{3,160}?)(?=[.;!?\n]|$)", re.IGNORECASE)
# a goal-type trigger states the final "why" more directly than a wish or a cause
WHY_PRIORITY = {"objetivo": 0, "meta": 0, "sonho": 0, "para": 1, "pra": 1, "porque": 2, "quero": 3, "desejo": 3}


def _fold(s: str) -> str:
    """Lower-case, accent-free copy with the SAME length (so indices map back to the text)."""
    out = []
    for ch in s:
        d = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        out.append((base or ch)[0].lower() if base else ch.lower())
    return "".join(out)


@dataclass
class Theme:
    name: str
    kind: str
    evidence: list[str]
    mentions: int


@dataclass
class ThemeExtraction:
    text: str
    themes: list[Theme] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)
    why: str | None = None            # verbatim clause stating the final "why" (goal-type trigger preferred)
    why_clauses: list[str] = field(default_factory=list)   # every verbatim "why"-like clause, in text order
    confidence: str = "nenhuma"      # alta | media | baixa | nenhuma
    notes: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def names(self) -> list[str]:
        return [t.name for t in self.themes]


CONTRAST = ("mas", "porem", "contudo", "todavia", "entretanto")


def _is_negated(folded: str, start: int) -> bool:
    """A mention is negated when a negation word occurs up to 5 words before it, within the
    same clause (punctuation and contrastive conjunctions such as "mas" end the scope):
    "não me importo com sustentabilidade" is negated; "não tenho filhos, mas quero educação"
    negates only "filhos"."""
    window = re.split(r"[,.;:!?\n]", folded[max(0, start - 60):start])[-1]
    words = re.findall(r"[a-z]+", window)[-5:]
    for i in range(len(words) - 1, -1, -1):
        if words[i] in CONTRAST:
            words = words[i + 1:]
            break
    return any(w in NEGATIONS for w in words)


def extract_themes(text: str | None) -> ThemeExtraction:
    raw = (text or "").strip()
    out = ThemeExtraction(text=raw)
    if not raw:
        out.notes.append("motivo_investidor vazio: score de alinhamento não calculado por falta de input")
        return out
    folded = _fold(raw)
    for theme, (kind, keywords) in LEXICON.items():
        ev: list[str] = []
        neg = False
        for kw in keywords:
            for m in re.finditer(r"\b" + re.escape(kw), folded):
                # extend to the end of the word so the evidence is a readable verbatim span
                end = m.end()
                while end < len(folded) and folded[end].isalpha():
                    end += 1
                span = raw[m.start():end]
                if _is_negated(folded, m.start()):
                    neg = True
                    continue
                if span not in ev:
                    ev.append(span)
        if ev:
            out.themes.append(Theme(theme, kind, ev, len(ev)))
        elif neg:
            out.negated.append(theme)
    found = []
    for m in WHY_PATTERN.finditer(raw):
        clause = raw[m.start():m.end()].strip()
        if clause not in out.why_clauses:
            out.why_clauses.append(clause)
            found.append((WHY_PRIORITY[m.group(1).lower()], m.start(), clause))
    if found:
        out.why = min(found)[2]
    n_words = len(re.findall(r"\w+", raw))
    n = len(out.themes)
    if n >= 2 and n_words >= 12:
        out.confidence = "alta"
    elif n >= 1 and n_words >= 6:
        out.confidence = "media"
    elif n >= 1:
        out.confidence = "baixa"
        out.notes.append("texto curto: temas extraídos com baixa confiança")
    else:
        out.confidence = "baixa"
        out.notes.append("nenhum tema reconhecido no texto: score de alinhamento não calculado (nada é inferido)")
    if out.negated:
        out.notes.append(f"menções negadas ignoradas: {', '.join(out.negated)}")
    return out
