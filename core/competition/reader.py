"""Reading the client case and the competition rulebook from pasted text — English or Portuguese — WITHOUT a
language model.

Deterministic patterns (below) find each item, and every value carries the exact sentence it came from (``trecho``,
a verbatim slice of the pasted text, checked by tests) so the team can confirm it. Nothing is inferred beyond the
text: what is not found is listed as missing, never guessed, and conflicting signals are shown with the rule used to
pick one (risk: the most conservative signal, the usual suitability practice of taking the lower of willingness and
ability to take risk; horizon: the longest goal, the shorter ones are listed as intermediate goals). The page fills
its form with what was found and the form stays editable — the reading is an aid, the team decides.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

from core.purpose.themes import _fold
from utils.br_format import pct

MAX_TEXT = 30_000

NUM_WORDS = {
    "one": 1, "an": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-five": 25, "thirty": 30, "forty": 40,
    "fifty": 50, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7,
    "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14,
    "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19, "vinte": 20, "trinta": 30,
    "quarenta": 40, "cinquenta": 50}
NUMW = "(\\d{1,3}(?:[.,]\\d+)?|" + "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True)) + ")"
PCT = r"(\d{1,3}(?:[.,]\d+)?)\s?(?:%|percent\b|por cento\b)"

MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
          "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
          "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12, "janeiro": 1,
          "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
          "outubro": 10, "novembro": 11, "dezembro": 12}
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_RX = (re.compile(rf"\b(?P<mon>{_MON})\.? (?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,? (?P<year>20\d\d))?"),
           re.compile(rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)? (?:of )?(?P<mon>{_MON})\b\.?(?:,? (?P<year>20\d\d))?"),
           re.compile(rf"\b(?P<day>\d{{1,2}}) de (?P<mon>{_MON})\b(?: de (?P<year>20\d\d))?"),
           re.compile(r"\b(?P<year>20\d\d)-(?P<mnum>\d{2})-(?P<day>\d{2})\b"))

_CUR = {"r$": "BRL", "brl": "BRL", "reais": "BRL", "us$": "USD", "u$s": "USD", "usd": "USD", "$": "USD",
        "dollars": "USD", "dolares": "USD", "eur": "EUR", "€": "EUR"}
_MULT = {"k": 1e3, "thousand": 1e3, "mil": 1e3, "m": 1e6, "mi": 1e6, "million": 1e6, "millions": 1e6, "milhao": 1e6,
         "milhoes": 1e6, "b": 1e9, "bn": 1e9, "bi": 1e9, "billion": 1e9, "billions": 1e9, "bilhao": 1e9,
         "bilhoes": 1e9}
_N = r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_MW = r"thousand|millions?|billions?|milhoes|milhao|bilhoes|bilhao|mil|mi|bi|bn|k|m|b"
MONEY_RX = re.compile(rf"(?P<cur>r\$|us\$|u\$s|usd|brl|eur|€|\$)\s?(?P<num>{_N})(?:\s?(?P<mult>{_MW})\b)?"
                      rf"|(?P<num2>{_N})\s?(?P<mult2>thousand|millions?|billions?|milhoes|milhao|bilhoes|bilhao|mil)?"
                      r"\s(?P<cur2>dollars|dolares|reais|usd|brl)\b")
PERIOD_RX = re.compile(r"^\W{0,3}(?:(?:per|a|an|each|every|/)\s?(?P<en>month|year|annum|yr|mo)\b|(?P<en2>monthly|annually|"
                       r"yearly)\b|(?:por|ao|a cada|todo|cada)\s(?P<pt>mes|ano)\b|(?P<pt2>mensais|mensal|anuais|anual)\b)")

_ABBREV = {"mr", "mrs", "ms", "dr", "sr", "sra", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "inc", "co",
           "no", "approx", "ltd", "corp"}
_EN_WORDS = {"the", "and", "is", "are", "to", "of", "with", "she", "he", "they", "their", "his", "her", "wants",
             "has", "would", "will", "for", "who"}
_PT_WORDS = {"de", "que", "os", "as", "do", "da", "dos", "das", "em", "para", "com", "nao", "uma", "um", "seu", "sua",
             "quer", "tem", "no", "na", "ele", "ela", "eles"}
NEG = {"en": re.compile(r"\b(?:not|no|never|nor|without|isn'?t|doesn'?t|don'?t|won'?t|cannot|can'?t|neither)\b"),
       "pt": re.compile(r"\b(?:nao|nem|nunca|jamais|sem)\b")}

# risk tolerance. Each rule: (pattern, level, whether a negation just before it cancels it, kind). "rotulo" = the
# client's own label (conservative, moderate, aggressive); "comportamento" = a described behaviour ("not comfortable
# with large swings"). A label, when there is one, decides (the most conservative if the text gives several labels);
# behaviours decide only when no label is written. Explicit negative phrases come first, so the words inside them
# ("comfortable with risk" in "not comfortable with risk") are not counted again.
RISK_RULES: tuple[tuple[str, str, bool, str], ...] = (
    ((r"\bnot (?:very |too |that |at all |really )?(?:comfortable|willing|keen|happy) (?:with|to take|taking|on|about) "
      r"(?:much |a lot of |high |significant |big |large |huge |wide |any |sharp |)(?:risks?|volatility|losses|swings|"
      r"declines|drops|ups and downs)"), "conservador", False, "comportamento"),
    ((r"\b(?:doesn'?t|does not|don'?t|do not|cannot|can'?t|could not|couldn'?t) (?:want to |afford to |like to |)"
      r"(?:take|lose|tolerate|handle|stomach|risk|accept|bear) (?:much |a lot of |big |significant |large |any |high |)"
      r"(?:risks?|volatility|losses|money|swings|declines)"), "conservador", False, "comportamento"),
    ((r"\b(?:risk[- ]averse|averse to risk|conservative(?:ly)?|cautious|low (?:risk )?tolerance|low tolerance for "
      r"risk)\b"), "conservador", True, "rotulo"),
    (r"\b(?:low[- ]risk|minimal risk|limited risk)\b", "conservador", True, "comportamento"),
    ((r"\b(?:moderate(?:ly)? (?:risk|risk tolerance|tolerance|investors?|approach)|moderate|medium[- ]risk|balanced "
      r"(?:approach|portfolio|investors?|risk))\b"), "moderado", True, "rotulo"),
    (r"\b(?:some risk|middle[- ]of[- ]the[- ]road)\b", "moderado", True, "comportamento"),
    ((r"\b(?:aggressive(?:ly)?|high (?:risk )?tolerance|high tolerance for risk|risk[- ]takers?|growth[- ]oriented)\b"),
     "arrojado", True, "rotulo"),
    ((r"\b(?:high[- ]risk|comfortable with (?:risk|volatility|market swings)|willing to (?:take|accept) (?:on )?(?:more |"
      r"higher |significant |substantial |considerable |)risks?)\b"), "arrojado", True, "comportamento"),
    ((r"\bnao (?:aceita|tolera|quer|pode|gosta de|suporta) (?:correr )?(?:muitos? |grandes? |altos? |nenhum )?"
      r"(?:riscos?|perdas|perder|oscilac\w*|quedas)"), "conservador", False, "comportamento"),
    (r"\b(?:conservador[ae]?s?|avess[oa]s? (?:a|ao) riscos?|cautelos[oa]s?)\b", "conservador", True, "rotulo"),
    (r"\b(?:baixo risco|pouco risco|risco baixo)\b", "conservador", True, "comportamento"),
    (r"\b(?:moderad[oa]s?|risco moderado)\b", "moderado", True, "rotulo"),
    (r"\b(?:equilibrad[oa]s?|risco medio|algum risco)\b", "moderado", True, "comportamento"),
    (r"\b(?:arrojad[oa]s?|agressiv[oa]s?)\b", "arrojado", True, "rotulo"),
    ((r"\b(?:alto risco|risco alto|aceita (?:correr )?(?:mais )?riscos?|tolerancia (?:alta|elevada) (?:a|ao) risco)\b"),
     "arrojado", True, "comportamento"),
)
RISK_ORDER = {"conservador": 0, "moderado": 1, "arrojado": 2}

OBJECTIVE_RULES: tuple[tuple[str, str], ...] = (
    ((r"\b(?:(?:generate|provide|produce|need|needs|require|requires|want|wants|supplement(?:al)?|steady|regular|"
     r"stable|reliable|passive|retirement|additional|extra|monthly|annual)\s(?:an?\s)?(?:steady\s|regular\s|stable\s|"
     r"reliable\s|passive\s|additional\s|extra\s|monthly\s|annual\s)?income|income[- ](?:generating|producing|"
     r"focused|stream|needs?)|dividends?|cash flow|living expenses|withdraw(?:al|als|s)?)\b"), "renda"),
    ((r"\b(?:preserv(?:e|ing|ation)|protect(?:ing)? (?:the |their |his |her |its )?(?:capital|savings|principal|"
     r"money|wealth|nest egg)|capital preservation|safety of (?:the )?principal|not lose (?:money|capital|"
     r"principal))\b"), "preservacao"),
    ((r"\b(?:grow(?:th|ing)?|capital appreciation|appreciat(?:e|ion)|maximi[sz]e (?:returns?|growth)|build (?:up )?"
     r"(?:wealth|savings)|accumulat\w*|increase (?:the |their |his |her )?(?:wealth|portfolio|savings))\b"),
     "crescimento"),
    ((r"\b(?:renda (?:mensal|passiva|extra|complementar|regular|periodica)|complementar a renda|dividendos?|viver de "
     r"renda|fluxo de caixa|sacar|saques|retiradas?)\b"), "renda"),
    ((r"\b(?:preservar|preservacao|proteger (?:o |seu |seus |a |sua )?(?:capital|patrimonio|dinheiro|poupanca)|nao "
     r"perder)\b"), "preservacao"),
    ((r"\b(?:crescer|crescimento|valorizacao|multiplicar|acumular|juntar|aumentar (?:o |seu |a |sua )?"
      r"(?:patrimonio|riqueza))\b"), "crescimento"),
)

EXCLUSION_TERMS: dict[str, str] = {
    "combustiveis_fosseis": (r"fossil[- ]fuels?|oil(?: and gas| & gas| companies| stocks| industry)?|coal|petroleum|"
                             r"combustiveis fosseis|petroleo|carvao|oleo e gas|petroleiras"),
    "tabaco": r"tobacco|cigarettes?|smoking|vaping|tabaco|cigarros?|fumo",
    "armas": (r"weapons?|firearms?|guns?|defen[cs]e (?:contractors?|companies|industry|stocks|firms)|military|"
              r"arms (?:manufacturers?|makers?|industry|dealers)|armas?|armamentos?|industria (?:belica|de defesa)"),
    "alcool": r"alcohol(?:ic)?(?: beverages| drinks)?|beer|liquor|spirits|wine|alcool|bebidas alcoolicas|cervejas?",
    "jogos": r"gambling|casinos?|(?:sports )?betting|jogos de azar|apostas|cassinos?",
}
EXCLUSION_PT = {"combustiveis_fosseis": "combustíveis fósseis (petróleo, gás, carvão)", "tabaco": "tabaco",
                "armas": "armas e defesa", "alcool": "bebidas alcoólicas", "jogos": "jogos de azar e apostas"}
AVOID = {"en": (r"\b(?:avoid\w*|exclud\w*|(?:does not|doesn'?t|do not|don'?t|would not|wouldn'?t|will not|won'?t|"
                r"never|refuses? to)\s(?:want|wish|like|support|invest|own|hold|put)\w*|not\s(?:want|wish|interested|"
                r"comfortable|invest)\w*|refus\w*|against|oppos\w*|free (?:of|from)|screen(?:ed|ing)? out|divest\w*|"
                r"stay away from|steer clear of|keep (?:away )?(?:out|clear) of)"),
         "pt": (r"\b(?:nao\s(?:quer|deseja|aceita|gosta|investe|investir|apoia|admite|investiria)\w*|evit\w*|exclu\w*|"
                r"contra|recusa\w*|nada de|longe de)")}
NEAR_AVOID = {"en": r"\b(?:no|without)\s", "pt": r"\b(?:sem|nada de)\s"}

THEMES: dict[str, str] = {
    "sustentabilidade": (r"\besg\b|sustainab\w*|environment\w*|climate|clean energy|renewable\w*|"
                         r"green energy|carbon|the planet|sustentab\w*|meio ambiente|ambienta\w*|energia limpa|"
                         r"renovav\w*|clima\w*"),
    "saude": r"health ?care|healthcare|medical (?:research|innovation|technology)|biotech\w*|saude|medicina",
    "tecnologia": (r"\btech(?:nology|nologies)?\b|innovation|artificial intelligence|software|semiconductors?|"
                   r"tecnologia|inovacao|inteligencia artificial"),
    "educacao": (r"(?:passionate about|cares? (?:deeply )?about|values?|interested in|supports?|supporting) "
                 r"(?:education|schools?|learning)|educacao"),
    "comunidade": (r"\bcommunity\b|charit\w*|philanthrop\w*|give back|giving back|social impact|comunidade|"
                   r"filantrop\w*|caridade|impacto social"),
    "familia_legado": (r"\blegacy\b|grandchildren|leave (?:money|an inheritance|something) (?:to|for)|heranca|legado|"
                       r"netos"),
}
THEME_PT = {"sustentabilidade": "sustentabilidade", "saude": "saúde", "tecnologia": "tecnologia",
            "educacao": "educação", "comunidade": "comunidade", "familia_legado": "família e legado"}

_GOAL_CUE = re.compile(
    r"\b(?:goal|target|aims?|hopes? to|wants? to (?:have|reach|accumulate|save|retire with|grow)|needs? to (?:have|"
    r"reach|accumulate|save|grow)|accumulat\w*|reach|save up|nest egg|retire with|grow (?:it |this |the portfolio |"
    r"their savings |his savings |her savings )?to|meta|objetivo|juntar|acumular|chegar a|atingir|alcancar|"
    r"(?:quer|precisa|deseja|gostaria de) ter)\b")
_WEALTH_CUE = re.compile(
    r"\b(?:has|have|had|holds?|owns?|savings|saved|inherit\w*|inheritance|portfolio (?:of|worth|valued)|investable|"
    r"lump sum|net worth|assets?|currently|sold|proceeds|bonus|windfall|tem|possui|economias|guardad\w*|heranca|"
    r"herdou|patrimonio|recebeu|ganhou|aplicar|investir|poupanca)\b")
_SALARY_CUE = re.compile(r"\b(?:salary|salaries|earns?|earning|makes|paid|wage|wages|income of|salario|ganha|recebe "
                         r"por mes)\b")
_OUTFLOW_CUE = re.compile(r"\b(?:withdraw\w*|spend\w*|expenses|draw|living|need|needs|income|sacar|retirar|gastar|"
                          r"despesas|precisa|renda|custo de vida)\b")
_INFLOW_CUE = re.compile(r"\b(?:contribut\w*|add\w*|invest\w*|save|saves|saving|deposit\w*|put\w*|set aside|aport\w*|"
                         r"guard\w*|poup\w*|deposit\w*)\b")
_EXPENSE_CUE = re.compile(r"\b(?:tuition|college|university|down payment|house|home|wedding|car|renovation|"
                          r"faculdade|universidade|estudos|entrada|casa|apartamento|casamento|carro|reforma)\b")
_LIQUIDITY_CUE = re.compile(r"\b(?:emergenc\w*|next year|within (?:a|one|the next) year|in (?:the next )?(?:\d{1,2}|"
                            r"six|three|twelve|nine) months|short[- ]term|proximo ano|proximos 12 meses|em (?:\d{1,2}|"
                            r"seis|tres|doze|nove) meses|curto prazo|reserva)\b")


@dataclass
class Finding:
    campo: str          # form field it fills
    rotulo: str         # what it is, in Portuguese
    valor: object
    trecho: str         # verbatim sentence of the pasted text
    nota: str = ""


@dataclass
class Reading:
    idioma: str = "en"
    achados: list[Finding] = field(default_factory=list)
    faltando: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    def get(self, campo: str) -> Finding | None:
        return next((f for f in self.achados if f.campo == campo), None)

    def value(self, campo: str) -> object:
        f = self.get(campo)
        return None if f is None else f.valor

    def as_dict(self) -> dict[str, object]:
        return {"idioma": self.idioma, "achados": [asdict(f) for f in self.achados], "faltando": self.faltando,
                "avisos": self.avisos, "sugestao": {f.campo: f.valor for f in self.achados if f.valor is not None}}


class _Text:
    """The pasted text and a lower-case, accent-free copy of the SAME length (so a match maps back to the text)."""

    def __init__(self, raw: str) -> None:
        self.raw = (raw or "")[:MAX_TEXT]
        folded = _fold(self.raw)
        self.f = folded.translate(str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-",
                                                 "“": '"', "”": '"', " ": " "}))
        words = re.findall(r"[a-z']+", self.f)
        en = sum(w in _EN_WORDS for w in words)
        pt = sum(w in _PT_WORDS for w in words)
        self.lang = "pt" if pt > en else "en"
        self.spans = self._sentences()

    def _sentences(self) -> list[tuple[int, int]]:
        out, start = [], 0
        for m in re.finditer(r"[.!?;]+(?=\s|$)|\n+", self.f):
            if m.group().startswith("."):
                w = re.search(r"([a-z.]+)$", self.f[start:m.start()])
                if w and w.group(1).strip(".") in _ABBREV:
                    continue
            if self.f[start:m.end()].strip():
                out.append((start, m.end()))
            start = m.end()
        if self.f[start:].strip():
            out.append((start, len(self.f)))
        return out

    def sentence(self, pos: int) -> tuple[int, int]:
        for a, b in self.spans:
            if a <= pos < b:
                return a, b
        return 0, len(self.f)

    def quote(self, pos: int, end: int | None = None) -> str:
        """The sentence around ``pos``, verbatim (a window of it when the sentence is very long)."""
        a, b = self.sentence(pos)
        if b - a > 320:
            a, b = max(a, pos - 140), min(b, (end or pos) + 160)
        return self.raw[a:b].strip()

    def negated(self, pos: int) -> bool:
        """A negation up to 4 words before ``pos`` in the same clause ("not interested in technology")."""
        a, _ = self.sentence(pos)
        window = re.split(r"[,:;]|\bbut\b|\bmas\b|\bhowever\b|\bporem\b", self.f[max(a, pos - 45):pos])[-1]
        return bool(NEG[self.lang].search(" ".join(window.split()[-4:])))


def _num(s: str) -> float:
    """'500,000' / '500.000' / '1,5' / '1.5' / '1,234.56' / '1.234,56' -> float."""
    s = s.strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        head, *rest = s.split(",")
        s = s.replace(",", "") if rest and all(len(r) == 3 for r in rest) and len(head) <= 3 else s.replace(",", ".")
    elif "." in s:
        head, *rest = s.split(".")
        if len(rest) > 1 or (rest and len(rest[0]) == 3 and len(head) <= 3):
            s = s.replace(".", "")
    return float(s)


def _count(s: str) -> float:
    return float(NUM_WORDS[s]) if s in NUM_WORDS else _num(s)


@dataclass
class _Money:
    value: float
    currency: str
    start: int
    end: int
    per_year: float | None = None       # 12 for "per month", 1 for "per year", None for a lump sum


def _moneys(t: _Text) -> list[_Money]:
    out = []
    for m in MONEY_RX.finditer(t.f):
        if m.group("num") is not None:
            cur, num, mult = m.group("cur"), m.group("num"), m.group("mult")
        else:
            cur, num, mult = m.group("cur2"), m.group("num2"), m.group("mult2")
        try:
            v = _num(num) * _MULT.get(mult or "", 1.0)
        except ValueError:
            continue
        per = None
        p = PERIOD_RX.match(t.f[m.end():m.end() + 30])
        if p:
            unit = p.group("en") or p.group("en2") or p.group("pt") or p.group("pt2") or ""
            per = 12.0 if unit.startswith(("mo", "mes", "mens")) else 1.0
        out.append(_Money(v, _CUR[cur], m.start(), m.end(), per))
    return out


def _fmt_money(v: float, ccy: str) -> str:
    sym = {"USD": "US$", "BRL": "R$", "EUR": "€"}.get(ccy, ccy)
    return f"{sym} {v:,.0f}".replace(",", ".")


_LOSS_WORDS = re.compile(r"\b(?:lose|losing|loss|losses|drop|decline|fall|drawdown|perder|perda|perdas|queda|cair)\b")


def _years_in(t: _Text, a: int, b: int, today: date) -> list[tuple[float, int]]:
    """(years, position) of every horizon stated in f[a:b], sentence by sentence (none from a sentence about losses:
    "cannot lose more than 20% in a year" states no goal horizon)."""
    out: list[tuple[float, int]] = []
    unit_rx = (rf"\bin (?:about |around |roughly |approximately |just )?{NUMW}[\s-](years?|months?)\b",
               rf"\bwithin (?:the next )?{NUMW} (years?|months?)\b",
               rf"\b(?:over|during|for) the next {NUMW} (years?|months?)\b",
               rf"\b{NUMW} (years?|months?) from (?:now|today)\b",
               rf"\b(?:time |investment )?horizon (?:of|is|of about|of around) (?:about |around )?{NUMW} (years?|months?)\b",
               rf"\b(?:em|dentro de|daqui a|nos proximos|pelos proximos) (?:cerca de |uns )?{NUMW} (anos?|meses)\b",
               rf"\b(?:horizonte|prazo) (?:de|e de) (?:cerca de )?{NUMW} (anos?|meses)\b")
    for sa, sb in t.spans:
        lo, hi = max(a, sa), min(b, sb)
        if lo >= hi:
            continue
        seg = t.f[lo:hi]
        if _LOSS_WORDS.search(seg):
            continue
        for rx in unit_rx:
            for m in re.finditer(rx, seg):
                try:
                    n = _count(m.group(1))
                except (ValueError, KeyError):
                    continue
                yrs = n / 12.0 if m.group(2).startswith(("month", "mes")) else n
                if 0 < yrs <= 60:
                    out.append((yrs, lo + m.start()))
        for m in re.finditer(rf"\b{NUMW}[- ]year (?:investment |time )?horizon\b", seg):
            try:
                n = _count(m.group(1))
            except (ValueError, KeyError):
                continue
            if 0 < n <= 60:
                out.append((n, lo + m.start()))
        for m in re.finditer(r"\b(?:by|in|until|around|em|ate|ate o ano de) (20[3-9]\d)\b", seg):
            yrs = int(m.group(1)) - today.year
            if 0 < yrs <= 60:
                out.append((float(yrs), lo + m.start()))
    out.sort(key=lambda x: x[1])
    dedup: list[tuple[float, int]] = []
    for y, p in out:
        if not any(abs(p - q) < 12 for _, q in dedup):
            dedup.append((y, p))
    return dedup


def _first(t: _Text, patterns: tuple[str, ...]) -> re.Match[str] | None:
    hits = [m for rx in patterns for m in re.finditer(rx, t.f)]
    return min(hits, key=lambda m: m.start()) if hits else None


# ----------------------------------------------------------------------------- the client case
def read_case(text: str, today: date | None = None) -> Reading:
    t = _Text(text)
    day = today or datetime.now(UTC).date()
    r = Reading(idioma=t.lang)
    if not t.raw.strip():
        r.faltando.append("texto do desafio vazio")
        return r

    # age and retirement age
    age_m = _first(t, (r"\b(\d{2})[- ]years?[- ]old\b", r"\baged (\d{2})\b",
                       r"(?<!retirement )(?<!retiring )\b(?:age|idade) (?:is |of |de )?(\d{2})\b",
                       r"\b(\d{2}) anos de idade\b", (r"\b(?:tem|com) (\d{2}) anos\b(?! de (?:prazo|horizonte|"
                       r"experiencia|carreira|empresa))")))
    age = None
    if age_m and 14 <= int(age_m.group(1)) <= 100:
        age = int(age_m.group(1))
        r.achados.append(Finding("idade", "idade do cliente", age, t.quote(age_m.start(), age_m.end())))
    ret_m = _first(t, (r"\bretire (?:at|by) (?:the )?(?:age )?(?:of )?(\d{2})\b",
                       r"\baposentar(?:-se)? (?:aos|com|a partir dos) (\d{2})\b"))

    # horizons: every one stated; the longest is the investment horizon, the others are intermediate goals
    horizons = _years_in(t, 0, len(t.f), day)
    if ret_m and age is not None and int(ret_m.group(1)) > age:
        horizons.append((float(int(ret_m.group(1)) - age), ret_m.start()))
    moneys = _moneys(t)

    # amounts: classified by the words of their own sentence (salaries are not investable money)
    wealth, goals, inflows, outflows, liquid = [], [], [], [], []
    prev_end = 0
    for mo in moneys:
        a, b = t.sentence(mo.start)
        s = t.f[a:b]
        pre = t.f[max(a, prev_end, mo.start - 90):mo.start]      # the words just before this amount
        prev_end = mo.end
        yrs = [y for y, _ in _years_in(t, a, b, day)]
        if mo.per_year is not None:
            if _SALARY_CUE.search(pre or s) and not re.search(r"\b(?:withdraw\w*|spend\w*|expenses|retirement "
                                                              r"income|sacar|retirar|despesas)\b", s):
                continue
            if _INFLOW_CUE.search(pre[-60:]):
                inflows.append(mo)
            elif _OUTFLOW_CUE.search(s):
                outflows.append(mo)
            elif _INFLOW_CUE.search(s):
                inflows.append(mo)
            continue
        if _LIQUIDITY_CUE.search(s) and (not yrs or min(yrs) <= 1):
            liquid.append(mo)
        elif _GOAL_CUE.search(pre):
            goals.append((mo, max(yrs) if yrs else None, bool(_EXPENSE_CUE.search(s))))
        elif _WEALTH_CUE.search(pre) and not _SALARY_CUE.search(pre):
            wealth.append(mo)
        elif _GOAL_CUE.search(s) or (_EXPENSE_CUE.search(s) and yrs):
            goals.append((mo, max(yrs) if yrs else None, bool(_EXPENSE_CUE.search(s))))
        elif _WEALTH_CUE.search(s) and not _SALARY_CUE.search(s):
            wealth.append(mo)
    ccys = {m.currency for m in moneys}
    if len(ccys) > 1:
        r.avisos.append("O texto usa mais de uma moeda (" + ", ".join(sorted(ccys)) + "): confira os valores.")
    if moneys:
        main_ccy = max(sorted(ccys), key=lambda c: sum(m.currency == c for m in moneys))
        first = next(m for m in moneys if m.currency == main_ccy)
        r.achados.append(Finding("moeda_caso", "moeda dos valores do caso", main_ccy, t.quote(first.start, first.end)))
    if wealth:
        top = max(wealth, key=lambda m: m.value)
        others = [m for m in wealth if m is not top]
        r.achados.append(Finding("patrimonio_cliente", "dinheiro do cliente hoje", top.value, t.quote(top.start, top.end),
                                 ("outros valores citados: " + ", ".join(_fmt_money(m.value, m.currency) for m in others)
                                  + " (usado o maior; some se forem partes do mesmo patrimônio)") if others else ""))
        r.achados[-1].nota = (r.achados[-1].nota + f" Moeda: {top.currency}.").strip()
    main_goal = None
    if goals:
        long_goals = sorted(goals, key=lambda g: (g[1] if g[1] is not None else -1, g[0].value), reverse=True)
        main_goal = long_goals[0]
        mo, yrs, _ = main_goal
        r.achados.append(Finding("valor_alvo_cliente", "meta em dinheiro", mo.value, t.quote(mo.start, mo.end),
                                 f"em {yrs:g} anos" if yrs else ""))
        for mo2, yrs2, _ in long_goals[1:]:
            r.achados.append(Finding("meta_intermediaria", "meta intermediária", mo2.value, t.quote(mo2.start, mo2.end),
                                     (f"em {yrs2:g} anos" if yrs2 else "sem prazo no texto") + ": a parte dessa meta "
                                     "costuma ficar em ativos de menor risco, à parte da carteira de longo prazo"))
    if inflows:
        tot = sum(m.value * (m.per_year or 1) for m in inflows)
        r.achados.append(Finding("aporte_anual_cliente", "aportes por ano", tot, t.quote(inflows[0].start, inflows[0].end),
                                 "valores mensais multiplicados por 12"))
    if outflows:
        tot = sum(m.value * (m.per_year or 1) for m in outflows)
        r.achados.append(Finding("retirada_anual_cliente", "retiradas/gastos por ano", tot,
                                 t.quote(outflows[0].start, outflows[0].end),
                                 "o motor não simula retiradas: elas aparecem no IPS e pesam no objetivo de renda"))
    pat = r.value("patrimonio_cliente")
    if liquid:
        amount = sum(m.value for m in liquid)
        frac = round(min(1.0, amount / float(pat)), 4) if isinstance(pat, float) and pat > 0 else None
        r.achados.append(Finding("necessidade_liquidez", "dinheiro que pode precisar em até 12 meses", frac,
                                 t.quote(liquid[0].start, liquid[0].end),
                                 f"{_fmt_money(amount, liquid[0].currency)}" + (f" = {pct(frac)} do patrimônio" if frac
                                                                               is not None else
                                                                               "; informe o patrimônio para virar %")))
    if horizons:
        yrs, pos = max(horizons, key=lambda h: h[0])
        note = ""
        if len({round(h[0], 2) for h in horizons}) > 1:
            note = ("prazos citados: " + ", ".join(f"{h[0]:g}" for h in sorted(horizons)) + " anos; usado o mais "
                    "longo (as metas mais curtas ficam como intermediárias)")
        if ret_m and age is not None and pos == ret_m.start():
            note = (note + f"; aposentadoria aos {ret_m.group(1)} com {age} anos hoje").strip("; ")
        r.achados.append(Finding("horizonte_anos", "horizonte do investimento (anos)", round(yrs, 2),
                                 t.quote(pos), note))

    # risk tolerance
    risk_hits: list[tuple[int, str, str, int]] = []
    taken: list[tuple[int, int]] = []
    for rx, level, cancel, kind in RISK_RULES:
        for m in re.finditer(rx, t.f):
            if any(a <= m.start() < b for a, b in taken):
                continue
            if cancel and t.negated(m.start()):
                continue
            risk_hits.append((m.start(), level, kind, m.end()))
            taken.append((m.start(), m.end()))
    if risk_hits:
        labels = [h for h in risk_hits if h[2] == "rotulo"]
        used = labels or risk_hits
        levels = sorted({h[1] for h in used}, key=RISK_ORDER.get)
        pick = levels[0]
        first = min((h for h in used if h[1] == pick), key=lambda h: h[0])
        notes = []
        if len(levels) > 1:
            notes.append("o texto dá mais de um rótulo (" + ", ".join(levels) + "): usado o mais conservador, prática "
                         "comum de adequação (o menor entre disposição e capacidade de correr risco)" if labels else
                         "sinais diferentes no texto (" + ", ".join(levels) + "): usado o mais conservador")
        behaviour = sorted({h[1] for h in risk_hits if h[2] == "comportamento"} - {pick}, key=RISK_ORDER.get)
        if labels and behaviour:
            notes.append("o texto também descreve comportamento de perfil " + ", ".join(behaviour) + "; vale o rótulo "
                         "que o cliente usa, e o limite de queda (se houver) entra como regra à parte")
        r.achados.append(Finding("perfil_risco", "tolerância a risco", pick, t.quote(first[0], first[3]), "; ".join(notes)))
    # objective
    obj: dict[str, list[re.Match[str]]] = {}
    for rx, kind in OBJECTIVE_RULES:
        for m in re.finditer(rx, t.f):
            if not t.negated(m.start()):
                obj.setdefault(kind, []).append(m)
    if obj:
        pref = {"preservacao": 0, "renda": 1, "crescimento": 2}
        kind = max(obj, key=lambda k: (len(obj[k]), -pref[k]))
        m0 = obj[kind][0]
        others = [k for k in obj if k != kind]
        r.achados.append(Finding("objetivo_tipo", "objetivo principal", kind, t.quote(m0.start(), m0.end()),
                                 ("também citados: " + ", ".join(others) + "; usado o mais citado") if others else ""))
    # loss tolerance and required return
    loss = _first(t, ((rf"\b(?:lose|losing|loss(?:es)?|drop|decline|fall|drawdown|down)\b[^.;]{{0,40}}?\b(?:more than|"
                      rf"over|beyond|exceeding|greater than|above)\s{PCT}"),
                      (rf"\b(?:max(?:imum)?|tolerate|accept|handle|stomach)\b[^.;]{{0,30}}?\b(?:loss|drop|decline|"
                      rf"drawdown) of\s(?:up to |about |around )?{PCT}"),
                      (rf"\b(?:perder|perda|queda|cair|desvalorizacao)\b[^.;]{{0,40}}?\b(?:mais de|acima de|alem de|"
                      rf"superior a)\s{PCT}"),
                      rf"\b(?:perda|queda) maxima (?:de|aceitavel de)\s{PCT}"))
    if loss:
        v = _num(loss.group(1)) / 100
        if 0 < v < 1:
            r.achados.append(Finding("tolerancia_perda", "maior queda que aceita", v, t.quote(loss.start(), loss.end())))
    ret = _first(t, ((rf"\b(?:return|returns|growth rate|earn|earning|grow)\b[^.;%]{{0,35}}?\b(?:of|at least|around|"
                     rf"about|approximately|roughly|by)\s{PCT}"),
                     rf"{PCT}\s(?:annual(?:ized)?|average annual|per year|a year|annually)\s(?:return|growth)",
                     rf"\b(?:retorno|rentabilidade|render|rendimento)\b[^.;%]{{0,35}}?\b(?:de|minimo de|pelo menos)\s{PCT}"))
    if ret:
        v = _num(ret.group(1)) / 100
        if 0 < v < 0.5:
            r.achados.append(Finding("retorno_exigido", "retorno anual exigido", v, t.quote(ret.start(), ret.end())))

    # exclusions (an avoidance word BEFORE the term, in the same sentence) and values
    excl: dict[str, Finding] = {}
    for key, rx in EXCLUSION_TERMS.items():
        for m in re.finditer(rf"\b(?:{rx})\b", t.f):
            a, _ = t.sentence(m.start())
            before = t.f[a:m.start()]
            strong = list(re.finditer(AVOID[t.lang], before))
            near = re.search(NEAR_AVOID[t.lang] + r"(?:\w+\s){0,2}$", before)
            if (strong and m.start() - (a + strong[-1].start()) <= 140) or near:
                excl.setdefault(key, Finding("exclusoes", "não quer investir em", key, t.quote(m.start(), m.end()),
                                             EXCLUSION_PT[key]))
    if excl:
        keys = sorted(excl)
        r.achados.append(Finding("exclusoes", "não quer investir em", keys,
                                 " … ".join(dict.fromkeys(excl[k].trecho for k in keys)),
                                 ", ".join(EXCLUSION_PT[k] for k in keys)))
    themes = {}
    for key, rx in THEMES.items():
        for m in re.finditer(rf"(?:{rx})", t.f):
            if not t.negated(m.start()):
                themes.setdefault(key, t.quote(m.start(), m.end()))
                break
    if themes:
        keys = sorted(themes)
        r.achados.append(Finding("temas", "valores e interesses do cliente", keys,
                                 " … ".join(dict.fromkeys(themes[k] for k in keys)),
                                 ", ".join(THEME_PT[k] for k in keys) + " (inclinam a carteira, sem passar dos limites "
                                                                          "de risco)"))
    if re.search(r"\besg\b|socially responsible|ethical invest\w*|impact invest\w*|investimento (?:responsavel|"
                 r"sustentavel|etico)", t.f) and "combustiveis_fosseis" not in excl:
        r.avisos.append("O texto fala em investimento responsável/ESG sem dizer o que excluir: decidam se combustíveis "
                        "fósseis, armas e tabaco ficam de fora (marque em 'Não quer investir em').")
    wanted = {"horizonte_anos": "horizonte do investimento (em anos)", "perfil_risco": "tolerância a risco",
              "objetivo_tipo": "objetivo principal (crescer, renda ou proteger)",
              "patrimonio_cliente": "quanto o cliente tem hoje", "valor_alvo_cliente": "meta em dinheiro (se houver)"}
    r.faltando = [label for campo, label in wanted.items() if r.get(campo) is None]
    return r


# ----------------------------------------------------------------------------- the competition rules
def _date(m: re.Match[str]) -> tuple[int, int, int | None]:
    mon = MONTHS[m.group("mon")] if "mon" in m.groupdict() and m.group("mon") else int(m.group("mnum"))
    year = int(m.group("year")) if m.group("year") else None
    return int(m.group("day")), mon, year


def _dates(t: _Text) -> list[tuple[int, int, int, int | None]]:
    """(position, day, month, year or None) of each date written with the month's name (or ISO)."""
    out = []
    for rx in DATE_RX:
        for m in rx.finditer(t.f):
            try:
                d, mo, y = _date(m)
            except (KeyError, ValueError):
                continue
            if 1 <= d <= 31 and not any(abs(p - m.start()) < 4 for p, *_ in out):
                out.append((m.start(), d, mo, y))
    return sorted(out)


def read_rules(text: str, today: date | None = None) -> Reading:
    t = _Text(text)
    day = today or datetime.now(UTC).date()
    r = Reading(idioma=t.lang)
    if not t.raw.strip():
        r.faltando.append("texto das regras vazio")
        return r
    f = t.f

    def add(campo: str, rotulo: str, valor: object, m: re.Match[str] | int, nota: str = "") -> None:
        if r.get(campo) is None:
            pos = m if isinstance(m, int) else m.start()
            end = None if isinstance(m, int) else m.end()
            r.achados.append(Finding(campo, rotulo, valor, t.quote(pos, end), nota))

    cap = [mo for mo in _moneys(t) if mo.value >= 1000 and mo.per_year is None and re.search(
        r"\b(?:virtual|starting|initial|beginning|start(?:s)? with|receives?|given|allocated|each team|portfolio of|cash "
        r"of|balance of|capital inicial|dinheiro virtual|saldo inicial|cada equipe|recebe\w*|carteira de)\b",
        f[t.sentence(mo.start)[0]:t.sentence(mo.start)[1]])]
    if cap:
        top = max(cap, key=lambda m: m.value)
        add("moeda", "moeda da competição", top.currency, top.start)
        add("capital", "capital da competição", top.value, top.start,
            f"moeda {top.currency}" + ("; outros valores: " + ", ".join(_fmt_money(m.value, m.currency) for m in cap
                                                                        if m is not top) if len(cap) > 1 else ""))
    m = _first(t, (r"\bpriced (?:at|of|above|over|at or above) (?:\$|us\$ ?)(\d+(?:[.,]\d+)?)",
                   r"\b(?:minimum|min\.?) (?:share |stock |security )?price (?:of |is )?(?:\$|us\$ ?)(\d+(?:[.,]\d+)?)",
                   (r"\b(?:below|under|less than) (?:\$|us\$ ?)(\d+(?:[.,]\d+)?)(?: per share| a share)?[^.;]{0,30}?"
                   r"\b(?:are not|is not|cannot|can'?t|may not|not (?:permitted|allowed|eligible)|prohibited|ineligible)"),
                   r"\bprice (?:of )?(?:\$|us\$ ?)(\d+(?:[.,]\d+)?) or (?:higher|more|above)",
                   r"\bpreco minimo (?:de |por acao de )?(?:us\$|\$) ?(\d+(?:[.,]\d+)?)",
                   r"\babaixo de (?:us\$|\$) ?(\d+(?:[.,]\d+)?)[^.;]{0,30}?\b(?:nao|proibid)"))
    if m:
        add("preco_minimo", "preço mínimo por ação", _num(m.group(1)), m)
    m = _first(t, (rf"\b(?:at least|minimum of|a minimum of|no fewer than) {NUMW} (?:approved |eligible |)etfs?\b",
                   (rf"\bmust (?:hold|include|own|have|contain|purchase|buy) (?:at least )?{NUMW} (?:approved |eligible |)"
                   r"etfs?\b"),
                   rf"\b(?:pelo menos|no minimo|minimo de) {NUMW} etfs?\b"))
    if m:
        add("min_etfs", "mínimo de ETFs", int(_count(m.group(1))), m)
    m = _first(t, (r"\bapproved (?:etf|etfs|stock|security|securities)(?: and etf)? list\b", r"\blist of approved etfs?\b",
                   r"\betfs? (?:from|on) (?:the|an) approved list\b", r"\blista (?:aprovada|de etfs aprovados)\b"))
    if m:
        add("lista_etfs_aprovados", "há lista de ETFs aprovados", True, m,
            "cole os códigos da lista no campo 'ETFs aprovados': ETFs fora dela ficam de fora")
    m = _first(t, ((rf"\b(?:no|any|each) (?:single |one |individual )?(?:stock|security|position|holding|company|asset|"
                   rf"investment|equity)s? (?:may|can|should|shall|must|will)? ?(?:not )?(?:exceed|be more than|represent "
                   rf"more than|make up more than|account for more than|be greater than|go above|be over|surpass)\s{PCT}"),
                   (rf"\b(?:maximum|max\.?|limit|cap)(?: position| single[- ]stock)?(?: size| weight| allocation| "
                   rf"concentration)? (?:of |is )?{PCT} (?:per|in any|in a single|for any|of the portfolio|in one|on any)"),
                   rf"{PCT} (?:maximum|max|limit|cap) (?:per|in any|for any|on any|on a single)",
                   rf"\bposition (?:size )?limit(?: of| is|:)?\s{PCT}",
                   (rf"\bno more than {PCT} (?:of (?:the|your|their|a|its) (?:total )?portfolio )?(?:value )?(?:in|into|on|to) "
                   r"(?:any|a|one) (?:single |one |individual )?(?:stock|security|position|holding|company|asset)"),
                   (rf"\b(?:no maximo|maximo de|ate|limite de) {PCT} (?:por|em (?:um|uma) (?:unica |so )?|em cada) "
                   r"(?:acao|ativo|empresa|posicao|papel)")))
    if m:
        v = _num(m.group(1)) / 100
        if 0 < v <= 1:
            add("peso_maximo", "peso máximo por ativo", v, m)
    noun = r"(?:different |individual |separate |unique |distinct )?(?:stocks|securities|positions|holdings|companies|equities|investments|assets|tickers)"
    m = _first(t, (rf"\bbetween {NUMW} and {NUMW} {noun}\b", rf"\bentre {NUMW} e {NUMW} (?:acoes|ativos|posicoes|empresas)\b"))
    if m:
        add("min_posicoes", "mínimo de posições", int(_count(m.group(1))), m)
        add("max_posicoes", "máximo de posições", int(_count(m.group(2))), m)
    m = _first(t, (rf"\b(?:at least|minimum of|a minimum of|no fewer than|min\.?) {NUMW} {noun}\b",
                   rf"\b(?:pelo menos|no minimo|minimo de) {NUMW} (?:acoes|ativos|posicoes|empresas|papeis)\b"))
    if m:
        add("min_posicoes", "mínimo de posições", int(_count(m.group(1))), m)
    m = _first(t, (rf"\b(?:no more than|maximum of|a maximum of|at most|up to|max\.?) {NUMW} {noun}\b",
                   rf"\b(?:no maximo|maximo de|ate) {NUMW} (?:acoes|ativos|posicoes|empresas|papeis)\b"))
    if m:
        add("max_posicoes", "máximo de posições", int(_count(m.group(1))), m)
    m = _first(t, ((rf"\b(?:at least|minimum of|a minimum of|no fewer than) {NUMW} (?:different |distinct |separate )?"
                   r"(?:sectors|industries)\b"),
                   rf"\b{NUMW} or more (?:different |distinct )?sectors\b",
                   rf"\b(?:pelo menos|no minimo|minimo de) {NUMW} setores\b"))
    if m:
        add("min_setores", "mínimo de setores", int(_count(m.group(1))), m)
    m = _first(t, (r"\bas many (?:different )?sectors as (?:there are )?(?:team )?members\b",
                   r"\btantos setores quanto (?:membros|integrantes)\b"))
    if m:
        add("min_setores", "mínimo de setores", None, m, "a regra pede tantos setores quanto integrantes da equipe: "
                                                       "preencha com o tamanho da sua equipe")
    m = _first(t, ((rf"\bcash (?:position |holdings? |balance |allocation )?(?:may|can|should|must|shall)? ?(?:not )?"
                   rf"(?:exceed|be more than|be greater than|go above|be over)\s{PCT}"),
                   rf"\bno more than {PCT} (?:of (?:the|your|a) portfolio )?(?:in|as|held in|held as) cash\b",
                   rf"\bcaixa (?:nao pode (?:passar de|ultrapassar|exceder)|maximo de|de no maximo|ate)\s{PCT}"))
    if m:
        add("caixa_maximo", "caixa máximo", _num(m.group(1)) / 100, m)
    else:
        m = _first(t, ((rf"\b(?:at least|minimum of|a minimum of) {PCT} (?:of (?:the|your|a) portfolio )?(?:must be |should "
                       r"be |has to be )?invested\b"),))
        if m:
            add("caixa_maximo", "caixa máximo", round(1 - _num(m.group(1)) / 100, 4), m,
                f"investir pelo menos {m.group(1)}% = caixa de no máximo {100 - _num(m.group(1)):g}%")
    m = _first(t, ((r"\bmargin(?: trading| accounts?| buying| purchases)? (?:is |are )?(?:not (?:permitted|allowed)|"
                   r"prohibited|forbidden|banned|disallowed)\b"), r"\bno margin\b", r"\bwithout margin\b",
                   r"\b(?:nao e permitid[oa]|proibid[oa]) (?:operar )?(?:com |a )?margem\b", r"\bmargem (?:e )?proibida\b"))
    if m:
        add("sem_margem", "sem compra com margem", True, m, "a carteira nunca usa dinheiro emprestado")
    m = _first(t, ((r"\bshort(?:[- ]selling| sales?| positions?)? (?:is |are )?(?:not (?:permitted|allowed)|prohibited|"
                   r"forbidden|banned|disallowed)\b"), r"\bno short(?:[- ]selling)?\b",
                   r"\bvenda a descoberto (?:nao e permitida|e proibida|proibida)\b"))
    if m:
        add("sem_venda_descoberto", "sem venda a descoberto", True, m, "a carteira só compra (nunca vende a descoberto)")
    m = _first(t, ((r"\b(?:maximum|max\.?|limit|cap|up to|no more than|at most|total) (?:of )?(\d{1,4}) (?:total )?"
                   r"(?:trades|transactions|orders)\b"),
                   r"\b(\d{1,4}) (?:total )?(?:trades|transactions) (?:maximum|max|limit|per team|in total|cap)\b",
                   r"\b(?:maximo de|ate|no maximo|limite de) (\d{1,4}) (?:operacoes|ordens|negociacoes|transacoes)\b"))
    if m:
        add("max_operacoes", "máximo de operações", int(m.group(1)), m)
    m = _first(t, ((r"\b(?:twice|two times|2x|2 x|double) (?:the |a |its |each )?(?:security'?s |stock'?s )?(?:current |"
                   r"average |)(?:daily |day'?s |)(?:trading )?volume\b"),))
    if m:
        add("limite_volume", "limite por ordem (múltiplo do volume diário)", 2.0, m)
    else:
        m = _first(t, ((r"\b(\d+(?:[.,]\d+)?) ?(?:x|times) (?:the |a |its )?(?:security'?s )?(?:current |average )?daily "
                       r"(?:trading )?volume\b"),))
        if m:
            add("limite_volume", "limite por ordem (múltiplo do volume diário)", _num(m.group(1)), m)
    m = _first(t, (r"\bfractional shares? (?:are |is )?(?:not (?:permitted|allowed|supported|available)|prohibited)\b",
                   r"\bno fractional shares?\b", (r"\b(?:fracoes|fracionario) (?:de acoes )?(?:nao (?:e|sao) "
                                                  r"permitid[oa]s?|proibid[oa]s?)\b")))
    if m:
        add("permitir_fracoes", "frações de ação", False, m)
    else:
        m = _first(t, (r"\bfractional shares? (?:are |is )?(?:permitted|allowed|supported|available)\b",))
        if m:
            add("permitir_fracoes", "frações de ação", True, m)
    m = _first(t, (r"\bany (?:stock|security|equity|etf)s? (?:from|on|listed on|traded on) any (?:exchange|market)\b",
                   r"\bany exchange\b"))
    if m:
        add("qualquer_bolsa", "bolsas permitidas", True, m, "o site cobre bolsas dos EUA (NYSE, Nasdaq) e a B3; ações de "
                                                           "outros países ficam fora da análise")
    # trading window: a date after an ending word is the end; after a starting word, the start
    dates = _dates(t)
    for pos, d_, mo_, y_ in dates:
        a, b = t.sentence(pos)
        before = f[max(a, pos - 60):pos]
        same = [x for x in dates if a <= x[0] < b and x[3] is not None]
        year = y_ or (same[-1][3] if same else None)
        note = ""
        if year is None:
            cand = [date(y, mo_, min(d_, 28 if mo_ == 2 else 30 if mo_ in (4, 6, 9, 11) else 31))
                    for y in (day.year - 1, day.year, day.year + 1)]
            pick = min(cand, key=lambda c: abs((c - (day + timedelta(days=60))).days))
            year, note = pick.year, "ano não escrito no texto: inferido pela data de hoje"
        try:
            when = date(year, mo_, d_)
        except ValueError:
            continue
        if re.search(r"\b(?:registration|register|sign[- ]?up|report|deliverable|submission|submit|inscri\w*|"
                     r"relatorio|entrega)\b", before) or not re.search(
                         r"\b(?:trad\w*|competition|simulat\w*|portfolio|negocia\w*|competicao|simulac\w*|carteira|"
                         r"operac\w*)\b", f[a:b]):
            continue
        if re.search(r"\b(?:end|ends|ending|close|closes|closing|conclude\w*|finish\w*|final (?:trading )?day|last "
                     r"(?:trading )?day|through|until|thru|to|termina\w*|encerra\w*|ate|fim)\b[^.;]*$", before):
            add("fim_negociacao", "fim das negociações", when.isoformat(), pos, note)
        elif re.search(r"\b(?:begin\w*|start\w*|open\w*|commence\w*|from|inicio|inicia\w*|comeca\w*|abre|a partir de)"
                       r"\b[^.;]*$", before):
            add("inicio_negociacao", "início das negociações", when.isoformat(), pos, note)
    wanted = {"capital": "capital da competição", "preco_minimo": "preço mínimo por ação (se houver)",
              "peso_maximo": "peso máximo por ativo (se houver)", "fim_negociacao": "data de fim das negociações"}
    r.faltando = [label for campo, label in wanted.items() if r.get(campo) is None]
    return r


_NOT_TICKERS = {"ETF", "ETFS", "AND", "THE", "LIST", "OF", "OR", "FOR", "USD", "NYSE", "NASDAQ", "AMEX", "ARCA",
                "TICKER", "SYMBOL", "NAME", "FUND", "TRUST", "INDEX", "SHARES", "ISHARES", "SPDR", "VANGUARD",
                "APPROVED", "E", "DE", "DO", "DA", "OS", "AS", "NA", "NO", "US", "USA", "UK", "EU", "EM", "REIT", "ESG",
                "TIPS", "MSCI", "FTSE", "CRSP", "SP", "DJ", "LLC", "INC", "LTD", "CORP"}
_CODE = re.compile(r"\b[A-Za-z]{1,5}(?:[.-][A-Za-z])?\b")


def parse_tickers(text: str, limit: int = 600) -> list[str]:
    """Codes from a pasted list. Pieces are split at line breaks, commas, semicolons, tabs and bars; a piece with only
    codes ("SPY, QQQ; VTI") gives all of them; a piece that also has a name ("AGG - iShares Core US Aggregate Bond
    ETF", "SPDR S&P 500 ETF Trust (SPY)") gives the code in parentheses or else its first upper-case code of two or more
    letters. Common words of such lists (ETF, US, SPDR...) are never codes."""
    out: list[str] = []
    for piece in re.split(r"[\n,;|\t]+", text or ""):
        paren = re.findall(r"\(([A-Z]{1,5}(?:[.-][A-Z])?)\)", piece)
        codes = [t for t in _CODE.findall(piece) if t == t.upper() and t not in _NOT_TICKERS]
        if paren:
            found = paren
        elif re.search(r"[a-z]{2}", piece):          # the piece has a name: one code, the first that looks like one
            found = [t for t in codes if len(t.split(".")[0].split("-")[0]) >= 2][:1]
        else:
            found = codes
        for up in found:
            if up not in out and up not in _NOT_TICKERS:
                out.append(up)
            if len(out) >= limit:
                return out
    return out
