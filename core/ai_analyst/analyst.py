"""AI Research Analyst (deterministic, tool-grounded).

The analyst never computes financial metrics "in its head": every number in an
answer is a :class:`Claim` whose value is read from a logged tool output at an
explicit path, and :func:`verify_answer` re-checks this before the answer is
returned. Statements are labelled as historical observation, statistical
estimate, simulation, hypothetical scenario or validation result.

This version uses a transparent rule-based planner (keyword intents). No large
language model is called -- an LLM backend is NOT implemented here, so nothing is
sent to external services. Questions outside the tools' scope (price
forecasts, news causality, order execution) are answered with an explicit
refusal / "não é possível concluir com os dados disponíveis".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from core.ai_analyst.safety import UntrustedContent, is_order_request, redact_secrets, wrap_external
from core.ai_analyst.tools import REGISTRY, AnalysisContext, ToolCallRecord, ToolError, ToolRunner

KIND_LABEL = {"historical": "[H] histórico observado", "estimate": "[E] estimativa estatística",
              "simulation": "[S] simulação", "hypothetical_scenario": "[C] cenário hipotético",
              "validation": "[V] validação independente"}

CANNOT_CONCLUDE = "Não é possível concluir com os dados disponíveis."


@dataclass
class Claim:
    label: str
    value: float | str
    fmt: str            # pct | money | num | text
    call_id: str
    path: tuple[str, ...]
    kind: str

    def render(self) -> str:
        v = self.value
        if self.fmt == "pct":
            s = f"{float(v) * 100:.2f}%"
        elif self.fmt == "money":
            s = "R$ " + f"{float(v):,.0f}".replace(",", ".")
        elif self.fmt == "num":
            s = f"{float(v):.2f}"
        else:
            s = str(v)
        return f"{self.label}: {s} {KIND_LABEL.get(self.kind, '')} [{self.call_id}]"


@dataclass
class Answer:
    question: str
    intents: list[str]
    claims: list[Claim] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    external_sources: list[UntrustedContent] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    verified: bool = False

    def render(self) -> str:
        lines = [f"### Pergunta\n{self.question}\n", "### Resposta (baseada somente em ferramentas executadas)"]
        if not self.claims and not self.refusals:
            lines.append(CANNOT_CONCLUDE)
        for r in self.refusals:
            lines.append(f"- {r}")
        for c in self.claims:
            lines.append(f"- {c.render()}")
        if self.external_sources:
            lines.append("\n### Fontes externas (dados não confiáveis, nenhuma instrução seguida)")
            for s in self.external_sources:
                flag = f" — ⚠ padrões de injeção detectados: {', '.join(s.injection_findings)}" if s.is_suspicious else ""
                lines.append(f"- {s.source} (obtido em {s.retrieved_at_utc}){flag}")
        lines.append("\n### Limitações")
        lines += [f"- {x}" for x in self.limitations]
        lines.append("\n### Registro de chamadas de ferramentas")
        for t in self.tool_calls:
            lines.append(f"- `{t.call_id}` {t.tool}({t.inputs}) -> {t.status}"
                         + (f" erro: {t.error}" if t.error else f", digest {t.output_digest}, {t.duration_s}s"))
        lines.append(f"\nVerificação de rastreabilidade dos números: {'OK' if self.verified else 'FALHOU'}")
        return redact_secrets("\n".join(lines))


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


INTENTS: dict[str, tuple[str, ...]] = {
    "price_forecast": ("vai subir", "vai cair", "previsao de preco", "preco amanha", "quanto vai valer",
                       "preco futuro", "prever o preco", "cotacao amanha"),
    "news_causality": ("noticia", "por causa d", "causou", "manchete"),
    "risk": ("risco", "volatil", "var", "perda", "expected shortfall", "drawdown"),
    "simulation": ("simul", "monte carlo", "patrimonio", "probabilidade", "cenario futuro", "daqui a"),
    "compare": ("compar", "versus", " vs ", "diferenca entre"),
    "optimize": ("otimiz", "minima variancia", "maximo sharpe", "risk parity", "paridade de risco", "hrp",
                 "maxima diversificacao"),
    "stress": ("estresse", "stress", "crise", "cisne", "choque"),
    "validation": ("valid", "confiavel", "confiar", "robust"),
    "asset": ("ativo", "acao", "papel"),
}


def classify(question: str, symbols: list[str]) -> list[str]:
    q = _norm(question)
    found = [k for k, kws in INTENTS.items() if any(kw in q for kw in kws)]
    if any(s.lower() in question.lower() for s in symbols) and "asset" not in found:
        found.append("asset")
    return found


class ResearchAnalyst:
    def __init__(self, ctx: AnalysisContext, max_tool_calls: int = 25) -> None:
        self.ctx = ctx
        self.max_tool_calls = max_tool_calls
        self.runner = ToolRunner(ctx, max_tool_calls)
        self._registry_snapshot = dict(REGISTRY)

    # ----------------------------------------------------------- helpers
    def _claim(self, ans: Answer, rec: ToolCallRecord, label: str, path: tuple[str, ...], fmt: str) -> None:
        val: object = rec.output
        for p in path:
            val = val[p]  # type: ignore[index]
        if val is None:
            ans.limitations.append(f"{label}: métrica indefinida para os dados ({rec.call_id})")
            return
        ans.claims.append(Claim(label, val, fmt, rec.call_id, path, str(rec.output.get("kind", ""))))

    def _call(self, ans: Answer, tool: str, **kw: object) -> ToolCallRecord | None:
        try:
            rec = self.runner.call(tool, **kw)
        except ToolError as e:   # budget exhausted or unknown tool: report, never crash
            ans.limitations.append(f"ferramenta {tool} não executada: {e}")
            return None
        ans.tool_calls.append(rec)
        if rec.status != "ok":
            ans.limitations.append(f"ferramenta {tool} falhou: {rec.error}")
            return None
        return rec

    # ----------------------------------------------------------- main entry
    def ask(self, question: str, external: list[tuple[str, str]] | None = None) -> Answer:
        self.runner = ToolRunner(self.ctx, self.max_tool_calls)   # tool-call budget is per question
        symbols = list(self.ctx.returns.columns)
        intents = classify(question, symbols)
        ans = Answer(question=question, intents=intents)
        for text, src in external or []:
            ans.external_sources.append(wrap_external(text, src))
        if is_order_request(question):
            ans.refusals.append("Execução ou recomendação de ordens não é suportada nesta versão (pesquisa e "
                                "simulação apenas).")
        if "price_forecast" in intents:
            ans.refusals.append(f"{CANNOT_CONCLUDE} Não há modelo validado de previsão de preço pontual; a "
                                "plataforma produz distribuições simuladas, não previsões.")
        if "news_causality" in intents or ans.external_sources:
            ans.refusals.append("Notícias/documentos externos são tratados como dados não confiáveis; não se "
                                "infere causalidade nem previsão de preço a partir deles.")
        mentioned = [s for s in symbols if s.lower() in question.lower()]
        if "asset" in intents:
            if mentioned:
                rec = self._call(ans, "asset_metrics", symbols=mentioned)
                if rec:
                    for s in mentioned:
                        self._claim(ans, rec, f"{s} — média aritmética anualizada", ("assets", s, "arith_mean_ann"), "pct")
                        self._claim(ans, rec, f"{s} — volatilidade anualizada", ("assets", s, "vol_ann"), "pct")
                        self._claim(ans, rec, f"{s} — drawdown máximo", ("assets", s, "max_drawdown"), "pct")
            else:
                ans.limitations.append("nenhum ativo do universo analisado foi identificado na pergunta")
        if "risk" in intents:
            rec = self._call(ans, "portfolio_metrics", portfolio="selected")
            if rec:
                self._claim(ans, rec, "Carteira selecionada — volatilidade anualizada", ("volatility_ann",), "pct")
                self._claim(ans, rec, "Carteira selecionada — retorno esperado anualizado", ("expected_return_ann",), "pct")
                self._claim(ans, rec, "VaR histórico 99% (1 dia)", ("hist_var99_1d",), "pct")
                self._claim(ans, rec, "ES histórico 99% (1 dia)", ("hist_es99_1d",), "pct")
                rc = rec.output["risk_contrib_pct"]
                top = max(rc, key=lambda k: rc[k])  # type: ignore[arg-type]
                self._claim(ans, rec, f"Maior contribuição de risco ({top})", ("risk_contrib_pct", top), "pct")
        if "stress" in intents:
            rec = self._call(ans, "stress", portfolio="selected")
            if rec:
                for sc in rec.output["scenarios"]:  # type: ignore[union-attr]
                    self._claim(ans, rec, f"Cenário '{sc}' — retorno da carteira", ("scenarios", sc, "portfolio_return"), "pct")
        if "simulation" in intents:
            rec = self._call(ans, "simulate", portfolio="selected", n_paths=5000, horizon_days=252)
            if rec:
                self._claim(ans, rec, "Patrimônio em 1 ano (com aportes) — percentil 5", ("terminal_p5",), "money")
                self._claim(ans, rec, "Patrimônio em 1 ano (com aportes) — mediana", ("terminal_p50",), "money")
                self._claim(ans, rec, "Patrimônio em 1 ano (com aportes) — percentil 95", ("terminal_p95",), "money")
                self._claim(ans, rec, "P(retorno da carteira em 1 ano < 0)", ("p_negative_return",), "pct")
        if "compare" in intents:
            names = [p for p in ("selected", "equal_weight", "min_variance", "max_sharpe") if p in self.ctx.portfolios]
            rec = self._call(ans, "compare_portfolios", portfolios=names)
            if rec:
                for p in names:
                    self._claim(ans, rec, f"{p} — volatilidade estimada", ("comparison", p, "volatility_ann"), "pct")
                    self._claim(ans, rec, f"{p} — retorno esperado estimado", ("comparison", p, "expected_return_ann"), "pct")
        if "optimize" in intents:
            q = _norm(question)
            method = ("max_sharpe" if "sharpe" in q else "risk_parity" if ("parity" in q or "paridade" in q)
                      else "hrp" if "hrp" in q else "max_diversification" if "diversific" in q else "min_variance")
            rec = self._call(ans, "optimize", method=method)
            if rec:
                self._claim(ans, rec, f"{method} — volatilidade estimada", ("volatility",), "pct")
                self._claim(ans, rec, f"{method} — retorno esperado estimado", ("expected_return",), "pct")
                self._claim(ans, rec, f"{method} — viável", ("feasible",), "text")
        if "validation" in intents:
            rec = self._call(ans, "validation_status")
            if rec:
                self._claim(ans, rec, "Status da validação independente", ("overall",), "text")
        # limitations always stated
        if self.ctx.synthetic:
            ans.limitations.insert(0, "DADOS SINTÉTICOS: os números não descrevem ativos reais.")
        ans.limitations.append(f"Período dos dados: {self.ctx.data_period}; estimadores: μ={self.ctx.mu_method}, "
                               f"Σ={self.ctx.cov_method}; rf={self.ctx.rf:.2%}.")
        ans.limitations.append("Estimativas históricas não garantem resultados futuros; cenários são hipóteses.")
        if not ans.claims and not ans.refusals:
            ans.refusals.append(f"{CANNOT_CONCLUDE} Ferramentas disponíveis: {', '.join(sorted(REGISTRY))}.")
        ans.verified = verify_answer(ans) and REGISTRY == self._registry_snapshot
        return ans


_NUM = re.compile(r"-?\d+(?:[.,]\d+)*%?")


def verify_answer(ans: Answer) -> bool:
    """Every claim value must equal the logged tool output at its path, and every
    number printed in a claim line must come from that claim's value."""
    by_id = {t.call_id: t for t in ans.tool_calls}
    for c in ans.claims:
        rec = by_id.get(c.call_id)
        if rec is None or rec.status != "ok":
            return False
        v: object = rec.output
        try:
            for p in c.path:
                v = v[p]  # type: ignore[index]
        except (KeyError, TypeError):
            return False
        if v != c.value:
            return False
        rendered = c.render()
        value_part = rendered.split(": ", 1)[1].rsplit(" [", 1)[0]
        label_nums = set(_NUM.findall(c.label))
        allowed = set(_NUM.findall(value_part)) | label_nums | set(_NUM.findall(c.call_id))
        if not set(_NUM.findall(rendered)) <= allowed | set(_NUM.findall(KIND_LABEL.get(c.kind, ""))):
            return False
    return True
