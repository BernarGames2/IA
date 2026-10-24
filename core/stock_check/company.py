"""What the company does and why it may — or may not — grow (context for the 'Vale a pena?' check).

Sources, each labelled in the output; any of them can fail and is then reported as unavailable
(nothing is filled in):

* catalogue one-liner — static, factual curation (``core/purpose/catalog.py``);
* B3 listed-companies service — official registered activity and sector classification, in
  Portuguese (``GetInitialCompanies`` -> ``GetDetail``, parameters as base64 JSON);
* Yahoo Finance ``quoteSummary`` — business summary (English), sector/industry, fundamentals,
  analyst consensus and the date of the last reported quarter. Needs the cookie + crumb handshake
  (``fc.yahoo.com`` -> ``/v1/test/getcrumb``); a stale crumb is refreshed once;
* the Nasdaq sector/industry already present in the full-market listings.

Reasons (:func:`growth_case`) are fixed rules, each tagged with its basis:

  fundamentos  revenue growth >= +10% (up) / < +3% (down); earnings growth >= +15% / < 0;
               net margin >= 15% / < 3% (or a loss); ROE >= max(15%, rf + 3 p.p.) / below rf;
               debt/equity < 0.3x / > 1.5x; P/E < 8 (with caveat) / > 35; dividend yield >= 6%
               or payout > 80% (less reinvestment); free cash flow >= 10% of revenue / negative.
               Margin, leverage and cash-flow rules are skipped for banks, insurers and funds.
  analistas    forward P/E below/above 85%/115% of trailing; mean target >= +15% / <= -5% versus the
               price, only with >= 3 analysts — third-party opinion, reported as such.
  empresa      short factual structural notes (e.g. state control).
  histórico    12-month move with the 200-session average; excess return over rf and its t-stat;
               volatility >= 40%; drawdown <= -50%; beta >= 1.3; daily turnover below 5 million.
  simulação    chance of beating rf in the projection (>= 60% up; otherwise down).
  juros/câmbio Brazilian rf >= 10%; USD-quoted asset seen from BRL.
  setor        general knowledge of the segment (``sectors.py``) — not a forecast for the company.
"""

from __future__ import annotations

import base64
import http.cookiejar
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.market_data.yahoo_chart import USER_AGENT, FetchFn
from core.stock_check.resolver import Resolved
from core.stock_check.sectors import (
    COMPANY_NOTES,
    SEGMENTS,
    Segment,
    applies,
    classify_segment,
    segment_for_class,
)
from utils.logging_config import get_logger

log = get_logger("stock_check.company")

B3_COMPANY_URL = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/{call}/{payload}"
YH_COOKIE_URL = "https://fc.yahoo.com"
YH_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
YH_SUMMARY_URL = ("https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
                  "?modules=assetProfile,financialData,defaultKeyStatistics,summaryDetail&crumb={crumb}")
PROFILE_TTL_S = 24 * 3600
CRUMB_TTL_S = 3600
TIMEOUT_S = 8.0
SECTOR_PT = {
    "healthcare": "Saúde", "health care": "Saúde", "technology": "Tecnologia", "financial services":
    "Serviços financeiros", "finance": "Finanças", "consumer cyclical": "Consumo cíclico", "consumer discretionary":
    "Consumo cíclico", "consumer defensive": "Consumo não cíclico", "consumer staples": "Consumo não cíclico",
    "energy": "Energia", "utilities": "Utilidade pública", "industrials": "Indústria", "basic materials":
    "Materiais básicos", "communication services": "Comunicações", "telecommunications": "Telecomunicações",
    "real estate": "Imobiliário",
}
COUNTRY_PT = {"brazil": "Brasil", "united states": "Estados Unidos", "netherlands": "Países Baixos",
              "taiwan": "Taiwan", "denmark": "Dinamarca", "ireland": "Irlanda", "switzerland": "Suíça",
              "united kingdom": "Reino Unido", "canada": "Canadá", "mexico": "México", "china": "China",
              "japan": "Japão", "germany": "Alemanha", "france": "França", "argentina": "Argentina",
              "uruguay": "Uruguai", "chile": "Chile", "cayman islands": "Ilhas Cayman", "luxembourg": "Luxemburgo"}
REC_PT = {"strong_buy": "compra forte", "buy": "compra", "hold": "manter", "underperform": "abaixo da média",
          "sell": "venda"}
SYM = {"BRL": "R$", "USD": "US$", "EUR": "€"}
BASE_LABEL = {"fundamentos": "Fundamentos", "analistas": "Analistas", "empresa": "Empresa", "historico": "Histórico",
              "simulacao": "Simulação", "juros": "Juros", "cambio": "Câmbio", "setor": "Setor"}
SECTOR_SOURCE = "características gerais do segmento (curadoria editorial; não é previsão para a empresa)"


class SourceError(Exception):
    """A public source answered with an error or with data in an unexpected format."""


# ----------------------------------------------------------------------------- HTTP with cookies
_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_JAR))


def cookie_fetch(url: str, timeout_s: float) -> tuple[int, bytes]:
    """HTTP GET that keeps cookies for the process (Yahoo's crumb is tied to its cookie)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json,text/plain,*/*"})
    try:
        with _OPENER.open(req, timeout=timeout_s) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:      # the cookie processor already stored any Set-Cookie
        try:
            body = e.read()
        except OSError:
            body = b""
        return int(e.code), body


# ----------------------------------------------------------------------------- B3
def b3_payload(params: dict[str, object]) -> str:
    return base64.b64encode(json.dumps(params, separators=(",", ":")).encode()).decode()


def _b3_json(fetch: FetchFn, call: str, params: dict[str, object], timeout_s: float) -> object:
    st, body = fetch(B3_COMPANY_URL.format(call=call, payload=b3_payload(params)), timeout_s)
    if st != 200:
        raise SourceError(f"B3 respondeu HTTP {st}")
    try:
        return json.loads(body)
    except ValueError as e:
        raise SourceError("B3: resposta não é JSON") from e


def _b3_clean(text: object, name: bool = False) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    t = " ".join(text.split())
    if not name:
        t = re.sub(r"\.\s+(?=[A-ZÀ-Úa-zà-ú])", ", ", t)  # B3 exports commas as ". "
    if t.isupper():
        t = t.title() if name else t.capitalize()
    return t if name else t.rstrip(" .")


def fetch_b3_company(code4: str, fetch: FetchFn, timeout_s: float = TIMEOUT_S) -> dict[str, str | None]:
    """Registered activity and sector classification of a B3 issuer (4-letter code, e.g. PGMN)."""
    data = _b3_json(fetch, "GetInitialCompanies",
                    {"language": "pt-br", "pageNumber": 1, "pageSize": 20, "company": code4}, timeout_s)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise SourceError("B3: resposta em formato inesperado")
    hit = next((r for r in results if isinstance(r, dict)
                and str(r.get("issuingCompany", "")).strip().upper() == code4 and r.get("codeCVM")), None)
    if hit is None:
        raise SourceError(f"B3: emissor {code4} não encontrado entre as empresas listadas")
    det = _b3_json(fetch, "GetDetail", {"codeCVM": str(hit["codeCVM"]), "language": "pt-br"}, timeout_s)
    if isinstance(det, list):
        det = next((d for d in det if isinstance(d, dict)), None)
    if not isinstance(det, dict):
        raise SourceError("B3: detalhe da empresa em formato inesperado")
    cls = det.get("industryClassification")
    return {
        "razao_social": _b3_clean(det.get("companyName") or hit.get("companyName"), name=True),
        "nome_pregao": _b3_clean(det.get("tradingName") or hit.get("tradingName"), name=True),
        "atividade": _b3_clean(det.get("activity")),
        "classificacao": " › ".join(p.strip() for p in _b3_clean(cls).split("/")) if _b3_clean(cls) else None,
        "site": (str(det.get("website")).strip() or None) if det.get("website") else None,
    }


# ----------------------------------------------------------------------------- Yahoo quoteSummary
class YahooFundamentals:
    """quoteSummary client with the cookie + crumb handshake (crumb cached for an hour)."""

    def __init__(self, fetch: FetchFn, timeout_s: float = TIMEOUT_S, clock=time.monotonic) -> None:
        self._fetch = fetch
        self.timeout_s = timeout_s
        self._clock = clock
        self._crumb: str | None = None
        self._t = 0.0

    def crumb(self, refresh: bool = False) -> str:
        if self._crumb and not refresh and self._clock() - self._t < CRUMB_TTL_S:
            return self._crumb
        try:
            self._fetch(YH_COOKIE_URL, self.timeout_s)    # sets the cookie; the page itself answers 404
        except OSError:
            pass
        st, body = self._fetch(YH_CRUMB_URL, self.timeout_s)
        text = body.decode("utf-8", "replace").strip()
        if st != 200 or not text or len(text) > 64 or any(c in text for c in "<>{} \n"):
            raise SourceError(f"Yahoo: autorização (crumb) indisponível (HTTP {st})")
        self._crumb, self._t = text, self._clock()
        return text

    def get_json(self, url_for: Callable[[str], str]) -> object:
        """GET another Yahoo JSON endpoint that needs the crumb (``url_for(crumb)``), renewing it once on 401/403."""
        for attempt in range(2):
            st, body = self._fetch(url_for(self.crumb(refresh=attempt > 0)), self.timeout_s)
            if st in (401, 403) and attempt == 0:
                continue
            if st == 404:
                raise SourceError("Yahoo: dado não publicado para este ativo")
            if st != 200:
                raise SourceError(f"Yahoo respondeu HTTP {st}")
            try:
                return json.loads(body)
            except ValueError as e:
                raise SourceError("Yahoo: resposta não é JSON") from e
        raise SourceError("Yahoo: acesso negado mesmo após renovar a autorização")

    def summary(self, symbol: str) -> dict[str, object]:
        for attempt in range(2):
            crumb = self.crumb(refresh=attempt > 0)
            st, body = self._fetch(YH_SUMMARY_URL.format(sym=urllib.parse.quote(symbol, safe=""),
                                                         crumb=urllib.parse.quote(crumb, safe="")), self.timeout_s)
            if st in (401, 403) and attempt == 0:
                continue                                    # stale crumb/cookie: refresh once
            if st == 404:
                raise SourceError(f"Yahoo: sem perfil para {symbol}")
            if st != 200:
                raise SourceError(f"Yahoo respondeu HTTP {st}")
            try:
                data = json.loads(body)
            except ValueError as e:
                raise SourceError("Yahoo: resposta não é JSON") from e
            qs = data.get("quoteSummary") if isinstance(data, dict) else None
            res = (qs.get("result") or [None])[0] if isinstance(qs, dict) else None
            if not isinstance(res, dict):
                raise SourceError("Yahoo: resposta sem resultado")
            return res
        raise SourceError("Yahoo: acesso negado mesmo após renovar a autorização")


def _num(d: object, key: str) -> float | None:
    v = d.get(key) if isinstance(d, dict) else None
    if isinstance(v, dict):
        v = v.get("raw")
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v) if math.isfinite(v) else None


def _ratio(d: object, key: str) -> float | None:
    """A fraction; when the source's formatted value is a percentage, that value wins."""
    v = d.get(key) if isinstance(d, dict) else None
    if isinstance(v, dict) and isinstance(v.get("fmt"), str) and v["fmt"].strip().endswith("%"):
        try:
            return float(v["fmt"].strip().rstrip("%").replace(",", "")) / 100.0
        except ValueError:
            pass
    return _num(d, key)


def parse_summary(res: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """(profile fields, fundamentals) from a quoteSummary result; missing values are omitted."""
    ap = res.get("assetProfile") or {}
    fd = res.get("financialData") or {}
    ks = res.get("defaultKeyStatistics") or {}
    sd = res.get("summaryDetail") or {}

    def s(d: object, k: str) -> str | None:
        v = d.get(k) if isinstance(d, dict) else None
        return v.strip() if isinstance(v, str) and v.strip() else None

    emp = _num(ap, "fullTimeEmployees")
    profile = {"setor_en": s(ap, "sector"), "industria_en": s(ap, "industry"),
               "resumo_en": s(ap, "longBusinessSummary"), "cidade": s(ap, "city"), "pais": s(ap, "country"),
               "site": s(ap, "website"), "funcionarios": int(emp) if emp else None}
    de = _num(fd, "debtToEquity")
    mrq = _num(ks, "mostRecentQuarter")
    rec = fd.get("recommendationKey") if isinstance(fd, dict) else None
    f = {
        "preco_atual": _num(fd, "currentPrice"),
        "receita_12m": _num(fd, "totalRevenue"),
        "crescimento_receita": _ratio(fd, "revenueGrowth"),
        "crescimento_lucro": _ratio(fd, "earningsGrowth"),
        "margem_bruta": _ratio(fd, "grossMargins"),
        "margem_operacional": _ratio(fd, "operatingMargins"),
        "margem_liquida": _ratio(fd, "profitMargins"),
        "roe": _ratio(fd, "returnOnEquity"),
        "divida_patrimonio": de / 100.0 if de is not None else None,     # Yahoo reports it in percent
        "fluxo_caixa_livre": _num(fd, "freeCashflow"),
        "pl": _num(sd, "trailingPE"),
        "pl_projetado": _num(sd, "forwardPE") if _num(sd, "forwardPE") is not None else _num(ks, "forwardPE"),
        "dividend_yield": _ratio(sd, "dividendYield"),
        "payout": _ratio(sd, "payoutRatio"),
        "valor_mercado": _num(sd, "marketCap"),
        "preco_alvo_medio": _num(fd, "targetMeanPrice"),
        "n_analistas": _num(fd, "numberOfAnalystOpinions"),
        "recomendacao": rec if rec in REC_PT else None,
        "moeda_balanco": s(fd, "financialCurrency"),
        "moeda_cotacao": s(sd, "currency"),
        "ultimo_trimestre": datetime.fromtimestamp(mrq, UTC).date().isoformat() if mrq else None,
    }
    return profile, {k: v for k, v in f.items() if v is not None}


# ----------------------------------------------------------------------------- profile
@dataclass
class CompanyProfile:
    ticker: str
    nome: str
    descricao: str | None = None
    descricao_fonte: str | None = None
    segmento: Segment | None = None
    segmento_fonte: str | None = None
    atividade_b3: str | None = None
    razao_social: str | None = None
    classificacao: str | None = None
    classificacao_fonte: str | None = None
    resumo_en: str | None = None
    sede: str | None = None
    funcionarios: int | None = None
    site: str | None = None
    fundamentos: dict[str, object] = field(default_factory=dict)
    fundamentos_fonte: str | None = None
    pais: str = "BR"
    fontes: list[str] = field(default_factory=list)
    indisponivel: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        seg = self.segmento
        d["segmento"] = None if seg is None else {
            "chave": seg.key, "rotulo": seg.rotulo, "como_ganha_dinheiro": seg.como_ganha}
        return d


_yahoo_default: YahooFundamentals | None = None


def _yahoo_client(fetch: FetchFn | None) -> YahooFundamentals:
    global _yahoo_default
    if fetch is not None:
        return YahooFundamentals(fetch)
    if _yahoo_default is None:
        _yahoo_default = YahooFundamentals(cookie_fetch)
    return _yahoo_default


def _cache_file(cache_dir: Path | None, ticker: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / "perfis" / (re.sub(r"[^A-Za-z0-9._-]", "_", ticker) + ".json")


def _read_cache(path: Path | None, now: float) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and now - float(d.get("t", 0)) < PROFILE_TTL_S else None


def _write_cache(path: Path | None, d: dict[str, object]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    except OSError as e:                                  # read-only disk: run without cache
        log.warning("perfil não gravado em cache: %s", e)


def _fetch_sources(res: Resolved, fetch: FetchFn | None, timeout_s: float, today: str) -> dict[str, object]:
    http = fetch or cookie_fetch
    b3_code = res.ticker[:-3][:4] if res.ticker.endswith(".SA") else None
    out: dict[str, object] = {"consultado_em": today, "b3": None, "yahoo": None, "erros": {}}

    def b3() -> dict[str, str | None]:
        return fetch_b3_company(b3_code, http, timeout_s)  # type: ignore[arg-type]

    def yahoo() -> dict[str, object]:
        prof, fund = parse_summary(_yahoo_client(fetch).summary(res.ticker))
        return {"perfil": prof, "fundamentos": fund}

    jobs = {"yahoo": yahoo}
    if b3_code is not None:
        jobs["b3"] = b3
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result()
            except (SourceError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                out["erros"][name] = str(e) or type(e).__name__  # type: ignore[index]
    return out


def load_company_profile(res: Resolved, offline: bool = False, fetch: FetchFn | None = None,
                         cache_dir: Path | None = None, timeout_s: float = TIMEOUT_S,
                         now: float | None = None) -> CompanyProfile:
    """Assemble the profile from the catalogue and, unless offline, from B3 and Yahoo Finance.

    ``fetch`` is the injectable HTTP GET used for every public source (tests pass a fake); the
    default keeps cookies, as Yahoo requires."""
    asset = res.asset
    country = asset.country if asset else ("BR" if res.ticker.endswith(".SA") or res.ticker.startswith("SINT_")
                                           else "US")
    p = CompanyProfile(res.ticker, res.name, pais="BR" if country == "BR" else "US")
    synthetic = res.ticker.startswith("SINT_")
    if res.business:
        p.descricao = res.business
        if synthetic:
            p.descricao_fonte = "catálogo sintético (empresa fictícia)"
        elif asset is None and res.listed is not None:
            p.descricao_fonte = "Nasdaq (setor e indústria da listagem)"
        else:
            p.descricao_fonte = "catálogo (curadoria factual)"
        p.fontes.append(p.descricao_fonte)
    seg = segment_for_class(asset.macro_class, asset.asset_class, asset.business) if asset else None
    seg_src = "classe do ativo" if seg else None
    if seg is None and res.ticker.endswith("-USD"):
        seg, seg_src = SEGMENTS["cripto"], "código do ativo"
    if seg is None and res.listed is not None and res.listed.kind in ("etf", "fii"):
        seg, seg_src = SEGMENTS["etf_generico" if res.listed.kind == "etf" else "fii_generico"], "listagem"

    data: dict[str, object] = {}
    if synthetic or offline:
        p.indisponivel.append("modo sintético: sem dados reais de empresa (B3/Yahoo não consultados)")
    elif res.ticker.endswith("-USD"):
        p.indisponivel.append("criptoativo: não há empresa nem demonstrações financeiras")
    else:
        t_now = time.time() if now is None else now
        path = _cache_file(cache_dir, res.ticker)
        cached = _read_cache(path, t_now)
        if cached is not None:
            data = cached
        else:
            today = datetime.now(ZoneInfo("America/Sao_Paulo")).date().isoformat()
            data = _fetch_sources(res, fetch, timeout_s, today)
            if data.get("b3") or data.get("yahoo"):
                _write_cache(path, {**data, "t": t_now})
    when = data.get("consultado_em", "")
    b3 = data.get("b3") if isinstance(data.get("b3"), dict) else None
    yh = data.get("yahoo") if isinstance(data.get("yahoo"), dict) else None
    errs = data.get("erros") if isinstance(data.get("erros"), dict) else {}
    if b3:
        src = f"B3 — empresas listadas (consultado em {when})"
        p.fontes.append(src)
        p.atividade_b3 = b3.get("atividade")  # type: ignore[assignment]
        p.razao_social = b3.get("razao_social")  # type: ignore[assignment]
        p.site = b3.get("site")  # type: ignore[assignment]
        if b3.get("classificacao"):
            p.classificacao, p.classificacao_fonte = b3["classificacao"], "B3"  # type: ignore[assignment]
    elif "b3" in errs:
        p.indisponivel.append(f"B3: {errs['b3']}")
    prof: dict[str, object] = {}
    if yh:
        prof = yh.get("perfil") or {}  # type: ignore[assignment]
        p.fundamentos = dict(yh.get("fundamentos") or {})  # type: ignore[arg-type]
        q = p.fundamentos.get("ultimo_trimestre")
        p.fundamentos_fonte = (f"Yahoo Finance (consultado em {when}" + (f"; último trimestre reportado: {q})"
                                                                          if q else ")"))
        p.fontes.append(f"Yahoo Finance — perfil e indicadores (consultado em {when})")
        p.resumo_en = _shorten(prof.get("resumo_en"))  # type: ignore[arg-type]
        city, ctry = prof.get("cidade"), prof.get("pais")
        ctry = COUNTRY_PT.get(str(ctry).lower(), ctry) if ctry else None
        p.sede = ", ".join(str(x) for x in (city, ctry) if x) or None
        p.funcionarios = prof.get("funcionarios")  # type: ignore[assignment]
        p.site = p.site or prof.get("site")  # type: ignore[assignment]
        if not p.classificacao and (prof.get("setor_en") or prof.get("industria_en")):
            sec = str(prof.get("setor_en") or "")
            parts = [SECTOR_PT.get(sec.lower(), sec), str(prof.get("industria_en") or "")]
            p.classificacao = " › ".join(x for x in parts if x)
            p.classificacao_fonte = "Yahoo Finance (indústria em inglês, como publicada)"
        if not p.fundamentos:
            p.indisponivel.append("Yahoo Finance: nenhum indicador fundamentalista publicado para este ativo")
    elif "yahoo" in errs:
        p.indisponivel.append(f"Yahoo Finance: {errs['yahoo']}")
    lst = res.listed
    if not p.classificacao and lst is not None and (lst.sector or lst.industry):
        p.classificacao = " › ".join(x for x in (SECTOR_PT.get(lst.sector.lower(), lst.sector), lst.industry) if x)
        p.classificacao_fonte = "Nasdaq (indústria em inglês, como publicada)"
    if seg is None:
        texts = [("catálogo", res.business if asset is not None or res.listed is None else None),
                 ("B3 — atividade registrada", p.atividade_b3), ("B3 — classificação setorial", p.classificacao
                                                                 if p.classificacao_fonte == "B3" else None),
                 ("Yahoo Finance — indústria", prof.get("industria_en")),  # type: ignore[list-item]
                 ("Nasdaq — indústria", lst.industry if lst else None),
                 ("Yahoo Finance — setor", prof.get("setor_en")),  # type: ignore[list-item]
                 ("Nasdaq — setor", lst.sector if lst else None)]
        found = classify_segment(texts)
        if found is not None:
            seg, seg_src = found
    p.segmento, p.segmento_fonte = seg, seg_src
    if p.descricao is None and p.atividade_b3:
        p.descricao, p.descricao_fonte = p.atividade_b3, "B3 — atividade registrada"
    if p.descricao is None and seg is not None:
        p.descricao, p.descricao_fonte = f"empresa do segmento: {seg.rotulo.lower()}", f"segmento ({seg_src})"
    if seg is None:
        p.indisponivel.append("segmento não identificado nas fontes: motivos do setor omitidos")
    return p


def _shorten(text: str | None, n: int = 600) -> str | None:
    if not text:
        return None
    t = " ".join(text.split())
    if len(t) <= n:
        return t
    cut = t[:n]
    end = cut.rfind(". ")
    return cut[:end + 1] if end > n // 2 else cut.rstrip() + "…"


def basic_profile(res: Resolved, why: str) -> CompanyProfile:
    """Profile from the catalogue only (used when the online sources could not be awaited)."""
    p = load_company_profile(res, offline=True)
    p.indisponivel = [why]
    return p


# ----------------------------------------------------------------------------- reasons
def _money(x: float, ccy: str) -> str:
    sym = SYM.get(ccy, ccy)
    a = abs(x)
    if a >= 1e9:
        return f"{sym} {x / 1e9:.1f} bi".replace(".", ",")
    if a >= 1e6:
        return f"{sym} {x / 1e6:.1f} mi".replace(".", ",")
    return f"{sym} {x:,.0f}".replace(",", ".")


def growth_case(p: CompanyProfile, st: dict[str, object], pj: dict[str, object], rf: float, rf_name: str,
                ccy: str, hist_source: str, rf_source: str = "taxa livre de risco usada neste relatório"
                ) -> dict[str, list[dict[str, str]]]:
    """Reasons the asset may grow / may not grow as much, each with its basis and source."""
    up: list[dict[str, str]] = []
    down: list[dict[str, str]] = []

    def add(side: list[dict[str, str]], text: str, base: str, fonte: str) -> None:
        side.append({"texto": text, "base": base, "rotulo_base": BASE_LABEL[base], "fonte": fonte})

    f = p.fundamentos
    fs = p.fundamentos_fonte or "Yahoo Finance"
    fin = p.segmento is not None and p.segmento.financeiro
    bal = str(f.get("moeda_balanco") or ccy)
    sym = SYM.get(bal, bal)

    def num(k: str) -> float | None:
        v = f.get(k)
        return float(v) if isinstance(v, int | float) else None

    g = num("crescimento_receita")
    if g is not None:
        if g >= 0.10:
            add(up, f"Receita cresceu {g:.0%} sobre o mesmo trimestre do ano anterior.", "fundamentos", fs)
        elif g < 0:
            add(down, f"Receita caiu {abs(g):.0%} sobre o mesmo trimestre do ano anterior.", "fundamentos", fs)
        elif g < 0.03:
            add(down, f"Receita praticamente parada ({g:+.1%} sobre o mesmo trimestre do ano anterior).",
                "fundamentos", fs)
    e = num("crescimento_lucro")
    if e is not None:
        note = " (a base de comparação pode exagerar a variação)" if abs(e) > 1 else ""
        if e >= 0.15:
            add(up, f"Lucro cresceu {e:.0%} sobre o mesmo trimestre do ano anterior{note}.", "fundamentos", fs)
        elif e < 0:
            add(down, f"Lucro caiu {abs(e):.0%} sobre o mesmo trimestre do ano anterior{note}.", "fundamentos", fs)
    m = num("margem_liquida")
    if m is not None and not fin:
        if m < 0:
            add(down, f"Teve prejuízo nos últimos 12 meses (margem líquida de {m:.1%}).", "fundamentos", fs)
        elif m < 0.03:
            add(down, f"Margem líquida baixa ({m:.1%}): de cada {sym} 100 vendidos sobram cerca de {sym} "
                      f"{m * 100:.1f} de lucro, pouca folga para imprevistos.", "fundamentos", fs)
        elif m >= 0.15:
            add(up, f"Margem líquida alta ({m:.0%}): de cada {sym} 100 vendidos sobram cerca de {sym} {m * 100:.0f} "
                    "de lucro, o que dá fôlego para investir.", "fundamentos", fs)
    roe = num("roe")
    if roe is not None:
        if roe >= max(0.15, rf + 0.03):
            add(up, f"Rentabilidade alta: retorno sobre o patrimônio (ROE) de {roe:.0%} ao ano; lucro reinvestido a "
                    "essa taxa tende a gerar valor.", "fundamentos", fs)
        elif roe < rf:
            add(down, f"Retorno sobre o patrimônio (ROE) de {roe:.0%} ao ano, abaixo do {rf_name} ({rf:.1%}): o "
                      "negócio rende menos que a renda fixa.", "fundamentos", fs)
    de = num("divida_patrimonio")
    if de is not None and not fin:
        if de > 1.5:
            add(down, f"Dívida alta: {de:.1f} vezes o patrimônio; os juros consomem caixa e reduzem a folga para "
                      "expandir.", "fundamentos", fs)
        elif de < 0.3:
            add(up, f"Endividamento baixo ({de:.2f} vez o patrimônio): há folga para investir sem depender de "
                    "crédito caro.", "fundamentos", fs)
    pe, fpe = num("pl"), num("pl_projetado")
    if pe is not None and pe > 0:
        if pe > 35:
            add(down, f"Preço alto em relação ao lucro (P/L de {pe:.0f}): o mercado já espera muito crescimento, e "
                      "resultados abaixo do esperado costumam provocar quedas.", "fundamentos", fs)
        elif pe < 8:
            add(up, f"Preço baixo em relação ao lucro (P/L de {pe:.1f}): se o lucro se mantiver, há espaço para "
                    "valorização — mas preço baixo também pode refletir riscos que o mercado enxerga.",
                "fundamentos", fs)
        if fpe is not None and fpe > 0:
            if fpe < 0.85 * pe:
                add(up, f"Analistas projetam lucro maior nos próximos 12 meses (P/L projetado de {fpe:.1f} contra "
                        f"{pe:.1f} hoje).", "analistas", fs)
            elif fpe > 1.15 * pe:
                add(down, f"Analistas projetam lucro menor nos próximos 12 meses (P/L projetado de {fpe:.1f} contra "
                          f"{pe:.1f} hoje).", "analistas", fs)
    dy, payout = num("dividend_yield"), num("payout")
    if (dy is not None and dy >= 0.06) or (payout is not None and payout > 0.8 and not fin):
        what = f"dividend yield de {dy:.1%} ao ano" if dy is not None and dy >= 0.06 else \
            f"{payout:.0%} do lucro distribuído"
        add(down, f"Distribui muito do lucro ({what}): bom para quem busca renda, mas sobra menos para reinvestir "
                  "e crescer.", "fundamentos", fs)
    fcf, rev = num("fluxo_caixa_livre"), num("receita_12m")
    if fcf is not None and not fin:
        if fcf < 0:
            add(down, f"Fluxo de caixa livre negativo nos últimos 12 meses ({_money(fcf, bal)}): a empresa consome "
                      "caixa e pode precisar de dívida ou de novas ações.", "fundamentos", fs)
        elif rev and rev > 0 and fcf / rev >= 0.10:
            add(up, f"Gerou caixa livre equivalente a {fcf / rev:.0%} da receita nos últimos 12 meses "
                    f"({_money(fcf, bal)}).", "fundamentos", fs)
    n, tgt, price = num("n_analistas"), num("preco_alvo_medio"), num("preco_atual")
    if n is not None and n >= 3 and tgt and price:
        upside = tgt / price - 1
        qs = SYM.get(str(f.get("moeda_cotacao") or ccy), ccy)
        rec = f.get("recomendacao")
        rec_txt = f"; recomendação média: {REC_PT[rec]}" if rec in REC_PT else ""  # type: ignore[index]
        text = (f"Média de {int(n)} analistas aponta preço-alvo de {qs} {tgt:.2f}, {upside:+.0%} sobre a cotação de "
                f"{qs} {price:.2f}{rec_txt} (opinião de terceiros, não do modelo; alvos de analistas erram com "
                "frequência).")
        if upside >= 0.15:
            add(up, text, "analistas", fs)
        elif upside <= -0.05:
            add(down, text, "analistas", fs)
    for note in COMPANY_NOTES.get(p.ticker, ()):
        add(down, note, "empresa", "curadoria factual do catálogo")

    # price history --------------------------------------------------------------
    r12, above = st.get("retorno_12m"), st.get("acima_media_200d")
    if isinstance(r12, float) and above is not None:
        if r12 >= 0.10 and above:
            add(up, f"Tendência de alta: subiu {r12:.0%} em 12 meses e está acima da média dos últimos 200 pregões; "
                    "tendências costumam persistir por alguns meses, mas revertem com frequência.", "historico",
                hist_source)
        elif r12 <= -0.10 and not above:
            add(down, f"Tendência de queda: caiu {abs(r12):.0%} em 12 meses e está abaixo da média dos últimos 200 "
                      "pregões.", "historico", hist_source)
    t, ex, yrs = st.get("t_excesso"), st.get("excesso_anual"), st.get("anos_de_dados")
    if isinstance(t, float) and isinstance(ex, float) and isinstance(yrs, float):
        if t >= 1:
            add(up, f"No histórico, rendeu {ex:.1%} ao ano acima do {rf_name} em {yrs:.1f} anos (t = {t:.2f}, "
                    f"evidência {'forte' if t >= 2 else 'moderada'}).", "historico", hist_source)
        elif t <= -1:
            add(down, f"No histórico, rendeu {abs(ex):.1%} ao ano abaixo do {rf_name} em {yrs:.1f} anos "
                      f"(t = {t:.2f}).", "historico", hist_source)
        else:
            add(down, f"No histórico, o retorno não se distingue do {rf_name} (diferença de {ex:+.1%} ao ano, "
                      f"t = {t:.2f}): pode ter sido sorte ou azar.", "historico", hist_source)
    vol, dd, beta, giro = (st.get("volatilidade"), st.get("queda_maxima"), st.get("beta"),
                           st.get("giro_diario_medio"))
    if isinstance(vol, float) and vol >= 0.40:
        add(down, f"Oscila muito (volatilidade de {vol:.0%} ao ano): o resultado pode ficar bem longe da média.",
            "historico", hist_source)
    if isinstance(dd, float) and dd <= -0.50:
        add(down, f"Já caiu {abs(dd):.0%} do pico ao fundo no período analisado: quedas grandes podem se repetir.",
            "historico", hist_source)
    if isinstance(beta, float) and beta >= 1.3:
        add(down, f"Beta de {beta:.2f}: costuma amplificar os movimentos da bolsa, inclusive nas quedas.",
            "historico", hist_source)
    if isinstance(giro, float) and giro < 5e6:
        add(down, f"Liquidez baixa (giro médio de {_money(giro, ccy)} por dia): comprar e vender pode mexer no "
                  "preço.", "historico", hist_source)

    # simulation, rates, currency ----------------------------------------------------
    pb, T = float(pj["p_superar_rf"]), float(pj["horizonte_anos"])  # type: ignore[arg-type]
    sim = "simulação Monte Carlo deste relatório [S]"
    if pb >= 0.60:
        add(up, f"Na simulação, {pb:.0%} de chance de render mais que o {rf_name} em {T:g} anos.", "simulacao", sim)
    elif pb <= 0.40:
        add(down, f"Na simulação, só {pb:.0%} de chance de render mais que o {rf_name} em {T:g} anos.", "simulacao",
            sim)
    else:
        add(down, f"Na simulação, a chance de render mais que o {rf_name} em {T:g} anos é de {pb:.0%}, perto de um "
                  "cara ou coroa.", "simulacao", sim)
    if ccy == "BRL" and rf >= 0.10:
        add(down, f"Juros altos: com o {rf_name} a {rf:.1%} ao ano, a renda fixa paga bem com pouco risco, o que "
                  "eleva a régua para a ação valer a pena; juros altos também encarecem dívidas e esfriam o "
                  "consumo.", "juros", f"{rf_name}: {rf_source}")
    if ccy == "USD":
        add(down, "Cotada em dólar: para quem investe em reais, uma queda do dólar reduz o retorno (e uma alta "
                  "aumenta).", "cambio", "moeda de negociação do ativo")

    # segment ----------------------------------------------------------------------
    seg = p.segmento
    if seg is not None:
        src = f"{SECTOR_SOURCE} — {seg.rotulo}"
        for text, scope in seg.cresce:
            if applies(scope, p.pais):
                add(up, text, "setor", src)
        for text, scope in seg.limita:
            if applies(scope, p.pais):
                add(down, text, "setor", src)
    return {"pode_crescer": up, "pode_nao_crescer_tanto": down}


# ----------------------------------------------------------------------------- presentation helpers
FUND_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Receita (últimos 12 meses)", "receita_12m", "money"),
    ("Crescimento da receita (último trimestre vs. um ano antes)", "crescimento_receita", "pct"),
    ("Crescimento do lucro (último trimestre vs. um ano antes)", "crescimento_lucro", "pct"),
    ("Margem líquida", "margem_liquida", "pct"),
    ("Retorno sobre o patrimônio (ROE)", "roe", "pct"),
    ("Dívida / patrimônio", "divida_patrimonio", "x"),
    ("Preço / lucro (P/L)", "pl", "num"),
    ("P/L projetado (analistas)", "pl_projetado", "num"),
    ("Dividend yield (12 meses)", "dividend_yield", "pct"),
    ("Fluxo de caixa livre (12 meses)", "fluxo_caixa_livre", "money"),
    ("Valor de mercado", "valor_mercado", "money_quote"),
    ("Preço-alvo médio dos analistas", "preco_alvo_medio", "price"),
)


def fundamentals_table(p: CompanyProfile, ccy: str) -> list[dict[str, str]]:
    """Formatted indicator rows; a missing value is shown as 'indisponível', never estimated."""
    f = p.fundamentos
    if not f:
        return []
    bal = str(f.get("moeda_balanco") or ccy)
    quote = str(f.get("moeda_cotacao") or ccy)
    rows = []
    for label, key, kind in FUND_ROWS:
        v = f.get(key)
        if not isinstance(v, int | float):
            txt = "indisponível"
        elif kind == "money":
            txt = _money(float(v), bal)
        elif kind == "money_quote":
            txt = _money(float(v), quote)
        elif kind == "pct":
            txt = f"{float(v):.1%}"
        elif kind == "x":
            txt = f"{float(v):.2f}×"
        elif kind == "price":
            n = f.get("n_analistas")
            rec = f.get("recomendacao")
            txt = f"{SYM.get(quote, quote)} {float(v):.2f}"
            if isinstance(n, int | float):
                txt += f" ({int(n)} analistas"
                txt += f"; recomendação média: {REC_PT[rec]})" if rec in REC_PT else ")"  # type: ignore[index]
        else:
            txt = f"{float(v):.1f}"
        rows.append({"indicador": label, "valor": txt})
    return rows
