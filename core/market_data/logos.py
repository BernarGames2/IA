"""Company logos for the page: the icon each company publishes on its own website, read through a public
favicon service (Google's, by domain) and served by this app from its own address.

* Which website. For a stock researched in "Vale a pena?", the official sources give it (the B3 company
  registry or the Yahoo Finance profile). For the curated catalogue, ``B3_SITES`` (by the 4-letter root of
  the B3 code, so PETR3 and PETR4 share it) and ``US_SITES`` (by ticker) list each company's official site;
  funds and ETFs show their manager's logo (iShares, Vanguard, Kinea...). Editorial curation: a wrong domain
  would show another company's logo, so a ticker without a verified site gets none. Assets outside the
  catalogue (unless researched), the synthetic SINT_* assets and the CDI cash have no logo: the page shows
  the ticker's initials instead.
* The browser never contacts the favicon service: ``/api/logo/site/<domain>`` fetches the icon on the
  server (fixed host; the domain is validated and only passed as a parameter), accepts only raster images
  (PNG, JPEG, GIF, WEBP, ICO; no SVG, which could carry scripts), keeps it in memory for a week and answers
  404 when there is no usable icon (the page then shows the initials).
* Logos are trademarks of their owners, shown only to identify the asset; showing one implies no
  relationship with the company and is not a recommendation.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable

FetchFn = Callable[[str, float], tuple[int, bytes]]

FAVICON_URL = "https://www.google.com/s2/favicons?domain={domain}&sz=128"
TIMEOUT_S = 5.0
MAX_BYTES = 200_000
MIN_SIDE_PX = 24             # a 16 px icon looks worse than the initials at the sizes the page uses
HIT_TTL_S = 7 * 24 * 3600
MISS_TTL_S = 3600
CACHE_SIZE = 512

# Official sites (company, or the fund/ETF manager), by the 4-letter root of the B3 code.
B3_SITES: dict[str, str] = {
    # bancos, seguros e bolsa
    "ITUB": "itau.com.br", "BBDC": "bradesco.com.br", "BBAS": "bb.com.br", "SANB": "santander.com.br",
    "BPAC": "btgpactual.com", "ITSA": "itausa.com.br", "B3SA": "b3.com.br", "BBSE": "bbseguridade.com.br",
    "PSSA": "portoseguro.com.br", "CXSE": "caixaseguridade.com.br", "IRBR": "irbre.com",
    # petróleo, gás e combustíveis
    "PETR": "petrobras.com.br", "PRIO": "prio3.com.br", "RECV": "petroreconcavo.com.br", "UGPA": "ultra.com.br",
    "VBBR": "vibraenergia.com.br", "CSAN": "cosan.com.br", "ENEV": "eneva.com.br",
    # mineração, siderurgia, papel e madeira
    "VALE": "vale.com", "GGBR": "gerdau.com", "CSNA": "csn.com.br", "USIM": "usiminas.com",
    "CMIN": "csnmineracao.com.br", "BRAP": "bradespar.com.br", "SUZB": "suzano.com.br", "KLBN": "klabin.com.br",
    "DXCO": "dexco.com.br",
    # energia elétrica e saneamento
    "EGIE": "engie.com.br", "ELET": "eletrobras.com", "AURE": "aurenenergia.com.br",
    "EQTL": "equatorialenergia.com.br", "CPFE": "cpfl.com.br", "NEOE": "neoenergia.com", "ENGI": "energisa.com.br",
    "CMIG": "cemig.com.br", "TAEE": "taesa.com.br", "ALUP": "alupar.com.br", "ISAE": "isaenergiabrasil.com.br",
    "SBSP": "sabesp.com.br", "SAPR": "sanepar.com.br", "CSMG": "copasa.com.br",
    # indústria, consumo, varejo e agro
    "WEGE": "weg.net", "ABEV": "ambev.com.br", "MGLU": "magazineluiza.com.br", "LREN": "lojasrenner.com.br",
    "ASAI": "assai.com.br", "GMAT": "grupomateus.com.br", "VIVA": "vivara.com.br", "AZZA": "azzas2154.com.br",
    "MDIA": "mdiasbranco.com.br", "SMTO": "saomartinho.com.br", "SLCE": "slcagricola.com.br",
    "BEEF": "minervafoods.com",
    # saúde
    "RADL": "rdsaude.com.br", "FLRY": "fleury.com.br", "RDOR": "rededorsaoluiz.com.br", "HAPV": "hapvida.com.br",
    "HYPE": "hypera.com.br", "PGMN": "paguemenos.com.br", "PNVL": "panvel.com", "ODPV": "odontoprev.com.br",
    "QUAL": "qualicorp.com.br",
    # educação e tecnologia
    "COGN": "cogna.com.br", "YDUQ": "yduqs.com.br", "CSED": "cruzeirodosul.edu.br", "ANIM": "animaeducacao.com.br",
    "SEER": "sereducacional.com", "TOTS": "totvs.com", "LWSA": "locaweb.com.br",
    "POSI": "positivotecnologia.com.br", "INTB": "intelbras.com",
    # logística, serviços e telecomunicações
    "RAIL": "rumolog.com", "ECOR": "ecorodovias.com.br", "RENT": "localiza.com", "VAMO": "vamos.com.br",
    "VIVT": "vivo.com.br", "TIMS": "tim.com.br",
    # construção e shoppings
    "CYRE": "cyrela.com.br", "MRVE": "mrv.com.br", "DIRR": "direcional.com.br", "CURY": "cury.net",
    "EZTC": "eztec.com.br", "MULT": "multiplan.com.br", "ALOS": "allos.co", "IGTI": "iguatemi.com.br",
    # fundos imobiliários e ETFs listados na B3: logo da gestora
    "KNRI": "kinea.com.br", "KNCR": "kinea.com.br", "KNIP": "kinea.com.br", "XPML": "xpi.com.br",
    "XPLG": "xpi.com.br", "VISC": "vincipartners.com", "BTLG": "btgpactual.com", "BOVA": "ishares.com",
    "IVVB": "ishares.com", "SMAL": "ishares.com", "IMAB": "ishares.com", "IRFM": "ishares.com",
    "DIVO": "itnow.com.br", "B5P2": "itnow.com.br", "HASH": "hashdex.com",
}
# Official sites by US ticker (ETFs: the manager; crypto: the project).
US_SITES: dict[str, str] = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "NVDA": "nvidia.com", "GOOGL": "google.com", "GOOG": "google.com",
    "AMZN": "amazon.com", "META": "meta.com", "TSLA": "tesla.com", "AVGO": "broadcom.com", "TSM": "tsmc.com",
    "ASML": "asml.com", "AMD": "amd.com", "ORCL": "oracle.com", "CRM": "salesforce.com", "ADBE": "adobe.com",
    "CSCO": "cisco.com", "QCOM": "qualcomm.com", "INTC": "intel.com", "IBM": "ibm.com", "NFLX": "netflix.com",
    "UBER": "uber.com", "SHOP": "shopify.com", "BKNG": "bookingholdings.com", "ABNB": "airbnb.com",
    "PYPL": "paypal.com", "BRK-B": "berkshirehathaway.com", "BRK-A": "berkshirehathaway.com",
    "JPM": "jpmorganchase.com", "V": "visa.com", "MA": "mastercard.com", "BAC": "bankofamerica.com",
    "WFC": "wellsfargo.com", "GS": "goldmansachs.com", "MS": "morganstanley.com", "AXP": "americanexpress.com",
    "BLK": "blackrock.com", "LLY": "lilly.com", "UNH": "unitedhealthgroup.com", "NVO": "novonordisk.com",
    "ABBV": "abbvie.com", "MRK": "merck.com", "PFE": "pfizer.com", "TMO": "thermofisher.com", "ABT": "abbott.com",
    "AMGN": "amgen.com", "ISRG": "intuitive.com", "MDT": "medtronic.com", "GILD": "gilead.com",
    "CVS": "cvshealth.com", "JNJ": "jnj.com", "DUOL": "duolingo.com", "COUR": "coursera.org",
    "LRN": "stridelearning.com", "NEE": "nexteraenergy.com", "FSLR": "firstsolar.com", "ENPH": "enphase.com",
    "DUK": "duke-energy.com", "SO": "southerncompany.com", "XOM": "exxonmobil.com", "CVX": "chevron.com",
    "PG": "pg.com", "KO": "coca-colacompany.com", "PEP": "pepsico.com", "WMT": "walmart.com", "COST": "costco.com",
    "HD": "homedepot.com", "MCD": "mcdonalds.com", "SBUX": "starbucks.com", "NKE": "nike.com", "DIS": "disney.com",
    "CAT": "caterpillar.com", "DE": "deere.com", "HON": "honeywell.com", "GE": "geaerospace.com",
    "BA": "boeing.com", "LMT": "lockheedmartin.com", "UPS": "ups.com", "O": "realtyincome.com",
    "PLD": "prologis.com", "AMT": "americantower.com", "EQIX": "equinix.com",
    "QQQ": "invesco.com", "TAN": "invesco.com", "VOO": "vanguard.com", "VTI": "vanguard.com", "VYM": "vanguard.com",
    "VNQ": "vanguard.com", "BND": "vanguard.com", "SPY": "ssga.com", "XLV": "ssga.com", "XLK": "ssga.com",
    "XLE": "ssga.com", "XLF": "ssga.com", "XLU": "ssga.com", "GLD": "spdrgoldshares.com", "SCHD": "schwab.com",
    "ARKK": "ark-invest.com", "ICLN": "ishares.com", "AGG": "ishares.com", "IWM": "ishares.com", "EFA": "ishares.com",
    "EEM": "ishares.com", "IBB": "ishares.com", "SHY": "ishares.com", "IEF": "ishares.com", "TLT": "ishares.com",
    "TIP": "ishares.com", "LQD": "ishares.com", "HYG": "ishares.com", "SLV": "ishares.com",
    "BTC-USD": "bitcoin.org", "ETH-USD": "ethereum.org",
}
# Hosts a source may give as a "website" that are not the company's own (investor-relations platforms,
# social networks): their icon would be someone else's logo.
NOT_COMPANY_HOSTS = ("mzweb.com.br", "mziq.com", "mzgroup.com", "q4cdn.com", "q4inc.com", "q4web.com",
                     "linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
                     "wixsite.com", "wordpress.com", "blogspot.com", "github.io")

_DOMAIN = re.compile(r"(?=.{4,100}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}")
_B3_CODE = re.compile(r"([A-Z][A-Z0-9]{3})\d{1,2}F?(?:\.SA)?")


def clean_domain(site: str | None) -> str | None:
    """'https://www.petrobras.com.br/ri' -> 'petrobras.com.br'; None when it is not a usable company domain."""
    if not site or not isinstance(site, str):
        return None
    s = site.strip().lower()
    try:
        host = urllib.parse.urlsplit(s if "://" in s else "http://" + s).hostname or ""
    except ValueError:
        return None
    host = host.removeprefix("www.").rstrip(".")
    if not _DOMAIN.fullmatch(host):
        return None
    if any(host == h or host.endswith("." + h) for h in NOT_COMPANY_HOSTS):
        return None
    return host


def domain_for(ticker: str, site: str | None = None) -> str | None:
    """The company's domain: the curated list first (verified), then the site the official sources gave."""
    t = (ticker or "").strip().upper()
    if t in US_SITES:
        return US_SITES[t]
    m = _B3_CODE.fullmatch(t)
    if m and m.group(1) in B3_SITES:
        return B3_SITES[m.group(1)]
    return clean_domain(site)


def logo_path(ticker: str, site: str | None = None) -> str | None:
    """Same-origin address of the logo, or None (the page then shows the ticker's initials)."""
    d = domain_for(ticker, site)
    return f"/api/logo/site/{d}" if d else None


def sniff_image(body: bytes) -> str | None:
    """Media type of a raster image from its first bytes; None for anything else (including SVG)."""
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if body.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if body.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    if body[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"
    return None


def _png_side(body: bytes) -> int | None:
    if len(body) >= 24 and body[12:16] == b"IHDR":
        return min(int.from_bytes(body[16:20], "big"), int.from_bytes(body[20:24], "big"))
    return None


def image_fetch(url: str, timeout_s: float) -> tuple[int, bytes]:
    """HTTP GET for an image with the standard library (redirects followed; HTTP errors returned)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (QPI logo fetch)",
                                               "Accept": "image/png,image/*;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return int(r.status), r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        return int(e.code), b""


_lock = threading.Lock()
_cache: OrderedDict[str, tuple[float, tuple[bytes, str] | None]] = OrderedDict()


def fetch_logo(domain: str, fetch: FetchFn | None = None, now: Callable[[], float] = time.time
               ) -> tuple[bytes, str] | None:
    """(image bytes, media type) of the domain's icon, or None when there is no usable one. Cached in memory
    (a week; an hour for a miss). Network failures count as a miss for this request only (not cached)."""
    if not _DOMAIN.fullmatch(domain or ""):
        return None
    t = now()
    with _lock:
        hit = _cache.get(domain)
        if hit is not None and t - hit[0] < (HIT_TTL_S if hit[1] is not None else MISS_TTL_S):
            _cache.move_to_end(domain)
            return hit[1]
    try:
        st, body = (fetch or image_fetch)(FAVICON_URL.format(domain=urllib.parse.quote(domain, safe="")), TIMEOUT_S)
    except (OSError, ValueError):
        return None
    out: tuple[bytes, str] | None = None
    mime = sniff_image(body) if st == 200 and body and len(body) <= MAX_BYTES else None
    if mime is not None:
        side = _png_side(body) if mime == "image/png" else None
        if side is None or side >= MIN_SIDE_PX:
            out = (body, mime)
    with _lock:
        _cache[domain] = (t, out)
        _cache.move_to_end(domain)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def clear_cache() -> None:
    with _lock:
        _cache.clear()
