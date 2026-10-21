"""Company logos: which site each asset's logo comes from, the same-origin endpoint (fetch, validation, cache,
fallback) and the logo addresses in the stock-check and portfolio results. No network: the favicon service is a
test double."""

from __future__ import annotations

import json
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from core.market_data import logos
from core.market_data.logos import (
    B3_SITES,
    US_SITES,
    clean_domain,
    domain_for,
    fetch_logo,
    logo_path,
    sniff_image,
)
from core.purpose.catalog import REAL_CATALOG
from core.purpose.engine import PurposeRequest, run_purpose
from core.purpose.report import to_dict
from core.stock_check.analyzer import analyze_stock
from core.stock_check.analyzer import to_dict as stock_dict
from tests.market_doubles import Prov, http, structured_market
from tests.test_company import Router, _Prov, _series
from webapp import server


def _png(side: int) -> bytes:
    """A valid PNG of side x side pixels (one colour)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * side for _ in range(side))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Icons:
    """Fake favicon service: a 64 px icon for known domains, 404 for the others; counts calls."""

    def __init__(self, known=("petrobras.com.br", "apple.com", "paguemenos.com.br"), body=None, status=200):
        self.known, self.body, self.status, self.calls = set(known), body, status, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if not url.startswith("https://www.google.com/s2/favicons?domain="):
            raise AssertionError(f"unexpected host: {url}")
        domain = url.split("domain=", 1)[1].split("&", 1)[0]
        if domain not in self.known:
            return 404, _png(16)
        return self.status, self.body if self.body is not None else _png(64)


@pytest.fixture(autouse=True)
def _fresh_cache():
    logos.clear_cache()
    yield
    logos.clear_cache()


# ----------------------------------------------------------------------------- which site
@pytest.mark.parametrize(("ticker", "domain"), [("PETR4.SA", "petrobras.com.br"), ("PETR3", "petrobras.com.br"),
                                                ("PETR4F", "petrobras.com.br"), ("ITUB4.SA", "itau.com.br"),
                                                ("BOVA11.SA", "ishares.com"), ("KNRI11.SA", "kinea.com.br"),
                                                ("AAPL", "apple.com"), ("BRK-B", "berkshirehathaway.com"),
                                                ("BTC-USD", "bitcoin.org"), ("SPY", "ssga.com")])
def test_catalogue_companies_have_their_official_site(ticker, domain):
    assert domain_for(ticker) == domain and logo_path(ticker) == f"/api/logo/site/{domain}"


@pytest.mark.parametrize("ticker", ["SINT_PETRO4", "CAIXA_CDI", "ZZZZ3.SA", "XYZQ", "", "AAPL34.SA"])
def test_assets_without_a_verified_site_have_no_logo(ticker):
    assert logo_path(ticker) is None


@pytest.mark.parametrize(("site", "domain"), [("https://www.petrobras.com.br/ri", "petrobras.com.br"),
                                              ("www.paguemenos.com.br", "paguemenos.com.br"),
                                              ("HTTP://WWW.Vale.COM/", "vale.com"), ("ri.empresa.com.br", "ri.empresa.com.br"),
                                              ("http://169.254.169.254/latest", None), ("http://localhost:8000", None),
                                              ("javascript:alert(1)", None), ("https://ri.mziq.com/x", None),
                                              ("https://www.linkedin.com/company/x", None), ("", None), (None, None)])
def test_source_sites_are_reduced_to_company_domains(site, domain):
    assert clean_domain(site) == domain


def test_curated_sites_are_valid_domains_and_cover_the_catalogue():
    for d in [*B3_SITES.values(), *US_SITES.values()]:
        assert clean_domain(d) == d
    us = [a for a in REAL_CATALOG if a.macro_class in ("acoes_eua", "etf_eua")]
    br = [a for a in REAL_CATALOG if a.macro_class == "acoes_br"]
    assert all(logo_path(a.ticker) for a in us)
    assert sum(1 for a in br if logo_path(a.ticker)) == len(br)


def test_the_curated_list_wins_over_a_source_site_and_a_source_site_fills_the_gaps():
    assert domain_for("PETR4.SA", "https://other.example.com") == "petrobras.com.br"
    assert domain_for("ZZZZ3.SA", "https://www.zzzz.com.br") == "zzzz.com.br"


# ----------------------------------------------------------------------------- fetching the icon
def test_icon_is_fetched_validated_and_cached():
    icons = Icons()
    img = fetch_logo("petrobras.com.br", icons)
    assert img is not None and img[1] == "image/png" and img[0].startswith(b"\x89PNG")
    assert fetch_logo("petrobras.com.br", icons) == img and len(icons.calls) == 1        # served from memory


def test_missing_tiny_and_unsafe_icons_are_refused():
    assert fetch_logo("nada.com.br", Icons()) is None                                 # 404 from the service
    assert fetch_logo("apple.com", Icons(body=_png(16))) is None                     # a 16 px icon: initials instead
    logos.clear_cache()
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    assert fetch_logo("apple.com", Icons(body=svg)) is None and sniff_image(svg) is None
    logos.clear_cache()
    assert fetch_logo("apple.com", Icons(body=b"\x89PNG\r\n\x1a\n" + b"0" * 300_000)) is None    # too large
    assert fetch_logo("169.254.169.254", Icons()) is None


def test_a_miss_is_retried_after_an_hour_and_a_network_failure_is_not_cached():
    icons, t = Icons(), [1_000.0]
    assert fetch_logo("nada.com.br", icons, now=lambda: t[0]) is None
    assert fetch_logo("nada.com.br", icons, now=lambda: t[0]) is None and len(icons.calls) == 1
    t[0] += logos.MISS_TTL_S + 1
    fetch_logo("nada.com.br", icons, now=lambda: t[0])
    assert len(icons.calls) == 2

    def down(url, timeout):
        raise OSError("rede fora do ar")

    assert fetch_logo("apple.com", down) is None
    assert fetch_logo("apple.com", Icons()) is not None                              # tried again, not a cached miss


# ----------------------------------------------------------------------------- endpoint
@pytest.fixture
def client(monkeypatch):
    icons = Icons()
    monkeypatch.setattr(server, "_logo_fetch", lambda: icons)
    return TestClient(server.app), icons


def test_logo_endpoint_serves_the_icon_from_the_apps_own_address(client):
    c, icons = client
    r = c.get("/api/logo/site/petrobras.com.br")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "s-maxage" in r.headers["cache-control"] and r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == "default-src 'none'"
    assert c.get("/api/logo/ativo/PETR4").status_code == 200 and len(icons.calls) == 1   # same domain, cached
    assert c.get("/api/logo/ativo/AAPL").status_code == 200


def test_logo_endpoint_answers_404_without_calling_anyone_for_unknown_or_invalid_input(client):
    c, icons = client
    for path in ("/api/logo/ativo/ZZZZ3", "/api/logo/ativo/SINT_PETRO4", "/api/logo/site/169.254.169.254",
                 "/api/logo/site/localhost", "/api/logo/site/WWW.PETROBRAS.COM.BR", "/api/logo/site/a..b"):
        r = c.get(path)
        assert r.status_code == 404 and "max-age" in r.headers["cache-control"], path
    assert icons.calls == []
    assert c.get("/api/logo/site/nada.com.br").status_code == 404 and len(icons.calls) == 1   # the service has none


# ----------------------------------------------------------------------------- results
def test_portfolio_result_carries_the_logo_of_each_asset(tmp_path):
    req = PurposeRequest(capital_inicial=100_000, aporte_periodico=1_000, valor_alvo=500_000, horizonte_anos=10,
                         offline=False, universo="completo", max_ativos=20, n_simulacoes=500,
                         carteira_atual={"PETR4": 5_000})
    r = run_purpose(req, output_dir=tmp_path, provider=Prov(structured_market()), cache_dir=tmp_path, rf_fetch=http,
                    fast=True)
    d = json.loads(json.dumps(to_dict(r)))
    for a in d["onde_investir"]:
        assert a["logo"] == logo_path(a["ticker"])
        if a["logo"]:
            assert d["logos"][a["ticker"]] == a["logo"]
    assert d["logos"]["PETR4.SA"] == "/api/logo/site/petrobras.com.br"
    assert any(a["logo"] for a in d["onde_investir"]) and "CAIXA_CDI" not in d["logos"]


def test_synthetic_demonstration_has_no_logos(tmp_path):
    r = run_purpose(PurposeRequest(capital_inicial=10_000, aporte_periodico=1_000, valor_alvo=300_000, horizonte_anos=8,
                                   n_simulacoes=500, offline=True), output_dir=tmp_path, cache_dir=tmp_path, fast=True)
    d = to_dict(r)
    assert d["logos"] == {} and all(a["logo"] is None for a in d["onde_investir"])


def test_stock_check_logo_comes_from_the_company_site(tmp_path, monkeypatch):
    prov = _Prov({"PGMN3.SA": _series("PGMN3.SA", 1250, 7), "BOVA11.SA": _series("BOVA11.SA", 1250, 8, 0.012)})
    monkeypatch.delitem(B3_SITES, "PGMN")            # not in the curated list: the B3 registry's site is used
    r = analyze_stock("pague menos", horizon_years=3, provider=prov, rf_fetch=Router(), cache_dir=tmp_path,
                      n_paths=1500)
    assert stock_dict(r)["logo"] == "/api/logo/site/paguemenos.com.br"
    syn = analyze_stock("SINT_ENERGIA3", horizon_years=3, amount=10_000, profile="moderado", offline=True,
                        n_paths=1500)
    assert stock_dict(syn)["logo"] is None


def test_page_falls_back_to_initials():
    page = server._INDEX_HTML
    assert "function logoEl(ticker, url, size = 24)" in page and 'addEventListener("error", mono' in page
    assert "/api/logo/ativo/" in page and "withLogo(a.ticker" in page and "logoEl(d.ticker, d.logo, 40)" in page
