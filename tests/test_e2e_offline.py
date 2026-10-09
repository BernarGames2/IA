"""End-to-end offline runs (no internet): report, tables, charts and manifest must be valid."""

import json

import pandas as pd
import pytest

import main as cli

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    code = cli.main(["--offline", "--fast", "--simulations", "2000", "--output-dir", str(out),
                     "--profile", "conservador"])
    return code, out


def test_exit_code_and_files(fast_run):
    code, out = fast_run
    assert code == 0
    reports = list((out / "reports").glob("*_report.md"))
    assert len(reports) == 1
    md = reports[0].read_text(encoding="utf-8")
    assert "DADOS SINTÉTICOS" in md and "## 14." in md
    for k in range(1, 15):
        assert f"## {k}." in md
    pngs = list((out / "charts").glob("*.png"))
    assert len(pngs) >= 6
    for p in pngs:
        assert p.read_bytes()[:8] == PNG_MAGIC and p.stat().st_size > 10_000
    man = json.loads(next((out / "reports").glob("*_manifest.json")).read_text())
    assert man["data"]["synthetic"] is True and man["seed"] == 42 and len(man["data"]["sha256_prices"]) == 64
    assert man["validated"] is True
    res = json.loads(next((out / "reports").glob("*_results.json")).read_text())
    w = res["candidates"][man["selected_portfolio"]]["weights"]
    assert abs(sum(w.values()) - 1) < 1e-6
    for csv in (out / "tables").glob("*.csv"):
        pd.read_csv(csv)


def test_reproducible_numbers(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        assert cli.main(["--offline", "--fast", "--simulations", "500", "--no-charts", "--output-dir", str(d)]) == 0
    ra = json.loads(next((a / "reports").glob("*_results.json")).read_text())
    rb = json.loads(next((b / "reports").glob("*_results.json")).read_text())
    assert ra["mc_main"]["terminal_wealth_percentiles"] == rb["mc_main"]["terminal_wealth_percentiles"]
    assert ra["candidates"] == rb["candidates"]


def test_invalid_input_exit_code(tmp_path):
    assert cli.main(["--offline", "--simulations", "10", "--output-dir", str(tmp_path)]) == 2


@pytest.mark.slow
def test_full_offline_run_with_heavy_analyses(tmp_path):
    code = cli.main(["--offline", "--simulations", "3000", "--resamples", "10", "--output-dir", str(tmp_path)])
    assert code == 0
    md = next((tmp_path / "reports").glob("*_report.md")).read_text(encoding="utf-8")
    for needle in ("Walk-forward", "Validação independente", "EVT/POT", "t-cópula", "Reverse stress",
                   "Incerteza de MODELO", "HMM"):
        assert needle in md
    assert len(list((tmp_path / "charts").glob("*.png"))) == 7
