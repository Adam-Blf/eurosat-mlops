"""The PDF report builds from the committed results and passes the layout checker."""

from __future__ import annotations

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("matplotlib")

from eurosat_mlops.report.build import build_pdf  # noqa: E402
from eurosat_mlops.report.results import fr, load, pct  # noqa: E402


def test_french_number_format():
    assert fr(97.8, 1) == "97,8"
    assert pct(0.978) == "97,80 %"


def test_results_come_from_committed_files():
    results = load()
    assert results.metrics["samples"] == int(results.confusion.sum())
    assert [r["replicas"] for r in results.scaling] == ["1", "2", "4"]
    assert results.top_confusions(1)[0][2] >= 1


def test_report_builds_and_passes_checker(tmp_path):
    pdf = build_pdf(tmp_path / "rapport.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    pytest.importorskip("pymupdf")
    import check_report

    assert check_report.check(pdf) == []
