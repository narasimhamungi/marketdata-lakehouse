"""Unit tests for the Wikipedia-vs-N-PORT constituents cross-check."""
from datetime import date

import pandas as pd
import pytest

from src.reconcile.constituents_reconciliation import (
    cross_check_constituents,
    normalize_company_name,
)


def test_normalize_company_name_handles_common_variants():
    assert normalize_company_name("Apple Inc.") == normalize_company_name("APPLE INC")
    assert normalize_company_name("Alphabet Inc.") == normalize_company_name("ALPHABET INC CLASS A")
    assert normalize_company_name("Berkshire Hathaway Inc.") == normalize_company_name("Berkshire Hathaway")


def test_cross_check_matches_despite_naming_variants():
    wiki = pd.DataFrame(
        {
            "ticker": ["AAPL", "ACN"],
            "company_name": ["Apple Inc.", "Accenture plc"],
        }
    )
    nport = pd.DataFrame(
        {
            "name": ["Apple Inc", "Accenture PLC"],
            "title": ["Apple Inc", "Accenture PLC"],
        }
    )

    report = cross_check_constituents(wiki, nport)

    assert report.matched_count == 2
    assert len(report.unmatched_wikipedia) == 0
    assert len(report.unmatched_nport) == 0


def test_cross_check_flags_wikipedia_entry_with_no_nport_match():
    wiki = pd.DataFrame({"ticker": ["AAPL", "GHOST"], "company_name": ["Apple Inc.", "Ghost Corp"]})
    nport = pd.DataFrame({"name": ["Apple Inc"], "title": ["Apple Inc"]})

    report = cross_check_constituents(wiki, nport)

    assert len(report.unmatched_wikipedia) == 1
    assert report.unmatched_wikipedia.iloc[0]["ticker"] == "GHOST"


def test_cross_check_flags_nport_holding_with_no_wikipedia_match():
    wiki = pd.DataFrame({"ticker": ["AAPL"], "company_name": ["Apple Inc."]})
    nport = pd.DataFrame({"name": ["Apple Inc", "Unlisted Holding Co"], "title": ["Apple Inc", "Unlisted Holding Co"]})

    report = cross_check_constituents(wiki, nport)

    assert len(report.unmatched_nport) == 1
    assert report.unmatched_nport.iloc[0]["name"] == "Unlisted Holding Co"


def test_report_render_includes_counts_and_unmatched_detail():
    wiki = pd.DataFrame({"ticker": ["GHOST"], "company_name": ["Ghost Corp"]})
    nport = pd.DataFrame({"name": ["Someone Else Inc"], "title": ["Someone Else Inc"]})

    text = cross_check_constituents(wiki, nport).render()

    assert "GHOST" in text
    assert "Someone Else Inc" in text


def test_run_cross_check_raises_when_no_snapshot_exists(monkeypatch, tmp_path):
    """A missing bronze snapshot should say which command to run, not fail
    with an opaque empty-frame error downstream."""
    from src.reconcile import constituents_reconciliation as cr

    monkeypatch.setattr("src.config.BRONZE_DIR", tmp_path)

    with pytest.raises(ValueError, match="Run src.ingest.constituents first"):
        cr.run_cross_check(date(1999, 1, 1))


def test_run_cross_check_uses_snapshot_and_nport(monkeypatch, tmp_path):
    """Wiring test: reads the bronze snapshot for the given date and passes
    it, with live N-PORT holdings, into the comparison."""
    from src.reconcile import constituents_reconciliation as cr

    partition = tmp_path / "constituents" / "ingest_date=2026-09-07"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {"ticker": ["AAPL"], "company_name": ["Apple Inc."]}
    ).to_parquet(partition / "constituents.parquet")

    monkeypatch.setattr("src.config.BRONZE_DIR", tmp_path)
    monkeypatch.setattr(
        "src.ingest.nport_source.fetch_latest_nport_holdings",
        lambda: pd.DataFrame({"name": ["Apple Inc"], "title": ["Apple Inc"]}),
    )

    report = cr.run_cross_check(date(2026, 9, 7))

    assert report.wikipedia_count == 1
    assert report.matched_count == 1