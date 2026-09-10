"""
Unit tests for constituents ingestion.

The network call (pd.read_html against Wikipedia) is mocked — these tests
verify our parsing/reshaping/validation logic, not Wikipedia's availability.
Run this project's actual ingestion (uncommitted, live) from a machine with
open internet access; see README "Running against live data".
"""
from datetime import date

import pandas as pd
import pytest
import requests

from src.ingest import constituents as c


class _FakeResponse:
    """Stand-in for requests.Response — just enough surface for our code path."""

    def __init__(self, text: str = "<html></html>"):
        self.text = text

    def raise_for_status(self):
        pass


def _fake_wikipedia_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["AAPL", "BRK.B", "MSFT"],
            "Security": ["Apple Inc.", "Berkshire Hathaway", "Microsoft Corp."],
            "GICS Sector": ["Information Technology", "Financials", "Information Technology"],
            "GICS Sub-Industry": ["Technology Hardware", "Multi-Sector Holdings", "Systems Software"],
            "Date added": ["1976-12-12", "1957-03-04", "1994-06-01"],
        }
    )


def test_fetch_constituents_renames_and_normalizes_tickers(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.constituents.requests.get",
        lambda url, headers=None, timeout=None: _FakeResponse(),
    )
    monkeypatch.setattr(
        "src.ingest.constituents.pd.read_html",
        lambda buf: [_fake_wikipedia_table()],
    )

    df = c.fetch_constituents_table()

    assert list(df.columns) == ["ticker", "company_name", "gics_sector", "gics_sub_industry", "date_added"]
    # BRK.B -> BRK-B: yfinance/Tiingo both use hyphens, the index page uses dots.
    assert "BRK-B" in df["ticker"].values
    assert "BRK.B" not in df["ticker"].values
    assert len(df) == 3


def test_fetch_constituents_raises_on_schema_drift(monkeypatch):
    broken_table = pd.DataFrame({"Ticker": ["AAPL"], "Name": ["Apple"]})  # wrong column names
    monkeypatch.setattr(
        "src.ingest.constituents.requests.get",
        lambda url, headers=None, timeout=None: _FakeResponse(),
    )
    monkeypatch.setattr(
        "src.ingest.constituents.pd.read_html",
        lambda buf: [broken_table],
    )

    with pytest.raises(ValueError, match="schema drift"):
        c.fetch_constituents_table()


def test_fetch_constituents_raises_on_http_error(monkeypatch):
    """A 403 (or any HTTP error) from Wikipedia should surface, not hang or
    silently return nothing — this is the exact failure mode hit against
    live data before the User-Agent header was added."""

    class _Forbidden(_FakeResponse):
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr(
        "src.ingest.constituents.requests.get",
        lambda url, headers=None, timeout=None: _Forbidden(),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        c.fetch_constituents_table()


def test_ingest_constituents_snapshot_writes_dated_partition(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.constituents.BRONZE_DIR", tmp_path)
    monkeypatch.setattr(
        "src.ingest.constituents.fetch_constituents_table",
        lambda: _fake_wikipedia_table().rename(
            columns={
                "Symbol": "ticker",
                "Security": "company_name",
                "GICS Sector": "gics_sector",
                "GICS Sub-Industry": "gics_sub_industry",
                "Date added": "date_added",
            }
        ),
    )

    as_of = date(2026, 9, 6)
    df = c.ingest_constituents_snapshot(as_of=as_of)

    expected_path = tmp_path / "constituents" / f"ingest_date={as_of.isoformat()}" / "constituents.parquet"
    assert expected_path.exists()
    assert (df["snapshot_date"] == pd.Timestamp(as_of)).all()
