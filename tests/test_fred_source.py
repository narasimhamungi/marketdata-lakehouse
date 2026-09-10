"""
Unit tests for FRED ingestion. `requests.get` is mocked throughout — these
verify JSON parsing (including FRED's "." placeholder-for-missing
convention) and failure handling, not FRED's live availability.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingest import fred_source as f


class _FakeResponse:
    def __init__(self, json_data=None, status_code: int = 200):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _fake_payload(values):
    return {
        "observations": [
            {"date": f"2024-01-{i+1:02d}", "value": v} for i, v in enumerate(values)
        ]
    }


def test_download_series_parses_and_types_correctly(monkeypatch):
    monkeypatch.setattr("src.ingest.fred_source.FRED_API_KEY", "fake_key")
    monkeypatch.setattr(
        "src.ingest.fred_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(_fake_payload(["5.25", "5.26", "5.24"])),
    )

    df = f._download_series("DGS10", date(2024, 1, 1), date(2024, 1, 3))

    assert list(df.columns) == ["date", "series_id", "value"]
    assert len(df) == 3
    assert df["value"].dtype.kind == "f"
    assert (df["series_id"] == "DGS10").all()


def test_download_series_drops_placeholder_values(monkeypatch):
    """FRED pads daily-cadence gaps (weekends/holidays) with '.' — these
    must be dropped, not coerced to NaN or crash float()."""
    monkeypatch.setattr("src.ingest.fred_source.FRED_API_KEY", "fake_key")
    monkeypatch.setattr(
        "src.ingest.fred_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(_fake_payload(["5.25", ".", "5.24"])),
    )

    df = f._download_series("DGS10", date(2024, 1, 1), date(2024, 1, 3))

    assert len(df) == 2
    assert "." not in df["value"].astype(str).values


def test_download_series_raises_without_api_key(monkeypatch):
    monkeypatch.setattr("src.ingest.fred_source.FRED_API_KEY", "")

    with pytest.raises(RuntimeError, match="MDL_FRED_API_KEY"):
        f._download_series("DGS10", date(2024, 1, 1), date(2024, 1, 3))


def test_download_series_raises_when_all_values_are_placeholders(monkeypatch):
    monkeypatch.setattr("src.ingest.fred_source.FRED_API_KEY", "fake_key")
    monkeypatch.setattr(
        "src.ingest.fred_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(_fake_payload([".", "."])),
    )

    with pytest.raises(ValueError, match="placeholder"):
        f._download_series("CPIAUCSL", date(2024, 1, 1), date(2024, 1, 3))


def test_ingest_fred_series_writes_bronze_and_returns_frame(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.fred_source.BRONZE_DIR", tmp_path)
    monkeypatch.setattr(
        "src.ingest.fred_source._download_series",
        lambda series_id, start, end: pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "series_id": [series_id, series_id],
                "value": [5.25, 5.26],
            }
        ),
    )

    result = f.ingest_fred_series(["DGS10", "FEDFUNDS"], ingest_date=date(2026, 9, 7))

    assert set(result["series_id"].unique()) == {"DGS10", "FEDFUNDS"}
    out_dir = tmp_path / "fred_series" / "ingest_date=2026-09-07"
    assert (out_dir / "DGS10.parquet").exists()
    assert (out_dir / "FEDFUNDS.parquet").exists()


def test_ingest_fred_series_records_failed_series(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.fred_source.BRONZE_DIR", tmp_path)

    def flaky(series_id, start, end):
        raise ValueError("simulated FRED failure")

    monkeypatch.setattr("src.ingest.fred_source._download_series", flaky)

    result = f.ingest_fred_series(["DGS10", "BADSERIES"], ingest_date=date(2026, 9, 7))

    assert result.empty
    failed_file = tmp_path / "fred_series" / "ingest_date=2026-09-07" / "_failed_series.txt"
    assert failed_file.exists()
    assert set(failed_file.read_text().splitlines()) == {"DGS10", "BADSERIES"}
