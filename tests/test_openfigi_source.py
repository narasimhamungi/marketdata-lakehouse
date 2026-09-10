"""
Unit tests for OpenFIGI ingestion. `requests.post` is mocked throughout —
these verify batch construction, the "no match" vs "request failed"
distinction, and the rate-limit-specific retry path, not OpenFIGI's live
availability.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingest import openfigi_source as o


class _FakeResponse:
    def __init__(self, json_data=None, status_code: int = 200):
        self._json_data = json_data if json_data is not None else []
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _match(ticker: str) -> dict:
    return {
        "data": [
            {
                "figi": f"BBG{ticker}FIGI",
                "compositeFIGI": f"BBG{ticker}COMP",
                "shareClassFIGI": f"BBG{ticker}SHRC",
                "name": f"{ticker} INC",
                "securityType": "Common Stock",
                "marketSector": "Equity",
            }
        ]
    }


def _no_match() -> dict:
    return {"warning": "No identifier found."}


def test_map_batch_raises_distinctly_on_429(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.openfigi_source.requests.post",
        lambda url, json=None, headers=None, timeout=None: _FakeResponse(status_code=429),
    )

    with pytest.raises(o.RateLimited):
        o._map_batch(["AAPL"])


def test_map_batch_raises_on_result_count_mismatch(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.openfigi_source.requests.post",
        lambda url, json=None, headers=None, timeout=None: _FakeResponse(
            json_data=[_match("AAPL")]  # 1 result for a 2-ticker request
        ),
    )

    with pytest.raises(ValueError, match="no longer lines up 1:1"):
        o._map_batch(["AAPL", "MSFT"])


def test_ingest_openfigi_mapping_writes_matches_and_tracks_misses(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.openfigi_source.BRONZE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.openfigi_source.OPENFIGI_API_KEY", "fake_key")  # skip the sleep(2) pacing
    monkeypatch.setattr(
        "src.ingest.openfigi_source._map_batch",
        lambda batch, exch_code="US": [_match(t) if t != "GHOST" else _no_match() for t in batch],
    )

    result = o.ingest_openfigi_mapping(["AAPL", "MSFT", "GHOST"], ingest_date=date(2026, 9, 7))

    assert set(result["ticker"]) == {"AAPL", "MSFT"}
    assert (result.set_index("ticker").loc["AAPL", "figi"]) == "BBGAAPLFIGI"

    out_dir = tmp_path / "openfigi_mapping" / "ingest_date=2026-09-07"
    assert (out_dir / "mapping.parquet").exists()
    failed_file = out_dir / "_failed_tickers.txt"
    assert failed_file.exists()
    assert failed_file.read_text().splitlines() == ["GHOST"]


def test_ingest_openfigi_mapping_records_batch_level_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.openfigi_source.BRONZE_DIR", tmp_path)
    monkeypatch.setattr("src.ingest.openfigi_source.OPENFIGI_API_KEY", "fake_key")

    def flaky(batch, exch_code="US"):
        raise ValueError("simulated OpenFIGI failure")

    monkeypatch.setattr("src.ingest.openfigi_source._map_batch", flaky)

    result = o.ingest_openfigi_mapping(["AAPL", "MSFT"], ingest_date=date(2026, 9, 7))

    assert result.empty
    failed_file = tmp_path / "openfigi_mapping" / "ingest_date=2026-09-07" / "_failed_tickers.txt"
    assert set(failed_file.read_text().splitlines()) == {"AAPL", "MSFT"}
