"""
Unit tests for yfinance ingestion.

`yf.download` is mocked throughout — these tests check our retry, reshape,
batching, and failure-handling logic, not Yahoo's live endpoints. The point
of a hand-built retry+fallback path is exactly that it needs to be verified
without depending on a flaky real network call to expose bugs in it.
"""
from datetime import date

import pandas as pd
import pytest

from src.ingest import yfinance_source as y


def _fake_multiticker_frame(tickers: list[str], n_days: int = 3) -> pd.DataFrame:
    """Mimic yf.download(group_by='ticker') output: MultiIndex columns."""
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    columns = pd.MultiIndex.from_product([tickers, fields])
    data = {}
    for ticker in tickers:
        for j, field in enumerate(fields):
            base = 100.0 + hash(ticker) % 50
            if field == "Volume":
                data[(ticker, field)] = [1_000_000 + i * 1000 for i in range(n_days)]
            elif field in ("Dividends", "Stock Splits"):
                data[(ticker, field)] = [0.0] * n_days
            else:
                data[(ticker, field)] = [base + i + j * 0.1 for i in range(n_days)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_reshape_batch_produces_tidy_long_format():
    tickers = ["AAPL", "MSFT"]
    raw = _fake_multiticker_frame(tickers)

    tidy = y._reshape_batch(raw, tickers)

    assert set(tidy["ticker"].unique()) == set(tickers)
    assert {"date", "open", "high", "low", "close", "adj_close", "volume"} <= set(tidy.columns)
    assert len(tidy) == 2 * 3  # 2 tickers x 3 trading days


def test_reshape_batch_drops_ticker_with_no_data_and_logs(caplog):
    tickers = ["AAPL", "GHOST"]
    raw = _fake_multiticker_frame(["AAPL"])  # GHOST never returned by the source

    with caplog.at_level("WARNING"):
        tidy = y._reshape_batch(raw, tickers)

    assert "GHOST" not in tidy["ticker"].unique()
    assert any("GHOST" in rec.message for rec in caplog.records)


def test_ingest_yfinance_batch_writes_bronze_and_returns_frame(monkeypatch, tmp_path):
    tickers = ["AAPL", "MSFT"]
    monkeypatch.setattr("src.ingest.yfinance_source.BRONZE_DIR", tmp_path)
    monkeypatch.setattr(
        "src.ingest.yfinance_source._download_batch",
        lambda batch, start, end: _fake_multiticker_frame(batch),
    )

    result = y.ingest_yfinance_batch(tickers, batch_size=50, ingest_date=date(2026, 9, 6))

    assert not result.empty
    assert set(result["ticker"].unique()) == set(tickers)
    out_files = list((tmp_path / "yfinance_prices" / "ingest_date=2026-09-06").glob("*.parquet"))
    assert len(out_files) == 1  # one batch, batch_size >= len(tickers)


def test_ingest_yfinance_batch_routes_partial_batch_failure_to_fallback(monkeypatch, tmp_path):
    """
    Regression test for the AMZN case seen against live data: yf.download()
    can succeed for a batch overall (no exception raised) while silently
    dropping one ticker (observed cause: yfinance's own SQLite cache
    raising 'database is locked' under concurrent access). That ticker must
    still end up in the Tiingo-fallback list, not just get logged and lost.
    """
    monkeypatch.setattr("src.ingest.yfinance_source.BRONZE_DIR", tmp_path)
    # Simulate: batch of 3 requested, only 2 (AAPL, MSFT) actually returned —
    # the batch call itself does not raise.
    monkeypatch.setattr(
        "src.ingest.yfinance_source._download_batch",
        lambda batch, start, end: _fake_multiticker_frame(["AAPL", "MSFT"]),
    )

    tickers = ["AAPL", "MSFT", "AMZN"]
    result = y.ingest_yfinance_batch(tickers, batch_size=50, ingest_date=date(2026, 9, 6))

    assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}  # what we got is still written
    failed_file = tmp_path / "yfinance_prices" / "ingest_date=2026-09-06" / "_failed_tickers.txt"
    assert failed_file.exists()
    assert failed_file.read_text().splitlines() == ["AMZN"]  # the dropped ticker, and only it


def test_ingest_yfinance_batch_records_failed_tickers_for_tiingo_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.yfinance_source.BRONZE_DIR", tmp_path)

    def flaky_download(batch, start, end):
        raise ConnectionError("simulated Yahoo rate limit")

    monkeypatch.setattr("src.ingest.yfinance_source._download_batch", flaky_download)
    # Skip real retry waits in this test.
    monkeypatch.setattr(y, "_download_batch", flaky_download)

    tickers = ["AAPL", "MSFT"]
    result = y.ingest_yfinance_batch(tickers, batch_size=50, ingest_date=date(2026, 9, 6))

    assert result.empty
    failed_file = tmp_path / "yfinance_prices" / "ingest_date=2026-09-06" / "_failed_tickers.txt"
    assert failed_file.exists()
    assert set(failed_file.read_text().splitlines()) == set(tickers)
