"""
Tests for the full-universe driver. Every ingestion function it calls is
mocked — this verifies the wiring (right functions called with the right
ticker list, constituents reuse/refresh logic, source selection, and the
stratified sampling logic), not the ingestion logic itself, which is
already covered by each source's own test file.
"""
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.orchestrate import run_full_universe as rfu


def _fake_constituents(n_per_sector=10) -> pd.DataFrame:
    sectors = ["Tech", "Healthcare", "Financials", "Energy"]
    rows = []
    for sector in sectors:
        for i in range(n_per_sector):
            rows.append({"ticker": f"{sector[:2].upper()}{i:03d}", "gics_sector": sector})
    return pd.DataFrame(rows)


def testget_constituents_reuses_existing_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr("src.orchestrate.run_full_universe.BRONZE_DIR", tmp_path)
    partition_dir = tmp_path / "constituents" / "ingest_date=2026-09-07"
    partition_dir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["MSFT", "AAPL"], "gics_sector": ["Tech", "Tech"]}).to_parquet(
        partition_dir / "constituents.parquet"
    )

    ingest_mock = MagicMock()
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_constituents_snapshot", ingest_mock)

    df = rfu.get_constituents(date(2026, 9, 7), refresh=False)

    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    ingest_mock.assert_not_called()


def testget_constituents_pulls_fresh_when_none_exists(monkeypatch, tmp_path):
    monkeypatch.setattr("src.orchestrate.run_full_universe.BRONZE_DIR", tmp_path)
    ingest_mock = MagicMock(return_value=pd.DataFrame({"ticker": ["AMZN"], "gics_sector": ["Tech"]}))
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_constituents_snapshot", ingest_mock)

    df = rfu.get_constituents(date(2026, 9, 7), refresh=False)

    assert list(df["ticker"]) == ["AMZN"]
    ingest_mock.assert_called_once()


def teststratified_sample_covers_every_sector():
    constituents = _fake_constituents(n_per_sector=10)  # 4 sectors x 10 = 40 tickers

    sample = rfu.stratified_sample(constituents, n=8)

    assert len(sample) == 8
    sampled_sectors = constituents.set_index("ticker").loc[sample, "gics_sector"].unique()
    assert len(sampled_sectors) == 4  # all 4 sectors represented, not clustered in one


def teststratified_sample_is_deterministic_across_calls():
    constituents = _fake_constituents(n_per_sector=10)

    sample_a = rfu.stratified_sample(constituents, n=12)
    sample_b = rfu.stratified_sample(constituents, n=12)

    assert sample_a == sample_b  # same seed -> reproducible, matters for a repeatable demo


def teststratified_sample_returns_everything_when_n_exceeds_universe():
    constituents = _fake_constituents(n_per_sector=2)  # 8 tickers total
    sample = rfu.stratified_sample(constituents, n=100)
    assert set(sample) == set(constituents["ticker"])


def test_run_full_universe_samples_for_tiingo_but_not_yfinance_or_openfigi(monkeypatch):
    constituents = _fake_constituents(n_per_sector=10)  # 40 tickers
    monkeypatch.setattr("src.orchestrate.run_full_universe.get_constituents", lambda d, refresh: constituents)

    yf_mock = MagicMock()
    tg_mock = MagicMock()
    fred_mock = MagicMock()
    figi_mock = MagicMock()
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_yfinance_batch", yf_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_tiingo_batch", tg_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_fred_series", fred_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_openfigi_mapping", figi_mock)

    d = date(2026, 9, 7)
    rfu.run_full_universe(d, tiingo_sample_size=8)

    yf_call_tickers = yf_mock.call_args[0][0]
    tg_call_tickers = tg_mock.call_args[0][0]
    figi_call_tickers = figi_mock.call_args[0][0]

    assert len(yf_call_tickers) == 40       # full universe
    assert len(tg_call_tickers) == 8        # sampled
    assert len(figi_call_tickers) == 40     # full universe
    assert set(tg_call_tickers) <= set(yf_call_tickers)  # sample is a genuine subset
    fred_mock.assert_called_once_with(ingest_date=d)


def test_run_full_universe_respects_sources_selection(monkeypatch):
    constituents = _fake_constituents(n_per_sector=5)
    monkeypatch.setattr("src.orchestrate.run_full_universe.get_constituents", lambda d, refresh: constituents)

    yf_mock = MagicMock()
    tg_mock = MagicMock()
    fred_mock = MagicMock()
    figi_mock = MagicMock()
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_yfinance_batch", yf_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_tiingo_batch", tg_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_fred_series", fred_mock)
    monkeypatch.setattr("src.orchestrate.run_full_universe.ingest_openfigi_mapping", figi_mock)

    rfu.run_full_universe(date(2026, 9, 7), sources=("tiingo",))

    tg_mock.assert_called_once()
    yf_mock.assert_not_called()
    fred_mock.assert_not_called()
    figi_mock.assert_not_called()
