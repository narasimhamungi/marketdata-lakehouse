"""
Tests for the bronze quality suite. No network involved — Great
Expectations validates dataframes purely locally, so these run against
real GE computation (not mocked), just on small fixtures instead of live
data. Each test checks both that clean data passes AND that a specific,
deliberately introduced violation is actually caught — a quality suite
that only ever reports success is worse than no quality suite, since it
creates false confidence.
"""
from datetime import date

import pandas as pd

from src.quality import bronze_checks as bc


def _clean_prices(ticker="AAPL", n=3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "adj_close": [101.0 + i for i in range(n)],
            "volume": [1_000_000 + i * 1000 for i in range(n)],
            "ticker": [ticker] * n,
        }
    )


def test_check_price_bronze_passes_on_clean_data():
    report = bc.check_price_bronze(_clean_prices(), "yfinance")
    assert report.passed


def test_check_price_bronze_tolerates_one_isolated_violation_at_scale():
    """
    Regression test for the exact real failure: a single physically-
    impossible row (HUBB, 2021-05-05, low > open — but still < high and
    < close, so only that one check is violated, not several at once) in
    ~950K real yfinance rows previously failed the entire bronze quality
    gate and blocked the whole DAG. One bad print out of hundreds of
    thousands of rows is expected vendor noise, not a pipeline defect —
    this must still report the violation (so it isn't silently lost) but
    not fail the check.

    n=20,000 here, not a smaller number: at 99.99% tolerance, one
    violation is only tolerated once row count is large enough that
    1/n <= 0.0001 — a smaller fixture would make even one real violation
    exceed the tolerance and this test would (correctly) still fail,
    which is exactly what the first version of this test did wrong.
    """
    df = _clean_prices(n=20_000)
    row = 500
    # Isolate the violation to exactly "low <= open", matching HUBB's real
    # pattern — nudge low just above open but keep it below high and close,
    # so this doesn't also trip the other four OHLC checks.
    df.loc[row, "low"] = df.loc[row, "open"] + 0.5

    report = bc.check_price_bronze(df, "yfinance")

    assert report.passed  # tolerated, not a pipeline-blocking failure
    low_open_check = next(c for c in report.checks if "low <= open" in c.description)
    assert low_open_check.unexpected_count == 1  # but still visible in the report, not silently dropped
    # And confirm it really is isolated — the other pair checks shouldn't
    # have been collaterally broken by this one nudge.
    for other in ("high >= low", "high >= open", "high >= close", "low <= close"):
        other_check = next(c for c in report.checks if other in c.description)
        assert other_check.success


def test_check_price_bronze_still_fails_on_genuinely_widespread_violations():
    """The tolerance must not become a loophole — a real systemic problem
    (many rows violating OHLC ordering, not one isolated bad print) has to
    still fail the gate."""
    df = _clean_prices(n=100)
    for i in range(20):  # 20% of rows broken — nothing resembling isolated noise
        df.loc[i, "low"] = df.loc[i, "open"] + 10

    report = bc.check_price_bronze(df, "yfinance")

    assert not report.passed


def test_check_price_bronze_catches_negative_price():
    df = _clean_prices()
    df.loc[0, "close"] = -5.0
    report = bc.check_price_bronze(df, "yfinance")
    assert not report.passed
    failed = [c for c in report.checks if not c.success]
    assert any("close > 0" in c.description for c in failed)


def test_check_price_bronze_catches_high_less_than_low():
    df = _clean_prices()
    df.loc[0, "high"] = df.loc[0, "low"] - 10  # physically impossible bar
    report = bc.check_price_bronze(df, "tiingo")
    assert not report.passed
    failed = [c for c in report.checks if not c.success]
    assert any("high >= low" in c.description for c in failed)


def test_check_price_bronze_catches_duplicate_ticker_date():
    df = _clean_prices(n=2)
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    report = bc.check_price_bronze(df, "yfinance")
    assert not report.passed
    failed = [c for c in report.checks if not c.success]
    assert any("(ticker, date) unique" in c.description for c in failed)


def test_check_price_bronze_reports_failure_on_empty_dataframe():
    report = bc.check_price_bronze(pd.DataFrame(), "yfinance")
    assert not report.passed


def _clean_constituents(n=500) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"T{i:04d}" for i in range(n)],
            "company_name": [f"Company {i}" for i in range(n)],
            "gics_sector": ["Information Technology"] * n,
        }
    )


def test_check_constituents_bronze_passes_on_clean_data():
    report = bc.check_constituents_bronze(_clean_constituents())
    assert report.passed


def test_check_constituents_bronze_catches_duplicate_ticker():
    df = _clean_constituents()
    df.loc[1, "ticker"] = df.loc[0, "ticker"]
    report = bc.check_constituents_bronze(df)
    assert not report.passed
    failed = [c for c in report.checks if not c.success]
    assert any("ticker unique" in c.description for c in failed)


def test_check_constituents_bronze_catches_row_count_out_of_range():
    report = bc.check_constituents_bronze(_clean_constituents(n=10))
    assert not report.passed
    failed = [c for c in report.checks if not c.success]
    assert any("row count" in c.description for c in failed)


def test_check_fred_bronze_passes_on_clean_rate_data():
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3), "series_id": ["DGS10"] * 3, "value": [4.1, 4.2, 4.15]})
    report = bc.check_fred_bronze(df, "DGS10")
    assert report.passed


def test_check_fred_bronze_catches_implausible_rate():
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=3), "series_id": ["DGS10"] * 3, "value": [4.1, 400.0, 4.15]})
    report = bc.check_fred_bronze(df, "DGS10")
    assert not report.passed


def test_check_fred_bronze_cpi_uses_positive_check_not_rate_bound():
    # CPI index levels run into the hundreds — a [0,25] bound would wrongly
    # fail every real CPI row if applied here, so CPI must use a different
    # check than the rate series do.
    df = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=2), "series_id": ["CPIAUCSL"] * 2, "value": [310.3, 311.1]})
    report = bc.check_fred_bronze(df, "CPIAUCSL")
    assert report.passed


def test_check_openfigi_bronze_passes_on_clean_data():
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "figi": ["BBG000B9XRY4", "BBG000BPH459"], "name": ["APPLE INC", "MICROSOFT CORP"]})
    report = bc.check_openfigi_bronze(df)
    assert report.passed


def test_check_openfigi_bronze_catches_wrong_length_figi():
    df = pd.DataFrame({"ticker": ["AAPL"], "figi": ["TOOSHORT"], "name": ["APPLE INC"]})
    report = bc.check_openfigi_bronze(df)
    assert not report.passed


def test_find_ohlc_violations_identifies_the_specific_bad_row():
    """The GE check reports a count; this returns the actual row, which is
    what you need to investigate a real failure rather than just know one
    exists."""
    df = _clean_prices(n=3)
    df.loc[1, "low"] = df.loc[1, "open"] + 5  # low above open — impossible bar

    violations = bc.find_ohlc_violations(df)

    assert len(violations) >= 1
    assert "low > open" in violations["violation"].values
    assert violations.iloc[0]["ticker"] == "AAPL"


def test_find_ohlc_violations_returns_empty_for_clean_data():
    assert bc.find_ohlc_violations(_clean_prices()).empty


def test_read_partition_concats_multiple_parquet_files(tmp_path):
    source_dir = tmp_path / "yfinance_prices"
    partition_dir = source_dir / "ingest_date=2026-09-07"
    partition_dir.mkdir(parents=True)
    _clean_prices("AAPL", n=2).to_parquet(partition_dir / "batch_00000.parquet")
    _clean_prices("MSFT", n=2).to_parquet(partition_dir / "batch_00001.parquet")

    df = bc._read_partition(source_dir, date(2026, 9, 7))

    assert len(df) == 4
    assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}


def test_read_partition_returns_empty_for_missing_partition(tmp_path):
    df = bc._read_partition(tmp_path / "nonexistent_source", date(2026, 9, 7))
    assert df.empty


def test_run_all_bronze_checks_integration(tmp_path, monkeypatch):
    """End-to-end: real files on disk, matching each source's actual bronze
    layout, read back and validated — not just unit-level function calls."""
    monkeypatch.setattr("src.quality.bronze_checks.BRONZE_DIR", tmp_path)
    ingest_date = date(2026, 9, 7)

    (tmp_path / "constituents" / f"ingest_date={ingest_date}").mkdir(parents=True)
    _clean_constituents().to_parquet(tmp_path / "constituents" / f"ingest_date={ingest_date}" / "constituents.parquet")

    (tmp_path / "yfinance_prices" / f"ingest_date={ingest_date}").mkdir(parents=True)
    _clean_prices("AAPL").to_parquet(tmp_path / "yfinance_prices" / f"ingest_date={ingest_date}" / "batch_00000.parquet")

    (tmp_path / "fred_series" / f"ingest_date={ingest_date}").mkdir(parents=True)
    pd.DataFrame({"date": pd.date_range("2024-01-02", periods=2), "series_id": ["DGS10"] * 2, "value": [4.1, 4.2]}).to_parquet(
        tmp_path / "fred_series" / f"ingest_date={ingest_date}" / "DGS10.parquet"
    )

    reports = bc.run_all_bronze_checks(ingest_date)

    sources = {r.source for r in reports}
    assert sources == {"constituents", "yfinance", "fred:DGS10"}
    assert all(r.passed for r in reports)
