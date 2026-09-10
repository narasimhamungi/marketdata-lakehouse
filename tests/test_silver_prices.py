"""
Tests for the silver price transform. No network — purely local
dataframe/parquet operations, so these run against real logic, not mocks.
"""
from datetime import date

import pandas as pd
import pytest

from src.transform import silver_prices as sp


def _yfinance_bronze(n=3) -> pd.DataFrame:
    """Only has adj_close, not per-field adjusted OHLC — matches real
    yfinance bronze shape."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0][:n],
            "high": [102.0, 103.0, 104.0][:n],
            "low": [99.0, 100.0, 101.0][:n],
            "close": [101.0, 102.0, 103.0][:n],
            "adj_close": [50.5, 51.0, 51.5][:n],  # e.g. reflecting a 2:1 split -> factor 0.5
            "volume": [1_000_000, 1_100_000, 1_200_000][:n],
            "ticker": ["AAPL"] * n,
        }
    )


def _tiingo_bronze_with_native_adj(n=3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0][:n],
            "high": [102.0, 103.0, 104.0][:n],
            "low": [99.0, 100.0, 101.0][:n],
            "close": [101.0, 102.0, 103.0][:n],
            "adj_close": [50.5, 51.0, 51.5][:n],
            "adj_open": [50.0, 50.5, 51.0][:n],
            "adj_high": [51.0, 51.5, 52.0][:n],
            "adj_low": [49.5, 50.0, 50.5][:n],
            "adj_volume": [2_000_000, 2_200_000, 2_400_000][:n],  # deliberately != volume, to prove passthrough
            "volume": [1_000_000, 1_100_000, 1_200_000][:n],
            "ticker": ["AAPL"] * n,
        }
    )


def test_type_columns_normalizes_date_and_numerics():
    df = _yfinance_bronze()
    df["close"] = df["close"].astype(str)  # simulate a parquet round-trip type slip
    typed = sp._type_columns(df)
    assert typed["date"].dtype.kind == "M"
    assert typed["close"].dtype.kind == "f"


def test_dedupe_drops_duplicate_ticker_date_keeping_last():
    df = _yfinance_bronze(n=2)
    dup = df.iloc[[0]].copy()
    dup["close"] = 999.0  # distinguishable so we can check "last" was kept
    df = pd.concat([df, dup], ignore_index=True)

    result = sp._dedupe(df, "test")

    assert len(result) == 2
    kept_row = result[result["date"] == df.iloc[0]["date"]]
    assert kept_row.iloc[0]["close"] == 999.0


def test_derive_adjusted_ohlc_applies_factor_uniformly():
    df = _yfinance_bronze(n=1)  # close=101.0, adj_close=50.5 -> factor 0.5
    result = sp._derive_adjusted_ohlc_from_factor(df)

    assert result.iloc[0]["adj_open"] == pytest.approx(50.0)   # 100.0 * 0.5
    assert result.iloc[0]["adj_high"] == pytest.approx(51.0)   # 102.0 * 0.5
    assert result.iloc[0]["adj_low"] == pytest.approx(49.5)    # 99.0 * 0.5
    assert result.iloc[0]["adj_volume"] == result.iloc[0]["volume"]  # passed through, not adjusted


def test_derive_adjusted_ohlc_raises_on_non_positive_close():
    df = _yfinance_bronze(n=1)
    df.loc[0, "close"] = 0.0
    with pytest.raises(ValueError, match="close <= 0"):
        sp._derive_adjusted_ohlc_from_factor(df)


def test_build_silver_prices_derives_for_yfinance(monkeypatch, tmp_path):
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path / "silver")
    bronze_partition = tmp_path / "bronze" / "yfinance_prices" / "ingest_date=2026-09-07"
    bronze_partition.mkdir(parents=True)
    _yfinance_bronze().to_parquet(bronze_partition / "batch_00000.parquet")

    result = sp.build_silver_prices("yfinance", date(2026, 9, 7))

    assert list(result.columns) == sp.FINAL_COLUMNS
    assert "adj_open" in result.columns
    # factor-derived, should differ meaningfully from raw open given the
    # 0.5 factor baked into the fixture
    assert result.iloc[0]["adj_open"] < result.iloc[0]["open"]
    out_path = tmp_path / "silver" / "yfinance_prices" / "ingest_date=2026-09-07" / "prices.parquet"
    assert out_path.exists()


def test_build_silver_prices_uses_native_adjusted_columns_for_tiingo(monkeypatch, tmp_path):
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path / "silver")
    bronze_partition = tmp_path / "bronze" / "tiingo_prices" / "ingest_date=2026-09-07"
    bronze_partition.mkdir(parents=True)
    _tiingo_bronze_with_native_adj().to_parquet(bronze_partition / "AAPL.parquet")

    result = sp.build_silver_prices("tiingo", date(2026, 9, 7))

    # adj_volume should be the source-native value (2,000,000...), NOT
    # equal to raw volume (1,000,000...) — proves it used Tiingo's own
    # column rather than silently re-deriving and overwriting it.
    assert result.iloc[0]["adj_volume"] == 2_000_000
    assert result.iloc[0]["adj_volume"] != result.iloc[0]["volume"]


def test_build_silver_prices_falls_back_to_derivation_for_old_tiingo_bronze(monkeypatch, tmp_path):
    """Regression-style test: Tiingo bronze pulled before adj_open/high/
    low/volume were added to that ingester shouldn't crash silver — it
    should fall back to the same factor-derivation yfinance uses."""
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path / "silver")
    bronze_partition = tmp_path / "bronze" / "tiingo_prices" / "ingest_date=2026-09-07"
    bronze_partition.mkdir(parents=True)
    old_shape = _tiingo_bronze_with_native_adj().drop(columns=["adj_open", "adj_high", "adj_low", "adj_volume"])
    old_shape.to_parquet(bronze_partition / "AAPL.parquet")

    result = sp.build_silver_prices("tiingo", date(2026, 9, 7))

    assert list(result.columns) == sp.FINAL_COLUMNS  # didn't crash, still produced the full shape
    assert result.iloc[0]["adj_volume"] == result.iloc[0]["volume"]  # derived fallback behavior


def test_build_silver_prices_raises_on_unknown_source():
    with pytest.raises(ValueError, match="Unknown price source"):
        sp.build_silver_prices("madeup_source")


def test_build_silver_prices_returns_empty_for_missing_partition(monkeypatch, tmp_path):
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", tmp_path / "bronze")
    result = sp.build_silver_prices("yfinance", date(2026, 9, 7))
    assert result.empty
