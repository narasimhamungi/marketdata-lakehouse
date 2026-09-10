from datetime import date

import pandas as pd

from src.transform import silver_prices as sp


def test_read_silver_prices_returns_written_data(monkeypatch, tmp_path):
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    out_dir = tmp_path / "yfinance_prices" / f"ingest_date={d.isoformat()}"
    out_dir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2024-01-02"]), "adj_close": [100.0]}).to_parquet(
        out_dir / "prices.parquet"
    )

    result = sp.read_silver_prices("yfinance", d)

    assert not result.empty
    assert result.iloc[0]["ticker"] == "AAPL"


def test_read_silver_prices_returns_empty_when_nothing_built_yet(monkeypatch, tmp_path):
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path)
    result = sp.read_silver_prices("yfinance", date(2026, 9, 7))
    assert result.empty


def test_read_silver_prices_does_not_call_build(monkeypatch, tmp_path):
    """The whole point of this function: it must never trigger a rebuild —
    that would defeat the reason it exists (avoiding redundant recompute
    across DAG tasks)."""
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", tmp_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("read_silver_prices must not call build_silver_prices")

    monkeypatch.setattr(sp, "build_silver_prices", _should_not_be_called)
    sp.read_silver_prices("yfinance", date(2026, 9, 7))  # should not raise
