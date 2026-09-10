"""
Tests for price reconciliation. No network — reads pre-built silver
parquet fixtures, purely local logic.
"""
from datetime import date

import pandas as pd
import pytest

from src.reconcile import price_reconciliation as pr


def _write_silver(tmp_path, source: str, ingest_date: date, rows: list[dict]):
    out_dir = tmp_path / f"{source}_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(out_dir / "prices.parquet", index=False)


def test_reconcile_flags_discrepancy_beyond_tolerance(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    # AAPL: sources agree closely (well within 0.5% tolerance)
    # MSFT: sources disagree by ~2% — should be flagged
    _write_silver(tmp_path, "yfinance", d, [
        {"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00},
        {"ticker": "MSFT", "date": "2024-01-02", "adj_close": 100.00},
    ])
    _write_silver(tmp_path, "tiingo", d, [
        {"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.05},
        {"ticker": "MSFT", "date": "2024-01-02", "adj_close": 102.00},
    ])

    report = pr.reconcile_prices(d)

    assert report.total_compared == 2
    assert report.flagged_count == 1
    assert report.flagged.iloc[0]["ticker"] == "MSFT"


def test_reconcile_does_not_flag_within_tolerance(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [{"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00}])
    _write_silver(tmp_path, "tiingo", d, [{"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.10}])  # 0.1% diff

    report = pr.reconcile_prices(d)

    assert report.flagged_count == 0
    assert report.total_compared == 1


def test_reconcile_tracks_coverage_gaps_separately_from_discrepancies(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [
        {"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00},
        {"ticker": "ONLY_YF", "date": "2024-01-02", "adj_close": 50.00},
    ])
    _write_silver(tmp_path, "tiingo", d, [
        {"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00},
        {"ticker": "ONLY_TG", "date": "2024-01-02", "adj_close": 75.00},
    ])

    report = pr.reconcile_prices(d)

    assert report.total_compared == 1  # only AAPL is in both
    assert report.yfinance_only_count == 1
    assert report.tiingo_only_count == 1
    assert report.flagged_count == 0  # coverage gaps are not discrepancies


def test_reconcile_symmetric_pct_diff_not_biased_toward_either_source(monkeypatch, tmp_path):
    """Discrepancy % should be relative to the average of both sources, not
    just one — swapping which source is 'higher' shouldn't change the
    computed pct_diff for the same absolute gap."""
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [
        {"ticker": "A", "date": "2024-01-02", "adj_close": 100.0},
        {"ticker": "B", "date": "2024-01-02", "adj_close": 102.0},  # yfinance higher this time
    ])
    _write_silver(tmp_path, "tiingo", d, [
        {"ticker": "A", "date": "2024-01-02", "adj_close": 102.0},  # tiingo higher
        {"ticker": "B", "date": "2024-01-02", "adj_close": 100.0},
    ])

    report = pr.reconcile_prices(d)
    diffs = report.full.set_index("ticker")["pct_diff"]
    assert diffs["A"] == pytest.approx(diffs["B"])


def test_reconcile_raises_when_silver_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    with pytest.raises(ValueError, match="Missing silver data"):
        pr.reconcile_prices(date(2026, 9, 7))


def test_reconcile_writes_parquet_output(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [{"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00}])
    _write_silver(tmp_path, "tiingo", d, [{"ticker": "AAPL", "date": "2024-01-02", "adj_close": 100.00}])

    pr.reconcile_prices(d)

    out_path = tmp_path / "reconciled_prices" / f"ingest_date={d.isoformat()}" / "reconciliation.parquet"
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert set(written.columns) == {"ticker", "date", "yfinance_adj_close", "tiingo_adj_close", "abs_diff", "pct_diff", "flagged"}


def test_render_includes_top_discrepancies(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [{"ticker": "MSFT", "date": "2024-01-02", "adj_close": 100.00}])
    _write_silver(tmp_path, "tiingo", d, [{"ticker": "MSFT", "date": "2024-01-02", "adj_close": 105.00}])

    report = pr.reconcile_prices(d)
    rendered = report.render()

    assert "MSFT" in rendered
    assert "Flagged" in rendered


def test_consensus_covers_every_ticker_including_single_source(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [
        {"ticker": "BOTH", "date": "2024-01-02", "adj_close": 100.0},
        {"ticker": "ONLY_YF", "date": "2024-01-02", "adj_close": 50.0},
    ])
    _write_silver(tmp_path, "tiingo", d, [
        {"ticker": "BOTH", "date": "2024-01-02", "adj_close": 100.05},
        {"ticker": "ONLY_TG", "date": "2024-01-02", "adj_close": 75.0},
    ])

    report = pr.reconcile_prices(d)
    consensus = report.consensus.set_index("ticker")

    assert set(consensus.index) == {"BOTH", "ONLY_YF", "ONLY_TG"}

    assert consensus.loc["BOTH", "reconciliation_flag"] == "agreed"
    assert consensus.loc["BOTH", "primary_source"] == "yfinance"
    assert consensus.loc["BOTH", "adj_close"] == 100.0

    assert consensus.loc["ONLY_YF", "reconciliation_flag"] == "single_source"
    assert consensus.loc["ONLY_YF", "primary_source"] == "yfinance"
    assert consensus.loc["ONLY_YF", "adj_close"] == 50.0
    assert consensus.loc["ONLY_YF", "sources_available"] == ["yfinance"]

    assert consensus.loc["ONLY_TG", "reconciliation_flag"] == "single_source"
    assert consensus.loc["ONLY_TG", "primary_source"] == "tiingo"
    assert consensus.loc["ONLY_TG", "adj_close"] == 75.0


def test_consensus_flags_disagreed_beyond_tolerance(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [{"ticker": "BADCO", "date": "2024-01-02", "adj_close": 100.0}])
    _write_silver(tmp_path, "tiingo", d, [{"ticker": "BADCO", "date": "2024-01-02", "adj_close": 110.0}])

    report = pr.reconcile_prices(d)
    row = report.consensus.set_index("ticker").loc["BADCO"]

    assert row["reconciliation_flag"] == "disagreed"
    assert row["sources_available"] == ["yfinance", "tiingo"]


def test_flagged_by_ticker_separates_concentrated_from_spread(monkeypatch, tmp_path):
    """
    The distinction that actually matters when triaging a reconciliation
    result: one ticker broken across all its rows (corporate-action
    handling) vs. every ticker slightly off (systemic comparison problem).
    A headline percentage can't tell those apart.
    """
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    yf_rows, tg_rows = [], []
    # BADCO: every row wildly off. CLEANCO: every row agrees.
    for day in range(1, 4):
        yf_rows.append({"ticker": "BADCO", "date": f"2024-01-0{day}", "adj_close": 100.0})
        tg_rows.append({"ticker": "BADCO", "date": f"2024-01-0{day}", "adj_close": 110.0})
        yf_rows.append({"ticker": "CLEANCO", "date": f"2024-01-0{day}", "adj_close": 50.0})
        tg_rows.append({"ticker": "CLEANCO", "date": f"2024-01-0{day}", "adj_close": 50.01})
    _write_silver(tmp_path, "yfinance", d, yf_rows)
    _write_silver(tmp_path, "tiingo", d, tg_rows)

    report = pr.reconcile_prices(d)
    per_ticker = report.flagged_by_ticker().set_index("ticker")

    assert per_ticker.loc["BADCO", "rows_flagged"] == 3
    assert per_ticker.loc["BADCO", "flagged_pct"] == 100.0
    assert per_ticker.loc["CLEANCO", "rows_flagged"] == 0
    # And the rendered report should surface that concentration explicitly
    assert "Affected tickers: 1 of 2" in report.render()


def test_flagged_by_ticker_reports_the_flagged_date_window(monkeypatch, tmp_path):
    """
    The window is what turns "these rows disagree" into a diagnosable
    finding: a discrepancy that stops abruptly on one date points at a
    corporate action on that date, which is a completely different
    conclusion from a discrepancy spanning the whole history.
    """
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    yf_rows, tg_rows = [], []
    # Disagree for the first 3 days, then agree exactly from the 4th on —
    # simulating a vendor restating history up to an event date.
    for day, (yf_val, tg_val) in enumerate(
        [(100.0, 110.0), (100.0, 110.0), (100.0, 110.0), (100.0, 100.0), (100.0, 100.0)], start=1
    ):
        yf_rows.append({"ticker": "EVENTCO", "date": f"2024-01-0{day}", "adj_close": yf_val})
        tg_rows.append({"ticker": "EVENTCO", "date": f"2024-01-0{day}", "adj_close": tg_val})
    _write_silver(tmp_path, "yfinance", d, yf_rows)
    _write_silver(tmp_path, "tiingo", d, tg_rows)

    report = pr.reconcile_prices(d)
    row = report.flagged_by_ticker().set_index("ticker").loc["EVENTCO"]

    assert pd.Timestamp(row["first_flagged"]).date() == date(2024, 1, 1)
    assert pd.Timestamp(row["last_flagged"]).date() == date(2024, 1, 3)  # window closes at the event


def test_flagged_by_ticker_returns_empty_when_nothing_compared(monkeypatch, tmp_path):
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", tmp_path)
    d = date(2026, 9, 7)
    _write_silver(tmp_path, "yfinance", d, [{"ticker": "ONLY_YF", "date": "2024-01-02", "adj_close": 100.0}])
    _write_silver(tmp_path, "tiingo", d, [{"ticker": "ONLY_TG", "date": "2024-01-02", "adj_close": 100.0}])

    report = pr.reconcile_prices(d)

    assert report.total_compared == 0
    assert report.flagged_by_ticker().empty
