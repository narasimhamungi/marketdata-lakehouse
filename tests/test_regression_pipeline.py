"""
End-to-end regression test using a small, FIXED snapshot of real data —
not live-pulled, not synthetic round numbers, but the actual values this
project validated against live yfinance/Tiingo earlier (AAPL, 2019-01-02
and 2019-01-03).

This exists because pure unit tests, which mostly use small round made-up
numbers (100.0, 50.5), can't catch every regression: a subtle bug in the
adjustment-factor math or the reconciliation calculation could still pass
every unit test while producing a wrong answer on real data with real
precision. Pinning expected values to numbers this project already proved
correct against live sources closes that gap.

Expected values below were computed independently, once, outside the
application code (see the computation this file's expected constants came
from — same formula the code uses, but evaluated separately, not imported
from it) — using the *same* formula inside both the code and this test
would make the test tautological, agreeing with itself no matter what bug
existed in that formula.
"""
from datetime import date

import pandas as pd
import pytest

from src.reconcile.price_reconciliation import reconcile_prices
from src.transform.silver_prices import build_silver_prices

INGEST_DATE = date(2026, 9, 7)  # arbitrary fixed partition date for this fixture

# Real values confirmed against live yfinance earlier in this project.
_YFINANCE_ROWS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
        "open": [38.7225, 35.994999],
        "high": [39.712502, 36.430000],
        "low": [38.557499, 35.500000],
        "close": [39.480000, 35.547501],
        "adj_close": [37.436916, 33.707928],
        "volume": [148158800, 365248800],
        "ticker": ["AAPL", "AAPL"],
    }
)

# Real values confirmed against live Tiingo earlier in this project.
# Deliberately no adj_open/adj_high/adj_low/adj_volume — this fixture also
# exercises silver's derivation fallback for Tiingo bronze pulled before
# that schema existed, a real historical case this project actually hit.
_TIINGO_ROWS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
        "open": [154.89, 143.98],
        "high": [158.85, 145.72],
        "low": [154.23, 142.00],
        "close": [157.92, 142.19],
        "adj_close": [37.437651, 33.708584],
        "volume": [37039737, 91312195],
        "ticker": ["AAPL", "AAPL"],
    }
)


def _write_bronze_fixtures(bronze_dir):
    yf_dir = bronze_dir / "yfinance_prices" / f"ingest_date={INGEST_DATE.isoformat()}"
    yf_dir.mkdir(parents=True)
    _YFINANCE_ROWS.to_parquet(yf_dir / "batch_00000.parquet", index=False)

    tg_dir = bronze_dir / "tiingo_prices" / f"ingest_date={INGEST_DATE.isoformat()}"
    tg_dir.mkdir(parents=True)
    _TIINGO_ROWS.to_parquet(tg_dir / "AAPL.parquet", index=False)


def test_silver_transform_matches_known_good_values_for_both_sources(monkeypatch, tmp_path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", bronze_dir)
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", silver_dir)
    _write_bronze_fixtures(bronze_dir)

    yf_silver = build_silver_prices("yfinance", INGEST_DATE)
    tg_silver = build_silver_prices("tiingo", INGEST_DATE)

    # yfinance: adj_close is given directly by the source, unchanged.
    assert yf_silver.iloc[0]["adj_close"] == pytest.approx(37.436916)
    # adj_open derived via the adjustment-factor technique — expected
    # value computed independently, not via the code's own formula.
    assert yf_silver.iloc[0]["adj_open"] == pytest.approx(36.718616509878416, rel=1e-9)

    # Tiingo: this fixture has no native adj_open, so it must fall back to
    # the same derivation technique as yfinance (confirmed: it does, and
    # lands on a very close but NOT identical value to yfinance's — the
    # two sources' raw open/close differ, so their derived adj_open should
    # too, even though adj_close itself nearly matches).
    assert tg_silver.iloc[0]["adj_open"] == pytest.approx(36.71933740748481, rel=1e-9)


def test_reconciliation_agrees_on_known_good_real_values(monkeypatch, tmp_path):
    """
    The actual real-world result this project found: yfinance and Tiingo's
    adj_close for AAPL 2019-01-02 agree to within ~0.002% — well inside
    the 0.5% tolerance. If a future change to the reconciliation math
    caused this specific, already-verified case to start disagreeing,
    that's exactly the kind of regression this test exists to catch.
    """
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    monkeypatch.setattr("src.transform.silver_prices.BRONZE_DIR", bronze_dir)
    monkeypatch.setattr("src.transform.silver_prices.SILVER_DIR", silver_dir)
    monkeypatch.setattr("src.reconcile.price_reconciliation.SILVER_DIR", silver_dir)
    _write_bronze_fixtures(bronze_dir)
    build_silver_prices("yfinance", INGEST_DATE)
    build_silver_prices("tiingo", INGEST_DATE)

    report = reconcile_prices(INGEST_DATE)

    assert report.total_compared == 2
    assert report.flagged_count == 0  # both known-good days agree, as they did against live data
    row = report.full[report.full["date"] == pd.Timestamp("2019-01-02")].iloc[0]
    # Expected value computed independently: ~0.00196%, comfortably under
    # the 0.5% tolerance — not just "small enough to pass" but pinned to
    # the actual real figure.
    assert row["pct_diff"] == pytest.approx(1.9632834738287712e-05, rel=1e-6)
