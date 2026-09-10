from datetime import date

import pandas as pd

from src.model import dim_date as dd
from src.model import dim_security as ds
from src.model import fact_price_daily as fpd


def _seed_dims(conn, start, end, tickers):
    dd.build_dim_date(start, end, set(), conn=conn)
    constituents = pd.DataFrame(
        {"ticker": tickers, "company_name": tickers, "gics_sector": ["Tech"] * len(tickers), "gics_sub_industry": ["X"] * len(tickers)}
    )
    ds.build_dim_security(constituents, pd.DataFrame(columns=["ticker", "figi"]), start, conn=conn)


def _silver_row(ticker="AAPL", d="2024-01-02"):
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "date": pd.to_datetime([d]),
            "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
            "adj_open": [100.0], "adj_high": [102.0], "adj_low": [99.0], "adj_close": [101.0],
            "volume": [1_000_000], "adj_volume": [1_000_000],
        }
    )


def test_build_fact_price_daily_loads_with_correct_source_tag(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    summary = fpd.build_fact_price_daily(_silver_row(), "yfinance", conn=pg_conn)

    assert summary == {"loaded": 1, "skipped_no_security_key": 0, "skipped_no_date_key": 0}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT source, close FROM fact_price_daily")
        source, close = cur.fetchone()
    assert source == "yfinance"
    assert float(close) == 101.0


def test_build_fact_price_daily_same_security_date_different_source_is_two_rows(pg_conn):
    """The whole point of this table's grain: yfinance and Tiingo for the
    same (ticker, date) must coexist, not overwrite each other."""
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    fpd.build_fact_price_daily(_silver_row(), "yfinance", conn=pg_conn)
    fpd.build_fact_price_daily(_silver_row(), "tiingo", conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM fact_price_daily")
        assert cur.fetchone()[0] == 2


def test_build_fact_price_daily_skips_ticker_missing_from_dim_security(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    summary = fpd.build_fact_price_daily(_silver_row(ticker="UNKNOWN"), "yfinance", conn=pg_conn)
    assert summary == {"loaded": 0, "skipped_no_security_key": 1, "skipped_no_date_key": 0}


def test_build_fact_price_daily_upsert_replaces_values(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    fpd.build_fact_price_daily(_silver_row(), "yfinance", conn=pg_conn)
    revised = _silver_row()
    revised["close"] = 999.0
    fpd.build_fact_price_daily(revised, "yfinance", conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), close FROM fact_price_daily GROUP BY close")
        count, close = cur.fetchone()
    assert count == 1
    assert float(close) == 999.0
