from datetime import date

import pandas as pd

from src.model import dim_date as dd
from src.model import dim_security as ds
from src.model import fact_price_daily_consensus as fpdc


def _seed_dims(conn, start, end, tickers):
    dd.build_dim_date(start, end, set(), conn=conn)
    constituents = pd.DataFrame(
        {"ticker": tickers, "company_name": tickers, "gics_sector": ["Tech"] * len(tickers), "gics_sub_industry": ["X"] * len(tickers)}
    )
    ds.build_dim_security(constituents, pd.DataFrame(columns=["ticker", "figi"]), start, conn=conn)


def _consensus_row(ticker, flag, sources, pct_diff=None):
    return {
        "ticker": ticker,
        "date": pd.Timestamp("2024-01-02"),
        "adj_close": 100.0,
        "primary_source": sources[0],
        "sources_available": sources,
        "pct_diff": pct_diff,
        "reconciliation_flag": flag,
    }


def test_build_fact_price_daily_consensus_loads_agreed_row(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    df = pd.DataFrame([_consensus_row("AAPL", "agreed", ["yfinance", "tiingo"], pct_diff=0.001)])

    summary = fpdc.build_fact_price_daily_consensus(df, conn=pg_conn)

    assert summary == {"loaded": 1, "skipped_no_security_key": 0, "skipped_no_date_key": 0}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT primary_source, sources_available, reconciliation_flag FROM fact_price_daily_consensus")
        source, sources_available, flag = cur.fetchone()
    assert source == "yfinance"
    assert sources_available == ["yfinance", "tiingo"]  # array round-trips correctly
    assert flag == "agreed"


def test_build_fact_price_daily_consensus_handles_null_pct_diff_for_single_source(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    df = pd.DataFrame([_consensus_row("AAPL", "single_source", ["yfinance"], pct_diff=None)])

    fpdc.build_fact_price_daily_consensus(df, conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT pct_diff, reconciliation_flag FROM fact_price_daily_consensus")
        pct_diff, flag = cur.fetchone()
    assert pct_diff is None
    assert flag == "single_source"


def test_build_fact_price_daily_consensus_upsert_replaces_row(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    df_v1 = pd.DataFrame([_consensus_row("AAPL", "agreed", ["yfinance", "tiingo"], pct_diff=0.001)])
    df_v2 = pd.DataFrame([_consensus_row("AAPL", "disagreed", ["yfinance", "tiingo"], pct_diff=0.02)])

    fpdc.build_fact_price_daily_consensus(df_v1, conn=pg_conn)
    fpdc.build_fact_price_daily_consensus(df_v2, conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), reconciliation_flag FROM fact_price_daily_consensus GROUP BY reconciliation_flag")
        assert cur.fetchall() == [(1, "disagreed")]  # replaced, not duplicated


def test_build_fact_price_daily_consensus_empty_input(pg_conn):
    assert fpdc.build_fact_price_daily_consensus(pd.DataFrame(), conn=pg_conn) == {
        "loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0,
    }
