from datetime import date

import pandas as pd

from src.model import dim_date as dd
from src.model import dim_security as ds
from src.model import fact_corporate_action as fca


def _seed_dims(conn, start, end, tickers):
    dd.build_dim_date(start, end, set(), conn=conn)
    constituents = pd.DataFrame(
        {"ticker": tickers, "company_name": tickers, "gics_sector": ["Tech"] * len(tickers), "gics_sub_industry": ["X"] * len(tickers)}
    )
    ds.build_dim_security(constituents, pd.DataFrame(columns=["ticker", "figi"]), start, conn=conn)


def test_build_fact_corporate_action_extracts_only_nonzero_events(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 5), ["AAPL"])
    bronze = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "dividends": [0.0, 0.24, 0.0],
            "stock_splits": [0.0, 0.0, 4.0],
            "close": [100.0, 101.0, 25.0],
        }
    )

    summary = fca.build_fact_corporate_action(bronze, conn=pg_conn)

    assert summary["loaded"] == 2  # one dividend row, one split row — 2024-01-01 has neither
    with pg_conn.cursor() as cur:
        cur.execute("SELECT action_type, value FROM fact_corporate_action ORDER BY action_type")
        rows = [(t, float(v)) for t, v in cur.fetchall()]
    assert rows == [("dividend", 0.24), ("split", 4.0)]


def test_build_fact_corporate_action_skips_ticker_not_in_dim_security(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 2), ["AAPL"])  # only AAPL seeded
    bronze = pd.DataFrame({"ticker": ["UNKNOWN"], "date": pd.to_datetime(["2024-01-01"]), "dividends": [0.5], "stock_splits": [0.0]})

    summary = fca.build_fact_corporate_action(bronze, conn=pg_conn)

    assert summary == {"loaded": 0, "skipped_no_security_key": 1, "skipped_no_date_key": 0}


def test_build_fact_corporate_action_no_events_in_range(pg_conn):
    _seed_dims(pg_conn, date(2024, 1, 1), date(2024, 1, 2), ["AAPL"])
    bronze = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2024-01-01"]), "dividends": [0.0], "stock_splits": [0.0]})

    summary = fca.build_fact_corporate_action(bronze, conn=pg_conn)

    assert summary == {"loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0}


def test_build_fact_corporate_action_empty_input(pg_conn):
    assert fca.build_fact_corporate_action(pd.DataFrame(), conn=pg_conn) == {
        "loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0,
    }
