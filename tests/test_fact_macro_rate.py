from datetime import date

import pandas as pd

from src.model import dim_date as dd
from src.model import fact_macro_rate as fmr


def _seed_dim_date(conn, start, end):
    dd.build_dim_date(start, end, set(), conn=conn)


def test_build_fact_macro_rate_loads_rows_with_resolved_date_key(pg_conn):
    _seed_dim_date(pg_conn, date(2024, 1, 1), date(2024, 1, 5))
    fred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "series_id": ["DGS10", "DGS10"],
            "value": [4.1, 4.2],
        }
    )

    summary = fmr.build_fact_macro_rate(fred_df, conn=pg_conn)

    assert summary == {"loaded": 2, "skipped_no_date_key": 0}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT series_id, value FROM fact_macro_rate ORDER BY date_key")
        # Postgres NUMERIC comes back as Decimal, not float — compare as
        # float explicitly rather than relying on Decimal == float, which
        # is False even for equal values due to float imprecision.
        rows = [(sid, float(v)) for sid, v in cur.fetchall()]
    assert rows == [("DGS10", 4.1), ("DGS10", 4.2)]


def test_build_fact_macro_rate_skips_and_counts_dates_missing_from_dim_date(pg_conn):
    _seed_dim_date(pg_conn, date(2024, 1, 1), date(2024, 1, 2))  # dim_date doesn't cover 2024-06-01
    fred_df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-06-01"]), "series_id": ["FEDFUNDS", "FEDFUNDS"], "value": [5.0, 5.5]})

    summary = fmr.build_fact_macro_rate(fred_df, conn=pg_conn)

    assert summary == {"loaded": 1, "skipped_no_date_key": 1}


def test_build_fact_macro_rate_upsert_updates_existing_value(pg_conn):
    _seed_dim_date(pg_conn, date(2024, 1, 1), date(2024, 1, 1))
    df_v1 = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "series_id": ["DGS10"], "value": [4.0]})
    df_v2 = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "series_id": ["DGS10"], "value": [4.5]})  # a revision

    fmr.build_fact_macro_rate(df_v1, conn=pg_conn)
    fmr.build_fact_macro_rate(df_v2, conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), value FROM fact_macro_rate GROUP BY value")
        count, value = cur.fetchone()
    assert count == 1
    assert float(value) == 4.5  # updated in place, not duplicated


def test_build_fact_macro_rate_empty_input(pg_conn):
    assert fmr.build_fact_macro_rate(pd.DataFrame(), conn=pg_conn) == {"loaded": 0, "skipped_no_date_key": 0}
