"""
fact_macro_rate loader.

Straight load from FRED bronze — no security dimension involved, and no
reconciliation logic: FRED is a single authoritative government source
here, not two vendors being compared against each other. date_key is
resolved via dim_date, which means dim_date must already cover the FRED
series' date range before this runs — any date without a dim_date row is
skipped and counted, not silently dropped.
"""
from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_values

from src.model.db import get_connection


def build_fact_macro_rate(fred_df: pd.DataFrame, conn=None) -> dict:
    """
    fred_df: columns date, series_id, value — the shape produced by
    src.ingest.fred_source. Returns {"loaded": n, "skipped_no_date_key": n}.
    """
    if fred_df.empty:
        return {"loaded": 0, "skipped_no_date_key": 0}

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date_key, date FROM dim_date")
            date_key_by_date = {row[1]: row[0] for row in cur.fetchall()}

        df = fred_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["date_key"] = df["date"].map(date_key_by_date)

        skipped = int(df["date_key"].isna().sum())
        df = df.dropna(subset=["date_key"])

        rows = [(int(r.date_key), r.series_id, float(r.value)) for r in df.itertuples(index=False)]

        if rows:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO fact_macro_rate (date_key, series_id, value) VALUES %s "
                    "ON CONFLICT (date_key, series_id) DO UPDATE SET value = EXCLUDED.value",
                    rows,
                    page_size=5000,
                )
            conn.commit()

        return {"loaded": len(rows), "skipped_no_date_key": skipped}
    finally:
        if own_conn:
            conn.close()
