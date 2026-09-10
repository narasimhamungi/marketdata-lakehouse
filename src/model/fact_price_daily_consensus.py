"""
fact_price_daily_consensus loader.

Takes a ReconciliationReport's .consensus DataFrame (src.reconcile.
price_reconciliation) and loads it directly — this is the one place in
the gold layer where reconciliation stops being a report you read and
becomes a queryable product: every row already carries which source it
came from, which sources were even available, and whether the two
vendors agreed, disagreed, or only one had data at all. A quant querying
this table doesn't need to know reconciliation happened; the outcome is
just a column.

sources_available (a Postgres TEXT[]) is passed through as a plain Python
list — psycopg2 adapts that to an array literal automatically, no manual
formatting needed.
"""
from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_values

from src.model.db import get_connection


def build_fact_price_daily_consensus(consensus_df: pd.DataFrame, conn=None) -> dict:
    if consensus_df.empty:
        return {"loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0}

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date_key, date FROM dim_date")
            date_key_by_date = {row[1]: row[0] for row in cur.fetchall()}
            cur.execute("SELECT security_key, ticker FROM dim_security WHERE is_current")
            security_key_by_ticker = {row[1]: row[0] for row in cur.fetchall()}

        df = consensus_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["security_key"] = df["ticker"].map(security_key_by_ticker)
        df["date_key"] = df["date"].map(date_key_by_date)

        skipped_security = int(df["security_key"].isna().sum())
        skipped_date = int(df["date_key"].isna().sum())
        df = df.dropna(subset=["security_key", "date_key"])

        rows = []
        for r in df.itertuples(index=False):
            pct_diff = float(r.pct_diff) if pd.notna(r.pct_diff) else None
            rows.append(
                (
                    int(r.security_key), int(r.date_key), float(r.adj_close),
                    r.primary_source, list(r.sources_available), pct_diff, r.reconciliation_flag,
                )
            )

        if rows:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO fact_price_daily_consensus
                        (security_key, date_key, adj_close, primary_source,
                         sources_available, pct_diff, reconciliation_flag)
                    VALUES %s
                    ON CONFLICT (security_key, date_key) DO UPDATE SET
                        adj_close = EXCLUDED.adj_close,
                        primary_source = EXCLUDED.primary_source,
                        sources_available = EXCLUDED.sources_available,
                        pct_diff = EXCLUDED.pct_diff,
                        reconciliation_flag = EXCLUDED.reconciliation_flag
                    """,
                    rows,
                    # See fact_price_daily.py for why this is here — same
                    # scale, same fix.
                    page_size=5000,
                )
            conn.commit()

        return {
            "loaded": len(rows),
            "skipped_no_security_key": skipped_security,
            "skipped_no_date_key": skipped_date,
        }
    finally:
        if own_conn:
            conn.close()
