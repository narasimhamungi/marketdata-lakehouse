"""
fact_price_daily loader.

Loads silver price data at (security, date, source) grain — called once
per source (yfinance, Tiingo), preserving both vendors distinctly. This is
what fact_price_daily_consensus's reconciliation-derived logic is built on
top of; this table itself carries no notion of which source is "right."
"""
from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_values

from src.model.db import get_connection


def build_fact_price_daily(silver_df: pd.DataFrame, source: str, conn=None) -> dict:
    if silver_df.empty:
        return {"loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0}

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date_key, date FROM dim_date")
            date_key_by_date = {row[1]: row[0] for row in cur.fetchall()}
            cur.execute("SELECT security_key, ticker FROM dim_security WHERE is_current")
            security_key_by_ticker = {row[1]: row[0] for row in cur.fetchall()}

        df = silver_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["security_key"] = df["ticker"].map(security_key_by_ticker)
        df["date_key"] = df["date"].map(date_key_by_date)

        skipped_security = int(df["security_key"].isna().sum())
        skipped_date = int(df["date_key"].isna().sum())
        df = df.dropna(subset=["security_key", "date_key"])

        rows = []
        for r in df.itertuples(index=False):
            rows.append(
                (
                    int(r.security_key), int(r.date_key), source,
                    float(r.open), float(r.high), float(r.low), float(r.close),
                    float(r.adj_open), float(r.adj_high), float(r.adj_low), float(r.adj_close),
                    int(r.volume) if pd.notna(r.volume) else None,
                    int(r.adj_volume) if pd.notna(r.adj_volume) else None,
                )
            )

        if rows:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO fact_price_daily
                        (security_key, date_key, source, open, high, low, close,
                         adj_open, adj_high, adj_low, adj_close, volume, adj_volume)
                    VALUES %s
                    ON CONFLICT (security_key, date_key, source) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, adj_open = EXCLUDED.adj_open,
                        adj_high = EXCLUDED.adj_high, adj_low = EXCLUDED.adj_low,
                        adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume,
                        adj_volume = EXCLUDED.adj_volume
                    """,
                    rows,
                    # Default page_size (100) means ~9,500 separate
                    # round-trips for a full-universe load (~950K rows) —
                    # each also paying real FK-constraint-check cost and,
                    # via Docker, real network latency per round-trip.
                    # Confirmed slow against live data; larger batches
                    # cut the round-trip count by ~50x.
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
