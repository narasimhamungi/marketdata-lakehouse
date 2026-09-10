"""
fact_corporate_action loader.

Built from yfinance bronze's dividends/stock_splits columns — captured at
ingestion time (yf.download(..., actions=True)) but unused until now. A
row only exists here where dividends != 0 or stock_splits != 0; every
other (ticker, date) combination implicitly had no corporate action that
day. This is deliberately sparse, not a dense fact table like
fact_price_daily — a corporate action is a rare event, not a daily
measurement.

Source is fixed at "yfinance" for now since that's the only source this
pipeline captures actions from (Tiingo's raw response has divCash/
splitFactor fields per the module docstring, but they're not currently
ingested — a natural extension if a second source for this fact is ever
wanted).
"""
from __future__ import annotations

import pandas as pd
from psycopg2.extras import execute_values

from src.model.db import get_connection


def build_fact_corporate_action(yfinance_bronze: pd.DataFrame, conn=None) -> dict:
    if yfinance_bronze.empty:
        return {"loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0}

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date_key, date FROM dim_date")
            date_key_by_date = {row[1]: row[0] for row in cur.fetchall()}
            cur.execute("SELECT security_key, ticker FROM dim_security WHERE is_current")
            security_key_by_ticker = {row[1]: row[0] for row in cur.fetchall()}

        df = yfinance_bronze.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        pieces = []
        if "dividends" in df.columns:
            divs = df[df["dividends"].fillna(0) != 0][["ticker", "date", "dividends"]].rename(columns={"dividends": "value"})
            divs["action_type"] = "dividend"
            pieces.append(divs)
        if "stock_splits" in df.columns:
            splits = df[df["stock_splits"].fillna(0) != 0][["ticker", "date", "stock_splits"]].rename(columns={"stock_splits": "value"})
            splits["action_type"] = "split"
            pieces.append(splits)

        if not pieces:
            return {"loaded": 0, "skipped_no_security_key": 0, "skipped_no_date_key": 0}

        combined = pd.concat(pieces, ignore_index=True)
        combined["security_key"] = combined["ticker"].map(security_key_by_ticker)
        combined["date_key"] = combined["date"].map(date_key_by_date)

        skipped_security = int(combined["security_key"].isna().sum())
        skipped_date = int(combined["date_key"].isna().sum())
        combined = combined.dropna(subset=["security_key", "date_key"])

        rows = [
            (int(r.security_key), int(r.date_key), r.action_type, float(r.value), "yfinance")
            for r in combined.itertuples(index=False)
        ]

        if rows:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO fact_corporate_action (security_key, date_key, action_type, value, source)
                    VALUES %s
                    ON CONFLICT (security_key, date_key, action_type, source)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    rows,
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
