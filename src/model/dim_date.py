"""
dim_date loader.

is_trading_day is derived empirically from silver price data (any date at
least one security actually traded on counts as a trading day) rather than
a hand-maintained NYSE holiday calendar — see schema.sql for the full
reasoning: a calendar needs yearly upkeep and is easy to get subtly wrong,
while the data itself can't be wrong about whether trading happened.

Practical consequence: dim_date must be built from a date range that
actually has silver data behind it. Building it against an empty or
wrong-range trading_days set would mark real trading days as non-trading —
wrong, not just incomplete — so this takes the trading-day set as an
explicit argument rather than computing it silently, forcing the caller to
be deliberate about what it's derived from.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from psycopg2.extras import execute_values

from src.model.db import get_connection


def trading_days_from_silver(silver_df: pd.DataFrame) -> set[date]:
    """The empirical definition of 'trading day' this schema uses: the set
    of distinct dates that actually appear in a silver price table."""
    if silver_df.empty:
        return set()
    return set(pd.to_datetime(silver_df["date"]).dt.date.unique())


def _date_range_rows(start: date, end: date, trading_days: set[date]) -> list[tuple]:
    rows = []
    d = start
    while d <= end:
        rows.append(
            (
                int(d.strftime("%Y%m%d")),
                d,
                d.year,
                (d.month - 1) // 3 + 1,
                d.month,
                d.strftime("%B"),
                d.day,
                d.weekday(),
                d.strftime("%A"),
                d.weekday() < 5,
                d in trading_days,
            )
        )
        d += timedelta(days=1)
    return rows


def build_dim_date(start: date, end: date, trading_days: set[date], conn=None) -> int:
    """Generate and upsert dim_date rows for [start, end] inclusive.
    Idempotent: re-running with an updated trading_days set (e.g. after
    ingesting more silver data) refreshes is_trading_day without duplicating
    rows. Returns the number of rows processed."""
    rows = _date_range_rows(start, end, trading_days)

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO dim_date (date_key, date, year, quarter, month, month_name,
                                       day, day_of_week, day_name, is_weekday, is_trading_day)
                VALUES %s
                ON CONFLICT (date_key) DO UPDATE SET is_trading_day = EXCLUDED.is_trading_day
                """,
                rows,
                page_size=5000,
            )
        conn.commit()
    finally:
        if own_conn:
            conn.close()
    return len(rows)
