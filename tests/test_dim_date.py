from datetime import date

from src.model import dim_date as dd


def test_trading_days_from_silver_extracts_distinct_dates():
    import pandas as pd

    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]), "ticker": ["A", "B", "A"]})
    days = dd.trading_days_from_silver(df)
    assert days == {date(2024, 1, 2), date(2024, 1, 3)}


def test_trading_days_from_silver_empty_returns_empty_set():
    import pandas as pd

    assert dd.trading_days_from_silver(pd.DataFrame()) == set()


def test_build_dim_date_inserts_full_range_with_correct_flags(pg_conn):
    trading_days = {date(2024, 1, 2), date(2024, 1, 3)}  # Tue, Wed

    n = dd.build_dim_date(date(2024, 1, 1), date(2024, 1, 7), trading_days, conn=pg_conn)

    assert n == 7
    with pg_conn.cursor() as cur:
        cur.execute("SELECT date, is_weekday, is_trading_day FROM dim_date ORDER BY date")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    assert rows[date(2024, 1, 1)] == (True, False)   # Monday, not a trading day (holiday, say)
    assert rows[date(2024, 1, 2)] == (True, True)     # Tuesday, traded
    assert rows[date(2024, 1, 6)] == (False, False)   # Saturday
    assert rows[date(2024, 1, 7)] == (False, False)   # Sunday


def test_build_dim_date_date_key_is_correct_yyyymmdd(pg_conn):
    dd.build_dim_date(date(2024, 3, 5), date(2024, 3, 5), set(), conn=pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT date_key FROM dim_date")
        assert cur.fetchone()[0] == 20240305


def test_build_dim_date_is_idempotent_and_refreshes_trading_flag(pg_conn):
    d = date(2024, 1, 2)
    dd.build_dim_date(d, d, set(), conn=pg_conn)  # first pass: not a trading day
    with pg_conn.cursor() as cur:
        cur.execute("SELECT is_trading_day FROM dim_date WHERE date_key = 20240102")
        assert cur.fetchone()[0] is False

    dd.build_dim_date(d, d, {d}, conn=pg_conn)  # re-run with updated trading_days
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_date")
        assert cur.fetchone()[0] == 1  # no duplicate row
        cur.execute("SELECT is_trading_day FROM dim_date WHERE date_key = 20240102")
        assert cur.fetchone()[0] is True  # flag refreshed, not stuck at first value
