from datetime import date

import pandas as pd

from src.model import dim_security as ds


def _constituents(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _figi(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "figi"])


def test_first_snapshot_inserts_all_as_new(pg_conn):
    constituents = _constituents([
        {"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Tech", "gics_sub_industry": "Hardware"},
        {"ticker": "MSFT", "company_name": "Microsoft Corp.", "gics_sector": "Tech", "gics_sub_industry": "Software"},
    ])
    figi = _figi([{"ticker": "AAPL", "figi": "BBG000B9XRY4"}, {"ticker": "MSFT", "figi": "BBG000BPH459"}])

    summary = ds.build_dim_security(constituents, figi, date(2026, 1, 1), conn=pg_conn)

    assert summary == {"new": 2, "changed": 0, "closed": 0, "unchanged": 0}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT ticker, figi, is_current, effective_to FROM dim_security ORDER BY ticker")
        rows = cur.fetchall()
    assert rows == [("AAPL", "BBG000B9XRY4", True, None), ("MSFT", "BBG000BPH459", True, None)]


def test_missing_figi_mapping_still_inserts_security_with_null_figi(pg_conn):
    constituents = _constituents([{"ticker": "GHOST", "company_name": "Ghost Co", "gics_sector": "Tech", "gics_sub_industry": "X"}])
    figi = _figi([])  # OpenFIGI had no match for this ticker

    ds.build_dim_security(constituents, figi, date(2026, 1, 1), conn=pg_conn)

    with pg_conn.cursor() as cur:
        cur.execute("SELECT ticker, figi FROM dim_security")
        assert cur.fetchone() == ("GHOST", None)


def test_unchanged_ticker_on_second_snapshot_is_left_alone(pg_conn):
    constituents = _constituents([{"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Tech", "gics_sub_industry": "Hardware"}])
    figi = _figi([{"ticker": "AAPL", "figi": "BBG000B9XRY4"}])

    ds.build_dim_security(constituents, figi, date(2026, 1, 1), conn=pg_conn)
    summary = ds.build_dim_security(constituents, figi, date(2026, 2, 1), conn=pg_conn)

    assert summary == {"new": 0, "changed": 0, "closed": 0, "unchanged": 1}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_security")
        assert cur.fetchone()[0] == 1  # no duplicate row created


def test_changed_attribute_closes_old_row_and_opens_new_one(pg_conn):
    """The actual SCD-2 behavior: a sector reclassification should be
    visible as history, not overwritten in place."""
    figi = _figi([{"ticker": "AAPL", "figi": "BBG000B9XRY4"}])
    v1 = _constituents([{"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Tech", "gics_sub_industry": "Hardware"}])
    v2 = _constituents([{"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Consumer Discretionary", "gics_sub_industry": "Hardware"}])

    ds.build_dim_security(v1, figi, date(2026, 1, 1), conn=pg_conn)
    summary = ds.build_dim_security(v2, figi, date(2026, 3, 1), conn=pg_conn)

    assert summary == {"new": 0, "changed": 1, "closed": 0, "unchanged": 0}
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT gics_sector, is_current, effective_from, effective_to FROM dim_security "
            "WHERE ticker = 'AAPL' ORDER BY effective_from"
        )
        rows = cur.fetchall()

    assert len(rows) == 2  # full history preserved, not overwritten
    assert rows[0] == ("Tech", False, date(2026, 1, 1), date(2026, 3, 1))
    assert rows[1] == ("Consumer Discretionary", True, date(2026, 3, 1), None)


def test_ticker_dropped_from_universe_closes_its_row_with_no_replacement(pg_conn):
    figi = _figi([{"ticker": "AAPL", "figi": "BBG000B9XRY4"}, {"ticker": "DELISTED", "figi": "BBG000XXXXXX"}])
    v1 = _constituents([
        {"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Tech", "gics_sub_industry": "Hardware"},
        {"ticker": "DELISTED", "company_name": "Delisted Co", "gics_sector": "Energy", "gics_sub_industry": "Oil"},
    ])
    v2 = _constituents([{"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Tech", "gics_sub_industry": "Hardware"}])

    ds.build_dim_security(v1, figi, date(2026, 1, 1), conn=pg_conn)
    summary = ds.build_dim_security(v2, figi, date(2026, 4, 1), conn=pg_conn)

    assert summary == {"new": 0, "changed": 0, "closed": 1, "unchanged": 1}
    with pg_conn.cursor() as cur:
        cur.execute("SELECT is_current, effective_to FROM dim_security WHERE ticker = 'DELISTED'")
        assert cur.fetchone() == (False, date(2026, 4, 1))


def test_build_dim_security_raises_clearly_on_empty_constituents(pg_conn):
    """Regression test: an empty constituents frame previously produced a
    confusing pandas KeyError deep inside merge() rather than a clear
    error naming the actual problem."""
    import pytest

    with pytest.raises(ValueError, match="constituents is empty"):
        ds.build_dim_security(pd.DataFrame(), pd.DataFrame(columns=["ticker", "figi"]), date(2026, 1, 1), conn=pg_conn)


def test_at_most_one_current_row_per_ticker_is_enforced_by_the_schema(pg_conn):
    """The partial unique index in schema.sql, not just application logic,
    should make a second concurrently-current row for the same ticker
    impossible — belt and suspenders against a future bug in the loader."""
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dim_security (ticker, effective_from, effective_to, is_current) "
            "VALUES ('DUPTEST', %s, NULL, TRUE)",
            (date(2026, 1, 1),),
        )
    pg_conn.commit()

    import pytest
    import psycopg2

    with pytest.raises(psycopg2.errors.UniqueViolation):
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO dim_security (ticker, effective_from, effective_to, is_current) "
                "VALUES ('DUPTEST', %s, NULL, TRUE)",
                (date(2026, 2, 1),),
            )
    pg_conn.rollback()
