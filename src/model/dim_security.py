"""
dim_security loader — SCD-2 over constituents snapshots.

Compares a new constituents snapshot against the current (is_current=TRUE)
row per ticker: an unchanged ticker is left alone, a changed ticker gets
its old row closed (effective_to = snapshot_date, is_current = FALSE) and
a new row opened, a new ticker gets a fresh row, and a ticker no longer
present in the snapshot gets its current row closed with no replacement
(index membership ended).

Tracked attributes: company_name, gics_sector, gics_sub_industry, figi.
figi comes from a separate OpenFIGI mapping snapshot, not the constituents
data itself, and is left-joined in — a security whose OpenFIGI mapping
failed still gets a dim_security row, just with figi = NULL, rather than
being silently dropped from the dimension because one upstream source had
a gap.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.model.db import get_connection

TRACKED_ATTRIBUTES = ["company_name", "gics_sector", "gics_sub_industry", "figi"]


def _fetch_current_rows(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT security_key, ticker, company_name, gics_sector, gics_sub_industry, figi "
            "FROM dim_security WHERE is_current"
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _attrs(row: pd.Series) -> dict:
    return {a: (row[a] if a in row.index and pd.notna(row[a]) else None) for a in TRACKED_ATTRIBUTES}


def build_dim_security(
    constituents: pd.DataFrame,
    openfigi_mapping: pd.DataFrame,
    snapshot_date: date,
    conn=None,
) -> dict:
    """
    Apply one constituents snapshot as an SCD-2 update.
    Returns {"new": n, "changed": n, "closed": n, "unchanged": n}.
    """
    if constituents.empty:
        raise ValueError(
            "constituents is empty — nothing to load. This usually means no "
            "bronze constituents data exists for the given date (check for a "
            "midnight-rollover date mismatch, or that ingestion has actually "
            "run for this date), not that the index genuinely has zero members."
        )

    figi_cols = openfigi_mapping[["ticker", "figi"]] if not openfigi_mapping.empty else pd.DataFrame(columns=["ticker", "figi"])
    merged = constituents.merge(figi_cols, on="ticker", how="left").drop_duplicates(subset="ticker")

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        current = _fetch_current_rows(conn)
        current_by_ticker = current.set_index("ticker") if not current.empty else current
        existing_tickers = set(current_by_ticker.index) if not current.empty else set()
        new_tickers = set(merged["ticker"])

        summary = {"new": 0, "changed": 0, "closed": 0, "unchanged": 0}

        with conn.cursor() as cur:
            for _, row in merged.iterrows():
                ticker = row["ticker"]
                new_attrs = _attrs(row)

                if ticker not in existing_tickers:
                    cur.execute(
                        """
                        INSERT INTO dim_security
                            (ticker, company_name, gics_sector, gics_sub_industry, figi,
                             effective_from, effective_to, is_current)
                        VALUES (%s, %s, %s, %s, %s, %s, NULL, TRUE)
                        """,
                        (ticker, new_attrs["company_name"], new_attrs["gics_sector"],
                         new_attrs["gics_sub_industry"], new_attrs["figi"], snapshot_date),
                    )
                    summary["new"] += 1
                    continue

                old_row = current_by_ticker.loc[ticker]
                old_attrs = _attrs(old_row)
                if old_attrs == new_attrs:
                    summary["unchanged"] += 1
                    continue

                cur.execute(
                    "UPDATE dim_security SET effective_to = %s, is_current = FALSE WHERE security_key = %s",
                    (snapshot_date, int(old_row["security_key"])),
                )
                cur.execute(
                    """
                    INSERT INTO dim_security
                        (ticker, company_name, gics_sector, gics_sub_industry, figi,
                         effective_from, effective_to, is_current)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, TRUE)
                    """,
                    (ticker, new_attrs["company_name"], new_attrs["gics_sector"],
                     new_attrs["gics_sub_industry"], new_attrs["figi"], snapshot_date),
                )
                summary["changed"] += 1

            for ticker in existing_tickers - new_tickers:
                old_row = current_by_ticker.loc[ticker]
                cur.execute(
                    "UPDATE dim_security SET effective_to = %s, is_current = FALSE WHERE security_key = %s",
                    (snapshot_date, int(old_row["security_key"])),
                )
                summary["closed"] += 1

        conn.commit()
    finally:
        if own_conn:
            conn.close()

    return summary
