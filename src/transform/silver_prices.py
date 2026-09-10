"""
Silver transform: bronze -> typed, deduplicated, fully-adjusted OHLCV.

Reads the latest (or a specified) bronze partition for one price source
and produces a clean per-source silver table. The two sources go through
this independently, on purpose: reconciliation (the next build step)
compares the two silver outputs against each other, so blending them here
would destroy the very thing reconciliation needs to check.

Each bronze snapshot already contains the full historical range (every
ingestion run re-pulls from DEFAULT_START_DATE, not an incremental delta),
so this operates on a single partition, not a union across ingest_dates.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.config import BRONZE_DIR, SILVER_DIR
from src.utils import read_bronze_partition

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "adj_close", "volume", "ticker"}
FINAL_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "adj_close",
    "adj_open", "adj_high", "adj_low", "volume", "adj_volume",
]


def _type_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # daily data — strip any stray time component
    numeric_cols = [c for c in df.columns if c not in ("date", "ticker")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["ticker"] = df["ticker"].astype(str)
    return df


def _dedupe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    dropped = before - len(df)
    if dropped:
        logger.warning("%s: dropped %d duplicate (ticker, date) rows", source, dropped)
    return df


def _derive_adjusted_ohlc_from_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    For sources that only give adj_close (yfinance), derive adjusted
    open/high/low the conventional way: apply that day's adj_close/close
    ratio uniformly to open/high/low — this is what a split or dividend
    adjustment does to a whole bar, and it's effectively what
    yfinance's own auto_adjust=True does internally. Doing it explicitly
    here keeps the adjustment visible and auditable rather than hidden
    behind a library flag.

    Volume is intentionally left as-reported, not split-adjusted — a known
    limitation (see README), not a guess dressed up as a real adjustment.
    """
    if (df["close"] <= 0).any():
        raise ValueError("Cannot derive adjustment factor: found close <= 0 in bronze data that should have already passed quality checks")

    df = df.copy()
    factor = df["adj_close"] / df["close"]
    df["adj_open"] = df["open"] * factor
    df["adj_high"] = df["high"] * factor
    df["adj_low"] = df["low"] * factor
    df["adj_volume"] = df["volume"]
    return df


def build_silver_prices(source: str, ingest_date: date | None = None) -> pd.DataFrame:
    if source not in ("yfinance", "tiingo"):
        raise ValueError(f"Unknown price source: {source!r} (expected 'yfinance' or 'tiingo')")

    ingest_date = ingest_date or date.today()
    bronze_dir = BRONZE_DIR / f"{source}_prices"
    df = read_bronze_partition(bronze_dir, ingest_date)
    if df.empty:
        logger.warning("No bronze data for %s at ingest_date=%s", source, ingest_date)
        return df

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source} bronze is missing expected columns: {missing}")

    df = _type_columns(df)
    df = _dedupe(df, source)

    # Use source-native adjusted OHLC when the bronze data actually has it
    # (current Tiingo bronze does; yfinance never provides per-field
    # adjusted OHLC, only adj_close) — checked by column presence, not by
    # hardcoding "yfinance always derives", so this also handles older
    # Tiingo bronze partitions pulled before adj_open/high/low/volume were
    # added to that ingester, without needing a re-run to unblock silver.
    native_adj_columns = {"adj_open", "adj_high", "adj_low", "adj_volume"}
    if native_adj_columns.issubset(df.columns):
        logger.info("%s: using source-native adjusted OHLC", source)
    else:
        logger.info("%s: source-native adjusted OHLC not present, deriving from adj_close/close factor", source)
        df = _derive_adjusted_ohlc_from_factor(df)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df[FINAL_COLUMNS]

    out_dir = SILVER_DIR / f"{source}_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d silver rows for %s to %s", len(df), source, out_path)

    return df


def read_silver_prices(source: str, ingest_date: date) -> pd.DataFrame:
    """
    Read already-built silver output without rebuilding it.

    build_silver_prices() both computes AND writes — calling it from every
    downstream task that needs the silver data (gold facts, corporate
    actions) would redundantly recompute the same transform each time.
    This exists specifically for the Airflow DAG: tasks communicate via
    the persisted parquet artifact, not by recalling the function that
    produced it, matching how Airflow tasks are meant to hand off data —
    through external state, not by re-executing upstream work.
    """
    path = SILVER_DIR / f"{source}_prices" / f"ingest_date={ingest_date.isoformat()}" / "prices.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    for src in ("yfinance", "tiingo"):
        result = build_silver_prices(src, target_date)
        if not result.empty:
            print(f"\n{src} silver sample:")
            print(result.head())
