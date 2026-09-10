"""
Build the entire gold layer for one ingest_date, in dependency order:
dims before facts, since every fact table's foreign keys reference
dim_security/dim_date. One shared connection across the whole run — each
loader still commits its own portion, but sharing the connection avoids
reconnecting per step and keeps this one logical unit of work.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

import pandas as pd

from src.config import BRONZE_DIR, DEFAULT_START_DATE
from src.model.db import apply_schema, get_connection
from src.model.dim_date import build_dim_date, trading_days_from_silver
from src.model.dim_security import build_dim_security
from src.model.fact_corporate_action import build_fact_corporate_action
from src.model.fact_macro_rate import build_fact_macro_rate
from src.model.fact_price_daily import build_fact_price_daily
from src.model.fact_price_daily_consensus import build_fact_price_daily_consensus
from src.reconcile.price_reconciliation import reconcile_prices
from src.transform.silver_prices import build_silver_prices
from src.utils import read_bronze_partition

logger = logging.getLogger(__name__)


def read_fred_bronze(ingest_date: date) -> pd.DataFrame:
    fred_dir = BRONZE_DIR / "fred_series" / f"ingest_date={ingest_date.isoformat()}"
    if not fred_dir.exists():
        return pd.DataFrame()
    files = sorted(fred_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_gold(ingest_date: date | None = None, conn=None) -> None:
    ingest_date = ingest_date or date.today()
    own_conn = conn is None
    conn = conn or get_connection()

    try:
        logger.info("--- Applying schema ---")
        apply_schema(conn)

        # Fail loud, before writing anything, if the day's data isn't
        # actually there — the alternative (proceeding anyway) previously
        # let dim_date silently build with every date marked
        # is_trading_day=False, since an empty silver frame produces an
        # empty trading-days set with no error at all. A bad row count
        # here is a much better failure mode than bad data sitting quietly
        # in the database.
        logger.info("--- dim_date ---")
        yf_silver = build_silver_prices("yfinance", ingest_date)
        if yf_silver.empty:
            raise ValueError(
                f"No yfinance silver data for ingest_date={ingest_date.isoformat()}. "
                "Refusing to build dim_date from this — it would mark every date as "
                "a non-trading day. Check the date argument (bronze/silver data is "
                "dated by when it was actually pulled, not necessarily today) and "
                "that ingestion has run for this date."
            )
        trading_days = trading_days_from_silver(yf_silver)
        n = build_dim_date(DEFAULT_START_DATE, ingest_date, trading_days, conn=conn)
        logger.info("dim_date: %d rows", n)

        logger.info("--- dim_security ---")
        constituents = read_bronze_partition(BRONZE_DIR / "constituents", ingest_date)
        openfigi = read_bronze_partition(BRONZE_DIR / "openfigi_mapping", ingest_date)
        summary = build_dim_security(constituents, openfigi, ingest_date, conn=conn)
        logger.info("dim_security: %s", summary)

        logger.info("--- fact_price_daily (yfinance) ---")
        summary = build_fact_price_daily(yf_silver, "yfinance", conn=conn)
        logger.info("fact_price_daily/yfinance: %s", summary)

        logger.info("--- fact_price_daily (tiingo) ---")
        tg_silver = build_silver_prices("tiingo", ingest_date)
        summary = build_fact_price_daily(tg_silver, "tiingo", conn=conn)
        logger.info("fact_price_daily/tiingo: %s", summary)

        logger.info("--- fact_price_daily_consensus ---")
        report = reconcile_prices(ingest_date)
        summary = build_fact_price_daily_consensus(report.consensus, conn=conn)
        logger.info("fact_price_daily_consensus: %s", summary)

        logger.info("--- fact_corporate_action ---")
        yf_bronze = read_bronze_partition(BRONZE_DIR / "yfinance_prices", ingest_date)
        summary = build_fact_corporate_action(yf_bronze, conn=conn)
        logger.info("fact_corporate_action: %s", summary)

        logger.info("--- fact_macro_rate ---")
        fred_df = read_fred_bronze(ingest_date)
        summary = build_fact_macro_rate(fred_df, conn=conn)
        logger.info("fact_macro_rate: %s", summary)

        logger.info("Gold layer build complete for ingest_date=%s", ingest_date.isoformat())
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    build_gold(target_date)
