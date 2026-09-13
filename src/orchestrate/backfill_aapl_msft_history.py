"""
One-off backfill: extends AAPL and MSFT's price history back to 2009-01-01
across both yfinance and Tiingo, reconciles them, and loads the result
into fact_price_daily_consensus (and fact_price_daily, per source).

Deliberately NOT a call to build_gold() or run_full_universe() — both
call build_dim_security(), which closes out any currently-tracked ticker
missing from the constituents snapshot passed in. Feeding it a 2-ticker
snapshot would incorrectly close every one of the ~500 OTHER current
constituents. dim_security is already correctly populated for AAPL/MSFT
(see quant-data-sdk/scripts/seed_historical_dim_security.py) — this
script does not touch it at all.

Also deliberately skips fact_corporate_action and fact_macro_rate — not
needed to unblock quant-data-sdk's price join, and each is its own
separate scope decision, not something to fold in silently.

dim_date IS touched, carefully: build_dim_date() recomputes
is_trading_day from whatever trading_days set is passed in. Using only
AAPL/MSFT's own dates for that would risk narrowing the ALREADY-CORRECT
2019+ range if some other ticker ever traded on a day these two didn't
(vanishingly unlikely for two mega-cap tickers, but not zero, so not
worth silently risking). Fixed by unioning with the most recent existing
full-universe yfinance silver partition — this can only ever extend the
trading-day set backward, never narrow it.

Run from the marketdata-lakehouse repo root:
  python -m src.orchestrate.backfill_aapl_msft_history

Prerequisite: MDL_TIINGO_API_KEY set in .env (same key already used for
the existing Tiingo sample ingestion — this costs 2 more unique symbols
against the monthly cap, trivial against the 500 limit).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.config import SILVER_DIR
from src.ingest.yfinance_source import ingest_yfinance_batch
from src.ingest.tiingo_source import ingest_tiingo_batch
from src.model.db import apply_schema, get_connection
from src.model.dim_date import build_dim_date, trading_days_from_silver
from src.model.fact_price_daily import build_fact_price_daily
from src.model.fact_price_daily_consensus import build_fact_price_daily_consensus
from src.reconcile.price_reconciliation import reconcile_prices
from src.transform.silver_prices import build_silver_prices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT"]
BACKFILL_START = date(2009, 1, 1)


def _latest_existing_yfinance_silver_date() -> date | None:
    """Most recent pre-existing full-universe yfinance silver partition,
    used only to union its trading-day set into this run's dim_date
    update so the existing 2019+ range can never be narrowed."""
    base = SILVER_DIR / "yfinance_prices"
    if not base.exists():
        return None
    existing = sorted(
        p.name.split("=", 1)[1] for p in base.iterdir()
        if p.is_dir() and p.name.startswith("ingest_date=")
    )
    return date.fromisoformat(existing[-1]) if existing else None


def main() -> None:
    today = date.today()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM fact_price_daily_consensus WHERE security_key IN "
                "(SELECT security_key FROM dim_security WHERE ticker IN ('AAPL','MSFT'))"
            )
            (before_count,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM fact_price_daily_consensus")
            (before_total,) = cur.fetchone()
        logger.info(
            "Before: %d AAPL/MSFT rows, %d total rows in fact_price_daily_consensus",
            before_count, before_total,
        )

        logger.info("--- yfinance: AAPL, MSFT, %s -> %s ---", BACKFILL_START, today)
        ingest_yfinance_batch(TICKERS, start=BACKFILL_START, end=today, ingest_date=today)

        logger.info("--- Tiingo: AAPL, MSFT, %s -> %s (2 requests total) ---", BACKFILL_START, today)
        ingest_tiingo_batch(TICKERS, start=BACKFILL_START, end=today, ingest_date=today)

        logger.info("--- silver: yfinance ---")
        yf_silver = build_silver_prices("yfinance", today)
        logger.info("--- silver: tiingo ---")
        tg_silver = build_silver_prices("tiingo", today)

        logger.info("--- dim_date: extending, unioned with latest existing full-universe silver ---")
        apply_schema(conn)
        new_trading_days = trading_days_from_silver(yf_silver)
        latest_existing = _latest_existing_yfinance_silver_date()
        if latest_existing is not None:
            existing_full_universe = pd.read_parquet(
                SILVER_DIR / "yfinance_prices" / f"ingest_date={latest_existing.isoformat()}" / "prices.parquet"
            )
            new_trading_days |= trading_days_from_silver(existing_full_universe)
            logger.info("Unioned with existing ingest_date=%s (full universe)", latest_existing)
        else:
            logger.warning("No pre-existing yfinance silver found to union — proceeding with AAPL/MSFT dates only")
        n = build_dim_date(BACKFILL_START, today, new_trading_days, conn=conn)
        logger.info("dim_date: %d rows upserted", n)

        logger.info("--- fact_price_daily (yfinance) ---")
        summary = build_fact_price_daily(yf_silver, "yfinance", conn=conn)
        logger.info("fact_price_daily/yfinance: %s", summary)

        logger.info("--- fact_price_daily (tiingo) ---")
        summary = build_fact_price_daily(tg_silver, "tiingo", conn=conn)
        logger.info("fact_price_daily/tiingo: %s", summary)

        logger.info("--- reconcile + fact_price_daily_consensus ---")
        report = reconcile_prices(today)
        print(report.render())
        summary = build_fact_price_daily_consensus(report.consensus, conn=conn)
        logger.info("fact_price_daily_consensus: %s", summary)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM fact_price_daily_consensus WHERE security_key IN "
                "(SELECT security_key FROM dim_security WHERE ticker IN ('AAPL','MSFT'))"
            )
            (after_count,) = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM fact_price_daily_consensus")
            (after_total,) = cur.fetchone()
        logger.info(
            "After: %d AAPL/MSFT rows (was %d), %d total rows (was %d)",
            after_count, before_count, after_total, before_total,
        )
        if after_total - before_total != after_count - before_count:
            logger.error(
                "Total row delta does NOT match AAPL/MSFT row delta — something "
                "outside AAPL/MSFT changed. Investigate before trusting this run."
            )
        else:
            logger.info("Total row delta matches AAPL/MSFT delta exactly — nothing else was touched.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
