"""
Bulk daily OHLCV ingestion from Yahoo Finance via yfinance.

yfinance wraps Yahoo's *unofficial* internal endpoints (there is no official
Yahoo Finance API — it was discontinued in 2017). It is free and has no
usage-based billing, but it is not a contracted service: Yahoo can and does
rate-limit or reshape these endpoints without notice. Two defensive choices
follow directly from that:

1. Batch requests via yf.download() with a ticker list, rather than looping
   Ticker-by-Ticker — this is both faster and less likely to trip rate limits.
2. Treat every call as fallible: retry with exponential backoff, and let the
   caller fall back to Tiingo (src/ingest/tiingo_source.py) for any ticker that
   still fails after retries. That fallback is also what makes Tiingo useful
   as a genuine second source for reconciliation, not just a formality.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import BRONZE_DIR, DEFAULT_START_DATE, MAX_RETRIES, BACKOFF_BASE_SECONDS

logger = logging.getLogger(__name__)

# yfinance raises plain Exception subclasses for HTTP/parsing failures rather
# than one specific type, so we retry broadly but log what actually failed.
RETRYABLE_EXCEPTIONS = (Exception,)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BACKOFF_BASE_SECONDS, min=BACKOFF_BASE_SECONDS, max=120),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)
def _download_batch(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """One retried batch call. Isolated from the rest of the module so tests
    can mock exactly this function without reimplementing retry logic."""
    df = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        group_by="ticker",
        auto_adjust=False,  # we want raw + adj-close separately, not pre-merged
        actions=True,       # include dividends/splits for corporate-action handling later
        threads=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"yfinance returned an empty frame for batch of {len(tickers)} tickers")
    return df


def _reshape_batch(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    yf.download's multi-ticker output is a wide frame with a MultiIndex on
    columns: (ticker, field). Reshape to long/tidy: one row per
    (ticker, date), which is what the silver/gold loaders expect.
    """
    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            logger.warning("No data returned for %s — dropped or delisted?", ticker)
            continue
        sub = raw[ticker].copy()
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        sub["ticker"] = ticker
        # yfinance names this index level "Date" in practice, but that's an
        # observed convention, not a documented contract — normalize it
        # explicitly instead of trusting reset_index() to produce "date".
        sub.index.name = "date"
        sub = sub.reset_index().rename(columns=str.lower)
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    tidy = pd.concat(frames, ignore_index=True)
    expected = {"date", "open", "high", "low", "close", "adj close", "volume", "ticker"}
    missing = expected - set(tidy.columns)
    if missing:
        # Dividends/Stock Splits columns are optional per-ticker; everything
        # else is required for a usable price row.
        core_missing = missing - {"dividends", "stock splits"}
        if core_missing:
            raise ValueError(f"yfinance reshape missing core columns: {core_missing}")

    return tidy.rename(columns={"adj close": "adj_close", "stock splits": "stock_splits"})


def ingest_yfinance_batch(
    tickers: list[str],
    start: date = DEFAULT_START_DATE,
    end: date | None = None,
    batch_size: int = 50,
    ingest_date: date | None = None,
) -> pd.DataFrame:
    """
    Pull OHLCV for `tickers` in batches, write one bronze parquet file per
    batch (partitioned by ingest_date), and return the concatenated frame.

    Batching at 50 tickers balances two failure modes: too large a batch and
    one bad ticker or a transient error forces a full-batch retry; too small
    and we make far more HTTP round trips than necessary.
    """
    end = end or date.today()
    ingest_date = ingest_date or date.today()
    out_dir = BRONZE_DIR / "yfinance_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    failed_tickers: list[str] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = _download_batch(batch, start, end)
            tidy = _reshape_batch(raw, batch)
        except Exception:
            logger.exception("Batch %d-%d failed after retries; marking for Tiingo fallback", i, i + len(batch))
            failed_tickers.extend(batch)
            continue

        # yf.download() does NOT raise when one ticker inside an otherwise-
        # successful batch fails — a known cause is a "database is locked"
        # error from yfinance's own SQLite-backed cache under concurrent
        # (threaded) access. It just logs a warning and omits that ticker.
        # A batch that "succeeded" can therefore still be silently missing
        # tickers; diff requested vs. returned explicitly rather than only
        # checking whether the batch call raised.
        succeeded = set(tidy["ticker"].unique()) if not tidy.empty else set()
        missing = [t for t in batch if t not in succeeded]
        if missing:
            logger.warning(
                "%d of %d tickers in batch %d-%d returned no data (delisted, "
                "or a transient error such as yfinance's SQLite cache lock) "
                "— routed to Tiingo fallback: %s",
                len(missing), len(batch), i, i + len(batch), missing,
            )
            failed_tickers.extend(missing)

        if tidy.empty:
            continue

        batch_path = out_dir / f"batch_{i:05d}.parquet"
        tidy.to_parquet(batch_path, index=False)
        all_frames.append(tidy)
        logger.info("Batch %d-%d: wrote %d rows to %s", i, i + len(batch), len(tidy), batch_path)

    if failed_tickers:
        failed_path = out_dir / "_failed_tickers.txt"
        failed_path.write_text("\n".join(failed_tickers), encoding="utf-8")
        logger.warning(
            "%d tickers failed on yfinance and were written to %s for Tiingo fallback",
            len(failed_tickers),
            failed_path,
        )

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    demo_tickers = sys.argv[1:] or ["AAPL", "MSFT", "AMZN"]
    result = ingest_yfinance_batch(demo_tickers)
    print(result.head())
