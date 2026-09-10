"""
Bulk daily OHLCV ingestion from Tiingo.

Replaces the originally planned Stooq source: Stooq's CSV download began
requiring an API key in March 2026 (confirmed via a pandas-datareader
GitHub issue after this pipeline's Stooq module returned "no usable data"
against live data), and separate reports suggest its historical server has
become unreliable even for users who do have a key. Tiingo serves the same
two roles this pipeline needed Stooq for:

1. A fallback for any ticker that fails on yfinance (see
   yfinance_source.py's _failed_tickers.txt mechanism).
2. A genuinely independent second source for cross-source reconciliation —
   Tiingo sources US equities data from IEX and other exchanges directly,
   not from Yahoo's feed, which is the actual point of a reconciliation
   check (Stooq's ultimate data provenance was less clear).

Requires a free Tiingo account and API token (https://api.tiingo.com/).
Confirmed free by multiple independent sources (the riingo R package's
docs, a QuantStart review, community free-tier trackers) but not personally
verified end-to-end — if signup asks for a card, stop and report back
before using it, same as any other resource in this project.

Free-tier limits to design around: 500 unique symbols/month, 1,000
requests/day, 50/hour. The initial ~500-ticker S&P 500 backfill uses nearly
the entire monthly unique-symbol allowance in a single run — daily
incremental updates on the same universe afterward don't count as new
uniques, so this is a one-time constraint at backfill, not an ongoing one.

The 50/hour limit is the actually binding one for a full backfill, not the
monthly quota: 503 tickers ÷ 50/hour is over 10 hours for one sitting.
Confirmed the hard way — an early full-universe run had no rate limiting
and blew straight through the hourly cap (visible as a negative "Hourly
Requests" count on Tiingo's own usage dashboard) before it was manually
stopped. Two responses to that, both in this file: a throttle below that
makes exceeding the limit physically impossible regardless of how many
tickers are requested, and a design decision (see
src/orchestrate/run_full_universe.py) to sample a sector-stratified subset
for Tiingo rather than force the full universe through a 10+ hour pull.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import BRONZE_DIR, DEFAULT_START_DATE, MAX_RETRIES, BACKOFF_BASE_SECONDS, TIINGO_API_KEY

logger = logging.getLogger(__name__)

TIINGO_PRICES_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

# Kept a little under the documented 50/hour cap for safety margin (retries,
# clock drift, any other process sharing the same account). Tracked as a
# rolling window of real request timestamps, not a fixed sleep-per-request —
# that would either be needlessly slow for short runs or still not safely
# cap a long one.
_TIINGO_SAFE_HOURLY_LIMIT = 45
_request_timestamps: deque[float] = deque()


def _throttle(now_fn=time.monotonic, sleep_fn=time.sleep) -> None:
    """Block until it's safe to make another request without exceeding the
    hourly cap. now_fn/sleep_fn are injectable so tests can simulate time
    passing without an actual multi-minute sleep."""
    now = now_fn()
    while _request_timestamps and now - _request_timestamps[0] > 3600:
        _request_timestamps.popleft()

    if len(_request_timestamps) >= _TIINGO_SAFE_HOURLY_LIMIT:
        wait_seconds = 3600 - (now - _request_timestamps[0]) + 1
        logger.info("Tiingo hourly rate-limit safety margin reached — waiting %.0fs", wait_seconds)
        sleep_fn(wait_seconds)
        now = now_fn()
        while _request_timestamps and now - _request_timestamps[0] > 3600:
            _request_timestamps.popleft()

    _request_timestamps.append(now_fn())

# Tiingo tokens are 40 lowercase hex characters. This is an observed
# pattern, not documented API contract, so it's a soft check with a helpful
# message rather than something to trust blindly — but it exists
# specifically because a copy-paste mistake (leftover placeholder text
# concatenated with the real token) previously reached Tiingo as a 403 with
# no useful diagnostic. Catching the malformed shape locally is strictly
# better than a round trip for an error that was always going to happen.
_TOKEN_PATTERN = re.compile(r"^[a-f0-9]{40}$")


def _validate_token_format(token: str) -> None:
    if not _TOKEN_PATTERN.match(token):
        preview = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else token
        raise RuntimeError(
            f"MDL_TIINGO_API_KEY doesn't look like a valid Tiingo token "
            f"(expected 40 lowercase hex characters, got {len(token)}: {preview}). "
            "Common cause: leftover placeholder text concatenated with the real "
            "token when setting the environment variable. Run "
            "`echo $env:MDL_TIINGO_API_KEY` and confirm it's exactly the token, "
            "nothing else, before retrying."
        )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BACKOFF_BASE_SECONDS, min=BACKOFF_BASE_SECONDS, max=60),
    reraise=True,
)
def _download_one(ticker: str, start: date, end: date) -> pd.DataFrame:
    if not TIINGO_API_KEY:
        raise RuntimeError(
            "MDL_TIINGO_API_KEY is not set. Sign up free at https://api.tiingo.com, "
            "get an API token from your account page, and set it as an environment "
            "variable before running this."
        )
    _validate_token_format(TIINGO_API_KEY)
    _throttle()

    url = TIINGO_PRICES_URL.format(ticker=ticker.lower())
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "token": TIINGO_API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 404:
        # Tiingo uses 404 for "no such ticker", not an empty 200 — surface
        # this distinctly rather than letting it fall into the generic
        # retry-then-fail path, since retrying a 404 five times is pointless.
        raise ValueError(f"Tiingo has no data for ticker {ticker} (404 — check symbol validity)")
    response.raise_for_status()

    records = response.json()
    if not records:
        raise ValueError(f"Tiingo returned no rows for {ticker} in range {start}..{end}")

    df = pd.DataFrame(records)
    expected = {"date", "open", "high", "low", "close", "volume", "adjClose", "adjOpen", "adjHigh", "adjLow", "adjVolume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Tiingo schema drift for {ticker}: missing {missing}, got {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker
    df = df.rename(
        columns={
            "adjClose": "adj_close",
            "adjOpen": "adj_open",
            "adjHigh": "adj_high",
            "adjLow": "adj_low",
            "adjVolume": "adj_volume",
        }
    )
    # Capture Tiingo's own adjusted OHLC directly rather than deriving it
    # later from a close/adjClose ratio — Tiingo already computed these
    # correctly; re-deriving them would just reintroduce our own rounding
    # on numbers that don't need it.
    return df[
        [
            "date", "open", "high", "low", "close", "adj_close",
            "adj_open", "adj_high", "adj_low", "adj_volume", "volume", "ticker",
        ]
    ]


def ingest_tiingo_batch(
    tickers: list[str],
    start: date = DEFAULT_START_DATE,
    end: date | None = None,
    ingest_date: date | None = None,
) -> pd.DataFrame:
    """
    Pull OHLCV for `tickers` from Tiingo. One HTTP request per ticker (no
    batch endpoint on the free tier), each independently retried. Writes
    one bronze parquet file per ticker, partitioned by ingest_date, and
    returns the concatenated frame.
    """
    end = end or date.today()
    ingest_date = ingest_date or date.today()
    out_dir = BRONZE_DIR / "tiingo_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    failed_tickers: list[str] = []

    for ticker in tickers:
        try:
            df = _download_one(ticker, start, end)
        except Exception:
            logger.exception("Tiingo download failed for %s after retries", ticker)
            failed_tickers.append(ticker)
            continue

        out_path = out_dir / f"{ticker}.parquet"
        df.to_parquet(out_path, index=False)
        frames.append(df)
        logger.info("Wrote %d rows for %s to %s", len(df), ticker, out_path)

    if failed_tickers:
        failed_path = out_dir / "_failed_tickers.txt"
        failed_path.write_text("\n".join(failed_tickers), encoding="utf-8")
        logger.warning("%d tickers failed on Tiingo: %s", len(failed_tickers), failed_tickers)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if not TIINGO_API_KEY:
        print("Set MDL_TIINGO_API_KEY first (free signup at https://api.tiingo.com).")
        sys.exit(1)
    demo_tickers = sys.argv[1:] or ["AAPL", "MSFT", "AMZN"]
    result = ingest_tiingo_batch(demo_tickers)
    print(result.head())
