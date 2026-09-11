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

import json
import logging
import re
import time
from collections import deque
from datetime import date

import pandas as pd
import requests
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

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

# _request_timestamps is in-memory only — it resets to empty every time this
# process restarts, but Tiingo's own server-side hourly counter does not
# reset just because our script restarted. A restarted run (after a crash,
# Ctrl+C, or simply re-running the command) previously looked "safe" to the
# throttle while the account could already be close to the real cap from
# the prior process's requests, risking a genuine 429 rather than our own
# defensive wait. This file persists the rolling window to disk so a new
# process picks up where the last one left off.
_RATE_LIMIT_STATE_FILENAME = "_rate_limit_state.json"


def _rate_limit_state_path():
    return BRONZE_DIR / "tiingo_prices" / _RATE_LIMIT_STATE_FILENAME


def _load_persisted_request_timestamps() -> list[float]:
    path = _rate_limit_state_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [float(ts) for ts in data.get("request_times", [])]
    except (json.JSONDecodeError, ValueError, OSError):
        logger.warning("Could not read %s — starting with no rate-limit history", path)
        return []


def _persist_request_timestamps() -> None:
    path = _rate_limit_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request_times": list(_request_timestamps)}))


def _seed_request_timestamps_from_disk() -> None:
    """Called once at the top of a batch run. Merges any still-fresh
    (< 1 hour old) persisted timestamps into the in-memory deque, so the
    throttle below is immediately aware of requests made by a previous,
    now-exited process — not just this one."""
    now = time.time()
    for ts in _load_persisted_request_timestamps():
        if now - ts <= 3600 and ts not in _request_timestamps:
            _request_timestamps.append(ts)


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
    retry=retry_if_not_exception_type(ValueError),
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
    # now_fn=time.time (wall clock, not the default time.monotonic) because
    # this is the real production call site: the persisted rate-limit state
    # above is written and read as wall-clock timestamps, which are
    # meaningful across process restarts. time.monotonic()'s origin is only
    # defined within a single process's lifetime, so comparing a monotonic
    # value persisted by one process against another process's clock is not
    # reliable. Tests inject their own fake now_fn and never touch this
    # default, so this doesn't affect _throttle's own test coverage.
    _throttle(now_fn=time.time)

    url = TIINGO_PRICES_URL.format(ticker=ticker.lower())
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "token": TIINGO_API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 404:
        # Tiingo uses 404 for "no such ticker", not an empty 200 — this is
        # deterministic (retrying will never change the answer), which is
        # exactly what excluding ValueError from the retry policy above is
        # for: this now actually stops on the first attempt instead of
        # burning 5 throttle slots retrying an error that was always going
        # to happen again.
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
    force: bool = False,
) -> pd.DataFrame:
    """
    Pull OHLCV for `tickers` from Tiingo. One HTTP request per ticker (no
    batch endpoint on the free tier), each independently retried. Writes
    one bronze parquet file per ticker, partitioned by ingest_date, and
    returns the concatenated frame.

    Resumable by default: a ticker whose output file already exists for
    this ingest_date is skipped rather than re-fetched, so a run
    interrupted partway through (rate limit, crash, Ctrl+C) can simply be
    re-run and only picks up where it left off, instead of re-spending
    quota on tickers that already succeeded. Pass force=True to re-fetch
    everything regardless.
    """
    end = end or date.today()
    ingest_date = ingest_date or date.today()
    out_dir = BRONZE_DIR / "tiingo_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_request_timestamps_from_disk()

    frames = []
    failed_tickers: list[str] = []
    skipped = 0

    for ticker in tickers:
        out_path = out_dir / f"{ticker}.parquet"
        if out_path.exists() and not force:
            skipped += 1
            frames.append(pd.read_parquet(out_path))
            continue

        try:
            df = _download_one(ticker, start, end)
        except Exception:
            logger.exception("Tiingo download failed for %s after retries", ticker)
            failed_tickers.append(ticker)
            continue
        finally:
            # Persist after every real attempt, success or failure — a
            # request was made against the account's real hourly counter
            # either way, and this must survive the process being killed
            # mid-run for the resume-on-restart logic above to be safe.
            _persist_request_timestamps()

        df.to_parquet(out_path, index=False)
        frames.append(df)
        logger.info("Wrote %d rows for %s to %s", len(df), ticker, out_path)

    if skipped:
        logger.info(
            "Skipped %d/%d tickers already present for ingest_date=%s (pass force=True to re-fetch)",
            skipped, len(tickers), ingest_date,
        )

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