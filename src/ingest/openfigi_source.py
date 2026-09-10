"""
Instrument identifier mapping via OpenFIGI.

Genuinely free — FIGI (Financial Instrument Global Identifier) is a
public-trust open standard maintained by the Object Management Group, and
OpenFIGI's mapping API carries no cost-recovery fee. An API key is
optional here: it only raises the rate limit above the unkeyed tier. This
module works without one at this pipeline's scale (~503 tickers batched
100-per-request is only ~6 HTTP calls for a full backfill); sign up free
at https://www.openfigi.com/api if 429s ever show up in practice.

Why this exists: tickers get reused and reassigned across delisted and
relisted companies, and across countries — they're not a stable identity.
A FIGI is. This module is what lets dim_security (built when the gold
layer lands) key off something durable instead of a raw ticker string.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import BRONZE_DIR, MAX_RETRIES, BACKOFF_BASE_SECONDS, OPENFIGI_API_KEY

logger = logging.getLogger(__name__)

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
_BATCH_SIZE = 100  # OpenFIGI's documented max jobs per mapping request


class RateLimited(Exception):
    """Distinct from other failures so a 429 gets retried; a genuine 4xx/5xx
    on a malformed job does not — retrying a bad request five times wastes
    calls for no benefit."""


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BACKOFF_BASE_SECONDS, min=BACKOFF_BASE_SECONDS, max=60),
    retry=retry_if_exception_type(RateLimited),
    reraise=True,
)
def _map_batch(tickers: list[str], exch_code: str = "US") -> list[dict]:
    jobs = [{"idType": "TICKER", "idValue": t, "exchCode": exch_code} for t in tickers]
    headers = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        headers["X-OPENFIGI-APIKEY"] = OPENFIGI_API_KEY

    response = requests.post(OPENFIGI_MAPPING_URL, json=jobs, headers=headers, timeout=30)
    if response.status_code == 429:
        raise RateLimited(f"OpenFIGI rate-limited this batch of {len(tickers)}")
    response.raise_for_status()

    results = response.json()
    if len(results) != len(tickers):
        raise ValueError(
            f"OpenFIGI returned {len(results)} results for {len(tickers)} requested "
            "tickers — response no longer lines up 1:1 with the request, don't trust the zip."
        )
    return results


def ingest_openfigi_mapping(
    tickers: list[str],
    exch_code: str = "US",
    ingest_date: date | None = None,
) -> pd.DataFrame:
    """
    Map `tickers` to FIGIs. Writes one bronze parquet file for the whole
    run (not per-ticker or per-batch — this is a small, single mapping
    table, unlike the per-day price series above), partitioned by
    ingest_date, and returns the frame.
    """
    ingest_date = ingest_date or date.today()
    out_dir = BRONZE_DIR / "openfigi_mapping" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed_tickers: list[str] = []

    for i in range(0, len(tickers), _BATCH_SIZE):
        batch = tickers[i : i + _BATCH_SIZE]
        try:
            results = _map_batch(batch, exch_code)
        except Exception:
            logger.exception("OpenFIGI mapping failed for batch %d-%d after retries", i, i + len(batch))
            failed_tickers.extend(batch)
            continue

        for ticker, result in zip(batch, results):
            data = result.get("data")
            if not data:
                # OpenFIGI returns {"warning": "No identifier found."} (or
                # occasionally {"error": ...}) for a job with no match,
                # rather than an HTTP error — a genuinely different case
                # from "the request itself failed", but the outcome for us
                # is the same: no FIGI for this ticker.
                logger.warning("No OpenFIGI match for %s: %s", ticker, result.get("warning") or result.get("error"))
                failed_tickers.append(ticker)
                continue

            best = data[0]  # OpenFIGI orders matches by its own relevance ranking
            rows.append(
                {
                    "ticker": ticker,
                    "figi": best.get("figi"),
                    "composite_figi": best.get("compositeFIGI"),
                    "share_class_figi": best.get("shareClassFIGI"),
                    "name": best.get("name"),
                    "security_type": best.get("securityType"),
                    "market_sector": best.get("marketSector"),
                }
            )

        # Cheap insurance against the unkeyed rate limit — at this scale
        # (a handful of batches for a one-time backfill) this costs at
        # most a few seconds total and avoids relying on 429 retries to
        # carry the whole run.
        if not OPENFIGI_API_KEY and i + _BATCH_SIZE < len(tickers):
            time.sleep(2)

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = out_dir / "mapping.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %d FIGI mappings to %s", len(df), out_path)

    if failed_tickers:
        failed_path = out_dir / "_failed_tickers.txt"
        failed_path.write_text("\n".join(failed_tickers), encoding="utf-8")
        logger.warning("%d tickers had no OpenFIGI match or failed: %s", len(failed_tickers), failed_tickers)

    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    demo_tickers = sys.argv[1:] or ["AAPL", "MSFT", "AMZN"]
    result = ingest_openfigi_mapping(demo_tickers)
    print(result.head())
