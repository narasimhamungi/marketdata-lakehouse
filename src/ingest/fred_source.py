"""
Macro and risk-free rate series ingestion from FRED (Federal Reserve
Economic Data, St. Louis Fed).

Free, requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
(registration only, no card — same pattern as Tiingo, but FRED is a
government service, not a commercial one, so this is on materially firmer
ground than Tiingo's card-free claim).

Pulls a small, deliberately fixed set of series rather than an open-ended
list: this pipeline needs a risk-free rate curve and a couple of standard
macro references for later feature/context columns, not a general-purpose
FRED mirror. Add to FRED_SERIES if a specific downstream need shows up;
don't grow this speculatively.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import BRONZE_DIR, DEFAULT_START_DATE, MAX_RETRIES, BACKOFF_BASE_SECONDS, FRED_API_KEY

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> human-readable label, kept together so a bronze file is
# self-describing without a second lookup table.
FRED_SERIES = {
    "DGS3MO": "3-Month Treasury Constant Maturity Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "CPI, All Urban Consumers, Seasonally Adjusted",
}


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BACKOFF_BASE_SECONDS, min=BACKOFF_BASE_SECONDS, max=60),
    reraise=True,
)
def _download_series(series_id: str, start: date, end: date) -> pd.DataFrame:
    if not FRED_API_KEY:
        raise RuntimeError(
            "MDL_FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set it as an "
            "environment variable before running this."
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"FRED returned no observations for {series_id} in range {start}..{end}")

    df = pd.DataFrame(observations)
    # FRED marks non-trading/no-release days with "." rather than omitting
    # the row (e.g. CPI is monthly but the series is padded to a daily
    # cadence with "." placeholders) — drop those rather than let astype
    # blow up on them.
    df = df[df["value"] != "."].copy()
    if df.empty:
        raise ValueError(f"FRED returned only placeholder ('.') values for {series_id}")

    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = series_id
    return df[["date", "series_id", "value"]]


def ingest_fred_series(
    series_ids: list[str] | None = None,
    start: date = DEFAULT_START_DATE,
    end: date | None = None,
    ingest_date: date | None = None,
) -> pd.DataFrame:
    """
    Pull each series in `series_ids` (defaults to all of FRED_SERIES),
    write one bronze parquet file per series, partitioned by ingest_date,
    and return the concatenated frame.
    """
    series_ids = series_ids or list(FRED_SERIES.keys())
    end = end or date.today()
    ingest_date = ingest_date or date.today()
    out_dir = BRONZE_DIR / "fred_series" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    failed_series: list[str] = []

    for series_id in series_ids:
        try:
            df = _download_series(series_id, start, end)
        except Exception:
            logger.exception("FRED download failed for %s after retries", series_id)
            failed_series.append(series_id)
            continue

        out_path = out_dir / f"{series_id}.parquet"
        df.to_parquet(out_path, index=False)
        frames.append(df)
        logger.info("Wrote %d rows for %s (%s) to %s", len(df), series_id, FRED_SERIES.get(series_id, "?"), out_path)

    if failed_series:
        failed_path = out_dir / "_failed_series.txt"
        failed_path.write_text("\n".join(failed_series), encoding="utf-8")
        logger.warning("%d series failed on FRED: %s", len(failed_series), failed_series)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not FRED_API_KEY:
        print("Set MDL_FRED_API_KEY first (free key at https://fred.stlouisfed.org/docs/api/api_key.html).")
        raise SystemExit(1)
    result = ingest_fred_series()
    print(result.head())
