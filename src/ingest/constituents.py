"""
Ingest the current S&P 500 constituent list.

Source: Wikipedia's "List of S&P 500 companies" table. This is the standard
free source for index membership (there is no free, authoritative, live API
for current S&P 500 membership) and is factual/tabular data, not creative
content: ticker, company name, GICS sector, and date-added, refreshed by
editors whenever the index changes.

This is written as a *snapshot* ingester deliberately: every run writes a
new dated file under bronze/constituents/, never overwrites the previous
one. That history is what makes dim_security's SCD-2 (Slowly Changing
Dimension type 2) possible later — we need to know not just today's
membership but when a ticker joined or left the index.
"""
from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import BRONZE_DIR, MAX_RETRIES, BACKOFF_BASE_SECONDS, WIKIPEDIA_SP500_URL

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Date added"}

# Wikipedia (like most sites) rejects requests carrying urllib's default
# User-Agent ("Python-urllib/x.y") as bot-like — pd.read_html(url) delegates
# straight to urllib and hits this. Fetching the page ourselves with a
# descriptive UA sidesteps it without depending on pandas' storage_options
# header-forwarding, which has been inconsistent across pandas/fsspec versions.
_USER_AGENT = "marketdata-lakehouse/0.1 (contact: narasimhamungi@gmail.com)"


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=BACKOFF_BASE_SECONDS, min=BACKOFF_BASE_SECONDS, max=60),
    reraise=True,
)
def fetch_constituents_table(source_url: str = WIKIPEDIA_SP500_URL) -> pd.DataFrame:
    """
    Pull the current constituents table.

    Raises if the page's table structure has drifted from what we expect,
    rather than silently ingesting a malformed frame — a wrong or missing
    column here corrupts dim_security downstream, so fail loud and early.
    """
    response = requests.get(source_url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Constituents source schema drift: missing columns {missing}. "
            f"Got columns: {list(df.columns)}. Source table structure may have changed."
        )

    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Date added": "date_added",
        }
    )
    # Yahoo/Stooq use '-' where the index page uses '.' (e.g. BRK.B -> BRK-B).
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    cols = ["ticker", "company_name", "gics_sector", "gics_sub_industry", "date_added"]
    return df[cols].drop_duplicates(subset="ticker").reset_index(drop=True)


def ingest_constituents_snapshot(as_of: date | None = None) -> pd.DataFrame:
    """Fetch constituents and write a dated bronze snapshot. Returns the frame."""
    as_of = as_of or date.today()
    df = fetch_constituents_table()
    df["snapshot_date"] = pd.Timestamp(as_of)

    out_dir = BRONZE_DIR / "constituents" / f"ingest_date={as_of.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constituents.parquet"
    df.to_parquet(out_path, index=False)

    logger.info("Wrote %d constituents to %s", len(df), out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_constituents_snapshot()
