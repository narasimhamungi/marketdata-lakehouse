"""
Run bronze ingestion sources against the S&P 500 universe.

Reads the most recent constituents snapshot for the ticker list (pulling a
fresh one if none exists yet for the target date), then calls each
price/mapping ingester. This is a thin driver, not new ingestion logic —
every function it calls already exists and is already tested individually;
this wires them together in roughly the order the eventual Airflow DAG
will use.

yfinance and OpenFIGI run against the FULL universe — both proved capable
of that at full scale against live data. Tiingo does not: its free tier
caps at 50 requests/hour, so a full 503-ticker pull takes over 10 hours in
one sitting. Rather than force that (or silently rate-limit into a very
long-running command), Tiingo gets a sector-stratified SAMPLE — enough
tickers spread across every GICS sector to make reconciliation meaningful,
without requiring a multi-hour unattended run. The ingest module itself
(src/ingest/tiingo_source.py) also throttles to stay under the hourly cap
regardless of sample size, as defense in depth.

FRED is unaffected by universe size (four fixed macro series) — included
for a one-command full run, not because it needs the ticker list.

The `sources` parameter lets you re-run a subset (e.g. just Tiingo, after
fixing something, without re-pulling yfinance/OpenFIGI that already
succeeded) instead of always repeating all four.
"""
from __future__ import annotations

import logging
import random
import sys
from datetime import date

import pandas as pd

from src.config import BRONZE_DIR
from src.ingest.constituents import ingest_constituents_snapshot
from src.ingest.yfinance_source import ingest_yfinance_batch
from src.ingest.tiingo_source import ingest_tiingo_batch
from src.ingest.fred_source import ingest_fred_series
from src.ingest.openfigi_source import ingest_openfigi_mapping
from src.utils import read_bronze_partition

logger = logging.getLogger(__name__)

ALL_SOURCES = ("yfinance", "tiingo", "fred", "openfigi")
DEFAULT_TIINGO_SAMPLE_SIZE = 45  # matches the ingest module's safe hourly cap


def get_constituents(ingest_date: date, refresh: bool) -> pd.DataFrame:
    """Reuse today's constituents snapshot if one already exists (avoids an
    unnecessary Wikipedia hit), unless refresh=True is explicitly asked
    for. Returns the full frame (not just tickers) — Tiingo sampling below
    needs the sector column."""
    if not refresh:
        df = read_bronze_partition(BRONZE_DIR / "constituents", ingest_date)
        if not df.empty:
            logger.info("Reusing existing constituents snapshot for %s (%d tickers)", ingest_date, len(df))
            return df.drop_duplicates(subset="ticker")

    logger.info("Pulling a fresh constituents snapshot for %s", ingest_date)
    return ingest_constituents_snapshot(as_of=ingest_date).drop_duplicates(subset="ticker")


def stratified_sample(constituents: pd.DataFrame, n: int, sector_col: str = "gics_sector", seed: int = 42) -> list[str]:
    """
    Pick n tickers spread across GICS sectors rather than an arbitrary
    slice (e.g. alphabetical, which is what a naive tickers[:n] would give)
    — round-robins through sectors so a small sample still gives
    reconciliation something representative to check, not an accidental
    cluster from one part of the sector list. Deterministic (seeded) so
    the same sample is reproducible run to run, which matters for a
    portfolio demo you might want to walk through in an interview.
    """
    if n >= len(constituents):
        return sorted(constituents["ticker"].unique().tolist())

    groups: dict[str, list[str]] = {
        sector: g["ticker"].tolist() for sector, g in constituents.groupby(sector_col)
    }
    rng = random.Random(seed)
    for tickers in groups.values():
        rng.shuffle(tickers)

    sectors = sorted(groups.keys())
    sample: list[str] = []
    i = 0
    while len(sample) < n and any(groups[s] for s in sectors):
        sector = sectors[i % len(sectors)]
        if groups[sector]:
            sample.append(groups[sector].pop())
        i += 1

    return sorted(sample)


def run_full_universe(
    ingest_date: date | None = None,
    refresh_constituents: bool = False,
    tiingo_sample_size: int = DEFAULT_TIINGO_SAMPLE_SIZE,
    sources: tuple[str, ...] = ALL_SOURCES,
) -> None:
    ingest_date = ingest_date or date.today()
    constituents = get_constituents(ingest_date, refresh_constituents)
    tickers = sorted(constituents["ticker"].unique().tolist())
    logger.info("Universe: %d tickers", len(tickers))

    if "yfinance" in sources:
        logger.info("--- yfinance (full universe) ---")
        ingest_yfinance_batch(tickers, ingest_date=ingest_date)

    if "tiingo" in sources:
        sample = stratified_sample(constituents, tiingo_sample_size)
        logger.info(
            "--- Tiingo: sampling %d of %d tickers, stratified across GICS sectors "
            "(free-tier hourly limit makes a full-universe pull impractical in one "
            "sitting — see README) ---",
            len(sample), len(tickers),
        )
        ingest_tiingo_batch(sample, ingest_date=ingest_date)

    if "fred" in sources:
        logger.info("--- FRED (fixed series, not ticker-scoped) ---")
        ingest_fred_series(ingest_date=ingest_date)

    if "openfigi" in sources:
        logger.info("--- OpenFIGI (full universe) ---")
        ingest_openfigi_mapping(tickers, ingest_date=ingest_date)

    logger.info("Ingestion complete for ingest_date=%s (sources: %s)", ingest_date.isoformat(), ", ".join(sources))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = sys.argv[1:]
    refresh = "--refresh-constituents" in args
    date_args = [a for a in args if not a.startswith("--")]
    target_date = date.fromisoformat(date_args[0]) if date_args else date.today()

    sources_arg = next((a for a in args if a.startswith("--sources=")), None)
    selected_sources = tuple(sources_arg.split("=", 1)[1].split(",")) if sources_arg else ALL_SOURCES

    run_full_universe(target_date, refresh_constituents=refresh, sources=selected_sources)
