"""Small shared helpers used by more than one module — kept here rather
than duplicated, since the first duplication (silver needing the same
bronze-partition reader quality checks already had) is exactly the signal
that it belongs in one place."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def read_bronze_partition(source_dir: Path, ingest_date: date) -> pd.DataFrame:
    """Bronze partitions hold one-or-more parquet files per ingest_date
    (one per ticker/series, or one for the whole run, depending on the
    source) — concat whatever's there, ignore any _failed_*.txt sidecars."""
    partition_dir = source_dir / f"ingest_date={ingest_date.isoformat()}"
    if not partition_dir.exists():
        return pd.DataFrame()
    files = sorted(partition_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
