"""
Freshness check for the gold layer — the DAG's final task.

Deliberately a real, queryable gate rather than Airflow's built-in
SLA-miss mechanism: an SLA miss just sends an email (needs SMTP
infrastructure to actually demonstrate) and doesn't fail the DAG run by
default. This checks the thing that actually matters — does the gold
layer for this ingest_date have enough data, and is it recent enough to
trust — and raises if not, so a broken upstream step (a vendor silently
stopping returning recent days, an ingestion that "succeeded" with near-
zero rows) fails the pipeline loudly instead of leaving stale or thin
data sitting in Postgres unnoticed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.model.db import get_connection

logger = logging.getLogger(__name__)

# A full S&P 500 pull should land close to 500; well below this signals a
# broken or badly truncated ingestion, not normal index turnover.
MIN_SECURITIES = 400

# DEFAULT_START_DATE backfills from 2019 — a healthy consensus table
# should average well over 1,000 rows per security. Set low enough not to
# false-positive on a legitimately smaller universe (e.g. testing with a
# handful of tickers), but high enough to catch a real truncation.
MIN_CONSENSUS_ROWS_PER_SECURITY = 200

# Generous enough to cover a long weekend plus a Monday holiday without
# false-flagging normal market closure as staleness.
MAX_STALENESS_DAYS = 5


@dataclass
class FreshnessCheckResult:
    ingest_date: date
    security_count: int
    consensus_row_count: int
    max_consensus_date: date | None
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def render(self) -> str:
        lines = [f"Gold freshness check — ingest_date={self.ingest_date.isoformat()}"]
        for description, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else '** FAIL **'}] {description} — {detail}")
        return "\n".join(lines)


def check_gold_freshness(ingest_date: date, conn=None) -> FreshnessCheckResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM dim_security WHERE is_current")
            security_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*), MAX(d.date) FROM fact_price_daily_consensus f "
                "JOIN dim_date d ON f.date_key = d.date_key"
            )
            consensus_row_count, max_consensus_date = cur.fetchone()
            consensus_row_count = consensus_row_count or 0

        rows_per_security = (consensus_row_count / security_count) if security_count else 0
        staleness_days = (ingest_date - max_consensus_date).days if max_consensus_date else None

        checks = [
            (
                f"security count >= {MIN_SECURITIES}",
                security_count >= MIN_SECURITIES,
                f"{security_count} securities",
            ),
            (
                f"consensus rows/security >= {MIN_CONSENSUS_ROWS_PER_SECURITY}",
                rows_per_security >= MIN_CONSENSUS_ROWS_PER_SECURITY,
                f"{consensus_row_count} rows across {security_count} securities "
                f"({rows_per_security:.0f}/security)",
            ),
            (
                f"latest data within {MAX_STALENESS_DAYS} days of ingest_date",
                staleness_days is not None and 0 <= staleness_days <= MAX_STALENESS_DAYS,
                f"latest consensus date {max_consensus_date}, {staleness_days} day(s) "
                f"before ingest_date {ingest_date}" if staleness_days is not None else "no consensus data at all",
            ),
        ]

        return FreshnessCheckResult(
            ingest_date=ingest_date,
            security_count=security_count,
            consensus_row_count=consensus_row_count,
            max_consensus_date=max_consensus_date,
            checks=checks,
        )
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    result = check_gold_freshness(target_date)
    print(result.render())
    sys.exit(0 if result.passed else 1)
