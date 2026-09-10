"""
Cross-source price reconciliation: yfinance silver vs Tiingo silver.

Compares only adj_close, deliberately: earlier live-data testing showed
yfinance's raw OHLC (auto_adjust=False) is not split-adjusted while
Tiingo's raw OHLC is — comparing raw close-to-close would flag every
post-split row as a false discrepancy. adj_close is genuinely comparable
across both sources (both are fully split+dividend adjusted), which is
also why it's the one field that agreed to ~0.002% in the first live spot
check, back when this pipeline only had three tickers.

Discrepancy is measured symmetrically — relative to the *average* of the
two sources' adj_close, not relative to either one specifically. Neither
yfinance nor Tiingo is ground truth here; this is a peer comparison
between two independent vendors, not a check against a known-correct
reference, so nothing in the metric should implicitly treat one as more
authoritative than the other.

A (ticker, date) pair present in only one source is tracked separately
from a flagged discrepancy — that's a coverage gap, not a disagreement,
and conflating the two would understate how many rows were actually
checked against anything at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import SILVER_DIR, OUTPUTS_DIR, RECONCILIATION_TOLERANCE

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    ingest_date: date
    total_compared: int
    flagged_count: int
    yfinance_only_count: int
    tiingo_only_count: int
    flagged: pd.DataFrame = field(repr=False)
    full: pd.DataFrame = field(repr=False)
    consensus: pd.DataFrame = field(repr=False)

    @property
    def flagged_pct(self) -> float:
        return (self.flagged_count / self.total_compared * 100) if self.total_compared else 0.0

    def flagged_by_ticker(self) -> pd.DataFrame:
        """
        Per-ticker breakdown of flagged rows.

        A headline "4.7% of rows flagged" is uninterpretable on its own:
        evenly spread across every ticker means something systemically wrong
        with the comparison, while concentrated in a handful of tickers
        means specific corporate-action handling differences between the
        two vendors. Those need completely different responses, so the
        report shouldn't force you to guess which one you're looking at.
        """
        if self.full.empty:
            return pd.DataFrame()

        per_ticker = (
            self.full.groupby("ticker")
            .agg(
                rows_compared=("flagged", "size"),
                rows_flagged=("flagged", "sum"),
                max_pct_diff=("pct_diff", "max"),
                median_pct_diff=("pct_diff", "median"),
            )
            .reset_index()
        )
        per_ticker["flagged_pct"] = per_ticker["rows_flagged"] / per_ticker["rows_compared"] * 100

        # Date range of the flagged rows specifically. This is what
        # distinguishes the two failure modes that a max/median comparison
        # only hints at: a constant offset across the whole history (one
        # vendor applying a different adjustment factor throughout) versus
        # a discrepancy confined to everything before a single date (one
        # vendor retroactively restating pre-event history for a corporate
        # action the other didn't). The second shows up as a flagged window
        # that ends abruptly — and that end date is the event date.
        flagged_only = self.full[self.full["flagged"]]
        if not flagged_only.empty:
            date_range = (
                flagged_only.groupby("ticker")["date"]
                .agg(first_flagged="min", last_flagged="max")
                .reset_index()
            )
            per_ticker = per_ticker.merge(date_range, on="ticker", how="left")
        else:
            per_ticker["first_flagged"] = pd.NaT
            per_ticker["last_flagged"] = pd.NaT

        return per_ticker.sort_values("rows_flagged", ascending=False)

    def render(self) -> str:
        lines = [
            f"Reconciliation report — ingest_date={self.ingest_date.isoformat()}",
            f"  Compared: {self.total_compared} (ticker, date) pairs present in both sources",
            f"  Flagged (> {RECONCILIATION_TOLERANCE * 100:.2f}% adj_close discrepancy): "
            f"{self.flagged_count} ({self.flagged_pct:.2f}%)",
            f"  Present only in yfinance (coverage gap, not a discrepancy): {self.yfinance_only_count}",
            f"  Present only in Tiingo (coverage gap, not a discrepancy): {self.tiingo_only_count}",
        ]

        if self.flagged_count:
            per_ticker = self.flagged_by_ticker()
            affected = per_ticker[per_ticker["rows_flagged"] > 0]
            lines.append(
                f"\n  Affected tickers: {len(affected)} of {len(per_ticker)} compared "
                f"— concentration matters more than the headline %, see below"
            )
            lines.append(f"  {'ticker':<8} {'flagged':>8} {'compared':>9} {'flagged%':>9} {'max diff':>9} {'median diff':>12}  {'flagged window':<25}")
            for _, r in affected.iterrows():
                first = pd.Timestamp(r["first_flagged"]).date() if pd.notna(r["first_flagged"]) else "—"
                last = pd.Timestamp(r["last_flagged"]).date() if pd.notna(r["last_flagged"]) else "—"
                lines.append(
                    f"  {r['ticker']:<8} {int(r['rows_flagged']):>8} {int(r['rows_compared']):>9} "
                    f"{r['flagged_pct']:>8.1f}% {r['max_pct_diff'] * 100:>8.2f}% {r['median_pct_diff'] * 100:>11.2f}%  "
                    f"{first} to {last}"
                )
            lines.append(
                "\n  Reading this: a flagged window spanning the full history with max ~ median\n"
                "  suggests a constant adjustment-factor difference; a window that ends on a\n"
                "  specific date with max >> median suggests one vendor restated pre-event\n"
                "  history for a corporate action on that date and the other didn't."
            )

            lines.append("\n  Top individual discrepancies:")
            top = self.flagged.sort_values("pct_diff", ascending=False).head(10)
            for _, row in top.iterrows():
                lines.append(
                    f"    {row['ticker']} {pd.Timestamp(row['date']).date()}: "
                    f"yfinance={row['yfinance_adj_close']:.4f}  tiingo={row['tiingo_adj_close']:.4f}  "
                    f"diff={row['pct_diff'] * 100:.3f}%"
                )
        return "\n".join(lines)


def _read_silver(source: str, ingest_date: date) -> pd.DataFrame:
    path = SILVER_DIR / f"{source}_prices" / f"ingest_date={ingest_date.isoformat()}" / "prices.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def reconcile_prices(ingest_date: date | None = None) -> ReconciliationReport:
    ingest_date = ingest_date or date.today()

    yf_silver = _read_silver("yfinance", ingest_date)
    tg_silver = _read_silver("tiingo", ingest_date)

    if yf_silver.empty or tg_silver.empty:
        raise ValueError(
            f"Missing silver data for ingest_date={ingest_date.isoformat()} "
            f"(yfinance empty={yf_silver.empty}, tiingo empty={tg_silver.empty}). "
            "Run src.transform.silver_prices for both sources first."
        )

    yf = yf_silver[["ticker", "date", "adj_close"]].rename(columns={"adj_close": "yfinance_adj_close"})
    tg = tg_silver[["ticker", "date", "adj_close"]].rename(columns={"adj_close": "tiingo_adj_close"})

    merged = yf.merge(tg, on=["ticker", "date"], how="outer", indicator=True)

    yfinance_only = merged[merged["_merge"] == "left_only"]
    tiingo_only = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True).copy()

    both["abs_diff"] = (both["yfinance_adj_close"] - both["tiingo_adj_close"]).abs()
    both["pct_diff"] = both["abs_diff"] / ((both["yfinance_adj_close"] + both["tiingo_adj_close"]) / 2)
    both["flagged"] = both["pct_diff"] > RECONCILIATION_TOLERANCE

    flagged = both[both["flagged"]].copy()

    # Consensus: every (ticker, date) pair from either source, not just the
    # ones present in both — this is what fact_price_daily_consensus loads
    # directly, so it needs a "trusted" adj_close and an outcome tag for
    # rows that only had one source too, not just the overlap.
    # yfinance is the default primary_source when both agree or disagree:
    # it has full-universe coverage in this pipeline, Tiingo is the sampled
    # second opinion, so preferring yfinance's value keeps the consensus
    # table's coverage matching yfinance's rather than the smaller sample.
    agreed_or_disagreed = both.copy()
    agreed_or_disagreed["adj_close"] = agreed_or_disagreed["yfinance_adj_close"]
    agreed_or_disagreed["primary_source"] = "yfinance"
    agreed_or_disagreed["sources_available"] = [["yfinance", "tiingo"]] * len(agreed_or_disagreed)
    agreed_or_disagreed["reconciliation_flag"] = agreed_or_disagreed["flagged"].map({True: "disagreed", False: "agreed"})

    yf_single = yfinance_only.drop(columns="_merge").copy()
    yf_single["adj_close"] = yf_single["yfinance_adj_close"]
    yf_single["primary_source"] = "yfinance"
    yf_single["sources_available"] = [["yfinance"]] * len(yf_single)
    yf_single["pct_diff"] = pd.NA
    yf_single["reconciliation_flag"] = "single_source"

    tg_single = tiingo_only.drop(columns="_merge").copy()
    tg_single["adj_close"] = tg_single["tiingo_adj_close"]
    tg_single["primary_source"] = "tiingo"
    tg_single["sources_available"] = [["tiingo"]] * len(tg_single)
    tg_single["pct_diff"] = pd.NA
    tg_single["reconciliation_flag"] = "single_source"

    consensus_cols = ["ticker", "date", "adj_close", "primary_source", "sources_available", "pct_diff", "reconciliation_flag"]
    consensus = pd.concat(
        [agreed_or_disagreed[consensus_cols], yf_single[consensus_cols], tg_single[consensus_cols]],
        ignore_index=True,
    )

    out_dir = SILVER_DIR / "reconciled_prices" / f"ingest_date={ingest_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    both.to_parquet(out_dir / "reconciliation.parquet", index=False)
    logger.info("Wrote %d reconciled rows to %s", len(both), out_dir / "reconciliation.parquet")

    return ReconciliationReport(
        ingest_date=ingest_date,
        total_compared=len(both),
        flagged_count=len(flagged),
        yfinance_only_count=len(yfinance_only),
        tiingo_only_count=len(tiingo_only),
        flagged=flagged,
        full=both,
        consensus=consensus,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    report = reconcile_prices(target_date)
    print(report.render())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / f"reconciliation_report_{target_date.isoformat()}.txt"
    report_path.write_text(report.render(), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    sys.exit(0 if report.flagged_count == 0 else 1)
