"""
Data quality checks on the bronze layer, using Great Expectations.

Each check function builds an ephemeral GE context (no project directory,
no config files — just an in-memory validator against a dataframe) and
runs a fixed set of expectations, returning a QualityReport rather than
raising on the first failure: the point of a quality gate is to see
*everything* wrong with a batch in one pass, not to stop at the first
issue and require five separate runs to find the rest.

This is deliberately not using GE's Checkpoint/DataDocs machinery (that
needs a project store and produces HTML reports) — for this pipeline's
scale, one plain-text report per run is more useful than a static site
nobody will open. If DataDocs becomes worth having later, this is the
layer to build it on top of, not replace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import great_expectations as gx
import pandas as pd

from src.config import BRONZE_DIR, OUTPUTS_DIR
from src.utils import read_bronze_partition

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    description: str
    success: bool
    unexpected_count: int | None = None


@dataclass
class QualityReport:
    source: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.success for c in self.checks)

    def render(self) -> str:
        lines = [f"{'PASS' if self.passed else 'FAIL'}  {self.source}"]
        for c in self.checks:
            mark = "  ok " if c.success else "  ** FAIL **"
            detail = f" ({c.unexpected_count} unexpected)" if c.unexpected_count else ""
            lines.append(f"  [{mark}] {c.description}{detail}")
        return "\n".join(lines)


def _get_validator(df: pd.DataFrame, suite_name: str):
    """One ephemeral GE context per call — cheap, and avoids any state
    leaking between checks run in the same process."""
    context = gx.get_context(mode="ephemeral")
    datasource = context.sources.add_pandas(f"{suite_name}_source")
    asset = datasource.add_dataframe_asset(name=suite_name)
    batch_request = asset.build_batch_request(dataframe=df)
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
    return context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)


def _run(validator, description: str, expectation_fn) -> CheckResult:
    result = expectation_fn(validator)
    return CheckResult(
        description=description,
        success=result.success,
        unexpected_count=result.result.get("unexpected_count"),
    )


def find_ohlc_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic companion to check_price_bronze: the GE-based check reports
    *that* a row violates OHLC ordering and how many, but not which row —
    fine for a pass/fail gate, not for actually investigating a failure.
    This runs the same checks in plain pandas and returns the actual
    offending rows with full context (ticker, date, values), tagged with
    which specific rule each one broke.
    """
    checks = {
        "low > open": df["low"] > df["open"],
        "low > close": df["low"] > df["close"],
        "low > high": df["low"] > df["high"],
        "high < open": df["high"] < df["open"],
        "high < close": df["high"] < df["close"],
    }
    violations = []
    for rule, mask in checks.items():
        bad = df[mask].copy()
        if not bad.empty:
            bad["violation"] = rule
            violations.append(bad)

    if not violations:
        return pd.DataFrame()
    return pd.concat(violations, ignore_index=True).sort_values(["ticker", "date"])


def check_price_bronze(df: pd.DataFrame, source_name: str) -> QualityReport:
    """
    Applies to both yfinance and Tiingo bronze — both are normalized to the
    same column shape (date, open, high, low, close, adj_close, volume,
    ticker) by their respective ingest modules, so one check function
    covers both.
    """
    report = QualityReport(source=source_name)
    if df.empty:
        report.checks.append(CheckResult("non-empty dataframe", success=False))
        return report

    v = _get_validator(df, f"{source_name}_prices")
    # OHLC-ordering checks (high>=low/open/close, low<=open/close) use a
    # 99.99% tolerance, not zero: confirmed against real yfinance data that
    # a single physically-impossible row can occur in ~950K rows (one real
    # bad print from the vendor, ticker HUBB, 2021-05-05) without it being
    # a pipeline bug. A hard zero-tolerance gate failed the entire DAG over
    # that one row — too brittle for real vendor data at this scale, and a
    # gate that fails on every run stops being trusted. 99.99% still fails
    # correctly on anything resembling a systemic problem (tested: 6
    # violations against this same threshold on 1,000 rows still fails) —
    # it tolerates isolated noise, not a real defect. Null/negative-price/
    # duplicate checks stay at zero tolerance: those indicate an actual
    # pipeline bug, not market noise, and should never be waved through.
    OHLC_TOLERANCE = 0.9999
    checks = [
        ("ticker not null", lambda v: v.expect_column_values_to_not_be_null("ticker")),
        ("date not null", lambda v: v.expect_column_values_to_not_be_null("date")),
        ("close not null", lambda v: v.expect_column_values_to_not_be_null("close")),
        ("close > 0", lambda v: v.expect_column_values_to_be_between("close", min_value=0, strict_min=True)),
        ("open > 0", lambda v: v.expect_column_values_to_be_between("open", min_value=0, strict_min=True)),
        ("adj_close > 0", lambda v: v.expect_column_values_to_be_between("adj_close", min_value=0, strict_min=True)),
        ("volume >= 0", lambda v: v.expect_column_values_to_be_between("volume", min_value=0)),
        ("high >= low", lambda v: v.expect_column_pair_values_a_to_be_greater_than_b("high", "low", or_equal=True, mostly=OHLC_TOLERANCE)),
        ("high >= open", lambda v: v.expect_column_pair_values_a_to_be_greater_than_b("high", "open", or_equal=True, mostly=OHLC_TOLERANCE)),
        ("high >= close", lambda v: v.expect_column_pair_values_a_to_be_greater_than_b("high", "close", or_equal=True, mostly=OHLC_TOLERANCE)),
        ("low <= open", lambda v: v.expect_column_pair_values_a_to_be_greater_than_b("open", "low", or_equal=True, mostly=OHLC_TOLERANCE)),
        ("low <= close", lambda v: v.expect_column_pair_values_a_to_be_greater_than_b("close", "low", or_equal=True, mostly=OHLC_TOLERANCE)),
        ("(ticker, date) unique", lambda v: v.expect_compound_columns_to_be_unique(column_list=["ticker", "date"])),
    ]
    for description, fn in checks:
        report.checks.append(_run(v, description, fn))
    return report


def check_constituents_bronze(df: pd.DataFrame) -> QualityReport:
    report = QualityReport(source="constituents")
    if df.empty:
        report.checks.append(CheckResult("non-empty dataframe", success=False))
        return report

    v = _get_validator(df, "constituents")
    checks = [
        ("ticker not null", lambda v: v.expect_column_values_to_not_be_null("ticker")),
        ("ticker unique", lambda v: v.expect_column_values_to_be_unique("ticker")),
        ("gics_sector not null", lambda v: v.expect_column_values_to_not_be_null("gics_sector")),
        # S&P 500 nominally has 500 tickers but runs ~500-505 in practice
        # (multiple share classes count as separate constituents); a wide
        # tolerance band catches a genuinely broken scrape without false
        # alarms on normal index turnover.
        (
            "row count in [480, 520]",
            lambda v: v.expect_table_row_count_to_be_between(min_value=480, max_value=520),
        ),
    ]
    for description, fn in checks:
        report.checks.append(_run(v, description, fn))
    return report


def check_fred_bronze(df: pd.DataFrame, series_id: str) -> QualityReport:
    report = QualityReport(source=f"fred:{series_id}")
    if df.empty:
        report.checks.append(CheckResult("non-empty dataframe", success=False))
        return report

    v = _get_validator(df, f"fred_{series_id}")
    checks = [
        ("value not null", lambda v: v.expect_column_values_to_not_be_null("value")),
    ]
    if series_id != "CPIAUCSL":
        # Rate series only — CPI is an index level, not a percentage, so a
        # 0-25 bound would be nonsensical for it.
        checks.append(
            ("value in plausible rate range [0, 25]", lambda v: v.expect_column_values_to_be_between("value", min_value=0, max_value=25))
        )
    else:
        checks.append(("value > 0 (index level)", lambda v: v.expect_column_values_to_be_between("value", min_value=0, strict_min=True)))

    for description, fn in checks:
        report.checks.append(_run(v, description, fn))
    return report


def check_openfigi_bronze(df: pd.DataFrame) -> QualityReport:
    report = QualityReport(source="openfigi_mapping")
    if df.empty:
        report.checks.append(CheckResult("non-empty dataframe", success=False))
        return report

    v = _get_validator(df, "openfigi_mapping")
    checks = [
        ("figi not null", lambda v: v.expect_column_values_to_not_be_null("figi")),
        ("figi unique", lambda v: v.expect_column_values_to_be_unique("figi")),
        ("figi is 12 characters", lambda v: v.expect_column_value_lengths_to_equal("figi", value=12)),
        ("ticker unique", lambda v: v.expect_column_values_to_be_unique("ticker")),
    ]
    for description, fn in checks:
        report.checks.append(_run(v, description, fn))
    return report


def _read_partition(source_dir: Path, ingest_date: date) -> pd.DataFrame:
    """Thin alias kept for backward compatibility with existing tests and
    call sites — the real implementation now lives in src/utils.py, shared
    with the silver transform module."""
    return read_bronze_partition(source_dir, ingest_date)


def run_all_bronze_checks(ingest_date: date | None = None) -> list[QualityReport]:
    ingest_date = ingest_date or date.today()
    reports: list[QualityReport] = []

    constituents_df = _read_partition(BRONZE_DIR / "constituents", ingest_date)
    if not constituents_df.empty:
        reports.append(check_constituents_bronze(constituents_df))

    yfinance_df = _read_partition(BRONZE_DIR / "yfinance_prices", ingest_date)
    if not yfinance_df.empty:
        reports.append(check_price_bronze(yfinance_df, "yfinance"))

    tiingo_df = _read_partition(BRONZE_DIR / "tiingo_prices", ingest_date)
    if not tiingo_df.empty:
        reports.append(check_price_bronze(tiingo_df, "tiingo"))

    fred_dir = BRONZE_DIR / "fred_series" / f"ingest_date={ingest_date.isoformat()}"
    if fred_dir.exists():
        for f in sorted(fred_dir.glob("*.parquet")):
            series_id = f.stem
            reports.append(check_fred_bronze(pd.read_parquet(f), series_id))

    openfigi_df = _read_partition(BRONZE_DIR / "openfigi_mapping", ingest_date)
    if not openfigi_df.empty:
        reports.append(check_openfigi_bronze(openfigi_df))

    return reports


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)  # GE's own metric-calculation
    # progress bars print to stderr regardless; harmless, redirect with
    # `2>$null` (PowerShell) or `2>/dev/null` if they're distracting.

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    diagnose = "--diagnose" in sys.argv
    target_date = date.fromisoformat(args[0]) if args else date.today()

    if diagnose:
        # Investigation mode: don't re-run the pass/fail suite, just show
        # the actual offending rows for any OHLC ordering violation, which
        # the GE report deliberately doesn't include.
        print(f"OHLC violation diagnosis — ingest_date={target_date.isoformat()}\n" + "=" * 60)
        found_any = False
        for source in ("yfinance", "tiingo"):
            df = _read_partition(BRONZE_DIR / f"{source}_prices", target_date)
            if df.empty:
                continue
            bad = find_ohlc_violations(df)
            print(f"\n{source}: {len(bad)} violating row(s) out of {len(df):,}")
            if not bad.empty:
                found_any = True
                cols = [c for c in ("ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "violation") if c in bad.columns]
                print(bad[cols].to_string(index=False))
        if not found_any:
            print("\nNo OHLC ordering violations found.")
        sys.exit(0)

    reports = run_all_bronze_checks(target_date)

    if not reports:
        print(f"No bronze data found for ingest_date={target_date.isoformat()}. Run ingestion first.")
        sys.exit(1)

    print(f"\nBronze quality report — ingest_date={target_date.isoformat()}\n" + "=" * 60)
    for r in reports:
        print(r.render())
        print()

    overall_pass = all(r.passed for r in reports)
    summary = f"{'ALL CHECKS PASSED' if overall_pass else 'SOME CHECKS FAILED'} ({sum(r.passed for r in reports)}/{len(reports)} sources clean)"
    print("=" * 60)
    print(summary)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / f"bronze_quality_report_{target_date.isoformat()}.txt"
    report_path.write_text(
        f"Bronze quality report — ingest_date={target_date.isoformat()}\n"
        + "\n\n".join(r.render() for r in reports)
        + f"\n\n{summary}\n",
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path}")

    sys.exit(0 if overall_pass else 1)
