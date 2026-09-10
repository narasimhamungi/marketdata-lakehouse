"""
marketdata-lakehouse orchestration DAG.

Every task here is a thin wrapper around an already-tested function from
src/ — this file's job is to wire them together correctly, not to
reimplement any ingestion, transform, reconciliation, or loading logic.

Uses `data_interval_start` (Airflow's own scheduling-aware date) for every
ingest_date argument, never date.today(). This is a deliberate, tested
lesson from building this pipeline by hand: date.today() inside pipeline
code broke silently across a real midnight boundary during manual runs.
Airflow's whole date-handling model exists specifically to prevent that
class of bug — this DAG is built to actually use it, not bypass it.

bronze_quality_gate and gold_freshness_gate are real gates, not
informational logs: both raise (AirflowFailException) on failure, which
stops the DAG rather than letting bad or thin data flow through silently
to downstream tasks.

Imports are deliberately deferred inside each task callable rather than
at module top level — the scheduler re-parses every DAG file on a short
interval, and importing pandas/great_expectations/psycopg2 etc. at parse
time would slow that down for no benefit; they're only needed when a task
actually runs.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param

default_args = {
    "owner": "marketdata-lakehouse",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="marketdata_lakehouse",
    description="Bronze -> quality -> silver -> reconciliation -> gold, end to end",
    schedule="0 22 * * 1-5",  # 22:00 UTC, weekdays — after US market close
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={
        "tiingo_sample_size": Param(
            default=45,
            type="integer",
            minimum=1,
            maximum=503,
            title="Tiingo sample size",
            description=(
                "Number of tickers to sample for Tiingo ingestion. Lower "
                "this (e.g. 3) via 'Trigger DAG w/ config' for a fast test "
                "iteration that doesn't burn through Tiingo's hourly rate "
                "limit while debugging the rest of the pipeline — no code "
                "change or redeploy needed."
            ),
        ),
    },
    default_args=default_args,
    tags=["marketdata-lakehouse"],
)
def marketdata_lakehouse():
    @task
    def ingest_constituents(data_interval_start=None) -> None:
        from src.orchestrate.run_full_universe import get_constituents

        get_constituents(data_interval_start.date(), refresh=False)

    @task
    def ingest_yfinance(data_interval_start=None) -> dict:
        from src.ingest.yfinance_source import ingest_yfinance_batch
        from src.orchestrate.run_full_universe import get_constituents

        ingest_date = data_interval_start.date()
        constituents = get_constituents(ingest_date, refresh=False)
        tickers = sorted(constituents["ticker"].unique().tolist())
        ingest_yfinance_batch(tickers, ingest_date=ingest_date)
        return {"tickers": len(tickers)}

    @task
    def ingest_tiingo(data_interval_start=None, params=None) -> dict:
        from src.ingest.tiingo_source import ingest_tiingo_batch
        from src.orchestrate.run_full_universe import get_constituents, stratified_sample

        ingest_date = data_interval_start.date()
        constituents = get_constituents(ingest_date, refresh=False)
        sample = stratified_sample(constituents, params["tiingo_sample_size"])
        ingest_tiingo_batch(sample, ingest_date=ingest_date)
        return {"tickers": len(sample)}

    @task
    def ingest_fred(data_interval_start=None) -> None:
        from src.ingest.fred_source import ingest_fred_series

        ingest_fred_series(ingest_date=data_interval_start.date())

    @task
    def ingest_openfigi(data_interval_start=None) -> None:
        from src.ingest.openfigi_source import ingest_openfigi_mapping
        from src.orchestrate.run_full_universe import get_constituents

        ingest_date = data_interval_start.date()
        constituents = get_constituents(ingest_date, refresh=False)
        tickers = sorted(constituents["ticker"].unique().tolist())
        ingest_openfigi_mapping(tickers, ingest_date=ingest_date)

    @task
    def bronze_quality_gate(data_interval_start=None) -> None:
        from src.quality.bronze_checks import run_all_bronze_checks

        ingest_date = data_interval_start.date()
        reports = run_all_bronze_checks(ingest_date)
        if not reports or not all(r.passed for r in reports):
            failed = [r.source for r in reports if not r.passed]
            raise AirflowFailException(
                f"Bronze quality gate failed for {ingest_date}: "
                f"{failed or 'no bronze data found at all'}. Refusing to "
                "proceed to silver/gold with unvalidated data."
            )

    @task
    def silver_yfinance(data_interval_start=None) -> None:
        from src.transform.silver_prices import build_silver_prices

        build_silver_prices("yfinance", data_interval_start.date())

    @task
    def silver_tiingo(data_interval_start=None) -> None:
        from src.transform.silver_prices import build_silver_prices

        build_silver_prices("tiingo", data_interval_start.date())

    @task
    def reconcile(data_interval_start=None) -> dict:
        from src.reconcile.price_reconciliation import reconcile_prices

        report = reconcile_prices(data_interval_start.date())
        return {"flagged": report.flagged_count, "compared": report.total_compared}

    @task
    def gold_dims(data_interval_start=None) -> None:
        from src.config import BRONZE_DIR, DEFAULT_START_DATE
        from src.model.db import apply_schema
        from src.model.dim_date import build_dim_date, trading_days_from_silver
        from src.model.dim_security import build_dim_security
        from src.transform.silver_prices import read_silver_prices
        from src.utils import read_bronze_partition

        ingest_date = data_interval_start.date()
        apply_schema()

        yf_silver = read_silver_prices("yfinance", ingest_date)
        trading_days = trading_days_from_silver(yf_silver)
        build_dim_date(DEFAULT_START_DATE, ingest_date, trading_days)

        constituents = read_bronze_partition(BRONZE_DIR / "constituents", ingest_date)
        openfigi = read_bronze_partition(BRONZE_DIR / "openfigi_mapping", ingest_date)
        build_dim_security(constituents, openfigi, ingest_date)

    @task
    def gold_fact_price_daily_yfinance(data_interval_start=None) -> dict:
        from src.model.fact_price_daily import build_fact_price_daily
        from src.transform.silver_prices import read_silver_prices

        ingest_date = data_interval_start.date()
        return build_fact_price_daily(read_silver_prices("yfinance", ingest_date), "yfinance")

    @task
    def gold_fact_price_daily_tiingo(data_interval_start=None) -> dict:
        from src.model.fact_price_daily import build_fact_price_daily
        from src.transform.silver_prices import read_silver_prices

        ingest_date = data_interval_start.date()
        return build_fact_price_daily(read_silver_prices("tiingo", ingest_date), "tiingo")

    @task
    def gold_fact_price_daily_consensus(data_interval_start=None) -> dict:
        from src.model.fact_price_daily_consensus import build_fact_price_daily_consensus
        from src.reconcile.price_reconciliation import reconcile_prices

        report = reconcile_prices(data_interval_start.date())
        return build_fact_price_daily_consensus(report.consensus)

    @task
    def gold_fact_corporate_action(data_interval_start=None) -> dict:
        from src.config import BRONZE_DIR
        from src.model.fact_corporate_action import build_fact_corporate_action
        from src.utils import read_bronze_partition

        ingest_date = data_interval_start.date()
        yf_bronze = read_bronze_partition(BRONZE_DIR / "yfinance_prices", ingest_date)
        return build_fact_corporate_action(yf_bronze)

    @task
    def gold_fact_macro_rate(data_interval_start=None) -> dict:
        from src.model.fact_macro_rate import build_fact_macro_rate
        from src.orchestrate.build_gold import read_fred_bronze

        fred_df = read_fred_bronze(data_interval_start.date())
        return build_fact_macro_rate(fred_df)

    @task
    def gold_freshness_gate(data_interval_start=None) -> None:
        from src.quality.freshness_check import check_gold_freshness

        result = check_gold_freshness(data_interval_start.date())
        if not result.passed:
            raise AirflowFailException(result.render())

    # --- dependency wiring ---
    constituents_task = ingest_constituents()
    yf_task = ingest_yfinance()
    tg_task = ingest_tiingo()
    fred_task = ingest_fred()
    figi_task = ingest_openfigi()
    constituents_task >> [yf_task, tg_task, figi_task]

    quality_gate = bronze_quality_gate()
    [yf_task, tg_task, fred_task, figi_task] >> quality_gate

    silver_yf = silver_yfinance()
    silver_tg = silver_tiingo()
    quality_gate >> [silver_yf, silver_tg]

    recon = reconcile()
    [silver_yf, silver_tg] >> recon

    dims = gold_dims()
    [silver_yf, quality_gate] >> dims

    fact_yf = gold_fact_price_daily_yfinance()
    fact_tg = gold_fact_price_daily_tiingo()
    fact_consensus = gold_fact_price_daily_consensus()
    fact_corp = gold_fact_corporate_action()
    fact_macro = gold_fact_macro_rate()

    dims >> [fact_yf, fact_tg, fact_corp, fact_macro]
    [recon, dims] >> fact_consensus

    freshness = gold_freshness_gate()
    [fact_yf, fact_tg, fact_consensus, fact_corp, fact_macro] >> freshness


marketdata_lakehouse()
