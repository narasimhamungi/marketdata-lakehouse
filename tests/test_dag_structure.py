"""
DAG structure tests using Airflow's real DagBag — the standard way to
test an Airflow DAG: load it the same way the scheduler would, check it
parsed with no import errors, and verify the dependency graph is what it's
supposed to be.

Skipped gracefully if apache-airflow isn't installed (it's intentionally
NOT in the main requirements.txt — the ingestion/silver/gold pipeline
doesn't need it, only orchestration does, so it's a separate
airflow/requirements.txt used inside the Airflow container). Run
`pip install apache-airflow` in a throwaway environment to run these
locally; they're expected to just skip otherwise.
"""
import os
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")

os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_test_home")
Path(os.environ["AIRFLOW_HOME"]).mkdir(parents=True, exist_ok=True)

from airflow.models import DagBag  # noqa: E402

DAG_FOLDER = str(Path(__file__).resolve().parents[1] / "airflow" / "dags")


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAG_FOLDER, include_examples=False)


def test_dag_folder_has_no_import_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_marketdata_lakehouse_dag_is_present(dag_bag):
    assert "marketdata_lakehouse" in dag_bag.dags


def test_dag_has_expected_task_count(dag_bag):
    dag = dag_bag.dags["marketdata_lakehouse"]
    assert len(dag.tasks) == 16


def test_dag_has_no_cycles(dag_bag):
    dag = dag_bag.dags["marketdata_lakehouse"]
    order = dag.topological_sort()  # raises AirflowDagCycleException on a cycle
    assert len(order) == len(dag.tasks)


def test_dag_has_exactly_two_root_tasks(dag_bag):
    """ingest_constituents and ingest_fred are the only tasks with no
    dependency — everything else needs at least one of the four ingestion
    sources or the constituents snapshot first."""
    dag = dag_bag.dags["marketdata_lakehouse"]
    roots = {t.task_id for t in dag.tasks if not t.upstream_list}
    assert roots == {"ingest_constituents", "ingest_fred"}


def test_dag_has_exactly_one_leaf_task(dag_bag):
    """gold_freshness_gate is the final gate everything funnels into."""
    dag = dag_bag.dags["marketdata_lakehouse"]
    leaves = {t.task_id for t in dag.tasks if not t.downstream_list}
    assert leaves == {"gold_freshness_gate"}


def test_bronze_quality_gate_depends_on_all_four_ingestion_tasks(dag_bag):
    """The gate must actually see every source before deciding whether
    it's safe to proceed — depending on only some of them would let it
    pass while a source silently failed."""
    dag = dag_bag.dags["marketdata_lakehouse"]
    gate = dag.get_task("bronze_quality_gate")
    upstream = {t.task_id for t in gate.upstream_list}
    assert upstream == {"ingest_yfinance", "ingest_tiingo", "ingest_fred", "ingest_openfigi"}


def test_silver_tasks_depend_on_quality_gate_not_directly_on_ingestion(dag_bag):
    """Silver must never run against unvalidated bronze — enforced by the
    dependency edge itself, not just by convention."""
    dag = dag_bag.dags["marketdata_lakehouse"]
    for task_id in ("silver_yfinance", "silver_tiingo"):
        upstream = {t.task_id for t in dag.get_task(task_id).upstream_list}
        assert upstream == {"bronze_quality_gate"}


def test_gold_freshness_gate_depends_on_every_fact_table_load(dag_bag):
    dag = dag_bag.dags["marketdata_lakehouse"]
    gate = dag.get_task("gold_freshness_gate")
    upstream = {t.task_id for t in gate.upstream_list}
    assert upstream == {
        "gold_fact_price_daily_yfinance",
        "gold_fact_price_daily_tiingo",
        "gold_fact_price_daily_consensus",
        "gold_fact_corporate_action",
        "gold_fact_macro_rate",
    }


def test_consensus_fact_depends_on_both_reconcile_and_dims(dag_bag):
    """Consensus needs dim_security/dim_date to resolve keys AND the
    reconciliation output itself — missing either dependency would let it
    run against stale dims or a stale reconciliation result."""
    dag = dag_bag.dags["marketdata_lakehouse"]
    upstream = {t.task_id for t in dag.get_task("gold_fact_price_daily_consensus").upstream_list}
    assert upstream == {"gold_dims", "reconcile"}
