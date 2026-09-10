"""
Shared fixtures for gold-layer tests. These use a REAL local Postgres
connection, not a mock — SQL correctness (constraint behavior, upsert
semantics, SCD-2 close/open logic) is exactly the kind of thing a mock
would happily let through wrong.

Connection params default to the SAME credentials docker-compose.yml's
postgres service already uses (mdl/mdl_local_dev), deliberately — so
there's nothing new to configure, just one additional database to create
once (see README "Gold layer" for the one-time setup command). A
DIFFERENT database name (marketdata_lakehouse_test, not
marketdata_lakehouse) is essential, not cosmetic: this fixture TRUNCATEs
every table before each test, and pointed at your real database that
would silently wipe actual gold-layer data built from real ingestion runs.
Overridable via MDL_TEST_DB_* env vars for CI or any other environment.
"""
import os

import psycopg2
import pytest

from src.model.db import apply_schema

TEST_DB_PARAMS = dict(
    host=os.environ.get("MDL_TEST_DB_HOST", "localhost"),
    port=int(os.environ.get("MDL_TEST_DB_PORT", "5432")),
    dbname=os.environ.get("MDL_TEST_DB_NAME", "marketdata_lakehouse_test"),
    user=os.environ.get("MDL_TEST_DB_USER", "mdl"),
    password=os.environ.get("MDL_TEST_DB_PASSWORD", "mdl_local_dev"),
)

TABLES = [
    "fact_corporate_action",
    "fact_macro_rate",
    "fact_price_daily_consensus",
    "fact_price_daily",
    "dim_security",
    "dim_date",
]


@pytest.fixture
def pg_conn():
    conn = psycopg2.connect(**TEST_DB_PARAMS)
    apply_schema(conn)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()
    yield conn
    conn.close()
