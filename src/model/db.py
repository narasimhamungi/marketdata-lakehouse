"""
Postgres connection helper for the gold layer.

Every loader function in src/model/ accepts an optional `conn` parameter
and only opens/closes its own connection when one isn't supplied — this is
what lets tests inject a real connection to a test database without
needing to patch config module internals, and what will let the eventual
Airflow DAG share one connection across a task instead of reconnecting
per function call.
"""
from __future__ import annotations

from pathlib import Path

import psycopg2

from src.config import GOLD_DB_HOST, GOLD_DB_PORT, GOLD_DB_NAME, GOLD_DB_USER, GOLD_DB_PASSWORD

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection():
    return psycopg2.connect(
        host=GOLD_DB_HOST,
        port=GOLD_DB_PORT,
        dbname=GOLD_DB_NAME,
        user=GOLD_DB_USER,
        password=GOLD_DB_PASSWORD,
    )


def apply_schema(conn=None) -> None:
    """Idempotent — schema.sql uses CREATE TABLE/INDEX IF NOT EXISTS, safe
    to call on every pipeline run rather than only once at setup."""
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        if own_conn:
            conn.close()
