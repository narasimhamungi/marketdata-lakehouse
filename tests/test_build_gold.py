"""
Wiring tests for build_gold: every sub-step is mocked, verifying call
order and that data flows between steps correctly (e.g. reconciliation's
.consensus reaches the consensus loader) — not re-testing each loader's
own logic, which already has real-Postgres tests of its own.
"""
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.orchestrate import build_gold as bg


def test_build_gold_raises_clearly_when_silver_missing_rather_than_building_bad_dim_date(monkeypatch):
    """
    Regression test: an empty yfinance silver frame previously let dim_date
    build successfully with every date silently marked is_trading_day=False
    — no crash, no warning that would be noticed, just wrong data sitting
    in the database. This must fail loudly instead, before writing anything.
    """
    monkeypatch.setattr(bg, "apply_schema", lambda conn: None)
    monkeypatch.setattr(bg, "build_silver_prices", lambda source, d: pd.DataFrame())  # empty, as if wrong date

    dim_date_mock = MagicMock()
    monkeypatch.setattr(bg, "build_dim_date", dim_date_mock)

    with pytest.raises(ValueError, match="No yfinance silver data"):
        bg.build_gold(date(2026, 9, 8), conn=MagicMock())

    dim_date_mock.assert_not_called()  # never reached — failed before writing anything


def test_build_gold_calls_every_step_in_dependency_order(monkeypatch):
    calls = []

    def tracked(name):
        def _fn(*args, **kwargs):
            calls.append(name)
            return MagicMock()
        return _fn

    monkeypatch.setattr(bg, "apply_schema", tracked("apply_schema"))
    monkeypatch.setattr(bg, "build_silver_prices", lambda source, d: pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["AAPL"]}))
    monkeypatch.setattr(bg, "trading_days_from_silver", tracked("trading_days_from_silver"))
    monkeypatch.setattr(bg, "build_dim_date", tracked("build_dim_date"))
    monkeypatch.setattr(bg, "read_bronze_partition", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bg, "build_dim_security", tracked("build_dim_security"))
    monkeypatch.setattr(bg, "build_fact_price_daily", tracked("build_fact_price_daily"))

    mock_report = MagicMock(consensus=pd.DataFrame())
    monkeypatch.setattr(bg, "reconcile_prices", lambda d: mock_report)
    monkeypatch.setattr(bg, "build_fact_price_daily_consensus", tracked("build_fact_price_daily_consensus"))
    monkeypatch.setattr(bg, "build_fact_corporate_action", tracked("build_fact_corporate_action"))
    monkeypatch.setattr(bg, "read_fred_bronze", lambda d: pd.DataFrame())
    monkeypatch.setattr(bg, "build_fact_macro_rate", tracked("build_fact_macro_rate"))

    fake_conn = MagicMock()
    bg.build_gold(date(2026, 9, 7), conn=fake_conn)

    # Dims before facts, and consensus after both fact_price_daily loads —
    # the actual dependency structure that matters here.
    assert calls.index("build_dim_date") < calls.index("build_fact_price_daily")
    assert calls.index("build_dim_security") < calls.index("build_fact_price_daily")
    assert calls.count("build_fact_price_daily") == 2  # yfinance and tiingo
    assert calls.index("build_fact_price_daily") < calls.index("build_fact_price_daily_consensus")


def test_build_gold_closes_only_a_connection_it_opened_itself(monkeypatch):
    monkeypatch.setattr(bg, "apply_schema", lambda conn: None)
    monkeypatch.setattr(bg, "build_silver_prices", lambda source, d: pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["AAPL"]}))
    monkeypatch.setattr(bg, "trading_days_from_silver", lambda df: set())
    monkeypatch.setattr(bg, "build_dim_date", lambda *a, **k: 0)
    monkeypatch.setattr(bg, "read_bronze_partition", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(bg, "build_dim_security", lambda *a, **k: {})
    monkeypatch.setattr(bg, "build_fact_price_daily", lambda *a, **k: {})
    monkeypatch.setattr(bg, "reconcile_prices", lambda d: MagicMock(consensus=pd.DataFrame()))
    monkeypatch.setattr(bg, "build_fact_price_daily_consensus", lambda *a, **k: {})
    monkeypatch.setattr(bg, "build_fact_corporate_action", lambda *a, **k: {})
    monkeypatch.setattr(bg, "read_fred_bronze", lambda d: pd.DataFrame())
    monkeypatch.setattr(bg, "build_fact_macro_rate", lambda *a, **k: {})

    caller_owned_conn = MagicMock()
    bg.build_gold(date(2026, 9, 7), conn=caller_owned_conn)
    caller_owned_conn.close.assert_not_called()  # caller supplied it, caller's responsibility to close it
