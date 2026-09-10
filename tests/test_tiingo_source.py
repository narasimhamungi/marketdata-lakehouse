"""
Unit tests for Tiingo ingestion. `requests.get` is mocked throughout — these
verify our JSON parsing and failure-handling logic, not Tiingo's live
availability, rate limits, or whether its free-tier signup genuinely
requires no card (documented as unverified in the module docstring).
"""
from datetime import date

import pandas as pd
import pytest

from src.ingest import tiingo_source as t


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    """The throttle's request-timestamp deque is module-level state, shared
    across tests unless reset — without this, tests would accumulate
    timestamps across the whole file and could eventually trigger a real
    sleep."""
    t._request_timestamps.clear()
    yield
    t._request_timestamps.clear()


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


_SAMPLE_RECORDS = [
    {
        "date": "2024-01-02T00:00:00.000Z",
        "open": 185.34,
        "high": 186.22,
        "low": 183.85,
        "close": 185.64,
        "adjClose": 185.64,
        "adjOpen": 185.34,
        "adjHigh": 186.22,
        "adjLow": 183.85,
        "adjVolume": 10000000,
        "volume": 10000000,
    },
    {
        "date": "2024-01-03T00:00:00.000Z",
        "open": 185.10,
        "high": 186.00,
        "low": 184.20,
        "close": 185.50,
        "adjClose": 185.50,
        "adjOpen": 185.10,
        "adjHigh": 186.00,
        "adjLow": 184.20,
        "adjVolume": 9500000,
        "volume": 9500000,
    },
]


def test_download_one_parses_json_correctly(monkeypatch):
    monkeypatch.setattr("src.ingest.tiingo_source.TIINGO_API_KEY", "a" * 40)
    monkeypatch.setattr(
        "src.ingest.tiingo_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(json_data=_SAMPLE_RECORDS),
    )

    df = t._download_one("AAPL", date(2024, 1, 1), date(2024, 1, 3))

    assert list(df.columns) == [
        "date", "open", "high", "low", "close", "adj_close",
        "adj_open", "adj_high", "adj_low", "adj_volume", "volume", "ticker",
    ]
    assert len(df) == 2
    assert (df["ticker"] == "AAPL").all()
    assert df["date"].dtype.kind == "M"


def test_download_one_raises_without_api_key(monkeypatch):
    monkeypatch.setattr("src.ingest.tiingo_source.TIINGO_API_KEY", "")

    with pytest.raises(RuntimeError, match="MDL_TIINGO_API_KEY"):
        t._download_one("AAPL", date(2024, 1, 1), date(2024, 1, 3))


def test_download_one_raises_on_malformed_token(monkeypatch):
    """
    Regression test for the exact real-world failure: placeholder text
    ("your_token_here") left concatenated with the real token when setting
    the environment variable, producing an invalid combined string that
    previously wasn't caught until Tiingo returned a bare 403.
    """
    monkeypatch.setattr(
        "src.ingest.tiingo_source.TIINGO_API_KEY",
        "your_token_heree8d1a177b331a019bef6bac232e4e574664b77d0",
    )

    with pytest.raises(RuntimeError, match="doesn't look like a valid Tiingo token"):
        t._download_one("AAPL", date(2024, 1, 1), date(2024, 1, 3))


def test_download_one_raises_distinctly_on_404(monkeypatch):
    monkeypatch.setattr("src.ingest.tiingo_source.TIINGO_API_KEY", "a" * 40)
    monkeypatch.setattr(
        "src.ingest.tiingo_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(status_code=404),
    )

    with pytest.raises(ValueError, match="404"):
        t._download_one("NOTAREALTICKER", date(2024, 1, 1), date(2024, 1, 3))


def test_download_one_raises_on_schema_drift(monkeypatch):
    monkeypatch.setattr("src.ingest.tiingo_source.TIINGO_API_KEY", "a" * 40)
    monkeypatch.setattr(
        "src.ingest.tiingo_source.requests.get",
        lambda url, params=None, timeout=None: _FakeResponse(json_data=[{"date": "2024-01-02", "wrong": 1}]),
    )

    with pytest.raises(ValueError, match="schema drift"):
        t._download_one("AAPL", date(2024, 1, 1), date(2024, 1, 3))


def test_ingest_tiingo_batch_writes_bronze_and_returns_frame(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.tiingo_source.BRONZE_DIR", tmp_path)
    monkeypatch.setattr(
        "src.ingest.tiingo_source._download_one",
        lambda ticker, start, end: pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "adj_close": [1.0, 2.0],
                "volume": [100, 200],
                "ticker": [ticker, ticker],
            }
        ),
    )

    result = t.ingest_tiingo_batch(["AAPL", "MSFT"], ingest_date=date(2026, 9, 6))

    assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}
    out_dir = tmp_path / "tiingo_prices" / "ingest_date=2026-09-06"
    assert (out_dir / "AAPL.parquet").exists()
    assert (out_dir / "MSFT.parquet").exists()


def test_ingest_tiingo_batch_records_failed_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingest.tiingo_source.BRONZE_DIR", tmp_path)

    def flaky(ticker, start, end):
        raise ValueError("simulated Tiingo failure")

    monkeypatch.setattr("src.ingest.tiingo_source._download_one", flaky)

    result = t.ingest_tiingo_batch(["AAPL", "GHOST"], ingest_date=date(2026, 9, 6))

    assert result.empty
    failed_file = tmp_path / "tiingo_prices" / "ingest_date=2026-09-06" / "_failed_tickers.txt"
    assert failed_file.exists()
    assert set(failed_file.read_text().splitlines()) == {"AAPL", "GHOST"}


def test_throttle_does_not_sleep_under_the_limit():
    fake_now = [1000.0]
    sleep_calls = []

    for _ in range(t._TIINGO_SAFE_HOURLY_LIMIT - 1):
        t._throttle(now_fn=lambda: fake_now[0], sleep_fn=sleep_calls.append)
        fake_now[0] += 1  # 1 second apart, well within the hour

    assert sleep_calls == []
    assert len(t._request_timestamps) == t._TIINGO_SAFE_HOURLY_LIMIT - 1


def test_throttle_waits_once_the_safe_hourly_limit_is_reached():
    fake_now = [1000.0]
    sleep_calls = []

    def now_fn():
        return fake_now[0]

    def sleep_fn(seconds):
        sleep_calls.append(seconds)
        fake_now[0] += seconds  # simulate time actually passing during the sleep

    # Fill the window right up to the safe limit, all within the same hour.
    for _ in range(t._TIINGO_SAFE_HOURLY_LIMIT):
        t._throttle(now_fn=now_fn, sleep_fn=sleep_fn)
        fake_now[0] += 1

    assert sleep_calls == []  # limit reached exactly, not yet exceeded

    # One more call should now trigger a wait rather than proceeding
    # immediately and risking a 429 — this is the exact scenario that
    # previously ran unthrottled against live data.
    t._throttle(now_fn=now_fn, sleep_fn=sleep_fn)

    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


def test_throttle_prunes_timestamps_older_than_an_hour():
    fake_now = [0.0]
    sleep_calls = []

    t._throttle(now_fn=lambda: fake_now[0], sleep_fn=sleep_calls.append)
    fake_now[0] += 3601  # advance past the 1-hour window

    t._throttle(now_fn=lambda: fake_now[0], sleep_fn=sleep_calls.append)

    # The first timestamp should have aged out, leaving only the second —
    # if pruning didn't work, stale entries would eventually force
    # unnecessary waits even after real time has clearly moved on.
    assert len(t._request_timestamps) == 1
    assert sleep_calls == []
