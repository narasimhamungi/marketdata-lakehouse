# marketdata-lakehouse

![CI](https://github.com/narasimhamungi/marketdata-lakehouse/actions/workflows/ci.yml/badge.svg)

Vendor-style multi-source equities data platform: ingests daily OHLCV and
index-membership data from independent free sources, lands it in a
bronze/silver/gold layered store, and proves it's correct through automated
cross-source reconciliation and data-quality checks — the kind of ingestion
and quality layer a market-data vendor (or a quant shop's internal data team)
builds on day one.

## Status

Early build — ingestion layer in progress. See `docs/architecture.md` (added
once the gold layer exists) for the full design.

- [x] Repo scaffold
- [x] Constituents ingestion (S&P 500 membership, dated snapshots) — confirmed against live data (503 constituents)
- [x] yfinance bulk OHLCV ingestion, batched, with retry/backoff and a
      recorded fallback list for tickers that fail — confirmed against live data
- [x] Tiingo ingestion (fallback source + reconciliation partner) — replaces
      the originally planned Stooq source, which began requiring an API key
      in March 2026 (see "Why these sources" below). Confirmed against live
      data. Adjusted-close agrees with yfinance to ~0.002% on AAPL — see
      the reconciliation note below before comparing any other field.
- [x] FRED ingestion (macro/risk-free rate series) — 4 fixed series
      (3-month & 10-year Treasury, Fed funds rate, CPI). Not yet tested
      against live data — needs a free FRED signup and API key first.
- [x] OpenFIGI ingestion (identifier mapping) — no API key required at
      this scale (unkeyed rate limit is enough for a one-time backfill).
      Not yet tested against live data.

- [x] Quality suite on bronze (Great Expectations) — schema, null, range,
      internal-consistency (high/low/open/close relationships), and
      uniqueness checks across all five sources. Confirmed against live
      data: 8/8 sources clean (all four FRED series checked individually).
- [x] Silver transform (typed, deduped, fully-adjusted OHLCV) — one clean
      table per price source (yfinance, Tiingo), kept separate on purpose
      since reconciliation (next) compares them against each other.
      yfinance only provides adj_close, so adj_open/high/low are derived
      via the standard adj_close/close factor technique, explicit and
      auditable rather than hidden inside a library flag. Tiingo actually
      provides adj_open/adj_high/adj_low/adj_volume natively — the ingest
      module was updated to capture them instead of discarding them, with
      a fallback to the same derivation technique for any older Tiingo
      bronze partitions pulled before that fix. Tested (9 tests, including
      the fallback path) but not yet run against your live bronze data.
- [ ] Cross-source reconciliation (tolerance-band comparison)
- [x] Cross-source reconciliation (tolerance-band comparison) — compares
      only adj_close between yfinance and Tiingo silver (raw OHLC isn't
      comparable, see the note above), symmetric % discrepancy (relative
      to the average of both sources, not biased toward either one), 0.5%
      tolerance. Coverage gaps (a ticker/date in only one source) are
      tracked separately from actual discrepancies — conflating the two
      would understate what was really checked. Tested (7 tests) but not
      yet run against your live silver data.
- [ ] Gold dimensional model (Postgres)
- [ ] Airflow orchestration
- [ ] CI (GitHub Actions)

## Why these sources

| Source | Role | Cost |
|---|---|---|
| yfinance | Primary OHLCV | Free (unofficial Yahoo endpoints — see note below) |
| Tiingo | Reconciliation + fallback OHLCV | Free tier (signup + API token required) |
| Wikipedia S&P 500 table | Index constituents | Free (factual/tabular data) |
| FRED | Macro series, risk-free rate | Free, requires a free API key |
| OpenFIGI | Instrument identifier mapping | Free |

**On the switch from Stooq to Tiingo:** the original plan used Stooq as a
no-registration free source. Running this against live data surfaced that
Stooq's CSV download has required an API key since March 2026 (confirmed
via a [pandas-datareader GitHub issue](https://github.com/pydata/pandas-datareader/issues/1012)),
and separate reports suggest its historical server has become unreliable
even with a key. Tiingo fills the same role — confirmed free tier by
several independent sources, not personally verified end-to-end, so flag
it if signup ever asks for a card. Free-tier limit: 500 unique symbols/
month, which the initial ~500-ticker backfill uses almost entirely in one
run; daily updates on the same universe afterward don't count against it.

**Reconciliation note, confirmed against live AAPL data:** yfinance's raw
OHLC (`auto_adjust=False`) is not split-adjusted; Tiingo's raw OHLC is
split-adjusted by default. Only `adj_close` is comparable across the two
sources — raw `close`-to-`close` will show AAPL off by ~4x post-2020-split
and flag as a false reconciliation failure. `adj_close` agreed to ~0.002%
on the one data point checked so far, which is the expected result.

**On yfinance:** there is no official Yahoo Finance API — `yfinance` wraps
undocumented internal endpoints that Yahoo can change or rate-limit without
notice. It also ships its own local SQLite-backed cache, which can throw
`database is locked` for an individual ticker inside an otherwise-successful
batch under concurrent (threaded) access — confirmed against live data. This
pipeline treats that as an expected failure mode, not an edge case: any
ticker missing from a batch's output, whether the whole call raised or just
that one ticker silently dropped, is written to a fallback list for Tiingo
to pick up. That's exactly why Tiingo exists here as more than a
reconciliation checkbox.

**On Wikipedia's constituents table:** `pd.read_html(url)` lets `urllib`
make the request with its default User-Agent, which Wikipedia returns a 403
for. Fixed by fetching the page with a descriptive User-Agent first, then
parsing the HTML — confirmed against live data.

## API keys — set once, forever, per machine

This project reads API keys from environment variables, but rather than
re-setting them in every new terminal (`$env:` only lasts for the session
that set it — the actual cause of the repeated Tiingo/FRED "why did my key
disappear" issues earlier), create a **`.env` file** in the project root:

```
MDL_TIINGO_API_KEY=your_real_tiingo_token
MDL_FRED_API_KEY=your_real_fred_key
```

This file is already covered by `.gitignore` — it never gets committed,
never leaves your machine, and is loaded automatically by every command in
this project from now on. **Put your real tokens directly into that local
file, not into chat** — pasting a real credential into a conversation
doesn't help fix anything here (nothing here can set an environment
variable on your machine remotely) and only exposes it unnecessarily.

If you've pasted a real token into this chat before, it's low-stakes (a
free-tier data API key, not a financial credential) but cheap to rotate on
the provider's dashboard if you'd rather not have it floating around.

## Gold layer (Postgres)

The gold layer needs a real Postgres running — `docker-compose.yml` already
defines it:

```powershell
docker compose up -d postgres
```

Apply the schema (idempotent, safe to re-run):
```powershell
python -c "from src.model.db import apply_schema; apply_schema()"
```

**dim_date**'s `is_trading_day` is derived empirically from silver price
data (a date counts as a trading day if any security actually traded on
it), not from a hardcoded NYSE holiday calendar — see `src/model/schema.sql`
for the reasoning. Build it from a real trading-day set:
```python
from datetime import date
from src.model.dim_date import build_dim_date, trading_days_from_silver
from src.transform.silver_prices import build_silver_prices

silver = build_silver_prices("yfinance", date(2026, 9, 7))
trading_days = trading_days_from_silver(silver)
build_dim_date(date(2019, 1, 1), date(2026, 12, 31), trading_days)
```

**dim_security** is a real SCD-2 — see `src/model/dim_security.py`'s
docstring. With only one constituents snapshot ingested so far, every row
will show `effective_to = NULL`; that's correct, history populates as more
snapshots get ingested over time, not from a single run.

Running the gold-layer tests locally requires a **separate test database**,
created once — deliberately not your real `marketdata_lakehouse` database,
since these tests `TRUNCATE` every table before each run and pointed at
real data that would silently wipe it:
```powershell
docker exec mdl_postgres psql -U mdl -d marketdata_lakehouse -c "CREATE DATABASE marketdata_lakehouse_test"
```
(`tests/conftest.py`'s `pg_conn` fixture connects to this — real Postgres,
not a mock, since SQL correctness is exactly what a mock would let through
wrong. Its defaults already match docker-compose's credentials, so nothing
else needs configuring.)

**Build the whole gold layer in one command**, once Postgres is running:
```powershell
python -m src.orchestrate.build_gold
```

Runs, in dependency order: schema apply → `dim_date` (trading days derived
from real silver data) → `dim_security` (SCD-2) → `fact_price_daily` for
both sources → `fact_price_daily_consensus` (reconciliation's actual
output, loaded as a queryable "trusted price" table — every row carries
`primary_source`, `sources_available`, `pct_diff`, and
`reconciliation_flag`, so a consumer doesn't need to know reconciliation
happened, just query the flag) → `fact_corporate_action` (from yfinance's
dividends/stock_splits, captured at ingestion but unused until now) →
`fact_macro_rate`.

## Orchestration (Airflow)

`airflow/dags/marketdata_lakehouse_dag.py` chains everything above into
one DAG: 5 ingestion tasks → a real bronze quality gate (fails the run,
doesn't just log, if any source fails its checks) → silver for both
sources → reconciliation → gold dimensions → all five gold fact loads →
a real gold freshness gate (fails the run if security count, history
depth, or data currency look wrong — not an SLA-miss email, an actual
check with real thresholds). Every task is a thin wrapper around a
function already in `src/` and already tested — the DAG's only job is
correct wiring, not new logic.

Uses Airflow's own `data_interval_start` for every date argument, never
`date.today()` — deliberately, after `date.today()` inside pipeline code
caused a real midnight-boundary bug earlier in this project. This is
exactly the class of bug Airflow's scheduling model exists to prevent.

**Start it:**
```powershell
docker compose up -d
```
Both Postgres and Airflow start. Airflow's first boot installs this
project's own dependencies into the container, then runs `airflow
standalone` — expect a couple of minutes before the UI is reachable.

**Find the admin login** (auto-generated on first run — note this changes
after the LocalExecutor switch below, since it's tied to a fresh metadata
database; your first password won't work anymore):
```powershell
docker compose logs airflow | Select-String password
```

**Open the UI:** http://localhost:8080 — the DAG (`marketdata_lakehouse`)
appears paused by default; toggle it on, or trigger a manual run directly
from the UI to test it without waiting for the schedule (weekdays,
22:00 UTC, after US market close).

**Validate the DAG structure without a running Airflow instance** (what I
used to verify the file before handing it over — real `DagBag` loading,
not a guess):
```powershell
pip install apache-airflow
python -m pytest tests/test_dag_structure.py -v
```
These tests are skipped automatically (`pytest.importorskip`) in your
normal `pytest tests/` runs, since apache-airflow isn't in the main
`requirements.txt` — it's only needed inside the Airflow container, not
for running the ingestion/silver/gold pipeline directly.

## Running the full universe

Everything above uses AAPL/MSFT/AMZN as a fast demo loop. To run the real
~500-ticker S&P 500 universe in one command:

```powershell
python -m src.orchestrate.run_full_universe
```

yfinance and OpenFIGI run against the **full universe** — both proved
capable of that at full scale against live data. Tiingo does **not**: its
free tier caps at 50 requests/hour, so a full 503-ticker pull takes over
10 hours in one sitting — confirmed the hard way when an early run had no
rate limiting and blew straight through the hourly cap. Tiingo instead
gets a **sector-stratified sample** (45 tickers by default, spread across
every GICS sector so reconciliation still has something representative to
check), and the Tiingo ingest module itself now throttles to stay under
the hourly limit regardless of how many tickers are requested — defense
in depth, not just a smaller ask.

Options:
```powershell
# Re-run just one source (e.g. after fixing something, without re-pulling
# yfinance/OpenFIGI that already succeeded):
python -m src.orchestrate.run_full_universe --sources=tiingo

# Force a fresh constituents pull instead of reusing today's snapshot:
python -m src.orchestrate.run_full_universe --refresh-constituents

# A specific date:
python -m src.orchestrate.run_full_universe 2026-09-07
```

## Running the ingestion locally

This sandbox environment has no network access to Yahoo, Tiingo, Wikipedia,
FRED, or SEC EDGAR — only the unit tests (which mock all network calls) run
here. Live ingestion needs to run on a machine with normal internet access:

```bash
python -m venv .venv && source .venv/bin/activate   # or your usual environment
pip install -r requirements.txt

# Pull current S&P 500 constituents
python -m src.ingest.constituents

# Pull OHLCV for a few tickers (defaults to 2019-01-01 -> today)
python -m src.ingest.yfinance_source AAPL MSFT AMZN

# Pull the full constituent list once the snapshot above exists:
python -c "
import pandas as pd
from src.ingest.yfinance_source import ingest_yfinance_batch
tickers = pd.read_parquet('data/bronze/constituents').ticker.tolist()
ingest_yfinance_batch(tickers)
"

# Pull the same tickers from Tiingo — requires a free API token first:
#   1. Sign up at https://api.tiingo.com (flag it here if it asks for a card)
#   2. Set the environment variable: setx MDL_TIINGO_API_KEY "your_token_here"
#      (then open a new terminal so it takes effect)
python -m src.ingest.tiingo_source AAPL MSFT AMZN

# Pull macro/risk-free rate series from FRED — needs a free API key first:
#   1. Get one at https://fred.stlouisfed.org/docs/api/api_key.html
#   2. $env:MDL_FRED_API_KEY = "your_key" (verify with echo before running)
python -m src.ingest.fred_source

# Map tickers to FIGIs — no key needed at this scale, but OpenFIGI's docs
# are at https://www.openfigi.com/api if you want one anyway
python -m src.ingest.openfigi_source AAPL MSFT AMZN

# Run the quality suite against today's bronze data (whatever you've
# ingested so far — it only checks sources that have data for the date):
python -m src.quality.bronze_checks
# Or a specific date:
python -m src.quality.bronze_checks 2026-09-07

# Build the silver layer (typed, deduped, adjusted OHLCV) from today's
# bronze. Re-run Tiingo ingestion first if you want Tiingo's own native
# adjusted OHLC rather than the derived fallback — the ingest module was
# updated after your last Tiingo pull to capture those fields directly.
python -m src.transform.silver_prices

# Reconcile the two sources against each other (requires silver for both
# yfinance and Tiingo to already exist for the date — run silver_prices
# first if you haven't):
python -m src.reconcile.price_reconciliation
# Or a specific date:
python -m src.reconcile.price_reconciliation 2026-09-07
```

Great Expectations prints its own metric-calculation progress bars to
stderr during this — harmless, redirect with `2>$null` if they're
distracting.

Paste the terminal output back and we'll debug from there against real data,
same as Bridgework and Trellis.

## Tests

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "."
python -m pytest tests/ --ignore=tests/test_dag_structure.py -v
```

118 tests, mocking every external network boundary (Wikipedia, yfinance,
Tiingo, FRED, OpenFIGI) so they verify this project's own logic, not a
third-party service's availability — except the ~40 gold-layer tests,
which deliberately use a real local Postgres instead of a mock (see "Gold
layer" above for the one-time test-database setup); SQL correctness is
exactly what a mock would let through wrong.

`tests/test_dag_structure.py` (10 tests, DAG parsing/wiring via Airflow's
real `DagBag`) is excluded above because it needs `apache-airflow`
installed, which is deliberately NOT in `requirements.txt` — it conflicts
with this project's own pandas version (confirmed in development, not
hypothetical; see the CI workflow's comments). Run it in its own throwaway
environment if you want to validate the DAG locally:
```powershell
pip install "apache-airflow==2.10.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.12.txt"
pip install psycopg2-binary python-dotenv pytest
$env:PYTHONPATH = "."
python -m pytest tests/test_dag_structure.py -v
```

**Regression tests** (`tests/test_regression_pipeline.py`) use a small,
fixed snapshot of *real* data — the actual AAPL values this project
validated against live yfinance and Tiingo earlier, not synthetic round
numbers — with expected outputs computed independently rather than
re-derived from the code's own formulas (which would make the test
tautological). This catches a class of bug pure unit tests can't: a subtle
error in the adjustment-factor math or reconciliation calculation could
still pass every small-number unit test while producing a wrong answer on
real-precision data.

Many bug-specific regression tests are scattered through the suite —
found running against live data, not hypothesized in advance: a Wikipedia
403 (bad User-Agent), a silently-dropped ticker inside an otherwise-
successful yfinance batch (yfinance's own SQLite cache locking under
concurrency), a stale/malformed Tiingo token, a Tiingo hourly rate-limit
breach, a Windows console encoding crash on a non-ASCII character, a
midnight ingest-date rollover, and a single real bad print in yfinance
data (`HUBB`, 2021-05-05, `low > open`) that the bronze quality gate now
tolerates without masking genuinely systemic problems.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`, as two
independent jobs:
- **test** — the full suite above, against a real `postgres:16` service
  container (GitHub Actions' standard pattern for this, not a workaround)
- **dag-validation** — `test_dag_structure.py`, in its own isolated
  environment for the same dependency-conflict reason as the local
  instructions above

Both run free on GitHub-hosted runners for a public repo.

