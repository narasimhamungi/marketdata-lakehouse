"""
Shared configuration for the marketdata-lakehouse pipeline.

Kept deliberately dumb: plain constants, no framework magic, so every module
that needs a path or a default can import from one place instead of
re-deriving it.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import date

from dotenv import load_dotenv

# Loads a local .env file (project root) if one exists — this is what lets
# API tokens survive across terminal sessions without re-setting $env:
# every time. override=True is deliberate: without it, python-dotenv
# skips a variable that's already set elsewhere in the environment (e.g.
# a leftover `setx` value from before this file existed), which defeats
# the entire point of having one authoritative place for these — .env
# should always win. .env is git-ignored; it never leaves your machine
# and is never something to paste into chat. See README "API keys".
load_dotenv(override=True)

# --- Root layout -------------------------------------------------------
# Everything under DATA_ROOT is git-ignored; only code and small fixtures
# (tests/fixtures/*) are committed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MDL_DATA_ROOT", PROJECT_ROOT / "data"))

BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _dir in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Pipeline defaults ---------------------------------------------------
DEFAULT_START_DATE = date(2019, 1, 1)  # ~5+ years of daily history
TODAY = date.today()

# Retry/backoff shared across all network sources.
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2  # exponential: 2, 4, 8, 16, 32

# Reconciliation tolerance on adjusted close, expressed as a fraction.
# 0.5% catches a genuinely wrong print or an unapplied split/dividend
# without flagging ordinary penny-level rounding differences between sources.
RECONCILIATION_TOLERANCE = 0.005

# User-Agent required by SEC EDGAR's fair-use policy (they will rate-limit
# or block requests without an identifying, non-generic User-Agent).
SEC_USER_AGENT = os.environ.get(
    "MDL_SEC_USER_AGENT",
    "marketdata-lakehouse (contact: narasimhamungi@gmail.com)",
)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Tiingo replaces the originally planned Stooq source (Stooq's CSV download
# began requiring an API key in March 2026 — confirmed against live data
# and via a pandas-datareader GitHub issue). Free tier: sign up at
# https://api.tiingo.com, no card confirmed by several independent sources
# but not personally verified end-to-end — if signup asks for one, stop.
TIINGO_API_KEY = os.environ.get("MDL_TIINGO_API_KEY", "")

# FRED (Federal Reserve Economic Data). Free API key, no card:
# https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get("MDL_FRED_API_KEY", "")

# OpenFIGI. Genuinely free, no cost-recovery ever (FIGI is a public-trust
# standard). An API key is optional here — it only raises the rate limit
# above the unkeyed tier; leave blank and this pipeline still works for a
# one-time backfill at this scale. Sign up free at
# https://www.openfigi.com/api if 429s show up.
OPENFIGI_API_KEY = os.environ.get("MDL_OPENFIGI_API_KEY", "")

# Gold layer (Postgres). Defaults match docker-compose.yml's postgres
# service — override via .env if connecting to a different instance.
GOLD_DB_HOST = os.environ.get("MDL_DB_HOST", "localhost")
GOLD_DB_PORT = int(os.environ.get("MDL_DB_PORT", "5432"))
GOLD_DB_NAME = os.environ.get("MDL_DB_NAME", "marketdata_lakehouse")
GOLD_DB_USER = os.environ.get("MDL_DB_USER", "mdl")
GOLD_DB_PASSWORD = os.environ.get("MDL_DB_PASSWORD", "mdl_local_dev")
