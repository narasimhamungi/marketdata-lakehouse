-- Gold layer: star schema over the reconciled silver price data.
--
-- dim_date's is_trading_day is deliberately NOT derived from a hardcoded
-- NYSE holiday calendar (fragile, needs yearly maintenance, easy to get
-- subtly wrong). It's derived empirically instead, by the loader, from
-- whether any security actually has a price row on that date in the
-- silver data — the data itself is the source of truth for what a
-- trading day was, not a calendar Claude wrote by hand.
--
-- dim_security is a real SCD-2: effective_from/effective_to/is_current,
-- driven by comparing each new constituents snapshot against the current
-- record per ticker. With only one snapshot ingested so far, every row
-- will show effective_to = NULL (nothing to detect a change against yet)
-- — that's correct, not a bug; the history populates as more snapshots
-- are ingested over time.
--
-- fact_price_daily preserves both vendors' data distinctly, at
-- (security, date, source) grain — this is what the reconciliation
-- module's comparisons are actually built on. fact_price_daily_consensus
-- is the derived, single "trusted" row per (security, date): the actual
-- product a quant would query, with the reconciliation outcome carried
-- as a first-class column rather than requiring a join back to the
-- reconciliation module's own output to know whether a row is trustworthy.

CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,          -- YYYYMMDD
    date DATE NOT NULL UNIQUE,
    year SMALLINT NOT NULL,
    quarter SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    month_name TEXT NOT NULL,
    day SMALLINT NOT NULL,
    day_of_week SMALLINT NOT NULL,         -- 0=Monday .. 6=Sunday
    day_name TEXT NOT NULL,
    is_weekday BOOLEAN NOT NULL,
    is_trading_day BOOLEAN NOT NULL        -- empirically derived, see above
);

CREATE TABLE IF NOT EXISTS dim_security (
    security_key SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    company_name TEXT,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    figi TEXT,
    effective_from DATE NOT NULL,
    effective_to DATE,                     -- NULL = currently in effect
    is_current BOOLEAN NOT NULL DEFAULT TRUE
);

-- Enforces at most one currently-effective row per ticker.
CREATE UNIQUE INDEX IF NOT EXISTS idx_dim_security_ticker_current
    ON dim_security (ticker) WHERE is_current;

CREATE UNIQUE INDEX IF NOT EXISTS idx_dim_security_ticker_effective
    ON dim_security (ticker, effective_from);

CREATE TABLE IF NOT EXISTS fact_price_daily (
    security_key INTEGER NOT NULL REFERENCES dim_security(security_key),
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    source TEXT NOT NULL,                  -- 'yfinance' | 'tiingo'
    open NUMERIC(18, 6),
    high NUMERIC(18, 6),
    low NUMERIC(18, 6),
    close NUMERIC(18, 6),
    adj_open NUMERIC(18, 6),
    adj_high NUMERIC(18, 6),
    adj_low NUMERIC(18, 6),
    adj_close NUMERIC(18, 6),
    volume BIGINT,
    adj_volume BIGINT,
    PRIMARY KEY (security_key, date_key, source)
);

CREATE TABLE IF NOT EXISTS fact_price_daily_consensus (
    security_key INTEGER NOT NULL REFERENCES dim_security(security_key),
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    adj_close NUMERIC(18, 6) NOT NULL,
    primary_source TEXT NOT NULL,          -- which source this adj_close came from
    sources_available TEXT[] NOT NULL,     -- which sources had a row at all
    pct_diff NUMERIC(9, 6),                -- NULL when only one source available
    reconciliation_flag TEXT NOT NULL,     -- 'agreed' | 'single_source' | 'disagreed'
    PRIMARY KEY (security_key, date_key)
);

CREATE TABLE IF NOT EXISTS fact_corporate_action (
    security_key INTEGER NOT NULL REFERENCES dim_security(security_key),
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    action_type TEXT NOT NULL,             -- 'dividend' | 'split'
    value NUMERIC(18, 6) NOT NULL,         -- dividend amount, or split ratio
    source TEXT NOT NULL,
    PRIMARY KEY (security_key, date_key, action_type, source)
);

CREATE TABLE IF NOT EXISTS fact_macro_rate (
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    series_id TEXT NOT NULL,
    value NUMERIC(18, 6) NOT NULL,
    PRIMARY KEY (date_key, series_id)
);
