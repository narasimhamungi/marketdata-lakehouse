"""
Cross-check the Wikipedia-sourced S&P 500 constituent list against SPY's
independently-filed SEC N-PORT holdings (see src/ingest/nport_source.py).

Name-matched, not ticker-matched: N-PORT holdings carry company name/CUSIP/
ISIN, not ticker symbols, so this compares on a normalized company name
rather than the ticker join used everywhere else in this pipeline. That
makes it a heuristic, not an exact reconciliation like
price_reconciliation.py's adj_close comparison — name variants
(punctuation, legal suffixes, share-class labels) are expected and
normalized for, but a name that still doesn't match after normalization is
reported, never silently dropped or auto-corrected. A human should look at
anything flagged here, the same way a flagged price discrepancy gets a
human look rather than an automatic pick of one source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

_SUFFIXES = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|PLC|LTD|LIMITED|LLC|"
    r"GROUP|HOLDINGS?|CLASS [A-Z])\b\.?"
)
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize_company_name(name: str) -> str:
    """Best-effort normalization for cross-source name matching — not a
    general-purpose company-name normalizer, tuned to the specific
    variance seen between Wikipedia's 'Security' column and N-PORT's
    name/title fields (case, punctuation, legal-suffix and share-class
    differences)."""
    if not name:
        return ""
    name = name.upper()
    name = _PUNCTUATION.sub(" ", name)
    name = _SUFFIXES.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


@dataclass
class ConstituentsCrossCheckReport:
    wikipedia_count: int
    nport_count: int
    unmatched_wikipedia: pd.DataFrame = field(repr=False)
    unmatched_nport: pd.DataFrame = field(repr=False)

    @property
    def matched_count(self) -> int:
        return self.wikipedia_count - len(self.unmatched_wikipedia)

    def render(self) -> str:
        lines = [
            f"Constituents cross-check: Wikipedia ({self.wikipedia_count}) vs SPY N-PORT ({self.nport_count})",
            f"  Matched by normalized name: {self.matched_count}",
            f"  Wikipedia entries with no N-PORT match: {len(self.unmatched_wikipedia)}",
            f"  N-PORT holdings with no Wikipedia match: {len(self.unmatched_nport)}",
        ]
        if len(self.unmatched_wikipedia):
            lines.append(
                "\n  Unmatched Wikipedia tickers (verify manually — may be a real "
                "recent edit, or a naming variant this heuristic missed):"
            )
            for _, r in self.unmatched_wikipedia.iterrows():
                lines.append(f"    {r['ticker']:<8} {r['company_name']}")
        if len(self.unmatched_nport):
            lines.append(
                "\n  Unmatched N-PORT holdings (SPY holds these; Wikipedia's list "
                "doesn't obviously include them):"
            )
            for _, r in self.unmatched_nport.iterrows():
                lines.append(f"    {r['name']}")
        return "\n".join(lines)


def cross_check_constituents(
    wikipedia_df: pd.DataFrame, nport_df: pd.DataFrame
) -> ConstituentsCrossCheckReport:
    wiki = wikipedia_df.copy()
    wiki["_norm"] = wiki["company_name"].map(normalize_company_name)

    nport = nport_df.copy()
    nport["_norm"] = nport["name"].map(normalize_company_name)
    # Some N-PORT rows carry a more specific title (e.g. class-share
    # detail) than name — match against either field.
    nport["_norm_title"] = nport["title"].map(normalize_company_name)

    nport_names = set(nport["_norm"]) | set(nport["_norm_title"])
    wiki_names = set(wiki["_norm"])

    unmatched_wikipedia = wiki[~wiki["_norm"].isin(nport_names)].drop(columns=["_norm"])
    unmatched_nport = nport[
        ~nport["_norm"].isin(wiki_names) & ~nport["_norm_title"].isin(wiki_names)
    ].drop(columns=["_norm", "_norm_title"])

    return ConstituentsCrossCheckReport(
        wikipedia_count=len(wiki),
        nport_count=len(nport),
        unmatched_wikipedia=unmatched_wikipedia.reset_index(drop=True),
        unmatched_nport=unmatched_nport.reset_index(drop=True),
    )


def run_cross_check(ingest_date: date | None = None) -> ConstituentsCrossCheckReport:
    """Read the bronze constituents snapshot for `ingest_date` and compare it
    against SPY's latest N-PORT filing. Fetches N-PORT live (it is quarterly
    and small, so there is nothing to gain from a bronze partition for it)."""
    from src.config import BRONZE_DIR
    from src.ingest.nport_source import fetch_latest_nport_holdings
    from src.utils import read_bronze_partition

    ingest_date = ingest_date or date.today()
    wiki = read_bronze_partition(BRONZE_DIR / "constituents", ingest_date)
    if wiki.empty:
        raise ValueError(
            f"No constituents snapshot for ingest_date={ingest_date.isoformat()}. "
            "Run src.ingest.constituents first."
        )
    return cross_check_constituents(wiki, fetch_latest_nport_holdings())


if __name__ == "__main__":
    import logging
    import sys

    from src.config import OUTPUTS_DIR

    logging.basicConfig(level=logging.INFO)
    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    report = run_cross_check(target_date)
    print(report.render())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / f"constituents_crosscheck_{target_date.isoformat()}.txt"
    report_path.write_text(report.render(), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    # Unmatched entries are worth a human look but are NOT a pipeline
    # failure: name-matching across two independent sources produces
    # expected variance, and N-PORT lags the index by ~60 days, so recent
    # index changes legitimately appear on one side only. Exit 0 either way
    # — unlike price_reconciliation.py, where a flagged row means two
    # vendors genuinely disagree about the same number.
    sys.exit(0)