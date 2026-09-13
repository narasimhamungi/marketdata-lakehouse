"""
SEC Form N-PORT holdings as an independent cross-check on the Wikipedia
constituents list.

Not a replacement: N-PORT's <invstOrSec> holdings carry company name,
CUSIP, and ISIN — no ticker symbol and no GICS sector, both of which
run_full_universe.py's stratified_sample() depends on (confirmed by
inspecting real N-PORT filings on sec.gov before writing this). Wikipedia
stays the ticker/sector source; this exists to address the actual concern
raised about it — an unnoticed bad edit, or a table structure drift that
EXPECTED_COLUMNS in constituents.py doesn't happen to catch — by comparing
against an independent, SEC-regulated source that's a much higher bar to
tamper with unnoticed. See src/reconcile/constituents_reconciliation.py
for the comparison logic.

Source: SPDR S&P 500 ETF Trust's (SPY, CIK 884394) most recent Form
N-PORT-P filing on SEC EDGAR. A physically-replicating S&P 500 ETF's
disclosed holdings are, definitionally, the S&P 500 constituent list as of
the filing's reporting date. Quarterly, ~60-day reporting lag — a
reconciliation check, not a live source.

Caution, matching this project's standing practice for anything not
personally verified end-to-end: the submission-lookup and document-fetch
path against live data has NOT been run in the environment this was
written in (no network access to sec.gov in that sandbox). The XML
parsing logic *is* tested, against a fixture built directly from real
N-PORT filings inspected on sec.gov (see tests/test_nport_source.py) —
but that is not the same as an end-to-end live run. Run this once
manually against live data and confirm the shape of what comes back
before wiring it into a schedule.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.config import SEC_USER_AGENT

logger = logging.getLogger(__name__)

SPY_CIK = "0000884394"  # SPDR S&P 500 ETF Trust
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/primary_doc.xml"


def _strip_ns(tag: str) -> str:
    """ElementTree qualifies tags with their namespace as '{uri}tag'.
    N-PORT's XML declares one; rather than hardcode a namespace URI this
    hasn't been verified live against, matching on the local tag name is
    robust regardless of the exact URI or whether a namespace is present
    at all."""
    return tag.rsplit("}", 1)[-1]


def find_latest_nport_accession(cik: str = SPY_CIK, session=None) -> tuple[str, str]:
    """Returns (accession_number_no_dashes, filing_date) for the most
    recent NPORT-P filing found in the filer's submissions history."""
    session = session or requests.Session()
    url = SUBMISSIONS_URL.format(cik=cik)
    response = session.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    response.raise_for_status()
    data = response.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    for form, accession, filed in zip(forms, accessions, dates):
        if form == "NPORT-P":
            return accession.replace("-", ""), filed

    raise ValueError(f"No NPORT-P filing found in the recent submissions for CIK {cik}")


def _parse_nport_xml(xml_bytes: bytes) -> pd.DataFrame:
    """Parse <invstOrSec> holdings out of an N-PORT primary_doc.xml.

    Raises if zero holdings are found — an empty result is far more likely
    to mean the parser didn't match the document's actual structure than
    that a fund genuinely holds nothing, so this fails loud rather than
    silently returning an empty frame that would look like "no
    discrepancies" to the cross-check downstream.
    """
    root = ET.fromstring(xml_bytes)
    rows = []
    for sec in root.iter():
        if _strip_ns(sec.tag) != "invstOrSec":
            continue
        fields = {
            _strip_ns(child.tag): (child.text or "").strip()
            for child in sec
            if len(child) == 0
        }
        isin = None
        for child in sec:
            if _strip_ns(child.tag) == "identifiers":
                for ident in child:
                    if _strip_ns(ident.tag) == "isin":
                        isin = ident.attrib.get("value")
        rows.append(
            {
                "name": fields.get("name"),
                "title": fields.get("title"),
                "cusip": fields.get("cusip"),
                "isin": isin,
                "asset_cat": fields.get("assetCat"),
                "pct_val": float(fields["pctVal"]) if fields.get("pctVal") else None,
                "val_usd": float(fields["valUSD"]) if fields.get("valUSD") else None,
            }
        )

    if not rows:
        raise ValueError(
            "Parsed zero <invstOrSec> holdings from the N-PORT document — the XML "
            "structure likely doesn't match what this parser expects (namespace, "
            "element names, or SEC has changed the N-PORT schema). Inspect the raw "
            "document before trusting an empty result as 'the fund holds nothing'."
        )

    df = pd.DataFrame(rows)
    # EC = equity, common — excludes cash, other funds held for liquidity
    # management, etc. that aren't part of the index-membership signal
    # this is used for.
    return df[df["asset_cat"] == "EC"].reset_index(drop=True)


def fetch_latest_nport_holdings(cik: str = SPY_CIK, session=None) -> pd.DataFrame:
    session = session or requests.Session()
    accession_nodash, filed = find_latest_nport_accession(cik, session=session)
    url = ARCHIVE_DOC_URL.format(cik_int=int(cik), accession_nodash=accession_nodash)
    response = session.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    response.raise_for_status()

    df = _parse_nport_xml(response.content)
    df["nport_filed_date"] = filed
    logger.info("Parsed %d equity holdings from NPORT-P filed %s", len(df), filed)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    holdings = fetch_latest_nport_holdings()
    print(holdings.head(10))
    print(f"\n{len(holdings)} equity holdings total")