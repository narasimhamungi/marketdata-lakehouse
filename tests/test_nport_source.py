"""
Unit tests for N-PORT ingestion. The XML parsing logic is tested against
a fixture built directly from real N-PORT filings inspected on sec.gov
(structure confirmed live; specific holdings below are illustrative, not
copied verbatim from any one filing). The HTTP fetch itself is mocked —
these do not verify sec.gov's live availability or exact response shape,
same caveat as this module's own docstring.
"""
from unittest.mock import MagicMock

import pytest

from src.ingest import nport_source as n

# Structure matches real N-PORT-P primary_doc.xml documents: invstOrSec
# elements under a namespaced root, each with name/title/cusip/identifiers
# (isin)/balance/units/curCd/valUSD/pctVal/assetCat/issuerCat.
_SAMPLE_NPORT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <formData>
    <invstOrSecs>
      <invstOrSec>
        <name>Accenture PLC</name>
        <lei>5493000EWHDSR3MZWH98</lei>
        <title>Accenture PLC</title>
        <cusip>000000000</cusip>
        <identifiers>
          <isin value="IE00B4BNMY34"/>
        </identifiers>
        <balance>866.00000000</balance>
        <units>NS</units>
        <curCd>USD</curCd>
        <valUSD>259063.90000000</valUSD>
        <pctVal>1.869879477081</pctVal>
        <assetCat>EC</assetCat>
        <issuerCat>CORP</issuerCat>
        <invCountry>IE</invCountry>
      </invstOrSec>
      <invstOrSec>
        <name>Alphabet Inc</name>
        <lei>5493006MHB84DD0ZWV18</lei>
        <title>ALPHABET INC CLASS A</title>
        <cusip>02079K305</cusip>
        <identifiers>
          <isin value="US02079K3059"/>
        </identifiers>
        <balance>101014.00000000</balance>
        <units>NS</units>
        <curCd>USD</curCd>
        <valUSD>16041023.20000000</valUSD>
        <pctVal>1.300118806747</pctVal>
        <assetCat>EC</assetCat>
        <issuerCat>CORP</issuerCat>
        <invCountry>US</invCountry>
      </invstOrSec>
      <invstOrSec>
        <name>SA LARGE CAP INDEX PORT</name>
        <lei>5493001KG7IB71632F87</lei>
        <title>SA LARGE CAP INDEX PORT</title>
        <cusip>000000000</cusip>
        <identifiers>
          <other otherDesc="Internal Identifier" value="971SAV906"/>
        </identifiers>
        <balance>179550.31500000</balance>
        <units>NS</units>
        <curCd>USD</curCd>
        <valUSD>5097433.44000000</valUSD>
        <pctVal>2.297042771440</pctVal>
        <assetCat>EC</assetCat>
        <issuerCat>RF</issuerCat>
        <invCountry>US</invCountry>
      </invstOrSec>
      <cashCollateral>
        <name>Not a security holding, must not be parsed</name>
      </cashCollateral>
    </invstOrSecs>
  </formData>
</edgarSubmission>
""".encode("utf-8")


def test_parse_nport_xml_extracts_expected_fields():
    df = n._parse_nport_xml(_SAMPLE_NPORT_XML)

    assert len(df) == 3
    assert set(df["name"]) == {"Accenture PLC", "Alphabet Inc", "SA LARGE CAP INDEX PORT"}
    accenture = df[df["name"] == "Accenture PLC"].iloc[0]
    assert accenture["isin"] == "IE00B4BNMY34"
    assert accenture["cusip"] == "000000000"
    assert accenture["pct_val"] == pytest.approx(1.869879477081)


def test_parse_nport_xml_ignores_non_invstorsec_elements():
    df = n._parse_nport_xml(_SAMPLE_NPORT_XML)
    assert "Not a security holding, must not be parsed" not in df["name"].values


def test_parse_nport_xml_raises_on_zero_holdings():
    empty_xml = b"""<?xml version="1.0"?>
    <edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
      <formData><invstOrSecs></invstOrSecs></formData>
    </edgarSubmission>"""

    with pytest.raises(ValueError, match="zero"):
        n._parse_nport_xml(empty_xml)


def test_find_latest_nport_accession_picks_nport_p_form():
    fake_session = MagicMock()
    fake_session.get.return_value.json.return_value = {
        "filings": {
            "recent": {
                "form": ["10-K", "NPORT-P", "N-CEN"],
                "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-24-000003"],
                "filingDate": ["2024-01-01", "2024-02-15", "2024-03-01"],
            }
        }
    }

    accession, filed = n.find_latest_nport_accession(session=fake_session)

    assert accession == "000124000002"
    assert filed == "2024-02-15"


def test_find_latest_nport_accession_raises_when_none_found():
    fake_session = MagicMock()
    fake_session.get.return_value.json.return_value = {
        "filings": {"recent": {"form": ["10-K"], "accessionNumber": ["0001-24-000001"], "filingDate": ["2024-01-01"]}}
    }

    with pytest.raises(ValueError, match="No NPORT-P filing found"):
        n.find_latest_nport_accession(session=fake_session)