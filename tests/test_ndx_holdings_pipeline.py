from pathlib import Path
import importlib.util
import sys

import pandas as pd


ROOT = Path(__file__).parents[1]
NDX_SRC = ROOT / "data" / "ndx_simulation" / "src"
NDX_SCRIPTS = ROOT / "data" / "ndx_simulation" / "scripts"
sys.path.insert(0, str(NDX_SRC))
sys.path.insert(0, str(NDX_SCRIPTS))

# The ETF X-Ray package also has a top-level module named ``config``.  Load the
# NDX config explicitly while importing this standalone script stack, then put
# the prior module back so test collection remains order-independent.
_prior_config = sys.modules.get("config")
_config_spec = importlib.util.spec_from_file_location("config", NDX_SRC / "config.py")
_ndx_config = importlib.util.module_from_spec(_config_spec)
sys.modules["config"] = _ndx_config
_config_spec.loader.exec_module(_ndx_config)
try:
    import holdings_snapshots  # noqa: E402
    import ndx_parser  # noqa: E402
    import reconstruct_weights  # noqa: E402
finally:
    if _prior_config is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = _prior_config


def test_extract_report_date_uses_schedule_date_not_filing_date(tmp_path):
    filing = tmp_path / "2025-01-27_filing.htm"
    filing.write_text(
        "<h2>Schedule of Investments</h2><p>September 30, 2024</p>",
        encoding="utf-8",
    )

    assert ndx_parser.extract_report_date(filing).isoformat() == "2024-09-30"


def test_repeated_schedule_pages_are_concatenated(tmp_path):
    filing = tmp_path / "old_filing.txt"
    filing.write_text(
        """
        Schedule of Investments\nAuditor reference only
        Schedule of Investments\nShares Value
        Alpha Corporation........ 1,000 100,000
        Beta Corporation......... 2,000 200,000
        Schedule of Investments (continued)\nShares Value
        Gamma Corporation........ 3,000 300,000
        Delta Corporation........ 4,000 400,000
        Total Investments
        """,
        encoding="utf-8",
    )

    parsed = ndx_parser.parse_repeated_schedule_pages(filing)

    assert [row[0] for row in parsed] == [
        "Alpha Corporation",
        "Beta Corporation",
        "Gamma Corporation",
        "Delta Corporation",
    ]


def test_nport_parser_keeps_equities_and_excludes_index_future(tmp_path):
    filing = tmp_path / (
        "2026-03-31_2026-05-28_0001067839-26-000024_primary_doc.xml"
    )
    filing.write_text(
        """<?xml version="1.0"?>
        <edgarSubmission><formData><genInfo><repPdDate>2026-03-31</repPdDate></genInfo>
        <invstOrSecs>
          <invstOrSec><name>Apple Inc.</name><title>Apple Inc.</title>
            <cusip>037833100</cusip><balance>10</balance><units>NS</units>
            <valUSD>2000</valUSD><pctVal>2</pctVal><assetCat>EC</assetCat>
          </invstOrSec>
          <invstOrSec><name>N/A</name><title>CME E-Mini NASDAQ 100 Index Future</title>
            <cusip>N/A</cusip><balance>1</balance><units>NC</units>
            <valUSD>100</valUSD><pctVal>0.1</pctVal><assetCat>DE</assetCat>
          </invstOrSec>
        </invstOrSecs></formData></edgarSubmission>""",
        encoding="utf-8",
    )

    rows = holdings_snapshots.parse_nport_file(filing)

    assert len(rows) == 1
    assert rows[0]["Company"] == "Apple Inc."
    assert rows[0]["Shares"] == 10


def test_dated_official_mapping_overrides_modern_successor_symbol():
    membership = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2006-03-20")],
            "Tickers": ["BRCM|ERTS"],
            "Names": ["BROADCOM CORPORATION|ELECTRONIC ARTS INC"],
        }
    )
    positions = [
        {"Company": "Broadcom Corporation", "Title": "Broadcom Corporation"},
        {"Company": "Electronic Arts, Inc.", "Title": "Electronic Arts, Inc."},
    ]
    stale_mapping = {
        "Broadcom Corporation": "AVGO",
        "Electronic Arts, Inc.": "EA",
    }

    matched, _ = holdings_snapshots.match_positions_to_official(
        positions,
        "2006-03-20",
        stale_mapping,
        membership,
    )

    assert {row["Ticker"] for row in matched} == {"BRCM", "ERTS"}
    assert all(row["MappingMethod"].startswith("dated-") for row in matched)


def test_historical_name_alias_beats_recycled_parser_symbol_without_roster():
    positions = [
        {
            "Company": "Apple Computer, Inc",
            "Title": "Apple Computer, Inc",
            "Ticker": "POAI",
        }
    ]

    matched, _ = holdings_snapshots.match_positions_to_official(
        positions,
        "2000-09-30",
        {"Apple Computer, Inc": "AAPL"},
        pd.DataFrame(columns=["Date", "Tickers", "Names"]),
    )

    assert matched[0]["Ticker"] == "AAPL"


def test_anchor_selection_prefers_nearby_complete_nport_snapshot():
    holdings = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-09-30", "2024-03-31", "2024-03-31"]),
            "Ticker": ["A", "A", "B"],
            "Form": ["485BPOS", "NPORT-P", "NPORT-P"],
        }
    )

    anchor = reconstruct_weights._choose_anchor_date(
        holdings,
        pd.Timestamp("2024-03-25"),
        ["A", "B"],
    )

    assert anchor[2] == pd.Timestamp("2024-03-31")
    assert anchor[3] == 1.0
    assert anchor[4] == "NPORT-P"


def test_price_history_alias_preserves_dated_membership_symbol():
    prices = pd.DataFrame(
        {"META": [100.0, 101.0], "FB": [pd.NA, pd.NA], "QQQ": [50.0, 51.0]},
        index=pd.to_datetime(["2018-01-02", "2018-03-19"]),
    )

    pricing_ticker = reconstruct_weights._price_history_ticker(
        "FB",
        prices,
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2018-03-19"),
    )

    assert pricing_ticker == "META"


def test_pre_official_roster_prefers_names_seen_on_both_sides():
    holdings = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2000-09-30",
                    "2001-03-31",
                    "2001-03-31",
                    "2001-09-30",
                    "2001-09-30",
                ]
            ),
            "Ticker": ["B", "A", "B", "B", "C"],
            "Company": ["B Corp", "A Corp", "B Corp", "B Corp", "C Corp"],
            "IsMapped": [True, True, True, True, True],
        }
    )
    filler = pd.DataFrame(
        {
            "Date": pd.Timestamp("2001-03-31"),
            "Ticker": [f"X{i}" for i in range(87)],
            "Company": [f"X{i}" for i in range(87)],
            "IsMapped": True,
        }
    )
    holdings = pd.concat([holdings, filler], ignore_index=True)
    anchor = holdings[holdings["Date"] == pd.Timestamp("2001-03-31")]

    roster, _ = reconstruct_weights._infer_pre_official_roster(
        holdings, pd.Timestamp("2001-06-30"), anchor
    )

    assert len(roster) == 90
    assert "B" in roster


def test_reconstruction_quality_gate_rejects_membership_gap():
    weights = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-18")] * 90,
            "Ticker": [f"T{i}" for i in range(90)],
            "Weight": [1 / 90] * 90,
            "PriceTicker": [f"T{i}" for i in range(90)],
            "AnchorDistanceDays": [20] * 90,
        }
    )
    alignment = pd.DataFrame(
        {
            "OfficialCount": [101],
            "MissingOfficialCount": [1],
            "RemovedExtraCount": [0],
        }
    )

    try:
        reconstruct_weights.validate_reconstruction_quality(weights, alignment)
    except RuntimeError as exc:
        assert "official memberships are missing" in str(exc)
    else:
        raise AssertionError("quality gate must reject incomplete official membership")
