from pathlib import Path
import sys

import pandas as pd


NDX_SRC = Path(__file__).parents[1] / "data" / "ndx_simulation" / "src"
sys.path.insert(0, str(NDX_SRC))

import free_price_history  # noqa: E402
import sec_index_fund_holdings  # noqa: E402


def test_rydex_parser_handles_schedule_before_fund_name():
    filing = b"""
    <html><body><pre>
    SCHEDULE OF INVESTMENTS March 31, 2005
    OTC FUND
    COMMON STOCKS 99.5%
    Microsoft Corp........ 2,000 $ 50,000
    Oracle Corp.* 1,000 12,000
    SCHEDULE OF INVESTMENTS
    NOVA FUND
    Microsoft Corp........ 9,000 $ 999,000
    </pre></body></html>
    """

    rows = sec_index_fund_holdings.parse_filing(filing, "SEC-RYDEX-NDX")

    assert {(row["Company"], row["Shares"], row["Value"]) for row in rows} == {
        ("Microsoft Corp", 2000.0, 50000.0),
        ("Oracle Corp", 1000.0, 12000.0),
    }


def test_token_parser_handles_one_html_cell_per_value():
    lines = [
        "Microsoft Corp.",
        "2,000",
        "$",
        "50,000",
        "Veritas Software",
        "Corp.*",
        "1,000",
        "12,000",
    ]

    rows = sec_index_fund_holdings._parse_token_rows(lines)

    assert [row["Company"] for row in rows] == [
        "Microsoft Corp",
        "Veritas Software Corp",
    ]


def test_complete_html_table_wins_over_flattened_subtotal_tokens():
    rows = []
    for number in range(20):
        rows.append(
            f"<tr><td>{1000 + number}</td><td>Company {number} Inc.</td>"
            f"<td>{20000 + number}</td></tr>"
        )
        rows.append(f"<tr><td></td><td></td><td>{900000 + number}</td></tr>")
    filing = ("<html><body><table>" + "".join(rows) + "</table></body></html>").encode()

    parsed = sec_index_fund_holdings.parse_filing(filing, "SEC-MORGAN-NDX")

    first = next(row for row in parsed if row["Company"] == "Company 0 Inc")
    assert len(parsed) == 20
    assert first["Shares"] == 1000.0
    assert first["Value"] == 20000.0


def test_reused_symbol_requires_sec_identity_anchor():
    raw = pd.Series(
        [20.0, 21.0],
        index=pd.to_datetime(["2004-03-30", "2004-03-31"]),
    )
    empty = pd.DataFrame(columns=["Date", "Ticker", "ImpliedPrice"])

    accepted, count, _, _ = free_price_history._identity_status(
        "VRTS", raw, empty
    )

    assert accepted is False
    assert count == 0


def test_matching_sec_anchor_validates_reused_symbol():
    raw = pd.Series(
        [20.0, 21.0],
        index=pd.to_datetime(["2004-03-30", "2004-03-31"]),
    )
    anchors = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2004-03-31"]),
            "Ticker": ["VRTS"],
            "ImpliedPrice": [21.0],
        }
    )

    accepted, count, median, _ = free_price_history._identity_status(
        "VRTS", raw, anchors
    )

    assert accepted is True
    assert count == 1
    assert median == 0.0


def test_cmu_split_normalization_removes_two_for_one_jump():
    raw = pd.Series(
        [100.0, 51.0, 52.0],
        index=pd.to_datetime(["2004-01-02", "2004-01-05", "2004-01-06"]),
    )

    normalized, corrections = free_price_history._conservative_split_adjust(raw)

    assert corrections == 1
    assert abs(normalized.iloc[1] / normalized.iloc[0] - 1.02) < 1e-12
