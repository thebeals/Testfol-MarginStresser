"""Build dated, ticker-resolved QQQ holdings snapshots for NDX reconstruction.

The raw annual parser and the structured quarterly NPORT parser intentionally
remain separate inputs.  This module creates the canonical, auditable layer
used by the reconstruction engine:

* 485BPOS schedules use their actual report date rather than filing date.
* NPORT-P XML supplies exact quarterly shares and values from September 2019.
* Names are resolved against the official NDX membership on each report date,
  preventing modern successor tickers from leaking into historical periods.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import pandas as pd

import config
from official_index_data import LOCAL_MEMBERSHIP_FILES
from sec_index_fund_holdings import load_index_fund_positions


OUTPUT_COLUMNS = [
    "Date",
    "FilingDate",
    "FilingID",
    "AccessionNumber",
    "Form",
    "Source",
    "Ticker",
    "Company",
    "Title",
    "CUSIP",
    "LEI",
    "Shares",
    "Value",
    "PctValue",
    "MappingMethod",
    "MappingScore",
    "IsMapped",
    "Confidence",
]

# Historical symbols cannot be recovered by asking a current-symbol service.
# Candidate lists are deliberately date-agnostic; intersection with the dated
# official roster selects the symbol that was valid for that report.
HISTORICAL_NAME_TICKER_CANDIDATES = {
    "ADC TELECOMMUNICATIONS": ("ADCT",),
    "ADAPTEC": ("ADPT",),
    "ADELPHIA COMMUNICATIONS": ("ADLAC",),
    "ADELPHIA COMMUNICATIONS CLASSA": ("ADLAC",),
    "ANDRX GROUP": ("ANDRX",),
    "ARIBA": ("ARBA",),
    "AMYLIN PHARMACEUTICALS": ("AMLN",),
    "AT HOME SERIES A": ("ATHM",),
    "ATI TECHNOLOGIES": ("ATYT",),
    "APPLE COMPUTER": ("AAPL",),
    "BEA SYSTEMS": ("BEAS",),
    "BED BATH AND BEYOND": ("BBBY",),
    "BROADCOM": ("BRCM", "AVGO"),
    "BROADVISION": ("BVSN",),
    "BROCADE COMMUNICATIONS SYSTEM": ("BRCD",),
    "CAREER EDUCATION": ("CECO",),
    "CAMBRIDGE TECHNOLOGY PARTNERS": ("CATP",),
    "DENTSPLY INTERNATIONAL": ("XRAY",),
    "3COM": ("COMS",),
    "CNET": ("CNET",),
    "CNET NETWORKS": ("CNET",),
    "COMCAST SPECIAL CLASSA": ("CMCSK", "CMCSA"),
    "CMGI": ("CMGI",),
    "CORPORATE EXPRESS": ("CEXP",),
    "ECHOSTAR COMMUNICATION": ("DISH",),
    "ECHOSTAR COMMUNICATIONS": ("DISH",),
    "ELECTRONICS FOR IMAGING": ("EFII",),
    "ELECTRONIC ARTS": ("ERTS", "EA"),
    "ERICSSON TELEPHONE": ("ERICY", "ERIC"),
    "ERICSSON ADR": ("ERICY", "ERIC"),
    "ERICSSON SP ADR": ("ERICY", "ERIC"),
    "HANSEN NATURAL": ("HANS", "MNST"),
    "FIRST HEALTH GROUP": ("FHCC",),
    "FLEXTRONICS INTERNATIONAL": ("FLEX",),
    "GEMSTAR INTERNATIONAL GROUP": ("GMST",),
    "GEMSTAR TV GUIDE INT L": ("GMST",),
    "GENZYME GENERAL DIVISION": ("GENZ",),
    "GENZYME GENL DIVISION": ("GENZ",),
    "GENZYME GENERAL": ("GENZ",),
    "IAC INTERACTIVE": ("IACI",),
    "IAC INTERACTIVECORP": ("IACI",),
    "I2 TECHNOLOGIES": ("ITWO",),
    "INTEGRATED DEVICE TECHNOLOGY": ("IDTI",),
    "INTERACTIVECORP": ("IACI",),
    "JUNIPER NETWORKS": ("JNPR",),
    "KMART HOLDING": ("KMRT",),
    "LIBERTY GLOBAL": ("LBTYA", "LBTYK"),
    "LEVEL 3 COMMUNICATIONS": ("LVLT",),
    "LINCARE HOLDINGS": ("LNCR",),
    "LM ERICSSON TELEPHONE": ("ERICY", "ERIC"),
    "LYCOS": ("LCOS",),
    "MCLEODUSA": ("MCLD",),
    "MCLEODUSA CLASSA": ("MCLD",),
    "MERCURY INTERACTIVE": ("MERQE",),
    "MILLICOM INT L CELLULAR": ("MICC",),
    "MILLICOM INTL CELLULAR S A": ("MICC",),
    "MILLENNIUM PHARMACEUTICALS": ("MLNM",),
    "MILLER HERMAN": ("MLHR",),
    "METROMEDIA FIBER NETWORK": ("MFNX",),
    "METROMEDIA FIBER NETWORK CLASSA": ("MFNX",),
    "MONSTER WORLDWIDE": ("TMPW", "MNST", "MWW"),
    "NEXTEL COMMUNICATIONS": ("NXTL",),
    "NEXTEL COMMUNICATIONS CLASSA": ("NXTL",),
    "NEXTLINK COMMUNICATIONS": ("NXLK", "XOXO"),
    "NEXTLINK COMMUNICATIONS CLASSA": ("NXLK", "XOXO"),
    "NETWORK ASSOCIATES": ("NET", "MFE"),
    "NETWORKS ASSOCIATES": ("NET", "MFE"),
    "NOVELLUS SYSTEM": ("NVLS",),
    "NOVELLUS SYSTEMS": ("NVLS",),
    "NVIDIA": ("NVDA",),
    "PATTERSON COMPANIES": ("PDCO",),
    "PALM": ("PALM",),
    "Q LOGIC": ("QLGC",),
    "QUINTILES TRANSNATIONAL": ("QTRN",),
    "REALNETWORKS": ("RNWK",),
    "REUTERS GROUP": ("RTRSY",),
    "REXALL SUNDOWN": ("RXSD",),
    "RF MICRO DEVISES": ("RFMD",),
    "RESEARCH IN MOTION": ("RIMM", "BBRY"),
    "RYANAIR HOLDINGS": ("RYAAY",),
    "SIEBEL SYSTEMS": ("SEBL",),
    "SUN MICROSYSTEMS": ("SUNW", "JAVA"),
    "TMP WORLDWIDE": ("TMPW", "MNST"),
    "TELEFONAKTIEBOLAGET LM ERICSSON": ("ERICY", "ERIC"),
    "TELEFONAKTIEBOLAGET LM ERICSSON SPONSORED ADR": ("ERICY", "ERIC"),
    "TECH DATA": ("TECD",),
    "VISX": ("VISX",),
    "VITESSE SEMICONDUCTOR": ("VTSS",),
    "USA NETWORKS": ("USAI",),
    "WASHINGTON": ("EXPD",),
    "WHOLE FOODS MARKET": ("WFMI",),
    "WORLDCOM WORLDCOM GROUP": ("WCOM",),
    "INKTOMI": ("INKT",),
    "ANDREW": ("ANDW",),
    "XO COMMUNICATIONS CLASSA": ("XOXO",),
}


def _split_pipe(value):
    if not isinstance(value, str) or not value:
        return []
    return [item for item in value.split("|") if item]


def normalize_security_name(value):
    """Normalize issuer names while retaining share-class information."""
    value = re.sub(r"\.{2,}.*$", "", str(value or "")).strip()
    value = re.sub(r"\*|\([a-z]\)", "", value, flags=re.IGNORECASE)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.upper().replace("&", " AND ")
    value = re.sub(r"\b(?:CLASS|CL|CM)\s+([ABC])\b", r" CLASS\1 ", value)
    value = re.sub(
        r"\b(?:INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|"
        r"PLC|N\.V|NV|S\.A|SA|SE|ORDINARY|ORD|COMMON|CMN|STOCK|SHARES?)\b",
        " ",
        value,
    )
    return re.sub(r"[^A-Z0-9]+", " ", value).strip()


def _name_score(left, right):
    left = normalize_security_name(left)
    right = normalize_security_name(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left.startswith(right) or right.startswith(left):
        return 0.94
    sequence = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    token_score = len(left_tokens & right_tokens) / max(
        1, len(left_tokens | right_tokens)
    )
    return max(sequence, token_score)


def load_official_membership(path=None):
    path = path or LOCAL_MEMBERSHIP_FILES["NDX"]
    frame = pd.read_csv(path, parse_dates=["Date"])
    return frame.sort_values("Date").reset_index(drop=True)


def official_name_map(membership, report_date):
    report_date = pd.Timestamp(report_date).normalize()
    eligible = membership[membership["Date"] <= report_date]
    if eligible.empty:
        return {}
    else:
        row = eligible.iloc[-1]
    tickers = _split_pipe(row.get("Tickers", ""))
    names = _split_pipe(row.get("Names", ""))
    return {
        ticker: names[index] if index < len(names) and names[index] else ticker
        for index, ticker in enumerate(tickers)
    }


def _mapping_index(mapping):
    normalized = {}
    for name, ticker in mapping.items():
        normalized.setdefault(normalize_security_name(name), set()).add(ticker)
    return normalized


def match_positions_to_official(positions, report_date, mapping, membership):
    """Resolve holdings to the dated official roster with one-to-one matching."""
    rows = [dict(position) for position in positions]
    official = official_name_map(membership, report_date)
    normalized_mapping = _mapping_index(mapping)
    used = set()

    for row in rows:
        ticker = str(row.get("Ticker") or "").strip()
        candidates = []
        for name in (row.get("Company"), row.get("Title")):
            if not name:
                continue
            normalized_name = normalize_security_name(name)
            candidates.extend(
                HISTORICAL_NAME_TICKER_CANDIDATES.get(normalized_name, ())
            )
            if name in mapping:
                candidates.append(mapping[name])
            normalized = normalized_mapping.get(normalized_name, set())
            # A normalized issuer can legitimately have several symbols over
            # time (ERTS/EA, RIMM/BBRY, BRCM/AVGO). The dated official roster
            # disambiguates those aliases safely.
            candidates.extend(sorted(normalized))
        if ticker:
            # Parser-provided symbols can come from current SEC registrant
            # lookups, where an old symbol has since been recycled. Prefer
            # issuer-name historical aliases and use the raw symbol only as a
            # fallback (for example AAPL over today's unrelated POAI).
            candidates.append(ticker)

        chosen = next(
            (
                candidate
                for candidate in candidates
                if candidate not in used and (not official or candidate in official)
            ),
            None,
        )
        if chosen:
            row["Ticker"] = chosen
            row["MappingMethod"] = "dated-direct"
            row["MappingScore"] = 1.0
            row["IsMapped"] = True
            used.add(chosen)
        else:
            row["Ticker"] = ""
            row["MappingMethod"] = "unmapped"
            row["MappingScore"] = 0.0
            row["IsMapped"] = False

    if official:
        pairs = []
        for row_index, row in enumerate(rows):
            if row["IsMapped"]:
                continue
            security_names = [row.get("Title", ""), row.get("Company", "")]
            for ticker, official_name in official.items():
                if ticker in used:
                    continue
                score = max(
                    _name_score(security_name, official_name)
                    for security_name in security_names
                )
                pairs.append((score, row_index, ticker))

        # Global greedy assignment avoids two similar share classes claiming
        # the same official ticker. Direct matches have already locked the easy
        # 85-95% of each roster.
        assigned_rows = set()
        for score, row_index, ticker in sorted(pairs, reverse=True):
            if score < 0.55 or row_index in assigned_rows or ticker in used:
                continue
            row = rows[row_index]
            row["Ticker"] = ticker
            row["MappingMethod"] = "dated-fuzzy"
            row["MappingScore"] = round(float(score), 6)
            row["IsMapped"] = True
            assigned_rows.add(row_index)
            used.add(ticker)

    return rows, official


def _local_name(tag):
    return tag.split("}")[-1]


def _first_text(root, name, default=""):
    return next(
        (element.text for element in root.iter() if _local_name(element.tag) == name),
        default,
    )


def parse_nport_file(path):
    root = ET.parse(path).getroot()
    report_date = _first_text(root, "repPdDate")
    if not report_date:
        raise ValueError(f"NPORT filing has no repPdDate: {path}")

    filename = os.path.basename(path)
    parts = filename.split("_")
    filing_date = parts[1] if len(parts) > 1 else ""
    accession = parts[2] if len(parts) > 2 else ""
    rows = []
    for investment in (
        element for element in root.iter() if _local_name(element.tag) == "invstOrSec"
    ):
        values = {_local_name(element.tag): element.text for element in investment.iter()}
        try:
            shares = float(values.get("balance") or 0)
            value = float(values.get("valUSD") or 0)
        except (TypeError, ValueError):
            continue
        # QQQ's NPORT equity schedule is entirely common shares. Exclude index
        # futures and other overlays that are not NDX constituents.
        if (
            values.get("assetCat") != "EC"
            or values.get("units") != "NS"
            or shares <= 0
            or value <= 0
        ):
            continue
        rows.append(
            {
                "Date": report_date,
                "FilingDate": filing_date,
                "FilingID": filename,
                "AccessionNumber": accession,
                "Form": "NPORT-P",
                "Source": "SEC-NPORT",
                "Ticker": "",
                "Company": values.get("name") or values.get("title") or "",
                "Title": values.get("title") or values.get("name") or "",
                "CUSIP": values.get("cusip") or "",
                "LEI": values.get("lei") or "",
                "Shares": shares,
                "Value": value,
                "PctValue": float(values.get("pctVal") or 0) / 100.0,
            }
        )
    return rows


def _annual_positions(frame):
    rows = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "Date": record["Date"],
                "FilingDate": record.get("FilingDate", record["Date"]),
                "FilingID": record.get("FilingID", ""),
                "AccessionNumber": "",
                "Form": record.get("Form", "485BPOS"),
                "Source": "SEC-485BPOS",
                "Ticker": record.get("Ticker", ""),
                "Company": record.get("Company", ""),
                "Title": record.get("Company", ""),
                "CUSIP": "",
                "LEI": "",
                "Shares": record.get("Shares"),
                "Value": record.get("Value"),
                "PctValue": 0.0,
            }
        )
    return rows


def build_snapshots():
    with open(
        os.path.join(config.ASSETS_DIR, "name_mapping.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        mapping = json.load(handle)
    membership = load_official_membership()

    annual = pd.read_csv(config.COMPONENTS_FILE)
    annual["Date"] = pd.to_datetime(annual["Date"]).dt.normalize()
    annual["Value"] = pd.to_numeric(
        annual["Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    annual["Shares"] = pd.to_numeric(
        annual["Shares"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    snapshots = []
    observed_dates = set()
    for report_date, group in annual.groupby("Date", sort=True):
        # Amendments and duplicate prospectuses can repeat the same schedule.
        # Select the most complete single filing rather than merging fund
        # shares from two copies of one report date.
        if "FilingID" in group and group["FilingID"].nunique() > 1:
            filing_counts = group.groupby("FilingID").size()
            selected_filing = filing_counts.sort_values(ascending=False).index[0]
            group = group[group["FilingID"] == selected_filing]
        matched, official = match_positions_to_official(
            _annual_positions(group), report_date, mapping, membership
        )
        # Once dated official membership exists, unmatched filing artifacts or
        # non-index positions must not enter the parent index.
        if official:
            matched = [row for row in matched if row["IsMapped"]]
        snapshots.extend(matched)
        observed_dates.add(pd.Timestamp(report_date).normalize())

    # Independent NDX-tracking funds fill the long gaps between QQQ's annual
    # schedules in 2000-2009.  Never blend two funds on the same date: their
    # share scales differ even though their normalized weights agree.
    index_fund_positions = load_index_fund_positions()
    candidates_by_date = {}
    for (report_date, source), group in index_fund_positions.groupby(
        ["Date", "Source"], sort=True
    ):
        report_date = pd.Timestamp(report_date).normalize()
        if report_date in observed_dates:
            continue
        matched, official = match_positions_to_official(
            group.to_dict("records"), report_date, mapping, membership
        )
        matched = [row for row in matched if row["IsMapped"]]
        mapped_tickers = {row["Ticker"] for row in matched}
        score = (
            len(mapped_tickers & set(official)) if official else len(mapped_tickers),
            source == "SEC-MORGAN-NDX",
        )
        current = candidates_by_date.get(report_date)
        if current is None or score > current[0]:
            candidates_by_date[report_date] = (score, matched)
    for report_date, (_, matched) in sorted(candidates_by_date.items()):
        if matched:
            snapshots.extend(matched)
            observed_dates.add(report_date)

    nport_dates = set()
    if os.path.isdir(config.NPORT_CACHE_DIR):
        for filename in sorted(os.listdir(config.NPORT_CACHE_DIR)):
            if not filename.lower().endswith(".xml"):
                continue
            path = os.path.join(config.NPORT_CACHE_DIR, filename)
            positions = parse_nport_file(path)
            if not positions:
                continue
            report_date = positions[0]["Date"]
            matched, official = match_positions_to_official(
                positions, report_date, mapping, membership
            )
            matched = [row for row in matched if row["IsMapped"]]
            mapped_tickers = {row["Ticker"] for row in matched}
            missing = sorted(set(official) - mapped_tickers)
            if missing:
                raise ValueError(
                    f"{report_date} NPORT mapping missing official members: "
                    + ", ".join(missing)
                )
            nport_dates.add(pd.Timestamp(report_date).normalize())
            snapshots.extend(matched)

    result = pd.DataFrame(snapshots)
    if result.empty:
        raise ValueError("No SEC holdings snapshots were parsed")
    result["Date"] = pd.to_datetime(result["Date"]).dt.normalize()
    result["FilingDate"] = pd.to_datetime(
        result["FilingDate"], errors="coerce"
    ).dt.normalize()
    result["Value"] = pd.to_numeric(
        result["Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    result["Shares"] = pd.to_numeric(
        result["Shares"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    result = result[(result["Value"] > 0) & (result["Shares"] > 0)].copy()

    # Structured NPORT replaces the same-date prospectus schedule.
    if nport_dates:
        duplicate_annual = result["Date"].isin(nport_dates) & (
            result["Form"] != "NPORT-P"
        )
        result = result[~duplicate_annual].copy()

    identity = result["Ticker"].where(
        result["Ticker"].astype(str).str.len() > 0,
        "UNMAPPED:" + result["Company"].map(normalize_security_name),
    )
    result["_Identity"] = identity
    result = result.sort_values(
        ["Date", "Form", "Value"], ascending=[True, False, False]
    ).drop_duplicates(["Date", "_Identity"], keep="first")
    totals = result.groupby("Date")["Value"].transform("sum")
    result["PctValue"] = result["Value"] / totals

    manifest_rows = []
    for report_date, group in result.groupby("Date", sort=True):
        official = official_name_map(membership, report_date)
        mapped = set(group.loc[group["IsMapped"].astype(bool), "Ticker"])
        missing = sorted(set(official) - mapped)
        form = str(group.iloc[0]["Form"])
        source = str(group.iloc[0]["Source"])
        coverage = len(mapped & set(official)) / len(official) if official else 0.0
        if form == "NPORT-P" and official and not missing:
            confidence = "A-observed-quarterly"
        elif source in {"SEC-RYDEX-NDX", "SEC-MORGAN-NDX"} and (
            (official and coverage >= 0.90)
            or (not official and len(mapped) >= 75)
        ):
            confidence = "A-observed-index-fund"
        elif official and coverage >= 0.95:
            confidence = "B-observed-annual"
        else:
            confidence = "C-partial-history"
        result.loc[group.index, "Confidence"] = confidence
        manifest_rows.append(
            {
                "Date": report_date.date().isoformat(),
                "Form": form,
                "Source": source,
                "Positions": len(group),
                "MappedPositions": int(group["IsMapped"].astype(bool).sum()),
                "OfficialCount": len(official),
                "OfficialCoverage": coverage,
                "MissingOfficialCount": len(missing),
                "MissingOfficial": "|".join(missing),
                "ValueTotal": float(group["Value"].sum()),
                "WeightSum": float(group["PctValue"].sum()),
                "Confidence": confidence,
            }
        )

    result = result.drop(columns=["_Identity"])
    for column in OUTPUT_COLUMNS:
        if column not in result:
            result[column] = ""
    result = result[OUTPUT_COLUMNS].sort_values(
        ["Date", "Value"], ascending=[True, False]
    )
    result.to_csv(config.HOLDINGS_SNAPSHOTS_FILE, index=False)
    pd.DataFrame(manifest_rows).to_csv(config.HOLDINGS_MANIFEST_FILE, index=False)
    print(
        f"Saved {len(result):,} holdings across {result['Date'].nunique()} "
        f"report dates to {config.HOLDINGS_SNAPSHOTS_FILE}"
    )
    return result


if __name__ == "__main__":
    build_snapshots()
