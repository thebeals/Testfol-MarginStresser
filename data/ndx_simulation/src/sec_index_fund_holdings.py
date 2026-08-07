"""Download and parse free SEC filings from funds that tracked the NDX.

QQQ's annual schedules are useful anchors, but they leave long gaps in the
dot-com era.  Rydex's OTC Fund and the Morgan Stanley Nasdaq-100 Index Fund
filed semiannual or quarterly schedules with the SEC.  Their stock weights
closely track the index, so these filings provide independent observed anchors
without relying on a modern-symbol price service.

Raw filings remain in the local cache.  The checked/generated CSV contains only
the selected fund positions plus filing provenance.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import time
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config


SEC_HEADERS = {
    "User-Agent": os.environ.get(
        "SEC_USER_AGENT",
        "Testfol NDX reconstruction research contact@example.com",
    ),
    "Accept-Encoding": "gzip, deflate",
}
ALLOWED_FORMS = {"N-30D", "N-CSR", "N-CSRS", "N-Q"}
START_DATE = pd.Timestamp("2000-01-01")
END_DATE = pd.Timestamp("2009-12-31")

FUND_SOURCES = {
    "SEC-RYDEX-NDX": {
        "cik": "0000899148",
        "fund_names": ("OTC FUND", "NASDAQ-100 FUND", "NASDAQ 100 FUND"),
    },
    "SEC-MORGAN-NDX": {
        "cik": "0001137676",
        "fund_names": ("MORGAN STANLEY NASDAQ-100 INDEX FUND",),
    },
}

POSITION_COLUMNS = [
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
]

_NAME_FIRST = re.compile(
    r"^(?P<name>.+?)\.{2,}\s+(?P<shares>[\d,]+)\s+\$?\s*"
    r"(?P<value>[\d,]+)(?:\s|$)"
)
_SHARES_FIRST = re.compile(
    r"^(?P<shares>[\d,]+)\s+(?P<name>.+?)\.{2,}\s+\$?\s*"
    r"(?P<value>[\d,]+)(?:\s|$)"
)
_NAME_COLUMNS = re.compile(
    r"^(?P<name>.*[A-Za-z].*?)\s+(?P<shares>[\d,]+)\s+\$?\s*"
    r"(?P<value>[\d,]+)(?:\s|$)"
)
_SHARES_COLUMNS = re.compile(
    r"^(?P<shares>[\d,]+)\s+(?P<name>.*[A-Za-z].*?)\s+\$?\s*"
    r"(?P<value>[\d,]+)(?:\s|$)"
)


def _request(url, *, as_json=False, attempts=3):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=SEC_HEADERS, timeout=60)
            response.raise_for_status()
            return response.json() if as_json else response.content
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch SEC resource {url}: {last_error}")


def _filing_rows(payload):
    rows = []
    for index, form in enumerate(payload.get("form", [])):
        if form not in ALLOWED_FORMS:
            continue
        report_date = pd.to_datetime(
            payload.get("reportDate", [""])[index], errors="coerce"
        )
        if pd.isna(report_date) or not START_DATE <= report_date <= END_DATE:
            continue
        rows.append(
            {
                "Date": report_date.normalize(),
                "FilingDate": pd.to_datetime(
                    payload.get("filingDate", [""])[index], errors="coerce"
                ),
                "Form": form,
                "AccessionNumber": payload["accessionNumber"][index],
                "FilingID": payload["primaryDocument"][index],
            }
        )
    return rows


def discover_filings():
    """Return all candidate NDX index-fund filings from SEC submissions JSON."""
    filings = []
    for source, spec in FUND_SOURCES.items():
        cik = spec["cik"]
        root_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        root = _request(root_url, as_json=True)
        payloads = [root.get("filings", {}).get("recent", {})]
        for supplemental in root.get("filings", {}).get("files", []):
            name = supplemental.get("name")
            if not name:
                continue
            payloads.append(
                _request(
                    f"https://data.sec.gov/submissions/{name}", as_json=True
                )
            )
        for payload in payloads:
            for row in _filing_rows(payload):
                accession_path = row["AccessionNumber"].replace("-", "")
                row.update(
                    {
                        "Source": source,
                        "CIK": cik,
                        "URL": (
                            "https://www.sec.gov/Archives/edgar/data/"
                            f"{int(cik)}/{accession_path}/{row['FilingID']}"
                        ),
                    }
                )
                filings.append(row)
    return pd.DataFrame(filings).sort_values(
        ["Date", "Source", "FilingDate", "AccessionNumber"]
    )


def _visible_lines(raw):
    text = BeautifulSoup(raw, "lxml").get_text("\n")
    lines = []
    for value in text.splitlines():
        value = html.unescape(value).replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value).strip()
        if value:
            lines.append(value)
    return lines


def _fund_schedule_lines(lines, source):
    if source != "SEC-RYDEX-NDX":
        return lines

    fund_names = FUND_SOURCES[source]["fund_names"]
    schedule_indexes = []
    for index, value in enumerate(lines):
        heading = " ".join(lines[index : index + 4]).upper()
        if value.upper().startswith("SCHEDULE OF INVESTMENTS") or (
            value.upper() == "SCHEDULE" and "SCHEDULE OF INVESTMENTS" in heading
        ):
            schedule_indexes.append(index)
    chunks = []
    for index, value in enumerate(lines):
        if value.upper() not in fund_names:
            continue
        nearby_schedules = [
            candidate
            for candidate in schedule_indexes
            if abs(candidate - index) <= 8
        ]
        if not nearby_schedules:
            continue
        schedule_index = min(
            nearby_schedules, key=lambda candidate: abs(candidate - index)
        )
        start = max(index, schedule_index) + 1
        end = next(
            (candidate for candidate in schedule_indexes if candidate >= start),
            min(len(lines), start + 600),
        )
        chunks.extend(lines[start:end])
    return chunks


def _clean_company(value):
    value = re.sub(r"\s+", " ", str(value)).strip(" -.$")
    value = re.sub(r"[*#\x86\u2020\u2021]+$", "", value).strip(" -.$")
    return value


def _number(value):
    value = str(value).strip().replace("$", "").replace(",", "")
    value = value.strip("() ")
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    return float(value)


def _position(name, shares, value):
    name = _clean_company(name)
    shares = _number(shares)
    value = _number(value)
    if not name or not re.search(r"[A-Za-z]", name):
        return None
    if shares is None or value is None or shares <= 0 or value <= 0:
        return None
    implied_price = value / shares
    if not 0.001 <= implied_price <= 100000:
        return None
    return {"Company": name, "Title": name, "Shares": shares, "Value": value}


def _parse_fixed_width(lines):
    rows = []
    prefix_parts = []
    for line in lines:
        match = (
            _NAME_FIRST.match(line)
            or _SHARES_FIRST.match(line)
            or _NAME_COLUMNS.match(line)
            or _SHARES_COLUMNS.match(line)
        )
        if match:
            name = match.group("name")
            if prefix_parts:
                name = " ".join(prefix_parts[-3:] + [name])
            row = _position(name, match.group("shares"), match.group("value"))
            if row:
                rows.append(row)
            prefix_parts = []
        elif (
            re.search(r"[A-Za-z]", line)
            and not re.search(r"\d", line)
            and len(line) < 80
            and not re.search(
                r"SCHEDULE|COMMON STOCK|MARKET|SHARES|VALUE|FUND$",
                line,
                re.IGNORECASE,
            )
        ):
            prefix_parts.append(line)
        else:
            prefix_parts = []
    return rows


def _parse_token_rows(lines):
    """Parse HTML filings whose table cells became one visible line per cell."""
    rows = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not re.search(r"[A-Za-z]", line) or "%" in line:
            index += 1
            continue
        if re.search(
            r"SCHEDULE|INVESTMENTS|COMMON STOCK|MARKET|SHARES|VALUE|NOTE \d",
            line,
            re.IGNORECASE,
        ):
            index += 1
            continue
        name_parts = []
        consumed_to = None
        for candidate in range(index, min(index + 4, len(lines))):
            shares = _number(lines[candidate])
            if shares is None:
                if not re.search(r"[A-Za-z]", lines[candidate]):
                    break
                name_parts.append(lines[candidate])
                continue
            value_index = candidate + 1
            while value_index < len(lines) and lines[value_index] in {"$", "—"}:
                value_index += 1
            if value_index >= len(lines):
                break
            value = _number(lines[value_index])
            if value is None:
                break
            row = _position(" ".join(name_parts), shares, value)
            if row:
                rows.append(row)
                consumed_to = value_index
            break
        index = consumed_to + 1 if consumed_to is not None else index + 1
    return rows


def _parse_html_tables(raw, source):
    rows = []
    soup = BeautifulSoup(raw, "lxml")
    for table in soup.find_all("table"):
        table_text = re.sub(r"\s+", " ", table.get_text(" ")).upper()
        if source == "SEC-RYDEX-NDX" and not any(
            name in table_text for name in FUND_SOURCES[source]["fund_names"]
        ):
            # Rydex reports contain many unrelated index and sector funds.
            continue
        for tr in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text(" ")).strip()
                for cell in tr.find_all(["td", "th"])
            ]
            cells = [cell for cell in cells if cell and cell != "$" and cell != "—"]
            numeric = [(index, _number(cell)) for index, cell in enumerate(cells)]
            numeric = [(index, value) for index, value in numeric if value is not None]
            if len(numeric) < 2:
                continue
            first_index, shares = numeric[0]
            last_index, value = numeric[-1]
            if first_index >= last_index:
                continue
            name = " ".join(
                cell
                for cell in cells[first_index + 1 : last_index]
                if re.search(r"[A-Za-z]", cell)
            )
            row = _position(name, shares, value)
            if row:
                rows.append(row)
    return rows


def parse_filing(raw, source):
    """Parse stock-like rows from one cached SEC filing."""
    lines = _fund_schedule_lines(_visible_lines(raw), source)
    if source == "SEC-RYDEX-NDX" and not lines:
        return []
    html_rows = _parse_html_tables(raw, source)
    if len(html_rows) >= 20:
        # Modern EDGAR HTML keeps shares, issuer and value in one <tr>.  Its
        # flattened visible text also exposes the following industry subtotal,
        # which made the token parser pair (market value, subtotal) as
        # (shares, value) and duplicate the issuer at a wildly wrong weight.
        # A complete table parse is structurally stronger, so do not blend the
        # lossy token stream into it.
        rows = html_rows
    else:
        rows = _parse_fixed_width(lines)
        rows.extend(_parse_token_rows(lines))
        rows.extend(html_rows)
    deduped = {}
    for row in rows:
        key = (
            re.sub(r"[^A-Z0-9]", "", row["Company"].upper()),
            row["Shares"],
            row["Value"],
        )
        deduped[key] = row
    return list(deduped.values())


def _cache_path(filing):
    safe_accession = filing["AccessionNumber"].replace("-", "")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filing["FilingID"])
    return os.path.join(
        config.INDEX_FUND_CACHE_DIR,
        filing["Source"].replace("SEC-", "").lower(),
        f"{filing['Date'].date()}_{safe_accession}_{safe_name}",
    )


def _load_raw_filing(filing, refresh=False):
    path = _cache_path(filing)
    if refresh or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw = _request(filing["URL"])
        with open(path, "wb") as handle:
            handle.write(raw)
        time.sleep(0.12)
    else:
        with open(path, "rb") as handle:
            raw = handle.read()
    return raw, path


def build_index_fund_positions(refresh=False):
    """Build the canonical parsed index-fund position cache."""
    filings = discover_filings()
    selected_rows = []
    manifest = []

    grouped = filings.groupby(["Source", "Date"], sort=True)
    for (source, report_date), candidates in grouped:
        parsed_candidates = []
        for filing in candidates.to_dict("records"):
            raw, cache_path = _load_raw_filing(filing, refresh=refresh)
            positions = parse_filing(raw, source)
            parsed_candidates.append((len(positions), filing, positions, cache_path))
        count, filing, positions, cache_path = max(
            parsed_candidates, key=lambda item: item[0]
        )
        if count < 20:
            manifest.append(
                {
                    "Date": report_date.date().isoformat(),
                    "Source": source,
                    "Form": filing["Form"],
                    "Positions": count,
                    "Status": "rejected-too-few-positions",
                    "AccessionNumber": filing["AccessionNumber"],
                    "FilingID": filing["FilingID"],
                    "URL": filing["URL"],
                    "CachePath": cache_path,
                }
            )
            continue
        for position in positions:
            position.update(
                {
                    "Date": report_date,
                    "FilingDate": filing["FilingDate"],
                    "FilingID": filing["FilingID"],
                    "AccessionNumber": filing["AccessionNumber"],
                    "Form": filing["Form"],
                    "Source": source,
                    "Ticker": "",
                    "CUSIP": "",
                    "LEI": "",
                    "PctValue": 0.0,
                }
            )
            selected_rows.append(position)
        manifest.append(
            {
                "Date": report_date.date().isoformat(),
                "Source": source,
                "Form": filing["Form"],
                "Positions": count,
                "Status": "selected",
                "AccessionNumber": filing["AccessionNumber"],
                "FilingID": filing["FilingID"],
                "URL": filing["URL"],
                "CachePath": cache_path,
            }
        )

    result = pd.DataFrame(selected_rows)
    if result.empty:
        raise ValueError("No SEC NDX index-fund positions were parsed")
    for column in POSITION_COLUMNS:
        if column not in result:
            result[column] = ""
    result = result[POSITION_COLUMNS].sort_values(
        ["Date", "Source", "Value"], ascending=[True, True, False]
    )
    result.to_csv(config.INDEX_FUND_POSITIONS_FILE, index=False)
    pd.DataFrame(manifest).to_csv(config.INDEX_FUND_MANIFEST_FILE, index=False)
    print(
        f"Saved {len(result):,} positions from {result['Date'].nunique()} "
        f"report dates to {config.INDEX_FUND_POSITIONS_FILE}"
    )
    return result


def load_index_fund_positions(refresh=False):
    if not refresh and os.path.exists(config.INDEX_FUND_POSITIONS_FILE):
        frame = pd.read_csv(
            config.INDEX_FUND_POSITIONS_FILE,
            parse_dates=["Date", "FilingDate"],
        )
        if not frame.empty:
            return frame
    return build_index_fund_positions(refresh=refresh)


def main():
    parser = argparse.ArgumentParser(
        description="Build free historical NDX index-fund holdings from SEC filings."
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build_index_fund_positions(refresh=args.refresh)


if __name__ == "__main__":
    main()
