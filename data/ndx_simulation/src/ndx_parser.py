import os
import re
import csv
import datetime
import html as html_lib
import logging
import sys
from html.parser import HTMLParser

import config

# Configuration
DOWNLOAD_DIR = config.DOWNLOAD_DIR
OUTPUT_FILE = config.COMPONENTS_FILE

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class InvestmentHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row_data = []
        self.current_cell_text = ""
        self.rows = [] 
        self.table_stack = [] # Stack of tables, each is a list of rows

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'table':
            self.in_table = True
            self.table_stack.append([]) # Start a new table context
        elif tag.lower() == 'tr':
            if self.in_table:
                self.in_row = True
                self.current_row_data = []
        elif tag.lower() == 'td':
            if self.in_row:
                self.in_cell = True
                self.current_cell_text = ""

    def handle_endtag(self, tag):
        if tag.lower() == 'table':
            if self.table_stack:
                finished_table_rows = self.table_stack.pop()
                # Analyze the table we just finished
                if self._is_investment_table(finished_table_rows):
                    self.rows.extend(finished_table_rows)
            
            if not self.table_stack:
                self.in_table = False
                
        elif tag.lower() == 'tr':
            if self.in_row:
                self.in_row = False
                if self.current_row_data:
                    # Add to current table (top of stack)
                    if self.table_stack:
                        self.table_stack[-1].append(self.current_row_data)
        elif tag.lower() == 'td':
            if self.in_cell:
                self.in_cell = False
                clean_text = self.current_cell_text.strip()
                # Remove common garbage
                clean_text = clean_text.replace('\xa0', ' ').replace('&nbsp;', ' ')
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                self.current_row_data.append(clean_text)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell_text += data + " "
            
    def _is_investment_table(self, rows):
        """Checks if a parsed table looks like a Schedule of Investments."""
        if not rows:
            return False
            
        # Check first few rows for Headers
        header_found = False
        global DEBUG_FILE
        if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
             print(f"--- Checking Table with {len(rows)} rows ---")
             
        for i in range(min(5, len(rows))):
            row_text = " ".join(rows[i]).lower()
            if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
                 print(f"DEBUG ROW {i}: {row_text}")
            if "value" in row_text and ("shares" in row_text or "principal" in row_text or "security" in row_text):
                 header_found = True
                 if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
                     print("!!! MATCH FOUND !!!")
                 break
        
        if not header_found:
             # DEBUG: Print failures for 2017 file specifically if needed
             # if "2017" in current_file: ...
             return False
            
        # Optional: Check if it has data rows?
        return True

def parse_html_file(filepath):
    """Parses an HTML filing."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    parser = InvestmentHTMLParser()
    parser.feed(content)
    
    # Process any remaining tables in the stack (handle unclosed tables)
    while parser.table_stack:
        finished_table_rows = parser.table_stack.pop()
        global DEBUG_FILE
        if hasattr(sys.modules[__name__], 'DEBUG_FILE') and DEBUG_FILE:
             print(f"--- Flushing/Closing Orphaned Table with {len(finished_table_rows)} rows ---")
        if parser._is_investment_table(finished_table_rows):
            parser.rows.extend(finished_table_rows)
            
    # Post-process extracted rows
    holdings = []
    
    # We expect rows with Name, Shares, Value
    for row in parser.rows:
        # cleanup row
        cleaned_row = [c for c in row if c and c not in ['$']]
        
        if len(cleaned_row) < 3:
            continue
            
        # Identify "Value" column and "Shares" column
        # Usually Value is last, Shares is second last.
        # But sometimes there are empty columns.
        
        try:
            # Filter out empty strings
            valid_cols = [c for c in cleaned_row if c.strip()]
            
            if len(valid_cols) < 3:
                continue
                
            val_str = valid_cols[-1].replace(',', '').replace('$', '').replace(')', '').replace('(', '')
            shares_str = valid_cols[-2].replace(',', '').replace('$', '')
            
            # Check if they are numbers
            if not (val_str.isdigit() and shares_str.isdigit()):
                continue
                
            name = " ".join(valid_cols[:-2]).strip()
            # Remove * or similar from name
            name = name.rstrip('*').strip()
            
            # Artifact blocklist: filter SEC filing non-holding rows
            ARTIFACT_NAMES = {
                'total', 'net assets', 'net realized gain', 'net change in unrealized',
                'net unrealized', 'interest', 'dividends', 'capital gains',
                'september', 'october', 'november', 'december',
                'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                'cash', 'amount due', 'investments purchased', 'marketing expenses',
                'ordinary income', 'other expenses', 'professional fees', 'shares sold',
                'trustee fees', 'undistributed net', 'other assets less liabilities',
                'net investment income', 'amount due to',
            }
            name_lower = name.lower().strip()
            if any(name_lower.startswith(a) for a in ARTIFACT_NAMES):
                continue
            if len(name) < 5 or not name[0].isalpha():
                continue

            holdings.append((name, shares_str, val_str))
        except Exception as e:
            # logging.debug(f"Skipping row: {cleaned_row} - Error: {e}")
            continue
            
    return holdings

def parse_html_as_text(filepath):
    """Fallback: Strip HTML and parse as text."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Strip tags
    text_content = re.sub(r'<[^>]+>', '   ', content)
    
    # Clean HTML entities
    text_content = text_content.replace('&nbsp;', ' ').replace('&#151;', ' ').replace('&amp;', '&')
    
    # Use array of lines
    lines = text_content.split('\n')
    
    # Use array of lines
    lines = text_content.split('\n')
    
    holdings = []
    
    # Reuse row pattern from text parser: Name ... Number ... Number
    # But HTML stripped might have extra spaces.
    # Regex patterns for fallback (single line)
    # 1. Name ... Shares ... Value (Standard Text)
    # 2. Shares ... Name ... Value (HTML 2017 style, if on same line)
    
    patterns = [
        (re.compile(r'^\s*([A-Za-z].*?)\s+([0-9,]+)\s+(?:\$\s*)?([0-9,]+)\s*$'), "NSV"),
        (re.compile(r'^\s*([0-9,]+)\s+([A-Za-z].*?)\s+(?:\$\s*)?([0-9,]+)\s*$'), "SNV")
    ]
    
    # Try Single Line Matching First
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Heuristics to skip junk
        if len(line) < 20: continue
        if "TOTAL" in line.upper(): continue
        
        match_found_single = False
        for regex, order in patterns:
            match = regex.match(line)
            if match:
                g1, g2, g3 = match.groups()
                if order == "NSV":
                    n, s, v = g1, g2, g3
                else: 
                    s, n, v = g1, g2, g3
                
                # Validation
                v_clean = v.replace(',', '')
                if len(v_clean) >= 6:
                     # Clean Name
                     n = n.strip()
                     n = re.sub(r'\*|\([a-z]\)', '', n).strip()
                     holdings.append((n, s, v))
                     match_found_single = True
                break
    
    # If single line matching worked well, return
    if len(holdings) > 10:
        return holdings
        
    logging.info("    Single-line text parse failed. Trying Multi-line Sequence Parser...")
    
    # Multi-Line Sequence Parser (Shares -> Name -> Value)
    # Trigger: Find valid Share count. Then look ahead for Name, then Value.
    
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if not l:
            i += 1
            continue
            
        # Check if line is a potential Share count (Number, e.g. 3,713,874)
        # Must be digits and commas, maybe dot.
        # Must be at least a reasonable size number?
        s_clean = l.replace(',', '').replace('.', '')
        if s_clean.isdigit() and len(s_clean) > 3: # > 1000 shares
             # Potential Shares found.
             shares = l
             
             # Look ahead for Name (within next 5 lines)
             name = None
             value = None
             j = i + 1
             name_idx = -1
             
             while j < min(len(lines), i + 6):
                 nl = lines[j].strip()
                 if nl and not nl.replace(',','').replace('.','').replace('$','').isdigit():
                      # Found text line. Is it a name?
                      if len(nl) > 3 and "TOTAL" not in nl.upper():
                          name = nl
                          name_idx = j
                          break
                 j += 1
                 
             if name:
                 # Look ahead for Value (within next 5 lines after Name)
                 k = name_idx + 1
                 while k < min(len(lines), name_idx + 6):
                     vl = lines[k].strip()
                     # Value might be "$" line, then Number line.
                     if vl == "$":
                         k += 1
                         continue
                         
                     vl_clean = vl.replace(',','').replace('$','')
                     if vl_clean.isdigit() and len(vl_clean) > 5: # Value > $100k
                         value = vl_clean
                         # Found complete set!
                         
                         name = name.strip()
                         name = re.sub(r'\*|\([a-z]\)', '', name).strip()
                         if len(name) > 2:
                             holdings.append((name, shares, value))
                         
                         # Advance main loop to k
                         i = k
                         break
                     k += 1
        i += 1
            
    return holdings


_MONTH_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)
_DATE_PATTERN = _MONTH_PATTERN + r"\s+\d{1,2},\s+\d{4}"


def extract_report_date(filepath):
    """Return the holdings as-of date embedded in a 485BPOS filing.

    The filename date is the SEC filing date, commonly four months after the
    actual schedule.  Using it as the valuation date materially distorts every
    projected weight.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()

    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", content))
    text = re.sub(r"\s+", " ", text).replace("\xa0", " ")
    patterns = [
        rf"Schedule of Investments.{{0,500}}?({_DATE_PATTERN})",
        rf"Statement of Investments.{{0,500}}?({_DATE_PATTERN})",
        rf"ESSENTIAL INFORMATION AS OF\s+({_DATE_PATTERN})",
        rf"as of\s+({_DATE_PATTERN})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return datetime.datetime.strptime(
                match.group(1).title(), "%B %d, %Y"
            ).date()
        except ValueError:
            continue
    return None


def parse_repeated_schedule_pages(filepath):
    """Parse every repeated page of old text-format holdings schedules.

    Early filings repeat the ``Schedule of Investments`` heading on each page.
    The old parser stopped at the first page-level total and retained only
    30-36 securities.  Actual schedule pages can be identified by their Shares
    and Value headers and safely concatenated through ``Total Investments``.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()

    text = html_lib.unescape(re.sub(r"<[^>]+>", "   ", content))
    text = text.replace("\xa0", " ")
    end_matches = list(re.finditer(r"Total\s+Investments", text, re.IGNORECASE))
    if not end_matches:
        return []
    schedule_end = end_matches[-1].end()
    starts = [
        match.start()
        for match in re.finditer(r"Schedule of Investments", text, re.IGNORECASE)
        if match.start() < schedule_end
    ]
    if not starts:
        return []

    patterns = [
        (re.compile(r"^\s*([A-Za-z].*?)\s+([0-9,]+)\s+(?:\$\s*)?([0-9,]+)\s*$"), "NSV"),
        (re.compile(r"^\s*([0-9,]+)\s+([A-Za-z].*?)\s+(?:\$\s*)?([0-9,]+)\s*$"), "SNV"),
    ]
    holdings = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else schedule_end
        segment = text[start:end]
        header = re.sub(r"\s+", " ", segment[:1200])
        if not (
            re.search(r"\bShares\b", header, re.IGNORECASE)
            and re.search(r"\bValue\b", header, re.IGNORECASE)
        ):
            continue

        for line in segment.splitlines():
            line = line.strip()
            for regex, order in patterns:
                match = regex.match(line)
                if not match:
                    continue
                first, second, third = match.groups()
                name, shares, value = (
                    (first, second, third)
                    if order == "NSV"
                    else (second, first, third)
                )
                if len(value.replace(",", "")) < 5:
                    break
                name = re.sub(r"\.{2,}.*$", "", name).strip()
                name = re.sub(r"\*|\([a-z]\)", "", name).strip(" .")
                holdings.append(
                    (name, shares.replace(",", ""), value.replace(",", ""))
                )
                break

    # Defensive de-duplication in case a filing repeats a page verbatim.
    return list(dict.fromkeys(holdings))

def parse_text_file(filepath):
    """Parses a text (ASCII) filing."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    holdings = []
    in_schedule = False
    pending_name_parts = []
    
    # Regex for a line like: "Microsoft Corporation*.......  3,347,177   $303,128,717"
    # Name can contain spaces. Separator is usually multiple dots or spaces.
    # We look for a line ending with two numbers.
    
    row_pattern = re.compile(r'^(.*?)\s+([0-9,]+)\s+(?:\$\s*)?([0-9,]+)$')

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "Schedule of Investments" in line:
            in_schedule = True
            pending_name_parts = []
            continue
            
        if not in_schedule:
            continue
            
        # Stop condition: total or end of table
        if "TOTAL" in line.upper() or "</TABLE>" in line.upper():
            in_schedule = False # Or just break if we assume one table
            pending_name_parts = []
            
        # Attempt to match row
        match = row_pattern.match(line)
        if match:
            name, shares, value = match.groups()

            # Older SEC schedules wrap issuer names over multiple lines. Keep
            # the prefix so "Adelphia Communications" / "Corporation" does
            # not become an unidentifiable position named only Corporation.
            if pending_name_parts:
                name = " ".join(pending_name_parts[-3:] + [name])
            
            # Cleanup
            name = name.strip('.').strip()
            name = name.rstrip('*').strip()
            shares = shares.replace(',', '')
            value = value.replace(',', '')
            
            holdings.append((name, shares, value))
            pending_name_parts = []
        elif (
            re.search(r"[A-Za-z]", line)
            and not re.search(r"\d", line)
            and len(line) < 100
            and not re.search(
                r"SHARES|MARKET|VALUE|COMMON STOCK|SCHEDULE|INVESTMENTS",
                line,
                re.IGNORECASE,
            )
        ):
            pending_name_parts.append(line.strip(". "))
        else:
            pending_name_parts = []
            
    return holdings

def process_files():
    files = sorted(os.listdir(DOWNLOAD_DIR))
    all_data = []
    
    for filename in files:
        if not (filename.endswith('.txt') or filename.endswith('.htm') or filename.endswith('.html')):
            continue
            
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # Extract date from filename (YYYY-MM-DD_...)
        file_date = filename.split('_')[0]
        
        logging.info(f"Processing {filename}...")
        
        holdings = []
        parser_method = "html-table"
        if filename.endswith('.txt'):
            text_holdings = parse_text_file(filepath)
            repeated_pages = parse_repeated_schedule_pages(filepath)
            if len(repeated_pages) > len(text_holdings):
                holdings = repeated_pages
                parser_method = "repeated-schedule-pages"
            else:
                holdings = text_holdings
                parser_method = "text-schedule"
        else:
            holdings = parse_html_file(filepath)
            
            # FALLBACK
            if len(holdings) < 10:
                  logging.info("  HTML parsing yielded few results. Using fallback text parsing.")
                  holdings = parse_html_as_text(filepath)
                  parser_method = "html-text-fallback"
            
        # Global artifact filter (catches artifacts from all parser paths)
        ARTIFACT_NAMES = {
            'total', 'net assets', 'net realized gain', 'net change in unrealized',
            'net unrealized', 'interest', 'dividends', 'capital gains',
            'september', 'october', 'november', 'december',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'cash', 'amount due', 'investments purchased', 'marketing expenses',
            'ordinary income', 'other expenses', 'professional fees', 'shares sold',
            'trustee fees', 'undistributed net', 'other assets less liabilities',
            'net investment income',
        }
        filtered = []
        for h in holdings:
            name_lower = h[0].lower().strip()
            if len(h[0]) < 5 or not h[0][0].isalpha():
                continue
            if any(name_lower.startswith(a) for a in ARTIFACT_NAMES):
                continue
            filtered.append(h)

        if len(filtered) < len(holdings):
            logging.info(f"  Filtered {len(holdings) - len(filtered)} artifact rows.")
        holdings = filtered

        logging.info(f"  Found {len(holdings)} holdings.")

        report_date = extract_report_date(filepath)
        if report_date is None:
            report_date = datetime.datetime.strptime(file_date, "%Y-%m-%d").date()
            date_quality = "filing-date-fallback"
        else:
            date_quality = "reported-as-of"

        for h in holdings:
            all_data.append([
                report_date.isoformat(),
                file_date,
                filename,
                h[0],
                h[1],
                h[2],
                "485BPOS",
                "SEC",
                parser_method,
                date_quality,
            ])

    # Write to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "Date",
            "FilingDate",
            "FilingID",
            "Company",
            "Shares",
            "Value",
            "Form",
            "Source",
            "ParserMethod",
            "DateQuality",
        ])
        writer.writerows(all_data)
        
    logging.info(f"Successfully wrote {len(all_data)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()
