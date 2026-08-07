import requests
import os
import time
import datetime
import config

# Configuration
CIK = "1067839"
FORM_TYPES = {"485BPOS", "NPORT-P"}
START_DATE = datetime.date(1999, 1, 1)
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Testfol NDX research testfol@example.com",
)

# SEC EDGAR URLs
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK.zfill(10)}.json"
BASE_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

def setup_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session

def download_filing(
    session,
    accession_number,
    primary_document,
    filing_date,
    *,
    form,
    report_date="",
):
    """
    Downloads a single filing.
    URL format: https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}
    Note: Accession number in URL usually DOES NOT have dashes.
    """
    accession_no_dashes = accession_number.replace("-", "")
    
    # Sometimes primary_document is empty or not what we want, but let's try the listing first
    # Actually, the bulk feed often gives the primary document.
    # The URL provided by the user: https://www.sec.gov/Archives/edgar/data/1067839/000091205700030669/a485bpos.txt
    # This implies we can just construct it.
    
    # NPORT submissions expose an XSL display path, while the raw XML lives at
    # the accession root.  Using basename works for both filing families.
    document_name = os.path.basename(primary_document)
    url = f"{BASE_ARCHIVE_URL}/{CIK}/{accession_no_dashes}/{document_name}"
    download_dir = (
        config.NPORT_CACHE_DIR if form == "NPORT-P" else config.NDX_CACHE_DIR
    )
    os.makedirs(download_dir, exist_ok=True)
    if form == "NPORT-P":
        filename = f"{report_date}_{filing_date}_{accession_number}_{document_name}"
    else:
        filename = f"{filing_date}_{accession_number}_{document_name}"
    filepath = os.path.join(download_dir, filename)
    refresh = os.environ.get("NDX_REFRESH_SEC_FILINGS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if os.path.exists(filepath) and not refresh:
        print(f"Already cached: {filepath}")
        return True
    
    # We might need to switch Host header for document download if it's different from data.sec.gov
    # The archives are usually on www.sec.gov
    print(f"Downloading {filing_date} - {url}...")
    
    try:
        response = session.get(url)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False
    finally:
        time.sleep(0.12) # Rate limit: SEC allows 10 req/s, so > 0.1s sleep is safe

def main():
    os.makedirs(config.NDX_CACHE_DIR, exist_ok=True)
    os.makedirs(config.NPORT_CACHE_DIR, exist_ok=True)

    session = setup_session()
    
    print(f"Fetching submissions for CIK {CIK}...")
    try:
        response = session.get(SUBMISSIONS_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching submissions: {e}")
        return

    filings = data.get("filings", {}).get("recent", {})
    
    if not filings:
        print("No recent filings found.")
        return

    # Filings data is a dictionary of lists. We need to iterate through them.
    # lists: accessionNumber, filingDate, reportDate, acceptanceDateTime, act, form, fileNumber, filmNumber, items, size, isXBRL, isInlineXBRL, primaryDocument, primaryDocDescription
    
    count = 0
    total_filings = len(filings["accessionNumber"])
    
    print(f"Processing {total_filings} filings...")
    
    for i in range(total_filings):
        form = filings["form"][i]
        filing_date_str = filings["filingDate"][i]
        filing_date = datetime.datetime.strptime(filing_date_str, "%Y-%m-%d").date()
        
        if form in FORM_TYPES and filing_date >= START_DATE:
            accession_number = filings["accessionNumber"][i]
            primary_document = filings["primaryDocument"][i]
            report_date = filings.get("reportDate", [""] * total_filings)[i]
            
            # primaryDocument is sometimes just a filename like "doc.xml". 
            # If we want the text version, it is often just the accession number + .txt?
            # User example: .../000091205700030669/a485bpos.txt
            # Let's trust primaryDocument for now. If it fails, we might need logic to find the .txt
            
            success = download_filing(
                session,
                accession_number,
                primary_document,
                filing_date_str,
                form=form,
                report_date=report_date,
            )
            if success:
                count += 1
                
    print(f"Done. Cached {count} SEC holdings filings ({', '.join(sorted(FORM_TYPES))}).")

if __name__ == "__main__":
    main()
