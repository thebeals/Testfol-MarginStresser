import json
import csv
import os
import requests
import difflib
import yfinance as yf
import time
import re
import config

# Configuration
INPUT_CSV = config.COMPONENTS_FILE
SEC_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_JSON_FILE = os.path.join(config.ASSETS_DIR, "company_tickers.json")
MAPPING_FILE = os.path.join(config.ASSETS_DIR, "name_mapping.json")
USER_AGENT = "Antigravity/1.0 (antigravity_agent@google.com)"

def download_sec_json():
    print("Downloading SEC Tickers JSON...")
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(SEC_JSON_URL, headers=headers)
        resp.raise_for_status()
        with open(SEC_JSON_FILE, "wb") as f:
            f.write(resp.content)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def load_sec_data():
    if not os.path.exists(SEC_JSON_FILE) or os.path.getsize(SEC_JSON_FILE) < 5000:
        if not download_sec_json():
            return {}
            
    with open(SEC_JSON_FILE, "r") as f:
        data = json.load(f)
    
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    # Convert to list of dicts
    tickers = []
    for k, v in data.items():
        tickers.append(v)
    return tickers

def get_unique_names():
    names = set()
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["Company"])
    return sorted(list(names))

def clean_name(name):
    # Simplify name for matching
    n = name.lower()
    # Strip (b), (a), (c) footnote markers from SEC filings
    n = re.sub(r'\s*\([a-z]\)\s*$', '', n)
    n = re.sub(r'[\.,]', '', n)
    n = re.sub(r'\b(inc|corp|corporation|ltd|limited|company|co|plc)\b', '', n)
    n = n.replace('*', '').strip()
    return n

def price_validate(ticker, date_str, shares, value):
    # Check if Price(date) * shares ~= value within 10%
    # shares and value are strings
    try:
        shares = float(shares)
        val = float(value)
        if shares == 0: return False
        
        implied_price = val / shares
        
        # Determine 5-day window around date
        # YF expects YYYY-MM-DD
        dt = date_str
        
        # We need Split-Adjusted Price from YF?
        # NO. The Shares in CSV are Raw. The Value is Raw.
        # But YF gives Split-Adjusted Price.
        # For validation, we need to know if the implied price matches the YF price (adjusted or raw).
        # Actually: Implied Price is RAW Price.
        # YF Price is ADJ Price.
        
        # If no split happened, match.
        # If split happened, mismatched.
        
        # Strategy: Fetch history + splits.
        # Calc Raw Price from YF data.
        # Raw Price = Adj Close / (Product of split factors since date?) 
        # Actually simpler: YF data has "Stock Splits".
        # We can detect splits.
        
        # But for quick validation, we can just fetch the data.
        # If implied price is 90 and YF is 22... maybe split 4:1.
        # If implied price is 90 and YF is 90... no split.
        
        # Let's fetch the data first.
        data = yf.download(ticker, start=dt, end=pd.to_datetime(dt) + pd.Timedelta(days=5), progress=False)
        if data.empty:
            return False
            
        # Get first available close
        yf_price_adj = data['Close'].iloc[0] # auto_adjust=True default in new YF
        if isinstance(yf_price_adj, pd.Series): yf_price_adj = yf_price_adj.iloc[0]
        
        ratio = implied_price / yf_price_adj
        
        # If ratio is close to 1.0, 2.0, 4.0, 0.5 etc. it's likely a match with split.
        # Allow crude validation.
        
        valid_ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 0.5, 0.25, 0.1]
        
        for r in valid_ratios:
            if 0.85 < (ratio / r) < 1.15:
                return True
                
        return False
    except:
        return False

# Manual overrides for difficult to map companies (ADRs, complex class structures)
# Includes fixes for ~32 confirmed wrong fuzzy matches and ~170 previously unmapped names
MANUAL_OVERRIDES = {
    # === EXISTING CORRECT OVERRIDES ===
    "ARM Holdings PLC, ADR (b)": "ARM",
    "ASML Holding N.V., New York Shares (Netherlands)": "ASML",
    "Airbnb, Inc., Class A (b)": "ABNB",
    "AstraZeneca PLC, ADR (United Kingdom)": "AZN",
    "Atlassian Corp., Class A (b)": "TEAM",
    "Atlassian Corp. PLC, Class A (b)": "TEAM",
    "Autodesk, Inc. (b)": "ADSK",
    "Baker Hughes Co., Class A": "BKR",
    "Biogen, Inc. (b)": "BIIB",
    "Charter Communications, Inc., Class A (b)": "CHTR",
    "Coca-Cola Europacific Partners PLC (United Kingdom)": "CCEP",
    "Copart, Inc. (b)": "CPRT",
    "CrowdStrike Holdings, Inc., Class A (b)": "CRWD",
    "Crowdstrike Holdings, Inc., Class A (b)": "CRWD",
    "Datadog, Inc., Class A (b)": "DDOG",
    "DexCom, Inc. (b)": "DXCM",
    "DoorDash, Inc., Class A (b)": "DASH",
    "Fortinet, Inc. (b)": "FTNT",
    "GLOBALFOUNDRIES, Inc. (b)": "GFS",
    "IDEXX Laboratories, Inc. (b)": "IDXX",
    "Illumina, Inc. (b)": "ILMN",
    "Intuitive Surgical, Inc. (b)": "ISRG",
    "Kraft Heinz Co. (The)": "KHC",
    "Monolithic Power Systems, Inc.": "MPWR",
    "Monolithic Power Systems": "MPWR",
    "Old Dominion Freight Line, Inc.": "ODFL",
    "Palo Alto Networks, Inc. (b)": "PANW",
    "Ross Stores, Inc.": "ROST",
    "Synopsys, Inc. (b)": "SNPS",
    "Take-Two Interactive Software, Inc. (b)": "TTWO",
    "The Trade Desk, Inc., Class A": "TTD",
    "Trade Desk, Inc. (The), Class A (b)": "TTD",
    "Warner Bros. Discovery, Inc. (b)": "WBD",
    "Workday, Inc., Class A (b)": "WDAY",
    "Workday, Inc., Class A": "WDAY",
    "Zscaler, Inc. (b)": "ZS",
    "lululemon athletica, inc. (b)": "LULU",
    "O\u2019Reilly Automotive, Inc. (b)": "ORLY",
    "Cadence Design Systems, Inc. (b)": "CDNS",
    "CoStar Group, Inc. (b)": "CSGP",
    "GE HealthCare Technologies, Inc.": "GEHC",
    "Marriott International, Inc., Class A": "MAR",
    "Marriott International, Inc., Class A (b)": "MAR",
    "MercadoLibre, Inc.": "MELI",
    "MercadoLibre, Inc. (Brazil) (b)": "MELI",
    "MercadoLibre, Inc. (Argentina) (b)": "MELI",
    "Microchip Technology, Inc.": "MCHP",
    "Microchip Technology Incorporated": "MCHP",
    "Micron Technology, Inc.": "MU",
    "Moderna, Inc.": "MRNA",
    "Moderna, Inc. (b)": "MRNA",
    "MongoDB, Inc. (b)": "MDB",
    "Monster Beverage Corp.": "MNST",
    "ON Semiconductor Corp. (b)": "ON",
    "PACCAR, Inc.": "PCAR",
    "Regeneron Pharmaceuticals, Inc.": "REGN",
    "Super Micro Computer, Inc. (b)": "SMCI",
    "T-Mobile US, Inc.": "TMUS",
    "Verisk Analytics, Inc.": "VRSK",
    "Verisk Analytics, Inc., Class A": "VRSK",
    "Vertex Pharmaceuticals, Inc.": "VRTX",
    "Tesla, Inc. (b)": "TSLA",
    "Amazon.com, Inc. (b)": "AMZN",
    "PDD Holdings, Inc., ADR (China) (b)": "PDD",
    "Pinduoduo, Inc., ADR (China) (b)": "PDD",
    "Netflix, Inc. (b)": "NFLX",
    "Alphabet, Inc., Class A": "GOOGL",
    "Alphabet, Inc., Class A (b)": "GOOGL",
    "Alphabet, Inc., Class C": "GOOG",
    "Alphabet, Inc., Class C (b)": "GOOG",
    "Meta Platforms, Inc., Class A": "META",
    "Meta Platforms, Inc., Class A (b)": "META",
    "Comcast Corp., Class A": "CMCSA",
    "Adobe, Inc. (b)": "ADBE",
    "ANSYS, Inc. (b)": "ANSS",

    # === FIXED WRONG MAPPINGS (were fuzzy-matched to wrong SEC registrants) ===
    "Apple Computer, Inc.": "AAPL",
    "Tesla Motors, Inc.": "TSLA",
    "Priceline Group, Inc.": "BKNG",
    "Priceline Group, Inc. (The)": "BKNG",
    "priceline.com, Inc.": "BKNG",
    "Comcast Corp.": "CMCSA",
    "Comcast Corporation": "CMCSA",
    "ASML Holding NV": "ASML",
    "Baidu, Inc.": "BIDU",
    "Baidu, Inc. ADR": "BIDU",
    "Baidu, Inc., ADR (China) (b)": "BIDU",
    "Baidu.com": "BIDU",
    "Teva Pharmaceutical Industries Limited": "TEVA",
    "Teva Pharmaceutical Industries Ltd": "TEVA",
    "Teva Pharmaceutical Industries Ltd. - ADR": "TEVA",
    "Teva Pharmaceutical Industries Ltd. ADR": "TEVA",
    "Altera Corp.": "ALTR",
    "Altera Corporation": "ALTR",
    "Linear Technology Corp.": "LLTC",
    "Linear Technology Corporation": "LLTC",
    "Juniper Networks, Inc.": "JNPR",
    "Seagate Technology": "STX",
    "Seagate Technology PLC": "STX",
    "Seagate Technology Holdings": "STX",
    "Symantec Corp.": "GEN",
    "Symantec Corporation": "GEN",
    "Avago Technologies Ltd.": "AVGO",
    "Avago Technologies Ltd., Class A": "AVGO",
    "Celgene Corp.": "CELG",
    "Celgene Corp. (b)": "CELG",
    "Nuance Communications, Inc.": "NUAN",
    "Costco Companies, Inc": "COST",
    "FLIR Systems, Inc.": "FLIR",
    "Flir Systems, Inc.": "FLIR",
    "Charter Communications, Inc.": "CHTR",
    "Discovery Communications, Inc.": "DISCA",
    "Discovery Communications, Inc., Class A": "DISCA",
    "Discovery Communications, Inc., Class C": "DISCK",
    "Kraft Foods Group, Inc.": "KRFT",
    "Kraft Foods, Inc., Class A": "KRFT",
    "Level 3 Communications, Inc.": "LVLT",

    # === PREVIOUSLY UNMAPPED: HIGH FREQUENCY ===
    "Xilinx, Inc.": "XLNX",
    "Maxim Integrated Products, Inc.": "MXIM",
    "Citrix Systems, Inc.": "CTXS",
    "Activision Blizzard, Inc.": "ATVI",
    "Staples, Inc.": "SPLS",
    "Cerner Corp.": "CERN",
    "Sun Microsystems, Inc.": "JAVA",
    "Expedia, Inc.": "EXPE",
    "Adobe Systems, Inc.": "ADBE",
    "Adobe Systems Incorporated": "ADBE",
    "Whole Foods Market, Inc.": "WFM",
    "JDS Uniphase Corporation": "JDSU",
    "Walgreens Boots Alliance, Inc.": "WBA",
    "Yahoo!, Inc.": "YHOO",
    "Yahoo! Inc.": "YHOO",
    "Yahoo Inc.": "YHOO",
    "Stericycle, Inc.": "SRCL",
    "Vodafone Group PLC ADR": "VOD",
    "Dell, Inc.": "DELL",
    "Dell Inc.": "DELL",
    "Dell Computer Corporation": "DELL",
    "Express Scripts, Inc.": "ESRX",
    "Express Scripts, Inc., Class A": "ESRX",
    "Express Scripts Holding Co.": "ESRX",
    "Mattel, Inc.": "MAT",
    "Network Appliance, Inc.": "NTAP",
    "Google, Inc., Class A": "GOOGL",
    "Google, Inc., Class C": "GOOG",
    "Google, Inc.": "GOOG",
    "Google Inc.": "GOOG",
    "Cephalon, Inc.": "CEPH",
    "Broadcom Corp., Class A": "BRCM",
    "KLA-Tencor Corp.": "KLAC",
    "KLA-Tencor Corporation": "KLAC",
    "KLA -Tencor Corporation": "KLAC",
    "Kla-Tencor Corp.": "KLAC",
    "KLA-Tencor    Corp.": "KLAC",
    "Apollo Group, Inc.": "APOL",
    "Apollo Group, Inc., Class A": "APOL",
    "Sigma-Aldrich Corp.": "SIAL",
    "Sigma-Aldrich Corporation": "SIAL",
    "Tellabs, Inc.": "TLAB",

    # === PREVIOUSLY UNMAPPED: MEDIUM FREQUENCY ===
    "MedImmune, Inc.": "MEDI",
    "Genzyme Corporation": "GENZ",
    "Genzyme Corp.": "GENZ",
    "Genzyme General": "GENZ",
    "Biogen Idec, Inc.": "BIIB",
    "Biogen IDEC, Inc.": "BIIB",
    "DISH Network Corp., Class A": "DISH",
    "DISH Network Corp.": "DISH",
    "Viacom, Inc., Class B": "VIAB",
    "Facebook, Inc., Class A": "META",
    "Facebook, Inc., Class A (b)": "META",
    "Liberty Global PLC, Class A": "LBTYA",
    "Liberty Global PLC, Series C": "LBTYK",
    "Liberty Global PLC, Class C": "LBTYK",
    "Liberty Global PLC, Class C (United Kingdom) (b)": "LBTYK",
    "Liberty Global PLC, Series A (United Kingdom) (b)": "LBTYA",
    "Liberty Global PLC, Class A (United Kingdom) (b)": "LBTYA",
    "Liberty Global PLC LiLAC, Class A": "LILA",
    "Liberty Global PLC LiLAC, Class C": "LILAK",
    "Liberty Global PLC Lilac, Class A": "LILA",
    "Liberty Global PLC Lilac, Class C": "LILAK",
    "Liberty Global, Inc., Class A": "LBTYA",
    "Chiron Corporation": "CHIR",
    "Comverse Technology, Inc.": "CMVT",
    "PETsMART, Inc.": "PETM",
    "PetSmart, Inc.": "PETM",
    "Virgin Media, Inc.": "VMED",
    "Research In Motion Ltd.": "BBRY",
    "Research In Motion, Ltd.": "BBRY",
    "Mylan, Inc.": "MYL",
    "Mylan NV": "MYL",
    "Mylan N.V. (b)": "MYL",
    "Twenty-First Century Fox, Inc., Class A": "FOXA",
    "Twenty-First Century Fox, Inc., Class B": "FOX",
    "Twenty-First Century Fox, Inc.": "FOXA",
    "Fox Corp., Class A": "FOXA",
    "Fox Corp., Class B": "FOX",
    "BMC Software, Inc.": "BMC",
    "VERITAS Software Corporation": "VRTS",
    "Veritas Software Corporation": "VRTS",
    "PeopleSoft, Inc.": "PSFT",
    "Dollar Tree Stores, Inc.": "DLTR",
    "Invitrogen Corporation": "IVGN",
    "Mercury Interactive Corporation": "MERQ",
    "Molex Incorporated": "MOLX",
    "Sanmina-SCI Corporation": "SANM",
    "Smurfit-Stone Container Corporation": "SSCC",
    "American Power Conversion Corporation": "APCC",
    "Biomet, Inc.": "BMET",
    "Biomet, Inc": "BMET",
    "Novellus System, Inc.": "NVLS",
    "Joy Global, Inc.": "JOY",
    "Joy Global Inc.": "JOY",
    "Sirius XM Radio, Inc.": "SIRI",
    "Sirius Satellite Radio Inc.": "SIRI",
    "Sirius Satellite Radio, Inc.": "SIRI",
    "XM Satellite Radio Holdings Inc.": "XMSR",
    "XM Satellite Radio Holdings, Inc.": "XMSR",
    "News Corp., Class A": "NWSA",
    "Warner Chilcott PLC, Class A": "WCRX",
    "F5 Networks, Inc.": "FFIV",
    "Liberty Interactive Corp., Class A": "QVCA",
    "Liberty Interactive Corp. QVC Group, Class A": "QVCA",
    "Liberty Interactive Corp. QVC Group, Series A": "QVCA",
    "JD.com, Inc. ADR": "JD",
    "JD.com, Inc., ADR (China) (b)": "JD",
    "JD.com, Inc., ADR (China)": "JD",

    # === PREVIOUSLY UNMAPPED: LOWER FREQUENCY ===
    "NTL Incorporated": "NTLI",
    "Compuware Corporation": "CPWR",
    "Pixar": "PIXR",
    "IAC / InterActiveCorp": "IAC",
    "IAC/InterActiveCorp": "IAC",
    "IAC/InterActive Corp.": "IAC",
    "InterActiveCorp": "IAC",
    "USA Interactive": "USAI",
    "Sepracor, Inc.": "SEPR",
    "Sepracor Inc.": "SEPR",
    "Monster Worldwide, Inc.": "MWW",
    "Monster Worldwide Inc.": "MWW",
    "Liberty Media Corp. - Interactive": "LINTA",
    "Liberty Media Corp., Class A": "LMCA",
    "Liberty Media Corp., Class C": "LMCK",
    "Infosys Technologies Ltd. ADR": "INFY",
    "Green Mountain Coffee Roasters, Inc.": "GMCR",
    "Keurig Green Mountain, Inc.": "GMCR",
    "DIRECTV": "DTV",
    "DIRECTV, Class A": "DTV",
    "DIRECTV Group, Inc./The": "DTV",
    "Liberty Ventures, Series A": "LVNTA",
    "Ctrip.com International Ltd. ADR": "CTRP",
    "Ctrip.Com International Ltd. ADR": "CTRP",
    "Ctrip.com International, Ltd., ADR (China) (b)": "CTRP",
    "Trip.com Group Ltd., ADR (China) (b)": "TCOM",
    "NetEase, Inc. ADR": "NTES",
    "NetEase, Inc., ADR (China)": "NTES",
    "Applied Micro Circuits Corporation": "AMCC",
    "Conexant Systems, Inc.": "CNXT",
    "Concord EFS, Inc.": "CEFT",
    "Brocade Communications Systems, Inc.": "BRCD",
    "Human Genome Sciences, Inc.": "HGSI",
    "ICOS Corporation": "ICOS",
    "PanAmSat Corporation": "SPOT",
    "QLogic Corporation": "QLGC",
    "Qlogic Corporation": "QLGC",
    "RF Micro Devices, Inc.": "RFMD",
    "Intersil Corporation": "ISIL",
    "Siebel System, Inc.": "SEBL",
    "CheckFree Corp.": "CKFR",
    "Discovery Holding Co.": "DISCA",
    "Patterson Cos, Inc.": "PDCO",
    "Patterson Cos., Inc.": "PDCO",
    "Patterson Dental Company": "PDCO",
    "Foster Wheeler AG": "FWLT",
    "Foster Wheeler, Ltd.": "FWLT",
    "Foster Wheeler Ltd.": "FWLT",
    "Catamaran Corp.": "CTRX",
    "VimpelCom Ltd. ADR": "VEON",
    "Randgold Resources Ltd. ADR": "GOLD",
    "Shire PLC ADR": "SHPG",
    "NXP Semiconductors N.V. (Netherlands)": "NXPI",

    # === MORE UNMAPPED: VARIOUS FREQUENCIES ===
    "Fiserv, Inc. (b)": "FISV",
    "VeriSign, Inc. (b)": "VRSN",
    "Zoom Video Communications, Inc., Class A (b)": "ZM",
    "Incyte Corp. (b)": "INCY",
    "DocuSign, Inc. (b)": "DOCU",
    "Splunk, Inc. (b)": "SPLK",
    "Seagen, Inc. (b)": "SGEN",
    "Seattle Genetics, Inc. (b)": "SGEN",
    "Okta, Inc. (b)": "OKTA",
    "Peloton Interactive, Inc., Class A (b)": "PTON",
    "Global Crossing Ltd.": "GLBC",
    "Global Crossings Ltd.": "GLBC",
    "PMC-Sierra, Inc.": "PMCS",
    "PMC - Sierra, Inc.": "PMCS",
    "SDL Incorporated": "SDLI",
    "VoiceStream Wireless Corporation": "VSTR",
    "MCI WorldCom, Inc.": "WCOM",
    "Novell, Inc.": "NOVL",
    "QUALCOMM Incorporated": "QCOM",
    "Qualcomm, Incorporated": "QCOM",
    "Qwest Communications International Inc.": "Q",
    "Gemstar-TV Guide International, Inc.": "GMST",
    "Gemstar-TV Guide International Inc.": "GMST",
    "Atmel Corporation": "ATML",
    "CDW Computer Centers, Inc.": "CDWC",
    "Cytyc Corporation": "CYTC",
    "ImClone Systems Incorporated": "IMCL",
    "Protein Design Labs, Inc.": "PDLI",
    "Rational Software Corporation": "RATL",
    "Red Hat, Inc.": "RHT",
    "UAL Corp.": "UAL",
    "Hansen Natural, Corp.": "MNST",
    "Hansen Natural Corp.": "MNST",
    "Leap Wireless International, Inc.": "LEAP",
    "Pharmaceutical Product Development, Inc.": "PPDI",
    "Fossil, Inc.": "FOSL",
    "Qurate Retail, Inc.": "QRTEA",
    "Ulta Salon Cosmetics & Fragrance, Inc.": "ULTA",
    "SBA Communications Corp.,   Class A": "SBAC",
    "Alnylam Pharmaceuticals": "ALNY",
    "Ferrovial SE": "FER",
    "Insmed Incorporated": "INSM",
    "Western Digital": "WDC",

    # Misc name variants that may appear with/without (b)
    "Automatic Data Processing, Inc. when-issued": "ADP",

    # === FIX WRONG FUZZY MATCHES (these were matched to wrong SEC registrants) ===
    "Bed Bath & Beyond, Inc.": "BBBY",
    "Bed Bath & Beyond Inc.": "BBBY",
    "T-Mobile    US, Inc.": "TMUS",
    "T-Mobile US, Inc. (b)": "TMUS",
    "Sears Holdings Corp.": "SHLD",
    "WorldCom, Inc.": "WCOM",
    "WorldCom, Inc": "WCOM",
    "CMGI, Inc.": "CMGI",
    "CMGI, Inc": "CMGI",
    "TMP Worldwide Inc.": "TMPW",
    "I2 Technologies, Inc.": "ITWO",
    "i2 Technologies, Inc.": "ITWO",
    "NII Holdings, Inc.": "NIHD",
    "IDEC Pharmaceuticals Corporation": "IDPH",
    "IDEC Pharmaceuticals Corp.": "IDPH",
    "BEA Systems, Inc.": "BEAS",
    "BEA Systems, Inc": "BEAS",
    "Flextronics International Ltd": "FLEX",
    "Flextronics International Ltd.": "FLEX",
    "Flextronics International, Ltd": "FLEX",
    "Integrated Device Technology, Inc": "IDTI",
    "Integrated Device Technology, Inc.": "IDTI",
    "Vitesse Semiconductor Corp": "VTSS",
    "Vitesse Semiconductor Corporation": "VTSS",
    "Adelphia Communications": "ADLAC",
    "Adelphia Communications Corporation": "ADLAC",
    "USA Networks, Inc": "USAI",
    "Level 3 Communications, Inc": "LVLT",
    "Apple Computer, Inc": "AAPL",
    "Juniper Networks, Inc": "JNPR",
    "Bed Bath & Beyond Inc": "BBBY",
    "Immunex Corporation": "IMNX",
    "Vertex Pharmaceuticals, Inc. (b)": "VRTX",
    "SanDisk Corp.": "SNDK",
    "SanDisk Corp": "SNDK",
}

# Tickers that were acquired — map to successor for price fallback
SUCCESSOR_TICKERS = {
    # === Original entries ===
    'CELG': {'successor': 'BMY', 'effective_date': '2019-11-20'},   # Celgene → Bristol-Myers
    'XLNX': {'successor': 'AMD', 'effective_date': '2022-02-14'},   # Xilinx → AMD
    'MXIM': {'successor': 'ADI', 'effective_date': '2021-08-26'},   # Maxim → Analog Devices
    'CTXS': None,     # Citrix → taken private
    'ATVI': {'successor': 'MSFT', 'effective_date': '2023-10-13'},  # Activision → Microsoft
    'CERN': {'successor': 'ORCL', 'effective_date': '2022-06-08'},  # Cerner → Oracle
    'ESRX': {'successor': 'CI', 'effective_date': '2018-12-20'},    # Express Scripts → Cigna
    'ALTR': {'successor': 'INTC', 'effective_date': '2015-12-28'},  # Altera → Intel
    'LLTC': 'ADI',    # Linear Tech → Analog Devices (2017)
    'NUAN': {'successor': 'MSFT', 'effective_date': '2022-03-04'},  # Nuance → Microsoft
    'FLIR': {'successor': 'TDY', 'effective_date': '2021-05-14'},   # FLIR → Teledyne
    'LVLT': 'LUMN',   # Level 3 → Lumen (2017)
    'WFM':  {'successor': 'AMZN', 'effective_date': '2017-08-28'},  # Whole Foods → Amazon
    'KRFT': 'KHC',    # Kraft → Kraft Heinz (2015)
    'DELL': 'DELL',   # Dell went private, came back as DELL
    'JDSU': 'VIAV',   # JDS Uniphase → Viavi Solutions (2015)
    'BRCM': 'AVGO',   # Broadcom Corp → Broadcom Inc (2016)
    'YHOO': None,     # Yahoo → taken private (Altaba liquidated)
    'SRCL': 'SRCL',   # Stericycle — still public
    'SIAL': None,     # Sigma-Aldrich → Merck KGaA (no US ticker)

    # === Expanded: more acquired companies ===
    'GENZ': 'SNY',    # Genzyme → Sanofi (2011)
    'CHIR': 'NVS',    # Chiron → Novartis (2006)
    'DTV':  'T',      # DirecTV → AT&T (2015)
    'BMET': 'ZBH',    # Biomet → Zimmer Biomet (2015)
    'PSFT': 'ORCL',   # PeopleSoft → Oracle (2005)
    'JAVA': 'ORCL',   # Sun Microsystems → Oracle (2010)
    'IMNX': 'AMGN',   # Immunex → Amgen (2002)
    'IDPH': 'BIIB',   # IDEC Pharma → Biogen IDEC (2003)
    'BEAS': 'ORCL',   # BEA Systems → Oracle (2008)
    'SNDK': 'WDC',    # SanDisk → Western Digital (2016)
    'MYL':  'VTRS',   # Mylan → Viatris (2020)
    'VIAB': 'PARA',   # Viacom → Paramount (2019)
    'BBRY': 'BB',     # BlackBerry ticker change
    'RHT':  {'successor': 'IBM', 'effective_date': '2019-07-09'},   # Red Hat → IBM
    'MEDI': 'AZN',    # MedImmune → AstraZeneca (2007)
    'SGEN': {'successor': 'PFE', 'effective_date': '2023-12-14'},   # Seagen → Pfizer
    'SPLK': {'successor': 'CSCO', 'effective_date': '2024-03-18'},  # Splunk → Cisco

    # === Additional acquired companies ===
    'ANSS': {'successor': 'SNPS', 'effective_date': '2025-07-17'},  # Ansys → Synopsys
    'CTRP': 'TCOM',   # Ctrip → Trip.com (ticker change 2019)
    'FOXA': 'FOX',    # 21st Century Fox → Fox Corp (2019, Disney deal)
    'WBA':  None,     # Walgreens → taken private by Sycamore (2025)
    'VRTS': 'GEN',    # Veritas Software → Symantec → NortonLifeLock → Gen Digital
    'PETM': 'CHWY',   # PetSmart → Chewy spin-off (proxy)
    'NOVL': None,     # Novell → Micro Focus (taken private)
    'LEAP': 'T',      # Leap Wireless → AT&T (2014)
    'NWSA': 'NWS',    # News Corp Class A (ticker variant)

    # === Bankrupt / taken private (no public successor) ===
    'SHLD': None,     # Sears → bankrupt
    'BBBY': None,     # Bed Bath & Beyond → bankrupt
    'WCOM': None,     # WorldCom → bankrupt (fraud)
    'SPLS': None,     # Staples → taken private
    'APOL': None,     # Apollo Group → taken private
    'ITWO': None,     # I2 Technologies → JDA (private)
    'NIHD': None,     # NII Holdings → bankrupt
}

def main():
    sec_data = load_sec_data()
    csv_names = get_unique_names()
    
    # Build lookup for SEC titles
    sec_map = {}
    for item in sec_data:
        c_title = clean_name(item['title'])
        sec_map[c_title] = item['ticker']
    
    mapping = {}
    
    # Pre-populate with manual overrides
    mapping.update(MANUAL_OVERRIDES)
    
    print(f"Mapping {len(csv_names)} names...")
    
    matches_found = 0
    
    for name in csv_names:
        # Check manual overrides first
        if name in mapping:
            matches_found += 1
            continue
            
        c_name = clean_name(name)
        ticker = None
        
        # 1. Exact cleaned match
        if c_name in sec_map:
            ticker = sec_map[c_name]
        else:
            # 2. Difflib match
            matches = difflib.get_close_matches(c_name, sec_map.keys(), n=1, cutoff=0.8)
            if matches:
                 ticker = sec_map[matches[0]]
        
        if ticker:
            mapping[name] = ticker
            matches_found += 1
            # print(f"Mapped '{name}' -> {ticker}")
        else:
            # print(f"Failed to map '{name}'")
            pass

    print(f"Mapped {matches_found}/{len(csv_names)} companies.")
    
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

if __name__ == "__main__":
    import pandas as pd
    main()
