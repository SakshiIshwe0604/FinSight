KNOWN_COMPANIES = {
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "wipro": "WIPRO.NS",
    "hdfc": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
    "reliance": "RELIANCE.NS",
    "icici": "ICICIBANK.NS", "sbi": "SBIN.NS",
    "bajaj": "BAJFINANCE.NS", "asian paints": "ASIANPAINT.NS"
}

# companies that actually have annual reports indexed in FAISS
# (your live ticker list is broader than your PDF coverage)
PDF_INDEXED_COMPANIES = {
    "infosys", "infy", "tcs", "tata consultancy",
    "wipro", "hdfc", "hdfc bank", "reliance"
}

def detect_ticker(query: str) -> str | None:
    q = query.lower()
    for name, ticker in KNOWN_COMPANIES.items():
        if name in q:
            return ticker
    return None

def is_pdf_covered(query: str) -> bool:
    q = query.lower()
    return any(name in q for name in PDF_INDEXED_COMPANIES)