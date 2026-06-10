import yfinance as yf
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

TICKER_MAP = {
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "wipro": "WIPRO.NS", "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS", "reliance": "RELIANCE.NS",
    "icici": "ICICIBANK.NS", "sbi": "SBIN.NS",
    "bajaj": "BAJFINANCE.NS", "asian paints": "ASIANPAINT.NS"
}

def extract_ticker(query: str) -> str:
    q = query.lower()
    for name, ticker in TICKER_MAP.items():
        if name in q:
            return ticker
    return "INFY.NS"  # default fallback

def get_stock_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        news = stock.news[:3] if stock.news else []
        headlines = [n.get("content", {}).get("title", "") for n in news]
        return {
            "ticker": ticker,
            "price": info.get("currentPrice", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "week52_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "week52_low": info.get("fiftyTwoWeekLow", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "day_change": info.get("regularMarketChangePercent", "N/A"),
            "headlines": headlines
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

def live_data_node(state: dict) -> dict:
    if "live" not in state.get("agents_to_use", []):
        return state

    question = state["sub_tasks"].get("live", state["query"])
    print(f"\n[Live Data] Fetching for: {question}")

    ticker = extract_ticker(state["query"])
    data = get_stock_data(ticker)

    if "error" in data:
        answer = f"Could not fetch live data: {data['error']}"
    else:
        prompt = f"""Based on this live market data, answer: {question}

Stock: {data['ticker']}
Current Price: {data['price']}
P/E Ratio: {data['pe_ratio']}
EPS: {data['eps']}
52-Week High: {data['week52_high']}
52-Week Low: {data['week52_low']}
Market Cap: {data['market_cap']}
Day Change: {data['day_change']}%
Recent Headlines: {', '.join(data['headlines'])}

Provide a concise financial analysis based on these numbers."""

        answer = llm.invoke(prompt).content

    print(f"[Live Data] Answer length: {len(answer)} chars")
    return {**state, "live_answer": answer}