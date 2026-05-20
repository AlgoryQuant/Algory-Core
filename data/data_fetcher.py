import yfinance as yf
import pandas as pd

def fetch_market_data(ticker, period="1y", interval="1d"):
    """
    Stáhne historická data pro daný ticker.
    """
    print(f"Fetching data for: {ticker}...")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            print("No data found.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Příklad použití (pro test)
if __name__ == "__main__":
    df = fetch_market_data("AAPL")
    if df is not None:
        print(df.head())
