"""Download S&P500 OHLCV data from yfinance"""
import yfinance as yf
import pandas as pd

def download_sp500(tickers, start='2015-01-01', end='2024-01-01'):
    panel = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end)
        df['Ticker'] = t
        panel[t] = df
    combined = pd.concat(panel)
    combined.to_csv('data/sp500_ohlcv.csv')
    print("Saved data/sp500_ohlcv.csv")

if __name__ == "__main__":
    # Example usage: tickers = ['AAPL','MSFT']
    pass
