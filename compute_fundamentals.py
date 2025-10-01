"""Fetch fundamentals for S&P500 tickers from yfinance"""
import yfinance as yf
import pandas as pd

def fundamentals(tickers):
    rows=[]
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append({
            'Ticker': t,
            'peRatio': info.get('trailingPE'),
            'debtToEquity': info.get('debtToEquity'),
            'profitMargins': info.get('profitMargins')
        })
    df=pd.DataFrame(rows)
    df.to_csv('data/sp500_fundamentals.csv',index=False)
    print("Saved data/sp500_fundamentals.csv")

if __name__=="__main__":
    pass
