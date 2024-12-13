import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
from colorama import init, Style, Fore
import sqlite3

init()

class YahooParser:
    """Parser class to fetch OHLC data from Yahoo"""

    def __init__(self):
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'LLY', 
            'TSM', 'V', 'UNH', 'XOM', 'WMT', 'JPM', 'MA', 'JNJ', 'PG', 'AVGO',
            'HD', 'CVX', 'MRK', 'ORCL', 'KO', 'PEP', 'ABBV', 'BAC', 'PFE',
            'TMO', 'COST', 'DHR', 'MCD', 'ACN', 'ABT', 'DIS', 'CSCO', 'VZ'
        ]

    def fetch_ohlc(self,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch OHLC data for specified tickers and date
        """

        if not start_date:
            start_date = (datetime.now() - timedelta(days = 365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not tickers:
            tickers = self.tickers
        
        all_data = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                df.reset_index(inplace=True)

                df = df[['Date', 'Open', 'High', 'Low', 'Close']]
                df.columns = ['date', 'open', 'high', 'low', 'close']

                df['ticker'] = ticker

                all_data.append(df)
            
            except Exception as e:
                print(Fore.RED + f"Error: could not fetch data for {ticker}: {str(e)}" + Style.RESET_ALL)
                continue

        if not all_data: raise ValueError(Fore.RED + "No data was fetched for any ticker" + Style.RESET_ALL)

        combined_df = pd.concat(all_data, ignore_index=True)

        combined_df['data'] = combined_df['date'].dt.strftime('%Y-%m-%d')

        return combined_df
    
    def save_to_sqlite(self, df: pd.DataFrame, db_path: str) -> None:
        """Save OHLC data to SQLite database"""

        conn = sqlite3.connect(db_path)

        conn.execute('''
        CREATE TABLE IF NOT EXISTS yahoo_ohlc (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            PRIMARY KEY (date, ticker)
        )
        ''')

        df.to_sql('stock_ohlc', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()