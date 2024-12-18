import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from yfinance_data import YahooParser

src_dir = Path(__file__).parent.parent / 'src'
db_path = src_dir / 'synthetic_data.db'

print(f"Creating database at: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS ohlc (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL
)
''')

dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='B')
data = {
    'date': dates,
    'open': np.random.uniform(75, 120, len(dates)),
    'high': np.random.uniform(120, 130, len(dates)),
    'low': np.random.uniform(50, 70, len(dates)),
    'close': np.random.uniform(80, 120, len(dates))
}
ohlc_df = pd.DataFrame(data)
ohlc_df.to_sql('ohlc', conn, if_exists='replace', index=False)

cursor.execute('''
CREATE TABLE IF NOT EXISTS fxrates (
    date TEXT PRIMARY KEY,
    usd_to_eur REAL,
    usd_to_gbp REAL,
    usd_to_jpy REAL
)
''')

fx_data = {
    'date': dates,
    'usd_to_eur': np.random.uniform(0.8, 1.2, len(dates)),
    'usd_to_gbp': np.random.uniform(0.7, 0.9, len(dates)),
    'usd_to_jpy': np.random.uniform(110, 180, len(dates))
}
fxrates_df = pd.DataFrame(fx_data)
fxrates_df.to_sql('fxrates', conn, if_exists='replace', index=False)

cursor.execute('''
CREATE TABLE IF NOT EXISTS treasury_yields (
    date TEXT PRIMARY KEY,
    yield_5_year REAL,
    yield_7_year REAL,
    yield_10_year REAL
)
''')

treasury_data = {
    'date': dates,
    'yield_5_year': np.random.uniform(1.2, 4.1, len(dates)),
    'yield_7_year': np.random.uniform(1.3, 4.5, len(dates)),
    'yield_10_year': np.random.uniform(1.4, 4.5, len(dates))
}
treasury_yields_df = pd.DataFrame(treasury_data)
treasury_yields_df.to_sql('treasury_yields', conn, if_exists='replace', index=False)

yahoo_parser = YahooParser()
yahoo_data = yahoo_parser.fetch_ohlc(
    start_date='2020-01-01',
    end_date='2024-01-01'
)

cursor.execute('''
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

yahoo_data.to_sql('yahoo_ohlc', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"Database created successfully at: {db_path}")
