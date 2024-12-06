# Pandas Cheat Sheet for Financial Applications

A quick reference for commonly used Pandas functions for time series analysis, financial calculations, and data manipulation.

---

## **DataFrame & Series Basics**

### **Creating DataFrame & Series**
```python
import pandas as pd

# DataFrame
df = pd.DataFrame(data, columns=['Column1', 'Column2'])

# Series
series = pd.Series([1, 2, 3, 4, 5])
```

### **Viewing Data**
```python
df.head()  # First 5 rows
df.tail()  # Last 5 rows
df.info()  # Summary of DataFrame
df.describe()  # Statistical summary
```

---

## **Time Series Data Handling**

### **Setting DateTime Index**
```python
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
```

### **Generating Date Ranges**
```python
date_range = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')  # Business days
```

### **Resampling Time Series**
```python
df.resample('M').mean()  # Monthly average
df.resample('A').sum()  # Annual sum
```

### **Handling Missing Data**
```python
df.fillna(method='ffill', inplace=True)  # Forward fill
df.dropna(inplace=True)  # Drop missing values
```

---

## **Financial Calculations**

### **Calculating Daily Returns**
```python
df['Daily_Return'] = df['Close'].pct_change()  # Percentage change
```

### **Cumulative Returns**
```python
df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
```

### **Simple Moving Average (SMA)**
```python
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day SMA
```

### **Exponential Moving Average (EMA)**
```python
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()  # 50-day EMA
```

### **Bollinger Bands**
```python
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['Upper_Band'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower_Band'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
```

---

## **Rolling Windows**

### **Rolling Mean**
```python
df['Rolling_Mean_50'] = df['Close'].rolling(window=50).mean()  # 50-day rolling mean
```

### **Exponential Weighted Moving Average (EWMA)**
```python
df['EWMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()  # EWMA
```

### **Rolling Volatility (Std Dev)**
```python
df['Rolling_Volatility_30'] = df['Daily_Return'].rolling(window=30).std()  # 30-day rolling volatility
```

---

## **Volatility Models**

### **Yang-Zhang Volatility**
```python
high = df['High']
low = df['Low']
close = df['Close']

rvi = (high - low) / (high.shift(1) - low.shift(1))  # Range-based component
return_vol = df['Daily_Return'].rolling(window=30).std()  # Return-based component

yang_zhang_vol = (1 / (1 + (1 / 2))) * return_vol + (1 / 2) * rvi
df['Yang_Zhang_Vol'] = yang_zhang_vol
```

### **Garman-Klass Volatility**
```python
log_ret = np.log(df['Close'] / df['Close'].shift(1))  # Log returns
garman_klass_vol = np.sqrt(
    0.5 * (np.log(df['High'] / df['Low']))**2 - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
)
df['Garman_Klass_Vol'] = garman_klass_vol
```

---

## **Returns Calculations**

### **Close-to-Close Returns**
```python
df['Close_to_Close_Return'] = df['Close'].pct_change()  # Simple close-to-close returns
```

---

## **Data Selection & Filtering**

### **Selecting Columns**
```python
df['Close']  # Single column
df[['Close', 'Open']]  # Multiple columns
```

### **Selecting Rows by Index**
```python
df.loc['2020-01-01']  # Row by index (date)
df.iloc[10]  # Row by integer index
```

### **Filtering by Condition**
```python
df[df['Close'] > 100]  # Rows where Close is greater than 100
```

### **Date Range Filtering**
```python
df['2020-01-01':'2020-06-30']  # Filter data by date range
```

---

## **Aggregation & Grouping**

### **GroupBy Example**
```python
df.groupby('Symbol').mean()  # Group by a column and calculate the mean
```

### **Aggregating Multiple Functions**
```python
df.groupby('Symbol').agg({'Close': ['mean', 'std'], 'Volume': 'sum'})
```

### **Pivot Table Example**
```python
df.pivot_table(values='Close', index='Date', columns='Symbol', aggfunc='mean')
```

---

## **Covariance & Correlation**

### **Covariance**
```python
df[['Close', 'Volume']].cov()  # Covariance between columns
```

### **Correlation**
```python
df[['Close', 'Volume']].corr()  # Correlation between columns
```

---

## **Plotting Financial Data**

### **Plotting Simple Line Chart**
```python
df['Close'].plot(figsize=(10, 6), title='Stock Price')
```

### **Plotting with Moving Averages**
```python
df[['Close', 'SMA_50']].plot(figsize=(10, 6))
```

### **Candlestick Chart with `mplfinance`**
```python
import mplfinance as mpf
mpf.plot(df, type='candle', volume=True, mav=(50, 200), figsize=(10, 6))
```

---

## **Exporting Data**

### **Export to CSV**
```python
df.to_csv('output.csv')
```

### **Export to Excel**
```python
df.to_excel('output.xlsx', index=False)
```

### **Export to SQL**
```python
df.to_sql('table_name', conn, if_exists='replace')
```

---

## **Additional Useful Functions**

### **DataFrame Operations**
```python
df['Log_Close'] = np.log(df['Close'])  # Log transformation
df['Close_Diff'] = df['Close'].diff()  # Difference between consecutive values
```

### **Rolling Windows**
```python
df['Rolling_Mean'] = df['Close'].rolling(window=30).mean()  # 30-day rolling mean
```

---

## **Documentation Reference**

For more detailed usage and advanced functionality, refer to the official Pandas documentation: [https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/).

