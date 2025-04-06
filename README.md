# Text-to-SQL Interface

This project provides a natural language interface to query financial data stored in a SQLite database. Users can ask questions in plain English, and the system will convert them into SQL queries and return the results.

## Features

- Interactive UI built with Streamlit
- Database schema exploration with hoverable table information
- Sample data preview for each table
- Support for both OpenAI and Google Gemini models
- Natural language processing to convert questions to SQL

## Database Tables

The application works with the following tables:

1. **ohlc**: Stock price data with open, high, low, and close prices for each date
2. **fxrates**: Foreign exchange rates for USD to EUR, GBP, and JPY
3. **treasury_yields**: Treasury yields for 5-year, 7-year, and 10-year bonds
4. **yahoo_ohlc**: Stock price data from Yahoo Finance with ticker symbols

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

1. Generate the synthetic database (if not already done):
   ```
   python data/sqlite-synthetic.py
   ```

2. Run the Streamlit UI:
   ```
   streamlit run src/ui.py
   ```

3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Use the sidebar to explore the database schema and sample data

5. Enter your question in the text area and click "Submit Question"

## Example Questions

- "What are the closing prices for last 30 dates in the ohlc table?"
- "What is the average open price over the last 10 days in the ohlc table?"
- "What is the stock volatility over the last 21 days from the ohlc table?"
- "Calculate the correlation between 7 year treasury yields and close stock prices over the last 30 days"
- "Find the days where the stock price movement was more than 2 standard deviations from the mean"

## Project Structure

- `src/ui.py`: Streamlit UI interface
- `src/main.py`: Command-line interface for running queries
- `data/sqlite-synthetic.py`: Script to generate synthetic financial data
- `agents/`: Contains the agents that process natural language queries
