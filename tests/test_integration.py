import pytest
import sys
import os
import sqlite3
import time
import pandas as pd
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def test_db():
    """Create a temporary test database with sample data"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    
    conn = sqlite3.connect(db_path)
    
    # Create OHLC data
    df_ohlc = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=100),  # Increased to 100 days
        'open': [100 + i for i in range(100)],
        'high': [110 + i for i in range(100)],
        'low': [90 + i for i in range(100)],
        'close': [105 + i for i in range(100)]
    })
    df_ohlc.to_sql('ohlc', conn, if_exists='replace', index=False)
    
    # Create Treasury Yields data
    df_treasury = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=100),
        'yield_5_year': [3.0 + i*0.01 for i in range(100)],
        'yield_7_year': [3.2 + i*0.01 for i in range(100)],
        'yield_10_year': [3.5 + i*0.01 for i in range(100)]
    })
    df_treasury.to_sql('treasury_yields', conn, if_exists='replace', index=False)
    
    conn.close()
    yield db_path
    
    try:
        time.sleep(0.1)
        shutil.rmtree(temp_dir)
    except PermissionError:
        print(f"Warning: Could not remove temporary directory {temp_dir}")

@pytest.mark.parametrize("use_gemini", [False, True])
def test_master_workflow_integration(test_db, use_gemini):
    """Test the complete workflow with different types of questions"""
    from agents.master import MasterWorkflow
    
    # Skip Gemini tests if GOOGLE_API_KEY is not set
    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    
    # Skip OpenAI tests if OPENAI_API_KEY is not set
    if not use_gemini and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    workflow = MasterWorkflow(db_path=test_db, use_gemini=use_gemini)
    
    # Test simple SQL query
    result = workflow.process_question("What is the minimum price in the ohlc table?")
    assert result['workflow_type'] == 'sql'
    assert result['complexity_score'] < 0.5
    assert result.get('error') is None
    assert result.get('results') is not None
    
    # Test complex Python analysis
    result = workflow.process_question("What is the stock volatility over the last 21 days from the ohlc table?")
    assert result['workflow_type'] == 'python'
    assert result['complexity_score'] > 0.5
    assert result.get('error') is None
    assert result.get('results') is not None
    
    # Test correlation analysis
    result = workflow.process_question("Calculate the correlation between 7 year treasury yields and close stock prices over the last 30 days")
    assert result['workflow_type'] == 'python'
    assert result['complexity_score'] > 0.5
    assert result.get('error') is None
    assert result.get('results') is not None