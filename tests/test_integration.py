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
    
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=30),
        'open': [100 + i for i in range(30)],
        'high': [110 + i for i in range(30)],
        'low': [90 + i for i in range(30)],
        'close': [105 + i for i in range(30)]
    })
    df.to_sql('ohlc', conn, if_exists='replace', index=False)
    
    conn.close()
    yield db_path
    
    try:
        time.sleep(0.1)
        shutil.rmtree(temp_dir)
    except PermissionError:
        print(f"Warning: Could not remove temporary directory {temp_dir}")

def test_master_workflow_integration(test_db):
    """Test the complete workflow with different types of questions"""
    from main import MasterWorkflow
    
    workflow = MasterWorkflow(db_path=test_db)
    
    result = workflow.process_question("What is the minimum price in the ohlc table?")
    assert result['workflow_type'] == 'sql'
    assert result['complexity_score'] < 0.5
    assert result.get('error') is None
    assert result.get('results') is not None
    
    result = workflow.process_question("What is the stock yang-zhang volatility over the last 21 days from the ohlc table?")
    assert result['workflow_type'] == 'python'
    assert result['complexity_score'] > 0.5
    assert result.get('error') is None
    assert result.get('results') is not None