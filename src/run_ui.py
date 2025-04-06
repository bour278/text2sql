#!/usr/bin/env python
"""
Script to run the Text-to-SQL UI
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Check if the database exists
    db_path = Path(__file__).parent / 'synthetic_data.db'
    
    if not db_path.exists():
        print("Database not found. Generating synthetic data...")
        data_dir = Path(__file__).parent.parent / 'data'
        subprocess.run([sys.executable, str(data_dir / 'sqlite-synthetic.py')], check=True)
        print("Synthetic data generated successfully.")
    
    # Run the Streamlit UI
    print("Starting Text-to-SQL UI...")
    ui_path = Path(__file__).parent / 'ui.py'
    subprocess.run(['streamlit', 'run', str(ui_path)], check=True)

if __name__ == "__main__":
    main() 