import streamlit as st
import sqlite3
import pandas as pd
import os
from pathlib import Path
from agents.master import MasterWorkflow
from agents.basic_agents.config import get_paths

# Set page configuration
st.set_page_config(
    page_title="Text-to-SQL Interface",
    page_icon="📊",
    layout="wide"
)

# Get database path
_, SQL_DB_FPATH = get_paths()

# Function to get table schemas
def get_table_schemas():
    conn = sqlite3.connect(SQL_DB_FPATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schemas = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schemas[table_name] = columns
    
    conn.close()
    return schemas

# Function to get sample data from a table
def get_sample_data(table_name, limit=5):
    conn = sqlite3.connect(SQL_DB_FPATH)
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to get table descriptions
def get_table_descriptions():
    return {
        "ohlc": "Stock price data with open, high, low, and close prices for each date.",
        "fxrates": "Foreign exchange rates for USD to EUR, GBP, and JPY.",
        "treasury_yields": "Treasury yields for 5-year, 7-year, and 10-year bonds.",
        "yahoo_ohlc": "Stock price data from Yahoo Finance with ticker symbols."
    }

# Main UI
def main():
    st.title("📊 Text-to-SQL Interface")
    st.markdown("""
    This interface allows you to ask natural language questions about the financial data in our database.
    The system will convert your questions into SQL queries and return the results.
    """)
    
    # Sidebar with database information
    with st.sidebar:
        st.header("Database Information")
        
        # Get table schemas
        schemas = get_table_schemas()
        table_descriptions = get_table_descriptions()
        
        # Display tables with expandable details
        for table_name, columns in schemas.items():
            with st.expander(f"📋 {table_name}"):
                st.markdown(f"**Description:** {table_descriptions.get(table_name, 'No description available.')}")
                st.markdown("**Schema:**")
                
                # Create a DataFrame for the schema
                schema_df = pd.DataFrame(columns, columns=['cid', 'name', 'type', 'notnull', 'default', 'pk'])
                schema_df = schema_df[['name', 'type', 'pk']]  # Only show relevant columns
                schema_df.columns = ['Column', 'Type', 'Primary Key']
                st.dataframe(schema_df, use_container_width=True)
                
                # Show sample data
                st.markdown("**Sample Data:**")
                sample_data = get_sample_data(table_name)
                st.dataframe(sample_data, use_container_width=True)
    
    # Main content area
    st.header("Ask a Question")
    
    # Text input for the question
    user_question = st.text_area(
        "Enter your question about the data:",
        placeholder="Example: What is the average closing price over the last 30 days?",
        height=100
    )
    
    # Model selection
    model_option = st.radio(
        "Select the model to use:",
        ["OpenAI", "Gemini"],
        horizontal=True
    )
    
    # Submit button
    if st.button("Submit Question"):
        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    # Initialize the workflow
                    use_gemini = model_option == "Gemini"
                    workflow = MasterWorkflow(db_path=SQL_DB_FPATH, use_gemini=use_gemini)
                    
                    # Process the question
                    result = workflow.process_question(user_question)
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Display workflow type and complexity score
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Workflow Type", result['workflow_type'])
                    with col2:
                        st.metric("Complexity Score", f"{result['complexity_score']:.2f}")
                    
                    # Display the results
                    if result.get('results') is not None:
                        if isinstance(result['results'], (list, tuple)):
                            # Convert to DataFrame if it's a list of tuples
                            if result['results'] and isinstance(result['results'][0], tuple):
                                df = pd.DataFrame(result['results'])
                                st.dataframe(df, use_container_width=True)
                            else:
                                for row in result['results']:
                                    st.write(row)
                        else:
                            st.write(result['results'])
                    
                    # Display any errors
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main() 