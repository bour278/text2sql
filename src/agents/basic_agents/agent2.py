import os
import time
import json
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

def get_schema_context(config):
    db = config.db
    tables = db.get_usable_table_names()

    schema_lines = []
    for table in tables:
        if table in ['ohlc', 'treasury_yields']:
            table_info = db.get_table_info([table])
            schema_lines.append(f"Table: {table}\n{table_info}\n")

    schema_context = (
    "DATABASE SCHEMA:\n" +
    "\n".join(schema_lines)
    )
    return schema_context

def extract_query(response, type='sql'):
    pattern = rf"```{type}\s+([\s\S]*?)\s+```"
    match = re.search(pattern, response)
    
    if match:
        return match.group(1).strip()
    else:
        print(f"Extracting Query Failed: returning response.strip():\n{response.strip()}")
        return response.strip()

def get_sql_prompt2(user_query, config):
    schema_context = get_schema_context(config)
    prompt = f'''Given a user query and a SQlite db schema, only write 
    QUERY DESCRIPTION: QUERY DESCRIPTION, where QUERY DESCRIPTION is a prompt that 
    describes the query (table, cols, new col name, joins, etc.) to get the 
    raw data necessary to answer this user query. Don't write any code or explain any 
    computation, but write the prompt such that if an independent SQL master with access 
    to the SQlite db + schema + your instructions could easily query the data.

    Be mindful of how many days of data to pull, as certain queries may specific n days but 
    require more than n to compute the result.  Make sure the column names in the resulting 
    table are clear. 

    Database Schema: {schema_context}

    Example User Query: Calculate the correlation between 7 year treasury yields and stocks 
    close over the last 30 days in the table. 

    Example Labeled Answer:
    QUERY DESCRIPTION:
        Tables Involved:
        - ohlc (Stock data with date and close price)
        - treasury_yields (Treasury yields with 7-year yield and date)
        Columns Required: 
        - from ohlc: date, close (rename stock_close)
        - from treasury_yields: date, yield_7_year (rename -tsy_yield_7_year)
        Filters: 
        - only consider the last 30 days of data in the table. 
        Joins: 
        - perform an inner join between ohlc and treasury_yields on the date column to 
        align stock data with treasury yields.

    Note that the correlation is not calculated here. The prompt should NOT include
    any math. no standard deviation, no avg, nothing more advanced than multiplication. DO NOT MAKE
    ANY NEW COLUMNS. SAY THAT CALCULATIONS WILL BE DONE LATER, BY THE MATH MASTER. No filters,
    this will be done later by the FILTER MASTER.
    
    User Query: {user_query}
    '''
    sql_prompt = config.llm.invoke(prompt).content
    print(f'Generated SQL Prompt: {sql_prompt}')
    return sql_prompt

def execute_sql2(sql_prompt, config):
    """generate and execute sql"""
    schema_context = get_schema_context(config)
    sql_prompt = f''' 
    Given a SQlite db schema and a query description, generate a syntactically correct SQLite
     query wrapped in ```sql QUERY``` that pulls all the data relevant to the query.

    Database Schema: {schema_context} 

    Example Input Prompt: 
    QUERY DESCRIPTION:
    Tables Involved:
    - ohlc (Stock data with date and close price)
    - treasury_yields (Treasury yields with 7-year yield and date)
    Columns Required: 
    - from ohlc: date, close (rename stock_close)
    - from treasury_yields: date, yield_7_year (rename tsy_yield_7_year) 
    Joins: 
    - perform an inner join between ohlc and treasury_yields on the date column to align 
    stock data with treasury yields.

    *Labeled Answer:*
    ``` sql 
    SELECT 
        y.date AS date,
        y.yield_7_year AS treasury_yield_7_year,
        o.close AS stock_close
    FROM 
        treasury_yields y
    INNER JOIN 
        ohlc o ON y.date = o.date
    ORDER BY 
        y.date DESC
    
    Input Prompt: {sql_prompt}
    '''
    sql_query = extract_query(config.llm.invoke(sql_prompt).content)
    print(f'SQL Query: {sql_query}')
    df = pd.read_sql(sql_query, config.engine)
    return df 

def get_python_prompt2(user_query, df, config):
    prompt = f''' 
    Given a user query and a pandas dataframe with the relevant data, only write 
    CODE DESCRIPTION: CODE DESCRIPTION, where CODE DESCRIPTION is a prompt that 
    describes how to take the dataframe (called df) and write python code to perform 
    relevant computations to answer the user query. Don't write any code, but write 
    the prompt such that if an independent python master with access to df + your instructions could 
    easily answer the original user query. Be specific about how to perform the computations,
    including any relevant math, what functions to use (assume pandas, numpy access). 

    Example User Query: Calculate the correlation between 7 year treasury yields and stocks close over the last 30 days
    in the table.
    Example Dataframe (df.head()): 
    Date,Treasury Yield (7-Year),Stock Close
    2024-01-01 00:00:00,4.113933,84.676268
    2023-12-29 00:00:00,4.117221,100.393128
    2023-12-28 00:00:00,2.391113,112.97598
    2023-12-27 00:00:00,1.482054,119.224503
    2023-12-26 00:00:00,4.187207,108.335695

    Example Answer:
    CODE DESCRIPTION: Given df with cols treasury_yield_7_year, stock_close, date, use pandas corr function
    to compute the correlation between treasury_yield_7_year and stock close over the most recent 30 days.

    df.head(): {df.head()}
    User Query: {user_query}
    '''
    py_prompt = config.llm.invoke(prompt).content
    print(f'Generated Python Prompt: {py_prompt}')
    return py_prompt

def execute_python2(py_prompt, df, config):
    """generate and execute python"""
    py_code = f''' 
    Given a pandas dataframe df and a description to perform a specific computation, 
    generate syntactically correct python code wrapped in ```python QUERY`` that takes 
    the raw dataframe and performs any computations to fully answer the user's query. 
    Assume access to NumPy (v{np.__version__}), Pandas (v{pd.__version__}) and that 
    the dataframe is called df. The output of the code should be the variable that 
    contains the result of the user's query (call this variable result)

    Example Dataframe (df): 
    Date,Treasury Yield (7-Year),Stock Close
    2024-01-01 00:00:00,4.113933,84.676268
    2023-12-29 00:00:00,4.117221,100.393128
    2023-12-28 00:00:00,2.391113,112.97598
    2023-12-27 00:00:00,1.482054,119.224503
    2023-12-26 00:00:00,4.187207,108.335695

    Example Prompt: Given df with cols treasury_yield_7_year, stock_close, date, use pandas corr function
    to compute the correlation between treasury_yield_7_year and stock close. 

    Example Labeled Answer: 
    ``` python
    ### calculate corr btwn 7yr tsy and stock closes 
    df     = df.sort_values('date')[:30]
    result = df['treasury_yield_7_year'].corr(df['stock_close'])
    ```
    df.head(): {df.head()}
    Prompt: {py_prompt}
    '''
    code = extract_query(config.llm.invoke(py_code).content, type='python')
    print(f'Python Code:\n {code}')
    
    namespace = {'pd': pd, 'np': np, 'df': df}
    exec(code, namespace)
    result = namespace.get('result', "No result variable found")
    print(f'Result:\n{result}')
    return result

def run_agent2(user_query, config):
    """main function to run agent 2"""
    print(f'User Query:\n {user_query}')
    start = time.time()
    sql_prompt = get_sql_prompt2(user_query, config)
    df = execute_sql2(sql_prompt, config)
    py_prompt = get_python_prompt2(user_query, df, config)
    result = execute_python2(py_prompt, df, config)
    end = time.time()

    print(f'runtime: {round(end-start,2)} seconds')
    return result