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

class Config:
    """SQL agent config"""
    def __init__(
        self,
        env_fpath,
        sql_db_fpath,
        openai_model = 'gpt-4-1106-preview',
        temperature  = 0.7
        ):
        load_dotenv(env_fpath)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        self.llm = ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            api_key=openai_api_key
            )
        self.db = SQLDatabase.from_uri(f"sqlite:///{sql_db_fpath}")
        self.engine = create_engine(f"sqlite:///{sql_db_fpath}")

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

def execute_sql1(user_query, config):
    """generate and execute sql"""
    schema_context = get_schema_context(config)
    sql_prompt = f''' 
    Given a user query, generate a single syntatically correct SQLite query wrapped in ```sql QUERY``` that pulls all the 
    data relevant to the query. Do not perform any computation, assume that the user is able to take the raw data and 
    perform the relevant computations to reach the desired result. Be mindful of how many days of data to pull, as 
    certain queries may specific n days but require more than n to compute the result. For days do LIMIT rather than now().
    Make sure the column names in the resulting table are clear. Here is the database schema: {schema_context}
    
    Here is an example user query and then correct output:
    User Query: Calculate the correlation between 7 year treasury yields and stocks close over the last 30 days in the table.
    Correct Output: ``` sql 
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
    LIMIT 30;
    ```
    Notice there is no calculation calculation. This is the correct response, as 
    the correlation will be calculated later.
    
    Give the correct output for user query: {user_query}
    '''
    sql_query = extract_query(config.llm.invoke(sql_prompt).content)
    print(f'SQL Query: {sql_query}')
    df = pd.read_sql(sql_query, config.engine)
    return df

def execute_python1(user_query, df, config):
    """generate and execute python"""
    python_prompt = f''' 
    Given a user query and a pandas dataframe with the relevant data, generate syntatically correct python code 
    wrapped in ```python QUERY`` that takes the raw dataframe and performs any computations to fully answer the 
    user's query. Assume access to NumPy (v1.26.4), Pandas (v2.2.3) and that the dataframe is called df. The output 
    variable should always be called result. 

    Here is an example user query and df and then correct output:
    User Query: Calculate the correlation between 7 year treasury yields and stocks close over the last 30 days
    in the table.
    Dataframe (df): 
    Date,Treasury Yield (7-Year),Stock Close
    2024-01-01 00:00:00,4.113933,84.676268
    2023-12-29 00:00:00,4.117221,100.393128
    2023-12-28 00:00:00,2.391113,112.97598
    2023-12-27 00:00:00,1.482054,119.224503
    2023-12-26 00:00:00,4.187207,108.335695

    *Labeled Answer:*
    ``` python
    ### calculate corr btwn 7yr tsy and stock closes 
    result = df['treasury_yield_7_year'].corr(df['stock_close'])
    ```

    User Query: \n{user_query}
    df.head: \n{df.head()}
    '''

    code = extract_query(config.llm.invoke(python_prompt).content, type='python')
    print(f'Python Code:\n {code}')
    
    namespace = {'pd': pd, 'np': np, 'df': df}
    exec(code, namespace)
    result = namespace.get('result', "No result variable found")
    print(f'Result:\n{result}')
    return result

def run_agent1(user_query, config):
    """main function to run agent 1"""
    print(f'User Query:\n {user_query}')
    start = time.time()
    df = execute_sql1(user_query, config)
    result = execute_python1(user_query, df, config)
    end = time.time()

    print(f'runtime: {round(end-start,2)} seconds')
    return result