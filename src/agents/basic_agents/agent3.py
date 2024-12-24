import os
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def get_sql_json3(user_query, config):
    """Get SQL query as JSON structure"""
    schema_context = get_schema_context(config)
    prompt = f'''Given a user query and a SQLite database schema, return ONLY a valid JSON describing the 
    data required to answer the user query. The JSON should be parsable and adhere to proper JSON syntax.

    Instructions:
    2. Ensure column names and new column names are strings, and there are no extraneous characters.
    3. Be mindful of how many days of data to pull, as certain queries may specify n days but 
    require more than n to compute the result.
    4. Avoid creating new columns or performing calculations; these will be handled in later steps.
    5. Ensure new column names are self-explanatory and clear.

    Answer ONLY with a JSON in this format:
    {{
        "tables": {{
            "table1": [
                ["original_column_name", "new_column_name"],
                ...
            ],
            "table2": [...],
        }},
        "joins": [
            ["tableA", "tableB", "tableA_join_column", "tableB_join_column", "join_type"]
        ]
    }}

    Constraints:
    - Join types must be one of: "inner", "left", "right", "outer".
    - Joins will be applied in order of the json. Note the order does matter here. For example 
    if the first join is inner(A,B) then next is outer(B, C), what is really happening is 
    outer(C, inner(A,B)), i.e. not inner(A, outer(B,C))

    Database Schema: {schema_context}

    Example User Query: Calculate the correlation between 7-year treasury yields and stocks' 
    close prices over the last 30 days.

    Example Labeled Answer:
    ```json
    {{
        "tables": {{
            "ohlc": [
                ["date", "date"],
                ["close", "stock_close"]
            ],
            "treasury_yields": [
                ["date", "date"],
                ["yield_7_year", "tsy_yield_7_year"]
            ]
        }},
        "joins": [
            ["ohlc", "treasury_yields", "date", "date", "inner"]
        ]
    }}

    Answer for the following user query: {user_query}
    ''' 
    llm_response = config.llm.invoke(prompt).content
    print('=' * 30)
    print(f'LLM Response: {llm_response}')
    sql_json = extract_query(llm_response, type='json')
    try:
        sql_json = json.loads(sql_json)
        print(f'=' * 30)
        print(f'SQL PARSED JSON: {sql_json}')
        valid_join_types = {"inner", "left", "right", "outer"}
        for join in sql_json.get("joins", []):
            if len(join) != 5 or join[-1] not in valid_join_types:
                raise ValueError(f"Invalid join type or structure: {join}")

        return sql_json
    except json.JSONDecodeError:
        raise ValueError("LLM response is not valid JSON.")
    except Exception as e:
        raise ValueError(f"Error validating SQL JSON: {e}")

def compile_sql(sql_json: dict, config) -> str:
    """
    Compiles a JSON definition of tables and joins into a SQLite SELECT query.
    """
    tables = sql_json.get("tables", {})
    joins = sql_json.get("joins", [])

    # 1) Build the projection columns (the SELECT part)
    select_columns = []
    for table_name, column_pairs in tables.items():
        for original, alias in column_pairs:
            select_columns.append(f'"{table_name}"."{original}" AS "{alias}"')

    # If no tables at all, we can't form a valid query
    if not tables:
        raise ValueError("No tables were provided. At least one table is required.")

    columns_str = ",\n    ".join(select_columns)

    # 2) Build the FROM/JOIN parts of the query
    if joins:
        base_table = joins[0][0]
        from_clause = f'"{base_table}"'

        for (tbl_left, tbl_right, left_on, right_on, join_type) in joins:
            join_type_upper = join_type.upper() + " JOIN"
            from_clause += (
                f'\n{join_type_upper} "{tbl_right}" '
                f'ON "{tbl_left}"."{left_on}" = "{tbl_right}"."{right_on}"'
            )
    else:
        base_table = list(tables.keys())[0]
        from_clause = f'"{base_table}"'

    # 3) Put it all together
    query = f'''
SELECT
    {columns_str}
FROM {from_clause}'''.strip()
    print(f'=' * 30)
    print(f'SQL Query: {query}')
    df = pd.read_sql(query, config.engine)
    print(f'Query Results:\n{df.to_string()}')
    return df

def get_python_prompt3(user_query, df, config):
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

def execute_python3(py_prompt, df, config):
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

def run_agent3(user_query, config):
    """main function to run agent 3"""
    print('=' * 30)
    print(f'User Query:\n {user_query}')
    start = time.time()
    sql_json = get_sql_json3(user_query, config)
    df = compile_sql(sql_json, config)
    py_prompt = get_python_prompt3(user_query, df, config)
    result = execute_python3(py_prompt, df, config)
    end = time.time()
    print('=' * 30)
    print(f'runtime: {round(end-start,2)} seconds')
    return result