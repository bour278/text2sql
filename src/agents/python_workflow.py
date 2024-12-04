from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import re
from colorama import Fore, Style

class State(TypedDict):
    messages: Annotated[list, "Message history"]
    data: Any
    code: str
    results: Any

class PythonWorkflow:
    def __init__(self, db_path: str = "../synthetic_data.db"):
        self.db_path = db_path
        self.setup_components()
        self.setup_graph()

    def setup_components(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.chat_gpt = ChatOpenAI(
            model='gpt-4-1106-preview',
            temperature=0.7,
            api_key=api_key
        )
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def setup_graph(self):
        self.graph = StateGraph(State)
        
        self.graph.add_node("identify_data_needs", self.identify_data_needs)
        self.graph.add_node("fetch_data", self.fetch_data)
        self.graph.add_node("generate_code", self.generate_code)
        self.graph.add_node("execute_code", self.execute_code)
        
        self.graph.add_edge(START, "identify_data_needs")
        self.graph.add_edge("identify_data_needs", "fetch_data")
        self.graph.add_edge("fetch_data", "generate_code")
        self.graph.add_edge("generate_code", "execute_code")
        self.graph.add_edge("execute_code", END)
        
        self.compiled_graph = self.graph.compile()

    def identify_data_needs(self, state: State) -> Dict:
        query = state['messages'][-1].content if hasattr(state['messages'][-1], 'content') else state['messages'][-1][1]
        
        messages = [
            SystemMessage(content=f"""You are a data analysis expert. 
            Identify what data we need to fetch from the database to answer this question.
            
            Available tables and columns:
            {self.get_schema()}
            
            For time series analysis:
            - Always include the date column
            - For recent data, use ORDER BY date DESC with LIMIT
            - For volatility, we need closing prices
            - For price movements, we need closing prices
            
            Return ONLY the SQL query wrapped in ```sql``` code blocks."""),
            HumanMessage(content=query)
        ]
        
        response = self.chat_gpt.invoke(messages)
        sql = self.extract_sql(response.content)
        
        return {
            "messages": [("assistant", f"Data needs identified: {sql}")],
            "code": sql
        }

    def fetch_data(self, state: State) -> Dict:
        sql = state['code']
        try:
            df = pd.read_sql(sql, self.engine)
            return {
                "messages": [("assistant", "Data fetched successfully")],
                "data": df
            }
        except Exception as e:
            return {
                "messages": [("assistant", f"Error fetching data: {str(e)}")],
                "data": None
            }

    def generate_code(self, state: State) -> Dict:
        if state['data'] is None:
            return {
                "messages": [("assistant", "Cannot generate code: No data available")],
                "code": "",
                "results": None
            }

        query = state['messages'][0].content if hasattr(state['messages'][0], 'content') else state['messages'][0][1]
        data_info = f"DataFrame columns: {list(state['data'].columns)}"
        
        messages = [
            SystemMessage(content="""You are a Python data analysis expert specializing in financial calculations.
            Generate Python code to analyze the data and answer the question.
            
            Required code structure:
            1. Perform your calculations
            2. Store the final answer in a variable called 'result'
            3. Format 'result' appropriately:
               - For volatility: a single percentage number
               - For outliers: a DataFrame with dates and values
               - For movements: a DataFrame with dates and changes
            
            Financial calculation notes:
            - Volatility is annualized std dev of log returns
            - Use df['close'].pct_change() for returns
            - Multiply by sqrt(252) to annualize
            - Convert final percentages by multiplying by 100
            
            Example volatility code:
            ```python
            returns = df['close'].pct_change()
            result = returns.std() * np.sqrt(252) * 100  # as percentage
            ```
            
            Example outliers code:
            ```python
            returns = df['close'].pct_change()
            mean = returns.mean()
            std = returns.std()
            result = df[abs(returns - mean) > 2 * std][['date', 'close']]
            ```
            
            Return ONLY the Python code wrapped in ```python``` code blocks."""),
            HumanMessage(content=f"Question: {query}\n\nAvailable data: {data_info}")
        ]
        
        response = self.chat_gpt.invoke(messages)
        code = self.extract_code(response.content)
        
        return {
            "messages": [("assistant", code)],
            "code": code
        }

    def execute_code(self, state: State) -> Dict:
        try:
            namespace = {
                'pd': pd,
                'np': np,
                'df': state['data']
            }
            
            exec(state['code'], namespace)
            
            result = namespace.get('result', "No result variable found")
            
            return {
                "messages": [("assistant", f"Result: {result}")],
                "results": result
            }
        except Exception as e:
            return {
                "messages": [("assistant", f"Error executing code: {str(e)}")],
                "results": None
            }

    def extract_sql(self, text: str) -> str:
        sql_pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(sql_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    def extract_code(self, text: str) -> str:
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    def process_question(self, question: str) -> Dict:
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "data": None,
            "code": "",
            "results": None
        }
        return self.compiled_graph.invoke(initial_state)

    def get_schema(self) -> str:
        """Get the current database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                print(f"{Fore.RED}Warning: No tables found in database{Style.RESET_ALL}")
                return "No tables found in database"

            schema_parts = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                columns_info = [f"    - {col[1]} ({col[2]})" for col in columns]
                schema_parts.append(f"Table: {table_name}\n" + "\n".join(columns_info))
            
            conn.close()
            return "\n\n".join(schema_parts)
        except Exception as e:
            print(f"{Fore.RED}Error getting schema:{Style.RESET_ALL} {str(e)}")
            return str(e)