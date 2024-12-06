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
from dotenv import load_dotenv
from pathlib import Path

current_dir = Path(__file__).parent
env_path = current_dir / "keys.env"
print("Looking for .env file at:", env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key and env_path.exists():
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")

print("API Key loaded:", bool(api_key))

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
        
        def get_code_from_llm(query_text: str) -> str:
            messages = [
                SystemMessage(content="""You are a Python data analysis expert specializing in financial calculations.
                Generate Python code to analyze the data and answer the question. 
                
                IMPORTANT: Your response must be either:
                1. Python code wrapped in ```python``` blocks
                2. OR a clarification question starting with 'CLARIFICATION:'
                
                Required code structure when providing code:
                1. Perform your calculations
                2. Store the final answer in a variable called 'result'
                3. Format 'result' appropriately:
                   - For volatility: a single percentage number
                   - For outliers: a DataFrame with dates and values
                   - For movements: a DataFrame with dates and changes"""),
                HumanMessage(content=f"Question: {query_text}\n\nAvailable data: {data_info}")
            ]
            
            response = self.chat_gpt.invoke(messages)
            return response.content

        content = get_code_from_llm(query)
        
        max_attempts = 3
        attempt = 0
        
        while content.strip().startswith("CLARIFICATION:") and attempt < max_attempts:
            try:
                print(f"\n{Fore.YELLOW}Clarification needed:{Style.RESET_ALL}")
                print(content.replace("CLARIFICATION:", "").strip())
                user_input = input(f"\n{Fore.GREEN}Your response:{Style.RESET_ALL} ")
                
                # Update query with additional context
                query = f"{query}\nAdditional context: {user_input}"
                content = get_code_from_llm(query)
                attempt += 1
            except Exception as e:
                print(f"Error during clarification: {e}")
                break

        if attempt >= max_attempts:
            return {
                "messages": [("assistant", "Maximum clarification attempts reached. Please try rephrasing your question.")],
                "code": "",
                "results": None
            }

        code = self.extract_code(content)
        if not code:
            return {
                "messages": [("assistant", "Failed to generate valid Python code")],
                "code": "",
                "results": None
            }

        print(f"{Fore.GREEN}Generated Python:{Style.RESET_ALL}")
        print(code)
        
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
            
            # Format the result for display
            if isinstance(result, (float, np.float64)):
                formatted_result = f"{result:.4f}"
            elif isinstance(result, pd.DataFrame):
                formatted_result = "\n" + result.to_string()
            else:
                formatted_result = str(result)
            
            print(f"\n{Fore.GREEN}Result:{Style.RESET_ALL}")
            print(formatted_result)
            
            return {
                "messages": [("assistant", f"Result: {formatted_result}")],
                "results": result
            }
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return {
                "messages": [("assistant", error_msg)],
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