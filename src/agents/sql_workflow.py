from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
import re
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from colorama import Fore, Style
from typing import Dict, Any
from pathlib import Path
from pydantic import BaseModel, field_validator
import os

current_dir = Path(__file__).parent
env_path = current_dir / "keys.env"
print("Looking for .env file at:", env_path)

api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv('GOOGLE_API_KEY')

if not api_key and env_path.exists():
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv('GOOGLE_API_KEY')
    os.environ["GOOGLE_API_KEY"] = google_api_key


print("API Key loaded:", bool(api_key))

class SQLExtractor(BaseModel):
    response: str

    @field_validator('response')
    def extract_sql(cls, v):
        sql_pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(sql_pattern, v, re.DOTALL)
        
        if matches:
            sql = matches[0].strip()
            return sql
            
        sql_pattern = r'SQLQuery:\s*(.*?);'
        matches = re.findall(sql_pattern, v, re.DOTALL)
        
        if matches:
            return matches[0].strip()
            
        return v.strip()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_sql: str
    results: Any
    
class SQLWorkflow:
    def __init__(self, db_path: str = "../synthetic_data.db", use_gemini: bool = False):
        self.db_path = db_path
        self.use_gemini = use_gemini
        self.setup_components()
        self.setup_graph()

    def setup_components(self):
        if self.use_gemini:
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
            self.chat_gpt = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=google_api_key
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.chat_gpt = ChatOpenAI(
                model='gpt-4-1106-preview',
                temperature=0.7,
                api_key=api_key
            )
        
        # Initialize SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def setup_graph(self):
        self.graph = StateGraph(State)
        self.chat_gpt = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        self.graph.add_node("parse_question", self.parse_question)
        self.graph.add_node("generate_sql", self.generate_sql)
        self.graph.add_node("validate_sql", self.validate_sql)
        self.graph.add_node("execute_sql", self.execute_sql)
        
        self.graph.add_edge(START, "parse_question")
        self.graph.add_edge("parse_question", "generate_sql")
        self.graph.add_edge("generate_sql", "validate_sql")
        self.graph.add_edge("validate_sql", "execute_sql")
        self.graph.add_edge("execute_sql", END)
        
        self.compiled_graph = self.graph.compile()

    def parse_question(self, state: State) -> Dict:
        query = state['messages'][-1].content
        print(f"{Fore.CYAN}Parsing Question:{Style.RESET_ALL} {query}")
        return {"messages": [("assistant", f"Parsed question: {query}")]}

    def get_schema(self) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name from sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        main_schema = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f'PRAGMA table_info({table_name});')
            columns = cursor.fetchall()
            info = [f"    - {col[1]} ({col[2]})" for col in columns]
            schema = f"Table: {table_name}\n" + "\n".join(info)
            main_schema.append(schema)
        
        conn.close()
        return "\n\n".join(main_schema)

    def generate_sql(self, state: State) -> Dict:
        query = state['messages'][-1].content if hasattr(state['messages'][-1], 'content') else state['messages'][-1][1]
        schema = self.get_schema()
        
        print(f"\n{Fore.BLUE}Database Schema:{Style.RESET_ALL}")
        print(schema)
        print()
        
        messages = [
            SystemMessage(content=f"""You are a SQL expert. Generate a SQL query to answer the question.
            Here is the database schema:
            {schema}
            
            Return ONLY the SQL query wrapped in ```sql``` code blocks. Do not include any explanations.
            Important: For recent data, use ORDER BY date DESC with LIMIT instead of date comparisons."""),
            HumanMessage(content=query)
        ]
        
        print(f"{Fore.BLUE}Question being sent to LLM:{Style.RESET_ALL} {query}\n")
        
        response = self.chat_gpt.invoke(messages)
        
        extractor = SQLExtractor(response=response.content)
        sql = extractor.response
        
        print(f"{Fore.GREEN}Generated SQL:{Style.RESET_ALL} {sql}")
        return {
            "messages": [("assistant", sql)],
            "current_sql": sql
        }

    def validate_sql(self, state: State) -> Dict:
        sql = state['current_sql']
        print(f"{Fore.YELLOW}Validating SQL...{Style.RESET_ALL}")
        
        return {"messages": [("assistant", "SQL validated")], "current_sql": sql}

    def execute_sql(self, state: State) -> Dict:
        sql = state['current_sql']
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                print(f"{Fore.MAGENTA}Query Results:{Style.RESET_ALL}")
                for row in rows:
                    print(row)
                return {"messages": [("assistant", "Query executed successfully")], "results": rows}
        except Exception as e:
            print(f"{Fore.RED}Error executing SQL:{Style.RESET_ALL} {str(e)}")
            return {"messages": [("assistant", f"Error: {str(e)}")], "results": None}

    def process_question(self, question: str) -> Dict:
        print(f"\n{Fore.CYAN}Processing Question:{Style.RESET_ALL} {question}")
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "current_sql": "",
            "results": None
        }
        result = self.compiled_graph.invoke(initial_state)
        return result

    def identify_data_needs(self, state: State) -> Dict:
        print(f"\n{Fore.YELLOW}=== IDENTIFY DATA NEEDS ==={Style.RESET_ALL}")
        
        schema = self.get_schema()
        prompt = f"""You are a SQL expert. Write a SQL query to fetch the raw data needed for Python analysis.
        Return ONLY the SQL query wrapped in ```sql``` blocks.
        
        Database Schema:
        {schema}
        
        Important:
        - DO NOT perform calculations in SQL
        - Just fetch the necessary columns needed for Python analysis
        - For N-day rolling calculations, fetch at least N+2 days of data
        - For volatility calculations:
          * Need N+2 days minimum (N days + 1 for pct_change + 1 for initial value)
          * For 21-day volatility, fetch at least 23 days
        - For correlation calculations, fetch at least 62 days
        - For date-based queries, use 'ORDER BY date DESC LIMIT X'
        - Include the date column in results
        - Always join tables using proper date matching
        - If user asks for last N days, fetch N+1 days
        - Example: if user asks for 21 days, fetch LIMIT 22
        """
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state['user_question'])
        ]
        
        response = self.chat_gpt.invoke(messages)
        sql = self.extract_sql(response.content)
        
        new_state = {
            "user_question": state['user_question'],
            "messages": state['messages'],
            "code": sql,
            "data": None,
            "results": None
        }
        
        print(f"\n{Fore.YELLOW}SQL Generated:{Style.RESET_ALL}\n{sql}")
        
        return new_state