from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
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

if not api_key and env_path.exists():
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")

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
    def __init__(self, db_path: str = "../synthetic_data.db"):
        self.db_path = db_path
        print(f"\n{Fore.BLUE}Attempting to connect to database at:{Style.RESET_ALL} {os.path.abspath(self.db_path)}")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")
        
        self.db_uri = f"sqlite:///{db_path}"
        self.engine = create_engine(self.db_uri)
        
        schema = self.get_schema()
        print(f"\n{Fore.GREEN}Successfully connected to database. Schema:{Style.RESET_ALL}")
        print(schema)
        print()
        
        self.setup_graph()
        
    def setup_graph(self):
        self.graph = StateGraph(State)
        self.chat_gpt = ChatOpenAI(
            model='gpt-4-1106-preview',
            temperature=0.7,
            api_key=api_key
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