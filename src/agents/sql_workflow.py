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
    user_question: str  # Original user question
    messages: Annotated[list, "Message history"]
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

    def process_question(self, question: str) -> Dict:
        print(f"\n{Fore.CYAN}=== Starting SQL Workflow ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}User Question:{Style.RESET_ALL} {question}")
        
        initial_state = {
            "user_question": question,
            "messages": [],
            "current_sql": "",
            "results": None
        }
        return self.compiled_graph.invoke(initial_state)

    def parse_question(self, state: State) -> Dict:
        print(f"{Fore.CYAN}Parsing Question:{Style.RESET_ALL} {state['user_question']}")
        return {
            "user_question": state['user_question'],
            "messages": [("assistant", f"Parsed question: {state['user_question']}")],
            "current_sql": state['current_sql'],
            "results": state['results']
        }

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
            HumanMessage(content=state['user_question'])
        ]
        
        print(f"{Fore.BLUE}Question being sent to LLM:{Style.RESET_ALL} {state['user_question']}\n")
        
        response = self.chat_gpt.invoke(messages)
        
        extractor = SQLExtractor(response=response.content)
        sql = extractor.response
        
        print(f"{Fore.GREEN}Generated SQL:{Style.RESET_ALL} {sql}")
        return {
            "user_question": state['user_question'],
            "messages": state['messages'] + [("assistant", sql)],
            "current_sql": sql,
            "results": state['results']
        }

    def validate_sql(self, state: State) -> Dict:
        print(f"{Fore.YELLOW}Validating SQL...{Style.RESET_ALL}")
        
        # Here you could add actual SQL validation logic
        return {
            "user_question": state['user_question'],
            "messages": state['messages'] + [("assistant", "SQL validated")],
            "current_sql": state['current_sql'],
            "results": state['results']
        }

    def execute_sql(self, state: State) -> Dict:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(state['current_sql']))
                rows = result.fetchall()
                print(f"{Fore.MAGENTA}Query Results:{Style.RESET_ALL}")
                for row in rows:
                    print(row)
                return {
                    "user_question": state['user_question'],
                    "messages": state['messages'] + [("assistant", "Query executed successfully")],
                    "current_sql": state['current_sql'],
                    "results": rows
                }
        except Exception as e:
            error_msg = f"Error executing SQL: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            return {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", f"Error: {str(e)}")],
                "current_sql": state['current_sql'],
                "results": None
            }