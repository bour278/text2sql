from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
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
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

class State(TypedDict):
    user_question: str  # Original user question
    messages: Annotated[list, "Message history"]
    data: Any
    code: str
    results: Any

class PythonWorkflow:
    def __init__(self, db_path: str = "../synthetic_data.db", use_gemini: bool = False, verbose: bool = True):
        self.db_path = db_path
        self.use_gemini = use_gemini
        self.verbose = verbose
        self.setup_components()
        self.setup_graph()

    def format_print(self, category: str, content: str, is_result: bool = False):
        if not self.verbose:
            return
            
        if is_result:
            print(f"\n{Fore.GREEN}{category}:{Style.RESET_ALL} {content}")
        else:
            print(f"\n\033[3m{Fore.CYAN}{category}:{Style.RESET_ALL}\033[3m {content}\033[0m")

    def print_state(self, state: Dict, title: str = "Current State"):
        if not self.verbose:
            return
            
        print(f"\n\033[3m{Fore.MAGENTA}{title}:{Style.RESET_ALL}")
        for key, value in state.items():
            if key in ['results', 'code', 'data']:
                print(f"{key}: {value}")  # Normal text for results, code, and data
            else:
                print(f"\033[3m{key}: {value}\033[0m")  # Italic for other state items

    def setup_components(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
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
            self.chat_gpt = ChatOpenAI(
                model='gpt-4-1106-preview',
                temperature=0.7,
                api_key=api_key
            )
        
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.setup_rag()

    def setup_rag(self):
        current_dir = Path(__file__).parent.parent
        pandas_path = current_dir / "rag-data" / "pandas-cookbook.md"
        sqlite_path = current_dir / "rag-data" / "sqlite-cookbook.md"
        
        if not pandas_path.exists():
            print(f"Warning: {pandas_path} not found")
        if not sqlite_path.exists():
            print(f"Warning: {sqlite_path} not found")
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        docs = []
        for path in [pandas_path, sqlite_path]:
            if path.exists():
                print(f"Processing {path}")
                content = path.read_text()
                header_splits = markdown_splitter.split_text(content)
                for doc in header_splits:
                    chunks = text_splitter.split_text(doc.page_content)
                    docs.extend([{"content": chunk, "source": path.stem} for chunk in chunks])
        
        if not docs:
            print("Warning: No documents were processed. Check if the cookbook files exist and contain content.")
            self.vectorstore = None
            return
        
        try:
            texts = [doc["content"] for doc in docs]
            metadatas = [{"source": doc["source"]} for doc in docs]
            print(f"Creating vector store with {len(texts)} documents")
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vectorstore = None

    def setup_graph(self):
        self.graph = StateGraph(State)

        print(f"\n{Fore.BLUE}Starting Python Workflow{Style.RESET_ALL}")
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

    def process_question(self, question: str) -> Dict:
        """Process a question through the workflow."""
        
        if self.verbose:
            print(f"{Fore.CYAN}     User Question:{Style.RESET_ALL} {question}")
        
        initial_state = {
            "user_question": question,
            "messages": [],
            "data": None,
            "code": "",
            "results": None
        }
        return self.compiled_graph.invoke(initial_state)

    def identify_data_needs(self, state: State) -> Dict:
        if self.verbose:
            self.format_print("Python Workflow - Identify Data Needs", "Starting")
            self.print_state(state)

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
        - Always join tables using proper date matching"""
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state['user_question'])
        ]
        
        response = self.chat_gpt.invoke(messages)
        sql = self.extract_sql(response.content)
        
        if self.verbose:
            self.format_print("Generated SQL for Data Fetch", sql, is_result=True)
        
        return {
            "user_question": state['user_question'],
            "messages": state['messages'],
            "code": sql,
            "data": None,
            "results": None
        }

    def fetch_data(self, state: State) -> Dict:
        if self.verbose:
            self.format_print("Python Workflow - Fetch Data", "Starting")
            self.print_state(state)
            
        try:
            sql = state['code']
            df = pd.read_sql(sql, self.engine)
            
            if self.verbose:
                self.format_print("Data Retrieved", f"\nShape: {df.shape}\nColumns: {df.columns.tolist()}", is_result=True)
                self.format_print("Sample Data", f"\n{df.to_string()}", is_result=True)
            
            return {
                "user_question": state['user_question'],
                "messages": state['messages'],
                "data": df,
                "code": state['code'],
                "results": state.get('results')
            }
            
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            self.format_print("Error", error_msg)
            return {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", error_msg)],
                "data": pd.DataFrame(),
                "code": state['code'],
                "results": None
            }

    def generate_code(self, state: State) -> Dict:
        if self.verbose:
            self.format_print("Python Workflow - Generate Code", "Starting")
            self.print_state(state)

        messages = [
            SystemMessage(content=f"""You are a Python data analysis expert specializing in financial calculations.
            Generate Python code to analyze the data and answer the question. 
            
            Available Variables:
            - 'df': DataFrame with columns {list(state['data'].columns)}
            - 'pd': pandas module
            - 'np': numpy module
            
            Required:
            1. Use 'df' for calculations
            2. Store final answer in 'result'
            3. For volatility: return as decimal (0.24 for 24%)
            4. Sort data by date ascending before calculations
            5. For rolling calculations, ensure enough data points to avoid NaN"""),
            HumanMessage(content=state['user_question'])
        ]
        
        response = self.chat_gpt.invoke(messages)
        code = self.extract_code(response.content)
        
        if self.verbose:
            self.format_print("Generated Python Code", code, is_result=True)
        
        return {
            "user_question": state['user_question'],
            "messages": state['messages'],
            "code": code,
            "data": state['data'],
            "results": state.get('results')
        }

    def execute_code(self, state: State) -> Dict:
        if self.verbose:
            self.format_print("Python Workflow - Execute Code", "Starting")
            self.print_state(state)
            
        try:
            namespace = {
                'pd': pd,
                'np': np,
                'df': state['data'],
                'engine': self.engine
            }
            
            exec(state['code'], namespace)
            result = namespace.get('result', "No result variable found")
            
            if isinstance(result, (float, np.float64)):
                formatted_result = f"{result:.4f}"
            elif isinstance(result, pd.DataFrame):
                formatted_result = "\n" + result.to_string()
            else:
                formatted_result = str(result)

            if self.verbose:
                self.format_print("Analysis Results", formatted_result, is_result=True)
            
            return {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", f"Result: {formatted_result}")],
                "code": state['code'],
                "data": state['data'],
                "results": result
            }
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            self.format_print("Error", error_msg)
            return {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", error_msg)],
                "code": state['code'],
                "data": state['data'],
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

    def get_relevant_context(self, query: str) -> str:
        if not self.vectorstore:
            return "Documentation context not available."
        
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in docs)
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "Error retrieving documentation context."