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
    def __init__(self, db_path: str = "../synthetic_data.db", use_gemini: bool = False):
        self.db_path = db_path
        self.use_gemini = use_gemini
        self.setup_components()
        self.setup_graph()

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
        pandas_path = Path("src/rag-data/pandas-cookbook.md")
        sqlite_path = Path("src/rag-data/sqlite-cookbook.md")
        
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
        print(f"\n{Fore.CYAN}=== Starting Python Workflow ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}User Question:{Style.RESET_ALL} {question}")
        
        initial_state = {
            "user_question": question,
            "messages": [],
            "data": None,
            "code": "",
            "results": None
        }
        return self.compiled_graph.invoke(initial_state)

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
        - For volatility calculations, fetch at least 30 days of data
        - For correlation calculations, fetch at least 60 days of data to account for rolling windows
        - For date-based queries, use 'ORDER BY date DESC LIMIT X'
        - Include the date column in results
        - Always join tables using proper date matching
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

    def fetch_data(self, state: State) -> Dict:
        print(f"\n{Fore.GREEN}=== FETCH DATA ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Current State:{Style.RESET_ALL}")
        for key, value in state.items():
            print(f"  {key}: {value}")
        
        try:
            # Use the SQL generated from identify_data_needs
            sql = state['code']
            print(f"\n{Fore.GREEN}Executing SQL:{Style.RESET_ALL}\n{sql}")
            
            df = pd.read_sql(sql, self.engine)
            
            new_state = {
                "user_question": state['user_question'],
                "messages": state['messages'],
                "data": df,
                "code": state['code'],
                "results": state.get('results')
            }
            
            print(f"\n{Fore.GREEN}Query Results:{Style.RESET_ALL}")
            print(f"Data Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Data Head:\n{df.head()}")
            print(f"\n{Fore.GREEN}New State:{Style.RESET_ALL}")
            for key, value in new_state.items():
                print(f"  {key}: {value}")
            
            return new_state
            
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            print(f"{Fore.RED}Error:{Style.RESET_ALL} {error_msg}")
            
            error_state = {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", error_msg)],
                "data": pd.DataFrame(),
                "code": state['code'],
                "results": None
            }
            
            print(f"\n{Fore.RED}Error State:{Style.RESET_ALL}")
            for key, value in error_state.items():
                print(f"  {key}: {value}")
            
            return error_state

    def generate_code(self, state: State) -> Dict:
        print(f"\n{Fore.BLUE}=== GENERATE CODE ==={Style.RESET_ALL}")
        
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
        
        new_state = {
            "user_question": state['user_question'],
            "messages": state['messages'],
            "code": code,
            "data": state['data'],
            "results": state.get('results')
        }
        
        print(f"{Fore.BLUE}Output State:{Style.RESET_ALL}")
        print(f"Generated Code:\n{code}")
        
        return new_state

    def execute_code(self, state: State) -> Dict:
        print(f"\n{Fore.MAGENTA}=== EXECUTE CODE ==={Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Input State:{Style.RESET_ALL}")
        print(f"User Question: {state['user_question']}")
        print(f"Code to Execute:\n{state['code']}")
        
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

            new_state = {
                "user_question": state['user_question'],
                "messages": state['messages'] + [("assistant", f"Result: {formatted_result}")],
                "code": state['code'],
                "data": state['data'],
                "results": result
            }
            
            print(f"{Fore.MAGENTA}Output State:{Style.RESET_ALL}")
            print(f"Execution Result: {formatted_result}")
            
            return new_state
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            
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