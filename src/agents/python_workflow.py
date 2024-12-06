from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
        df = state['data']
        data_info = f"DataFrame columns: {list(df.columns)}"
        
        def get_code_from_llm(query_text: str) -> str:
            messages = [
                SystemMessage(content=f"""You are a Python data analysis expert specializing in financial calculations.
                Generate Python code to analyze the data and answer the question. 
                
                Available Variables:
                - 'df': DataFrame with columns {list(df.columns)}
                - 'pd': pandas module
                - 'np': numpy module
                
                For volatility calculations:
                - Use log returns: np.log(df['close'] / df['close'].shift(1))
                - Annualize by multiplying by sqrt(252)
                
                Required:
                1. Use 'df' for calculations
                2. Store final answer in 'result'
                3. For volatility: return as decimal (0.15 for 15%)"""),
                HumanMessage(content=query_text)
            ]
            
            response = self.chat_gpt.invoke(messages)
            return response.content

        # Initial attempt
        content = get_code_from_llm(query)
        code = self.extract_code(content)
        
        if not code:
            return {
                "messages": [("assistant", "Failed to generate valid Python code")],
                "code": "",
                "results": None
            }

        return {
            "messages": [("assistant", code)],
            "code": code
        }

    def execute_code(self, state: State) -> Dict:
        try:
            namespace = {
                'pd': pd,
                'np': np,
                'df': state['data'],
                'engine': self.engine  # Add engine to namespace for SQL queries
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
        """Process a question through the workflow."""
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