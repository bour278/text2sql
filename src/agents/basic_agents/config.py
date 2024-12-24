import os
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
        openai_model='gpt-4-1106-preview',
        temperature=0.7
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

def get_paths():
    """Get the paths for environment and database files."""
    SRC_PATH = os.path.dirname(os.getcwd())
    ENV_FPATH = os.path.join(SRC_PATH, 'agents', 'keys.env')
    SQL_DB_FPATH = os.path.join(SRC_PATH, 'src', 'synthetic_data.db')
    
    return ENV_FPATH, SQL_DB_FPATH