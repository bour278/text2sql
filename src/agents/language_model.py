import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine, text
from pydantic import BaseModel, field_validator
from langchain_core.runnables.base import Runnable
import pydantic
from colorama import Fore, Style
from pygments import highlight
from pygments.lexers import SqliteConsoleLexer
from pygments.formatters import NullFormatter

load_dotenv('keys.env')

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

class SQLValidator:
    @staticmethod
    def validate_sql(sql_query: str) -> bool:
        try:
            highlight(sql_query, SqliteConsoleLexer(), NullFormatter())
            return True
        except Exception as e:
            print(f"SQL Validation Error: {e}")
            return False

class TextToSQLConverter(BaseModel):
    uri: str = "sqlite:///../synthetic_data.db"
    model: str = "gpt-4-1106-preview"
    temperature: int = 0

    class Config:
        arbitrary_types_allowed = True

    @pydantic.computed_field()
    @property
    def database(self) -> SQLDatabase:
        return SQLDatabase(engine=create_engine(self.uri))

    @pydantic.computed_field()
    @property
    def language_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    @pydantic.computed_field()
    @property 
    def run_language_model(self, message):
        return self.language_model.invoke(message)

    @pydantic.computed_field()
    @property
    def query_chain(self) -> Runnable:
        return create_sql_query_chain(
            llm=self.language_model,
            db=self.database
        )

    def generate_query(self, user_question: str):
        print(Fore.CYAN + "User Question: " + Style.RESET_ALL + user_question)
        formatted_question = f"""
        Given a table where dates are stored as TEXT in YYYY-MM-DD format,
        please write a SQL query to answer this question: {user_question}
        Important: For recent data, use ORDER BY date DESC with LIMIT instead of date comparisons.
        """
        response = self.query_chain.invoke({"question": formatted_question})
        
        print(Fore.MAGENTA + "Raw Response: " + Style.RESET_ALL + str(response))
        
        extractor = SQLExtractor(response=response)
        sql_query = extractor.response

        sql_query = sql_query.replace("SQLQuery:", "").strip().rstrip(';')

        if SQLValidator.validate_sql(sql_query):
            print(Fore.GREEN + "Generated SQL Query: " + Style.RESET_ALL + sql_query)
            return sql_query
        else:
            print("Invalid SQL Query.")
            return ""

    def execute_sql(self, sql_query: str):
        with self.database._engine.connect() as connection:
            result = connection.execute(text(sql_query))
            return result.fetchall()

if __name__ == "__main__":
    text_to_sql = TextToSQLConverter()

    questions = [
        "What are the closing prices for all dates in the ohlc table?",
        "What is the minimum price in the ohlc table?",
        "What is the maximum price in the ohlc table?",
        "What is the average open price over the last 10 days in the ohlc table ORDER BY date DESC",
        "What is the stock volatility over the last 21 days from the ohlc table?"
    ]

    for user_question in questions:
        sql_query = text_to_sql.generate_query(user_question)

        if sql_query:
            results = text_to_sql.execute_sql(sql_query)
            print(Fore.YELLOW + "Query Results: " + Style.RESET_ALL)
            for row in results:
                print(row)