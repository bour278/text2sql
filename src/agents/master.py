from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from sender import ChatGPTSender
import yaml
import os
from langchain_openai import ChatOpenAI
import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from IPython.display import Image, display
from dotenv import load_dotenv

load_dotenv('keys.env')

class Agent:
    pass 

class State(TypedDict):
    messages : Annotated[list, add_messages] 

graph    = StateGraph(State)
chat_gpt = ChatOpenAI(
    model='gpt-4-1106-preview',
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
    )
query_gpt = lambda msg : chat_gpt.invoke(msg)


def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
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

def parse_question(self, state: dict) -> dict:
    """Parse user question and identify relevant tables and columns."""
    question = state['question']
    schema = self.db_manager.get_schema(state['uuid'])

    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a data analyst that can help summarize SQL tables and parse user questions about a database. 
            Given the question and database schema, identify the relevant tables and columns. 
            If the question is not relevant to the database or if there is not enough information to answer the question, set is_relevant to false.

            Your response should be in the following JSON format:
            {{
            "is_relevant": boolean,
            "relevant_tables": [
                {{
                    "table_name": string,
                    "columns": [string],
                    "noun_columns": [string]
                }}
            ]
            }}

            The "noun_columns" field should contain only the columns that are relevant to the question and contain nouns or names, for example, the column "Artist name" contains nouns relevant to the question "What are the top selling artists?", but the column "Artist ID" is not relevant because it does not contain a noun. Do not include columns that contain numbers.
            '''),
            ("human", "===Database schema:\n{schema}\n\n===User question:\n{question}\n\nIdentify relevant tables and columns:")
    ])

    output_parser = JsonOutputParser()
    
    response = self.llm_manager.invoke(prompt, schema=schema, question=question)
    parsed_response = output_parser.parse(response)
    return {"parsed_question": parsed_response}




def parse_question(state: State):
    query  = state['messages'][-1]
    prompt = f'Is this relevant: {query}'
    return {"messages" : [chat_gpt.send_request(prompt)]}

graph.add_node('question_parser', parse_question)
graph.add_edge(START, 'question_parser')

def break_into_components(state: State):
    query  = state['messages'][-1]
    prompt = f'Break into components: {query}'
    return {"messages" : [chat_gpt.send_request(prompt)]}

graph.add_node('break_into_components', break_into_components)
graph.add_edge("question_parser", "break_into_components")


def write_sql_code(state: State):
    query  = state['messages'][-1]
    prompt = f'Write sql code for this task: {query}'
    return {"messages" : [chat_gpt.send_request(prompt)]}

graph.add_node('write_sql_code', write_sql_code)
graph.add_edge("break_into_components", "write_sql_code")

def validate_sql(state: State):
    query  = state['messages'][-1]
    prompt = f'Check if this sql query makes sense: {query}'
    return {"messages" : [chat_gpt.send_request(prompt)]}

graph.add_node('validate_sql', validate_sql)
graph.add_edge('write_sql_code', 'validate_sql')
graph.add_edge('validate_sql', END)
compiled_graph = graph.compile()

try:
    graph_image = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_image)
    display(Image(graph_image))
except Exception as e:
    print(f"Could not display graph: {str(e)}")
    print("Make sure you have graphviz installed on your system")
