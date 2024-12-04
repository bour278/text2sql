from langgraph.graph import StateGraph, START, END 
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from .sql_workflow import SQLWorkflow
from .python_workflow import PythonWorkflow
import os
from pathlib import Path
import json
from typing import Any, Dict

current_dir = Path(__file__).parent
env_path = current_dir / "keys.env"
print("Looking for .env file at:", env_path)
load_dotenv(env_path)
api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded:", bool(api_key))

class State(TypedDict):
    messages: Annotated[list, "Message history"]
    complexity_score: float
    workflow_type: str
    results: Any

class MasterWorkflow:
    def __init__(self, db_path: str = "../synthetic_data.db"):
        self.db_path = db_path
        self.setup_components()
        self.setup_graph()

    def setup_components(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.chat_gpt = ChatOpenAI(
            model='gpt-4-1106-preview',
            temperature=0.7,
            api_key=api_key  # Explicitly pass the API key
        )
        self.sql_workflow = SQLWorkflow(self.db_path)
        self.python_workflow = PythonWorkflow(self.db_path)

    def setup_graph(self):
        self.graph = StateGraph(State)
        
        self.graph.add_node("evaluate_complexity", self.evaluate_complexity)
        self.graph.add_node("route_workflow", self.route_workflow)
        
        self.graph.add_edge(START, "evaluate_complexity")
        self.graph.add_edge("evaluate_complexity", "route_workflow")
        self.graph.add_edge("route_workflow", END)
        
        self.compiled_graph = self.graph.compile()

    def evaluate_complexity(self, state: State) -> Dict:
        """Evaluate query complexity using LLM."""
        query = state['messages'][-1].content
        
        messages = [
            SystemMessage(content="""You are a SQL complexity evaluator. 
            Analyze the given query and return a JSON object in this exact format:
            {
                "complexity_score": <float between 0 and 1>,
                "reasoning": "<brief explanation>",
                "recommended_workflow": "<either 'sql' or 'python'>"
            }
            
            Use these criteria:
            - SQL if it's a simple query needing only basic SQL operations (score < 0.5)
            - Python if it requires complex calculations, multiple steps, or data analysis (score >= 0.5)
            
            Return ONLY the JSON object, no other text."""),
            HumanMessage(content=query)
        ]
        
        try:
            response = self.chat_gpt.invoke(messages)
            print("Raw LLM Response:", response.content)  # Debug print
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            
            evaluation = json.loads(content)
            
            required_fields = ["complexity_score", "reasoning", "recommended_workflow"]
            if not all(field in evaluation for field in required_fields):
                raise ValueError("Missing required fields in evaluation")
            
            return {
                "complexity_score": float(evaluation["complexity_score"]),
                "workflow_type": evaluation["recommended_workflow"],
                "messages": [("assistant", f"Complexity evaluation: {evaluation['reasoning']}")],
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return {
                "complexity_score": 0.1,
                "workflow_type": "sql",
                "messages": [("assistant", "Failed to parse complexity, defaulting to SQL workflow")],
            }
        except Exception as e:
            print(f"Evaluation Error: {e}")
            return {
                "complexity_score": 0.1,
                "workflow_type": "sql",
                "messages": [("assistant", f"Error in evaluation: {str(e)}")],
            }

    def route_workflow(self, state: State) -> Dict:
        """Route to appropriate workflow based on complexity."""
        first_message = state['messages'][0]
        original_question = first_message.content if hasattr(first_message, 'content') else first_message[1]
        
        if state["workflow_type"] == "sql":
            result = self.sql_workflow.process_question(original_question)
        else:
            result = self.python_workflow.process_question(original_question)
        
        return {
            "messages": result["messages"],
            "results": result["results"]
        }

    def process_question(self, question: str) -> Dict:
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "complexity_score": 0.0,
            "workflow_type": "",
            "results": None
        }
        result = self.compiled_graph.invoke(initial_state)
        
        return {
            "workflow_type": result["workflow_type"],
            "complexity_score": result["complexity_score"],
            "results": result.get("results"),
            "error": result.get("error"),
            "messages": result.get("messages", [])
        }
