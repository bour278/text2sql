from langgraph.graph import StateGraph, START, END 
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from .sql_workflow import SQLWorkflow
from .python_workflow import PythonWorkflow
import os
from pathlib import Path
import json
from typing import Any, Dict
from colorama import init, Fore, Style

init()

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
    messages: Annotated[list, "Message history"]
    complexity_score: float
    workflow_type: str
    results: Any

class MasterWorkflow:
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
                print(f"{key}: {value}")  # Normal text for results and code
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
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.chat_gpt = ChatOpenAI(
                model='gpt-4-1106-preview',
                temperature=0.7,
                api_key=api_key
            )
        
        self.sql_workflow = SQLWorkflow(self.db_path, self.use_gemini, self.verbose)
        self.python_workflow = PythonWorkflow(self.db_path, self.use_gemini, self.verbose)

    def setup_graph(self):
        self.graph = StateGraph(State)
        
        if self.verbose:
            print(f"\n{Fore.BLUE}=== Activating SQL Agent ==={Style.RESET_ALL}")
        
        self.graph.add_node("evaluate_complexity", self.evaluate_complexity)
        self.graph.add_node("route_workflow", self.route_workflow)
        
        self.graph.add_edge(START, "evaluate_complexity")
        self.graph.add_edge("evaluate_complexity", "route_workflow")
        self.graph.add_edge("route_workflow", END)
        
        self.compiled_graph = self.graph.compile()

    def evaluate_complexity(self, state: State) -> Dict:
        """Evaluate query complexity using LLM."""
        query = state['messages'][-1].content
        
        if self.verbose:
            self.format_print("Evaluating Query Complexity", query)
            self.print_state(state)
        
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
            
            if self.verbose:
                self.format_print("LLM Evaluation", response.content)
                
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            
            evaluation = json.loads(content)
            
            required_fields = ["complexity_score", "reasoning", "recommended_workflow"]
            if not all(field in evaluation for field in required_fields):
                raise ValueError("Missing required fields in evaluation")
            
            result = {
                "complexity_score": float(evaluation["complexity_score"]),
                "workflow_type": evaluation["recommended_workflow"],
                "messages": [("assistant", f"Query: {query}", f"Evaluation: {evaluation['reasoning']}")],
            }

            if self.verbose:
                self.format_print("Complexity Analysis Result", 
                                f"Score: {result['complexity_score']}, Workflow: {result['workflow_type']}", 
                                is_result=True)
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            error_msg = f"Evaluation Error: {str(e)}"
            self.format_print("Error", error_msg)
            return {
                "complexity_score": 0.1,
                "workflow_type": "sql",
                "messages": [("assistant", "Failed to parse complexity, defaulting to SQL workflow")],
            }

    def route_workflow(self, state: State) -> Dict:
        """Route to appropriate workflow based on complexity."""
        if self.verbose:
            self.format_print("Routing Workflow", f"Selected: {state['workflow_type']}")
            self.print_state(state)

        first_message = state['messages'][0]
        original_question = first_message.content if hasattr(first_message, 'content') else first_message[1]
        
        if state["workflow_type"] == "sql":
            result = self.sql_workflow.process_question(original_question)
        else:
            result = self.python_workflow.process_question(original_question)
        
        if self.verbose:
            self.format_print("Final Results", str(result.get("results", "No results")), is_result=True)
        
        return {
            "messages": result["messages"],
            "results": result["results"]
        }

    def process_question(self, question: str) -> Dict:
        if self.verbose:
            self.format_print("Processing New Question", question)
            
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "complexity_score": 0.0,
            "workflow_type": "",
            "results": None
        }
        
        result = self.compiled_graph.invoke(initial_state)
        
        if self.verbose:
            self.format_print("Process Complete", "", is_result=True)
        
        return {
            "workflow_type": result["workflow_type"],
            "complexity_score": result["complexity_score"],
            "results": result.get("results"),
            "error": result.get("error"),
            "messages": result.get("messages", [])
        }
