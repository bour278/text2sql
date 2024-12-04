from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import inspect
from typing import Type
import os
from colorama import Fore, Style, init
from .master import MasterWorkflow, State as MasterState
from .python_workflow import PythonWorkflow, State as PythonState
from .sql_workflow import SQLWorkflow, State as SQLState

# Initialize colorama
init(autoreset=True)

def extract_graph_structure(workflow_class: Type):
    """Extract graph structure from a workflow class by inspecting its setup_graph method."""
    print(f"{Fore.CYAN}Extracting graph structure for {workflow_class.__name__}{Style.RESET_ALL}")
    
    setup_graph_source = inspect.getsource(workflow_class.setup_graph)
    
    nodes = []
    edges = []
    
    for line in setup_graph_source.split('\n'):
        if 'add_node' in line:
            node_name = line.split('"')[1]
            nodes.append(node_name)
            print(f"{Fore.GREEN}Found node:{Style.RESET_ALL} {node_name}")
        elif 'add_edge' in line:
            parts = line.split('(')[1].split(')')[0].split(',')
            from_node = parts[0].strip().strip('"')
            to_node = parts[1].strip().strip('"')
            edges.append((from_node, to_node))
            print(f"{Fore.BLUE}Found edge:{Style.RESET_ALL} {from_node} -> {to_node}")
    
    return nodes, edges

def visualize_workflow(workflow_class: Type, state_class: Type, filename: str):
    """Create visualization for a workflow class."""
    print(f"\n{Fore.YELLOW}Creating visualization for {workflow_class.__name__}{Style.RESET_ALL}")
    
    nodes, edges = extract_graph_structure(workflow_class)
    
    graph = StateGraph(state_class)
    
    for node in nodes:
        graph.add_node(node, lambda x: x)
    
    for from_node, to_node in edges:
        source = START if from_node == 'START' else from_node
        target = END if to_node == 'END' else to_node
        graph.add_edge(source, target)
    
    print(f"{Fore.MAGENTA}Compiling graph...{Style.RESET_ALL}")
    compiled_graph = graph.compile()
    
    try:
        graph_image = compiled_graph.get_graph().draw_mermaid_png()
        
        output_dir = 'workflow_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{filename}.png")
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        print(f"{Fore.GREEN}Graph saved as:{Style.RESET_ALL} {output_path}")
        display(Image(graph_image))
    except Exception as e:
        print(f"{Fore.RED}Error creating/displaying graph for {filename}:{Style.RESET_ALL} {str(e)}")
        print(f"{Fore.YELLOW}Make sure you have graphviz installed on your system{Style.RESET_ALL}")

def visualize_all_workflows():
    """Create and save visualizations for all workflows."""
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Creating workflow visualizations...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Visualizing Master Workflow...{Style.RESET_ALL}")
    visualize_workflow(MasterWorkflow, MasterState, "master_workflow")
    
    print(f"\n{Fore.YELLOW}Visualizing Python Workflow...{Style.RESET_ALL}")
    visualize_workflow(PythonWorkflow, PythonState, "python_workflow")
    
    print(f"\n{Fore.YELLOW}Visualizing SQL Workflow...{Style.RESET_ALL}")
    visualize_workflow(SQLWorkflow, SQLState, "sql_workflow")
    
    print(f"\n{Fore.GREEN}All visualizations created successfully!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    visualize_all_workflows()