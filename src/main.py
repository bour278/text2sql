import argparse
import os
from agents.basic_agents.agent1 import run_agent1
from agents.basic_agents.agent2 import run_agent2
from agents.basic_agents.agent3 import run_agent3
from agents.master import MasterWorkflow
from colorama import Fore, Style
from dotenv import load_dotenv
from agents.basic_agents.config import Config, get_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gemini", action="store_true", help="Use Google's Gemini model instead of OpenAI")
    parser.add_argument("-basic", type=int, choices=[1, 2, 3], help="Specify the agent number to run (1, 2, or 3)")
    args = parser.parse_args()
    
    load_dotenv()

    ENV_FPATH, SQL_DB_FPATH = get_paths()

    print(f'env fpath: {ENV_FPATH}')
    print(f'sql db fpath: {SQL_DB_FPATH}')

    config = Config(
        env_fpath=ENV_FPATH,
        sql_db_fpath=SQL_DB_FPATH  
    )

    if args.basic:
        agent_number = args.basic
        
        if agent_number == 1:
            user_query = '''Find the days where stock price movement := close-open was more 
            than 2 standard deviations from the mean'''
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Running Agent {agent_number} for Query:{Style.RESET_ALL} {user_query}")
            result = run_agent1(user_query, config)

        elif agent_number == 2:
            user_query = '''Find the days where stock price movement := close-open was more 
            than 2 standard deviations from the mean'''
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Running Agent {agent_number} for Query:{Style.RESET_ALL} {user_query}")
            result = run_agent2(user_query, config)

        elif agent_number == 3:
            user_query = '''For s in [1,2], of the days where the stock price 
            movement := close - open was more than s std deviations from the mean, look at the distribution 
            of 7yr tsy yield - 5yr tsy yield. To visualize this, assume access to matplotlib.pyplot as plt and 
            make a 2 plots, the left where s = 1 and a histogram of 7yr - 5yr tsy yields with lines at 25 percentile,
            50th percentile, 75th, and then right same with s=2'''
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Running Agent {agent_number} for Query:{Style.RESET_ALL} {user_query}")
            result = run_agent3(user_query, config)

        print(f"{Fore.MAGENTA}Results from Agent {agent_number}:{Style.RESET_ALL}")
        print(result)

    else:
        # Existing code for running the MasterWorkflow
        workflow = MasterWorkflow(db_path="synthetic_data.db", use_gemini=args.gemini)
        
        questions = [
            # Simple SQL queries
            "What are the closing prices for last 30 dates in the ohlc table?",
            "What is the minimum price in the ohlc table?",
            "What is the maximum price in the ohlc table?",
            
            # Moderate complexity
            "What is the average open price over the last 10 days in the ohlc table ORDER BY date DESC",
            
            # Complex analysis requiring Python
            "What is the stock volatility over the last 21 days from the ohlc table?",
            "Calculate the correlation between 7 year treasury yields and close stock prices over the last 30 days",
            "Find the days where the stock price movement was more than 2 standard deviations from the mean",
        ]
        
        for question in questions:
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Question:{Style.RESET_ALL} {question}")
            print(f"{Fore.YELLOW}Using Model:{Style.RESET_ALL} {'Gemini' if args.gemini else 'OpenAI'}")
            
            result = workflow.process_question(question)
            
            print(f"{Fore.MAGENTA}Workflow Type:{Style.RESET_ALL} {result['workflow_type']}")
            print(f"{Fore.MAGENTA}Complexity Score:{Style.RESET_ALL} {result['complexity_score']:.2f}")
            
            if result.get('results') is not None:
                print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
                if isinstance(result['results'], (list, tuple)):
                    for row in result['results']:
                        print(row)
                else:
                    print(result['results'])
            
            if result.get('error'):
                print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {result['error']}")

if __name__ == "__main__":
    main() 