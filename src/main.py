import argparse
from agents.master import MasterWorkflow
from colorama import Fore, Style

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gemini", action="store_true", help="Use Google's Gemini model instead of OpenAI")
    args = parser.parse_args()
    
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