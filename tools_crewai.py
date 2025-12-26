# pip install crewai litellm

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
import logging

# Load environment variables
load_dotenv()

# --- Best Practice: Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set up Groq LLM ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ Error: GROQ_API_KEY not found in .env")
    exit(1)

# Configure Groq LLM for CrewAI
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
)


# --- 1. Refactored Tool: Returns Clean Data ---
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not found.
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price
    else:
        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")


# --- 2. Define the Agent with Groq LLM ---
financial_analyst_agent = Agent(
    role='Senior Financial Analyst',
    goal='Analyze stock data using provided tools and report key prices.',
    backstory="You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.",
    verbose=True,
    tools=[get_stock_price],
    allow_delegation=False,
    llm=groq_llm,  # Use Groq instead of OpenAI
)

# --- 3. Refined Task ---
analyze_aapl_task = Task(
    description=(
        "What is the current simulated stock price for Apple (ticker: AAPL)? "
        "Use the 'Stock Price Lookup Tool' to find it. "
        "If the ticker is not found, you must report that you were unable to retrieve the price."
    ),
    expected_output=(
        "A single, clear sentence stating the simulated stock price for AAPL. "
        "For example: 'The simulated stock price for AAPL is $178.15.' "
        "If the price cannot be found, state that clearly."
    ),
    agent=financial_analyst_agent,
)

# --- 4. Formulate the Crew ---
financial_crew = Crew(
    agents=[financial_analyst_agent],
    tasks=[analyze_aapl_task],
    verbose=True
)

# --- 5. Run the Crew ---
def main():
    """Main function to run the crew."""
    print("\n## Starting the Financial Crew (using Groq)...")
    print("---------------------------------")

    result = financial_crew.kickoff()

    print("\n---------------------------------")
    print("## Crew execution finished.")
    print("\nFinal Result:\n", result)

if __name__ == "__main__":
    main()
