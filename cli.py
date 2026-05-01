# main.py is designed for a Command Line Interface (CLI) where it uses input() and print() in a terminal. Your app.py is the graphical Streamlit interface.

from dotenv import load_dotenv
from crewai import Crew, Process
from src.agents import get_shopping_task, get_support_task, shopper_agent, support_agent

load_dotenv()

def run_assistant():
    print("--- Retail AI Assistant (Simulation: April 29, 2026) ---")
    user_query = input("How can I help you today? ")

    # Basic intent routing
    if "return" in user_query.lower() or "order" in user_query.lower():
        task = get_support_task(user_query)
        selected_agent = support_agent
    else:
        task = get_shopping_task(user_query)
        selected_agent = shopper_agent

    # Create the Crew
    crew = Crew(
        agents=[selected_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    print("\n\nFINAL RESPONSE:\n", result)

if __name__ == "__main__":
    run_assistant()