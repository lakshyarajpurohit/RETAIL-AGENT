from dotenv import load_dotenv
from src.agents import run_shopping_crew, run_support_crew

load_dotenv()

SUPPORT_KEYWORDS = [
    "return", "refund", "exchange", "cancel", "order",
    "bought", "purchase", "receipt", "policy", "eligible"
]

def classify_intent(text: str) -> str:
    text_lower = text.lower()
    import re
    if re.search(r'\bo\d{3,}\b', text_lower):
        return "support"
    if any(kw in text_lower for kw in SUPPORT_KEYWORDS):
        return "support"
    return "shopping"

def print_thought_process(thought_steps: list):
    if not thought_steps:
        return
    print("\n" + "="*60)
    print("  AGENT THOUGHT PROCESS")
    print("="*60)
    for i, step in enumerate(thought_steps, 1):
        print(f"\n[Step {i}] {step['type']}")
        print("-" * 40)
        print(step['content'])
    print("="*60)

def run_assistant():
    print("\n--- Retail AI Assistant (Simulation: May 01, 2026) ---")
    print("Type 'quit' to exit\n")

    while True:
        user_query = input("You: ").strip()

        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        intent = classify_intent(user_query)
        agent_label = "Personal Shopper" if intent == "shopping" else "Customer Support"
        print(f"\n[Agent: {agent_label}] Thinking...\n")

        try:
            if intent == "support":
                response, thought_steps = run_support_crew(user_query)
            else:
                response, thought_steps = run_shopping_crew(user_query)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print("\nFINAL RESPONSE:")
        print("-" * 60)
        print(response)

        # Show thought process
        show = input("\nShow agent thought process? (y/n): ").strip().lower()
        if show == "y":
            print_thought_process(thought_steps)

        print()

if __name__ == "__main__":
    run_assistant()