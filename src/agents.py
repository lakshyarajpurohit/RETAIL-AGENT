import os
import sys
import io
import re
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from src.tools import search_products, get_product, get_order, evaluate_return
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm_instance = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

# ── Agent 1: Personal Shopper ─────────────────────────────────────────────────
shopper_agent = Agent(
    role="Personal Shopper",
    goal=(
        "Help customers find the perfect product by searching the catalog intelligently. "
        "Always use search_products for style/occasion queries. "
        "Always use get_product when a specific Product ID (P-prefix) is mentioned. "
        "Never make up product details — only use tool results."
    ),
    backstory=(
        "You are an expert luxury fashion consultant with deep knowledge of the store's catalog. "
        "You listen carefully to what the customer wants — size, budget, occasion, style — "
        "and find the best matches. You always justify your recommendations by referencing "
        "actual product data: price, tags, stock, and bestseller score. "
        "You never recommend a product without checking availability first."
    ),
    tools=[search_products, get_product],
    llm=llm_instance,
    max_iter=5,
    verbose=True,
    allow_delegation=False
)

# ── Agent 2: Customer Support Specialist ──────────────────────────────────────
support_agent = Agent(
    role="Customer Support Specialist",
    goal=(
        "Accurately determine return or exchange eligibility by looking up the order "
        "and applying the correct policy rules. "
        "Always call evaluate_return for any return or exchange request. "
        "If the order does not exist, say so clearly. Never guess."
    ),
    backstory=(
        "You are a precise customer operations analyst. You handle return and exchange requests "
        "by strictly following company policy. You always look up the order first using get_order, "
        "then call evaluate_return to get the official decision. "
        "You explain the outcome clearly: the order date, how many days have passed, "
        "which policy rule applied, and whether the customer is approved or denied. "
        "You never fabricate decisions — every answer comes from a tool call."
    ),
    tools=[get_order, evaluate_return],
    llm=llm_instance,
    max_iter=5,
    verbose=True,
    allow_delegation=False
)


# ── Log Capture Helpers ───────────────────────────────────────────────────────

def _strip_ansi(text):
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def _parse_thought_log(raw_log):
    """
    Parse CrewAI verbose stdout into clean thought steps.
    Returns list of dicts: [{type, content}, ...]
    """
    steps = []
    raw_log = _strip_ansi(raw_log)

    patterns = [
        (r"Thought\s*:",        "🧠 Thought"),
        (r"Action\s*:",         "⚡ Action — Tool Selected"),
        (r"Action Input\s*:",   "📥 Tool Input"),
        (r"Observation\s*:",    "📦 Tool Result"),
        (r"Final Answer\s*:",   "✅ Final Answer"),
    ]

    matches = []
    for pattern, label in patterns:
        for m in re.finditer(pattern, raw_log, re.IGNORECASE):
            matches.append((m.start(), m.end(), label))

    if not matches:
        cleaned = raw_log.strip()
        if cleaned:
            steps.append({"type": "📋 Agent Log", "content": cleaned})
        return steps

    matches.sort(key=lambda x: x[0])

    for i, (start, end, label) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(raw_log)
        content = raw_log[end:next_start].strip()
        if content:
            steps.append({"type": label, "content": content})

    return steps


def _run_crew_with_logs(agent, task_obj):
    """
    Run a Crew, capture stdout verbose logs, return (response, thought_steps).
    """
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        crew = Crew(
            agents=[agent],
            tasks=[task_obj],
            process=Process.sequential,
            verbose=True
        )
        result = crew.kickoff()
        response = str(result)
    finally:
        sys.stdout = old_stdout

    raw_log = captured.getvalue()
    thought_steps = _parse_thought_log(raw_log)
    return response, thought_steps


# ── Task Factories ────────────────────────────────────────────────────────────

def _make_shopping_task(user_input):
    return Task(
        description=(
            f"A customer has made the following request: '{user_input}'\n\n"
            "Your job:\n"
            "1. Parse what they want: occasion, style tags, size, budget, sale preference.\n"
            "2. Call search_products with the appropriate filters (size as a string, e.g. '8').\n"
            "3. Present the top results clearly, listing out the product ID, title, price, tags, and stock for the requested size in the Final Answer.\n"
            "4. Explain WHY each product fits their request — reference tags, price, stock, score.\n"
            "5. Never invent product details. Only use what the tools return."
        ),
        expected_output=(
            "A warm, clear shopping recommendation listing matching products with their "
            "names, prices, tags, sale status, and stock for the requested size. "
            "You MUST include the specific product data (ID, Title, Price, and Stock) in the final response."
        ),
        agent=shopper_agent
    )


def _make_support_task(user_input):
    return Task(
        description=(
            f"A customer has submitted the following support request: '{user_input}'\n\n"
            "Your job:\n"
            "1. Extract the Order ID from the request (starts with O, e.g. O0005).\n"
            "2. Call get_order to retrieve the order details.\n"
            "3. Call evaluate_return to get the official return eligibility decision.\n"
            "4. Present the decision clearly: approved or denied, with the reason.\n"
            "5. State the order date, how many days have passed, and which policy rule was applied.\n"
            "6. If the order does not exist, inform the customer clearly and do not guess.\n"
            "7. Never make up a return decision — only use what evaluate_return returns."
        ),
        expected_output=(
            "A professional support response stating: the order details, "
            "days since purchase, the policy rule applied, "
            "and whether the return/exchange is APPROVED or DENIED with full reasoning."
        ),
        agent=support_agent
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_shopping_crew(user_input):
    """Returns (response_text, thought_steps_list)."""
    return _run_crew_with_logs(shopper_agent, _make_shopping_task(user_input))


def run_support_crew(user_input):
    """Returns (response_text, thought_steps_list)."""
    return _run_crew_with_logs(support_agent, _make_support_task(user_input))