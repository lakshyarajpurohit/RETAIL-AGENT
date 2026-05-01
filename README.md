# 🛍️ Retail AI Assistant

An agentic AI system simulating a **Personal Shopper** and **Customer Support Specialist** for a fashion retail boutique — built with CrewAI, Groq LLaMA 3.3, and Streamlit.

---

## Features

- **Two specialized AI agents** — each with its own role, tools, and decision logic
- **True tool-calling** — agents decide which tools to use; no hardcoded routing
- **Deterministic return policy engine** — clearance, sale, vendor exceptions all handled in code
- **Thought process viewer** — see the agent's full reasoning chain after every response
- **Streamlit UI** — chat interface + live inventory + order history tabs
- **CLI mode** — terminal interface for quick testing

---

## Project Structure

```
retail-ai-assistant/
│
├── app.py                  ← Streamlit web UI (main entry point)
├── cli.py                  ← Command-line interface
├── requirements.txt
├── .env                    ← Your API keys (create this)
├── .env.example            ← Template for .env
│
├── data/
│   ├── product_inventory.csv
│   ├── orders.csv
│   └── policy.txt
│
└── src/
    ├── __init__.py
    ├── data_loader.py      ← CSV loading + data access methods
    ├── tools.py            ← 4 LangChain tools (search, get_product, get_order, evaluate_return)
    └── agents.py           ← CrewAI agents, tasks, thought log capture
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd retail-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then open `.env` and add your key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

---

## Running the App

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### CLI

```bash
python cli.py
```

---

## Demo Scenarios

Use these to test all parts of the assignment:

| # | Type | Query |
|---|------|-------|
| 1 | Shopping | `I need a modest evening gown under $300 in size 8, preferably on sale` |
| 2 | Shopping | `Find me a sale cocktail dress in size 12 under $250` |
| 3 | Support | `Can I return order O0005? It doesn't fit` |
| 4 | Support | `I want to exchange order O0003 for a different size` |
| 5 | Edge case | `Can I return order O9999?` |

---

## The 4 Tools

| Tool | When It's Called |
|------|-----------------|
| `search_products` | Customer describes style, occasion, size, or budget |
| `get_product` | Customer mentions a specific Product ID (e.g. P0042) |
| `get_order` | Customer references an Order ID (e.g. O0005) |
| `evaluate_return` | Customer asks about returning or exchanging an item |

---

## Return Policy (Simulation Date: May 1, 2026)

| Item Type | Window | Refund Type |
|-----------|--------|-------------|
| Normal | 14 days | Full refund |
| Sale | 7 days | Store credit only |
| Clearance | — | Final sale, no return |
| Nocturne (vendor) | 21 days | Full refund |
| Aurelia Couture (vendor) | 14 days | Exchange only |

---

## Requirements

- Python 3.10+
- Groq API key (free tier works)
- Dependencies listed in `requirements.txt`
