import streamlit as st
import re
from dotenv import load_dotenv
from src.tools import data_manager
from src.agents import run_shopping_crew, run_support_crew

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail AI Assistant",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #1f77b4; }

    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e0e0e0;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #333333 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
        font-weight: bold;
    }

    /* Agent badge */
    .agent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .badge-shopper { background: #d4edda; color: #155724; }
    .badge-support { background: #cce5ff; color: #004085; }

    /* Thought step cards inside the expander */
    .thought-step {
        background: #ffffff;
        border-left: 4px solid #007bff;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 13px;
    }
    .thought-step.thought  { border-color: #6f42c1; }
    .thought-step.action   { border-color: #fd7e14; }
    .thought-step.input    { border-color: #20c997; }
    .thought-step.result   { border-color: #0dcaf0; }
    .thought-step.answer   { border-color: #198754; }
    .step-label {
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ── Intent classifier ─────────────────────────────────────────────────────────
SUPPORT_KEYWORDS = [
    "return", "refund", "exchange", "cancel", "order", "bought",
    "purchase", "receipt", "policy", "eligible"
]

def classify_intent(text: str) -> str:
    text_lower = text.lower()
    if re.search(r'\bo\d{3,}\b', text_lower):
        return "support"
    if any(kw in text_lower for kw in SUPPORT_KEYWORDS):
        return "support"
    return "shopping"


# ── Thought process renderer ──────────────────────────────────────────────────
STEP_CSS_CLASS = {
    "🧠 Thought":              "thought",
    "⚡ Action — Tool Selected": "action",
    "📥 Tool Input":           "input",
    "📦 Tool Result":          "result",
    "✅ Final Answer":          "answer",
    "📋 Agent Log":            "result",
}

def render_thought_expander(thought_steps: list, intent: str):
    """Render the collapsible 'Agent Thought Process' section after each reply."""
    if not thought_steps:
        return

    agent_label = "🛍️ Personal Shopper" if intent == "shopping" else "🎧 Customer Support"
    step_count = len(thought_steps)

    with st.expander(
        f"🔍 Agent Thought Process — {agent_label}  "
        f"({step_count} step{'s' if step_count != 1 else ''})",
        expanded=False
    ):
        st.caption(
            "This shows exactly how the agent reasoned — what it thought, "
            "which tool it chose, what it sent, what it got back, and how it formed its answer."
        )
        st.divider()

        for i, step in enumerate(thought_steps, 1):
            step_type = step["type"]
            content   = step["content"]
            css_class = STEP_CSS_CLASS.get(step_type, "result")

            # Use Streamlit native components for clean rendering
            col_icon, col_body = st.columns([0.08, 0.92])

            with col_icon:
                st.markdown(f"**{i}**")

            with col_body:
                st.markdown(f"**{step_type}**")

                # Tool Result and Tool Input get a code block — raw data looks cleaner
                if step_type in ("📦 Tool Result", "📥 Tool Input"):
                    st.code(content, language="text")
                else:
                    st.markdown(content)

            if i < step_count:
                st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛠️ Admin Dashboard")

    inv_data = data_manager.get_inventory()
    ord_data = data_manager.orders

    col1, col2 = st.columns(2)
    col1.metric("Products", len(inv_data))
    col2.metric("Orders", len(ord_data))

    st.info("**Simulation Date:** May 01, 2026")

    with st.expander("📄 Return Policy"):
        st.write("**Normal items:** 14 days — Full refund")
        st.write("**Sale items:** 7 days — Store credit only")
        st.write("**Nocturne:** 21 days — Extended window")
        st.write("**Aurelia Couture:** Exchange only, 14 days")
        st.write("**Clearance:** Final sale — No returns")

    csv_bytes = inv_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Inventory CSV",
        data=csv_bytes,
        file_name='inventory.csv',
        mime='text/csv'
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("💡 Try these:")
    st.caption("• I need a modest dress under $300 in size 8")
    st.caption("• Can I return order O0005?")
    st.caption("• Show me product P0042")
    st.caption("• Return order O9999 (edge case)")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🛍️ Retail AI Assistant")
st.caption("Powered by CrewAI · Groq LLaMA 3.3 · Two specialized agents")

tab1, tab2, tab3 = st.tabs(["💬 AI Assistant", "📊 Inventory Explorer", "📦 Order History"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Personal Shopper & Customer Support")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render conversation history (responses + their thought dropdowns)
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                badge_class = "badge-shopper" if message.get("agent_type") == "shopping" else "badge-support"
                label = "🛍️ Personal Shopper" if message.get("agent_type") == "shopping" else "🎧 Customer Support"
                st.markdown(f'<span class="agent-badge {badge_class}">{label}</span>', unsafe_allow_html=True)

            st.markdown(message["content"])

            # Re-render thought expander from saved steps in history
            if message["role"] == "assistant" and message.get("thought_steps"):
                render_thought_expander(message["thought_steps"], message.get("agent_type", "shopping"))

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask me anything — shop, returns, orders…"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            intent = classify_intent(prompt)

            badge_class   = "badge-shopper" if intent == "shopping" else "badge-support"
            label         = "🛍️ Personal Shopper" if intent == "shopping" else "🎧 Customer Support"
            spinner_msg   = "Searching catalog…" if intent == "shopping" else "Checking order and applying policy…"

            st.markdown(f'<span class="agent-badge {badge_class}">{label}</span>', unsafe_allow_html=True)

            with st.spinner(spinner_msg):
                try:
                    if intent == "support":
                        response, thought_steps = run_support_crew(prompt)
                    else:
                        response, thought_steps = run_shopping_crew(prompt)
                except Exception as e:
                    response = (
                        f"⚠️ The agent encountered an error: {str(e)}\n\n"
                        "Please check your GROQ_API_KEY and try again."
                    )
                    thought_steps = []

            # Decision banner
            if "APPROVED" in response:
                st.success("✅ Return / Exchange Approved")
            elif "DENIED" in response:
                st.error("🛑 Return Request Denied")

            # Main answer
            st.markdown(response)

            # ── Thought process dropdown ──────────────────────────────────
            render_thought_expander(thought_steps, intent)

        # Save full message including thought steps for history re-render
        st.session_state.messages.append({
            "role":         "assistant",
            "content":      response,
            "agent_type":   intent,
            "thought_steps": thought_steps,
        })

# ── Tab 2: Inventory ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Real-Time Inventory Status")
    search_inv = st.text_input("Filter by product ID, name, or vendor:", key="inv_search")

    df_inv = inv_data.copy()
    df_inv['total_stock'] = df_inv['stock_per_size'].apply(
        lambda x: sum(x.values()) if isinstance(x, dict) else 0
    )
    display_cols = ['product_id', 'title', 'vendor', 'price', 'compare_at_price',
                    'tags', 'is_sale', 'is_clearance', 'bestseller_score', 'total_stock']
    df_inv = df_inv[[c for c in display_cols if c in df_inv.columns]]

    if search_inv:
        mask = (
            df_inv['product_id'].str.contains(search_inv, case=False, na=False) |
            df_inv['title'].str.contains(search_inv, case=False, na=False) |
            df_inv['vendor'].str.contains(search_inv, case=False, na=False)
        )
        df_inv = df_inv[mask]

    st.dataframe(df_inv, hide_index=True, use_container_width=True)
    st.caption(f"Showing {len(df_inv)} of {len(inv_data)} products")

# ── Tab 3: Orders ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Customer Order History")
    search_ord = st.text_input("Search by order ID or customer ID:", key="ord_search")

    df_ord = ord_data.copy()
    if search_ord:
        mask = (
            df_ord['order_id'].astype(str).str.contains(search_ord, case=False, na=False) |
            df_ord['customer_id'].astype(str).str.contains(search_ord, case=False, na=False)
        )
        df_ord = df_ord[mask]

    st.dataframe(df_ord, hide_index=True, use_container_width=True)
    st.caption(f"Showing {len(df_ord)} of {len(ord_data)} orders")