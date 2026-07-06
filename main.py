import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
import time

load_dotenv()

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Comparison Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0a10; color: #e8e8f0; }

    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a2a4a;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle { color: #7777aa; font-size: 0.95rem; margin-top: 0.4rem; }

    /* Category Tabs */
    .cat-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .cat-pill {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid #2a2a4a;
        background: #1a1a2e;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .cat-pill.active { background: #7c3aed; color: white; border-color: #7c3aed; }

    /* Model Cards */
    .model-card {
        background: #13131f;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        height: 100%;
    }
    .model-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #2a2a4a;
    }
    .model-badge {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .badge-groq    { background: #0d2a1a; color: #34d399; border: 1px solid #34d399; }
    .badge-mistral { background: #1a0d2a; color: #a78bfa; border: 1px solid #a78bfa; }
    .badge-gemini  { background: #0d1a2a; color: #60a5fa; border: 1px solid #60a5fa; }

    .model-name { font-size: 0.9rem; font-weight: 600; color: #c8c8e0; }

    .response-box {
        background: #0a0a14;
        border: 1px solid #1e1e35;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        font-size: 0.88rem;
        line-height: 1.75;
        color: #b8b8d0;
        min-height: 120px;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .response-meta {
        display: flex;
        gap: 12px;
        margin-top: 0.6rem;
        font-size: 0.72rem;
        color: #4a4a6a;
        font-family: 'JetBrains Mono', monospace;
    }
    .meta-chip {
        background: #1a1a2e;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #2a2a4a;
    }

    .error-box {
        background: #1a0a0a;
        border: 1px solid #5a1a1a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #f87171;
        font-size: 0.83rem;
    }

    /* Chat bubble */
    .chat-user {
        background: #1e1a3a;
        border: 1px solid #3a2a6a;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        color: #c8b8f0;
        font-size: 0.9rem;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-ai {
        background: #13131f;
        border: 1px solid #2a2a4a;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        color: #b8b8d0;
        font-size: 0.88rem;
        max-width: 85%;
    }
    .chat-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }

    /* Stats */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: #13131f;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #7c3aed; font-family: 'JetBrains Mono', monospace; }
    .stat-label  { font-size: 0.7rem; color: #5a5a7a; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Fastest badge */
    .fastest-badge {
        display: inline-block;
        background: #0d2a1a;
        color: #34d399;
        border: 1px solid #34d399;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Input */
    .stTextArea textarea {
        background: #13131f !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 10px !important;
        color: #e8e8f0 !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.15) !important;
    }
    .stSelectbox > div > div {
        background: #13131f !important;
        border: 1px solid #2a2a4a !important;
        color: #e8e8f0 !important;
        border-radius: 8px !important;
    }

    hr { border-color: #2a2a4a !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #4a4a7a;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e1e35;
    }
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #5a5a8a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #13131f;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #2a2a4a;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6666aa !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #7c3aed !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "chat_groq"     not in st.session_state: st.session_state.chat_groq     = []
if "chat_mistral"  not in st.session_state: st.session_state.chat_mistral  = []
if "chat_gemini"   not in st.session_state: st.session_state.chat_gemini   = []
if "active_cat"    not in st.session_state: st.session_state.active_cat    = "General"
if "quick_prompt"  not in st.session_state: st.session_state.quick_prompt  = ""


# ── Categories & Prompts ──────────────────────────────────────
CATEGORIES = {
    "General": {
        "icon": "💬",
        "prompts": [
            "What is Artificial Intelligence?",
            "Explain Machine Learning simply",
            "What is the future of AI?",
            "Difference between AI and human intelligence",
        ]
    },
    "Coding": {
        "icon": "💻",
        "prompts": [
            "Write a Python function to reverse a string",
            "Explain what is an API in simple terms",
            "What is the difference between a list and tuple in Python?",
            "Write a simple REST API example",
        ]
    },
    "Science": {
        "icon": "🔬",
        "prompts": [
            "Explain quantum computing in simple words",
            "How does DNA store information?",
            "What is the theory of relativity?",
            "How does the human brain work?",
        ]
    },
    "Creative": {
        "icon": "🎨",
        "prompts": [
            "Write a short poem about technology",
            "Tell me a creative story about a robot",
            "Write a product description for a smart watch",
            "Create a tagline for an AI startup",
        ]
    },
    "Math": {
        "icon": "📐",
        "prompts": [
            "Explain the Pythagorean theorem with example",
            "What is calculus and where is it used?",
            "Explain probability with a real life example",
            "What is a matrix and how is it used in AI?",
        ]
    },
    "Business": {
        "icon": "📊",
        "prompts": [
            "How can AI help small businesses?",
            "What is a SWOT analysis?",
            "Explain digital marketing in simple words",
            "What skills are needed for data science jobs?",
        ]
    },
}

# ── LLM Response ──────────────────────────────────────────────
def get_response(llm, messages):
    start = time.time()
    try:
        resp    = llm.invoke(messages)
        elapsed = time.time() - start
        return resp.content, elapsed, None
    except Exception as e:
        return None, 0, str(e)

def build_llm(key, temperature):
    if key == "groq":
        return ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    elif key == "mistral":
        return ChatMistralAI(model="mistral-small-latest", temperature=temperature)
    elif key == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    st.markdown('<div class="sidebar-label">Active Models</div>', unsafe_allow_html=True)
    use_groq    = st.checkbox("⚡ Groq — Llama 3.1",      value=True)
    use_mistral = st.checkbox("🌟 Mistral — Small Latest", value=True)
    use_gemini  = st.checkbox("💎 Gemini — 1.5 Flash",    value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Temperature</div>', unsafe_allow_html=True)
    st.caption("Low = focused · High = creative")
    temperature = st.slider("", 0.0, 1.0, 0.7, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Response Length</div>', unsafe_allow_html=True)
    length_prompt = st.selectbox("", ["Short (2-3 lines)", "Medium (1 paragraph)", "Detailed (full explanation)"], label_visibility="collapsed")

    length_map = {
        "Short (2-3 lines)":           "Answer in 2-3 lines only.",
        "Medium (1 paragraph)":        "Answer in one clear paragraph.",
        "Detailed (full explanation)": "Give a detailed, thorough explanation.",
    }

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    active_models_count = sum([use_groq, use_mistral, use_gemini])
    st.markdown(f"Models active: **{active_models_count}**")
    st.markdown(f"Queries run: **{st.session_state.total_queries}**")
    st.markdown(f"History items: **{len(st.session_state.history)}**")

    st.markdown("---")
    if st.button("🗑 Clear All History"):
        st.session_state.history      = []
        st.session_state.total_queries = 0
        st.session_state.chat_groq    = []
        st.session_state.chat_mistral = []
        st.session_state.chat_gemini  = []
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">LLM Comparison Studio</div>
    <div class="hero-subtitle">Compare Groq · Mistral · Gemini — side by side, in real time</div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-number">{active_models_count}</div><div class="stat-label">Models</div></div>
    <div class="stat-card"><div class="stat-number">{st.session_state.total_queries}</div><div class="stat-label">Queries</div></div>
    <div class="stat-card"><div class="stat-number">{len(CATEGORIES)}</div><div class="stat-label">Categories</div></div>
    <div class="stat-card"><div class="stat-number">{len(st.session_state.history)}</div><div class="stat-label">History</div></div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Compare", "💬 Chat Mode", "📋 History"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — COMPARE
# ═══════════════════════════════════════════════════════════════
with tab1:
    # Category selector
    st.markdown('<div class="section-title">Question Category</div>', unsafe_allow_html=True)
    cat_cols = st.columns(len(CATEGORIES))
    for i, (cat_name, cat_data) in enumerate(CATEGORIES.items()):
        with cat_cols[i]:
            if st.button(f"{cat_data['icon']} {cat_name}", key=f"cat_{cat_name}"):
                st.session_state.active_cat   = cat_name
                st.session_state.quick_prompt = ""

    active_cat = st.session_state.active_cat
    st.markdown(f"**{CATEGORIES[active_cat]['icon']} {active_cat} — Quick Prompts:**")

    prompt_cols = st.columns(2)
    for j, p in enumerate(CATEGORIES[active_cat]["prompts"]):
        with prompt_cols[j % 2]:
            if st.button(f"  {p}", key=f"qp_{j}_{p}"):
                st.session_state.quick_prompt = p

    st.markdown("---")

    # Question input
    st.markdown('<div class="section-title">Your Question</div>', unsafe_allow_html=True)
    default_q = st.session_state.quick_prompt or f"Explain {active_cat} concepts in simple words."
    question  = st.text_area("", value=default_q, height=110,
                              placeholder="Type any question — coding, science, creative, math, business...",
                              label_visibility="collapsed")

    run_btn = st.button("🚀 Generate & Compare All Models")

    # Generate
    if run_btn and question.strip():
        length_instruction = length_map[length_prompt]
        full_question      = f"{question}\n\n[{length_instruction}]"

        st.markdown("---")
        st.markdown('<div class="section-title">Live Responses</div>', unsafe_allow_html=True)

        models_config = []
        if use_groq:    models_config.append(("Groq",    "llama-3.1-8b-instant",  "groq",    "badge-groq",    "⚡"))
        if use_mistral: models_config.append(("Mistral", "mistral-small-latest",   "mistral", "badge-mistral", "🌟"))
        if use_gemini:  models_config.append(("Gemini",  "gemini-2.0-flash",       "gemini",  "badge-gemini",  "💎"))

        if not models_config:
            st.warning("Please select at least one model from the sidebar.")
        else:
            cols    = st.columns(len(models_config))
            results = {}

            for idx, (model_name, model_id, model_key, badge_class, icon) in enumerate(models_config):
                with cols[idx]:
                    with st.spinner(f"{icon} {model_name} thinking..."):
                        llm     = build_llm(model_key, temperature)
                        content, elapsed, error = get_response(llm, [HumanMessage(content=full_question)])
                        results[model_name] = {"content": content, "time": elapsed, "error": error}

                    if error:
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-header">
                                <span class="model-badge {badge_class}">{model_name}</span>
                                <span class="model-name">{icon} {model_id}</span>
                            </div>
                            <div class="error-box">❌ {error}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        word_count  = len(content.split())
                        char_count  = len(content)
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-header">
                                <span class="model-badge {badge_class}">{model_name}</span>
                                <span class="model-name">{icon} {model_id}</span>
                            </div>
                            <div class="response-box">{content}</div>
                            <div class="response-meta">
                                <span class="meta-chip">⏱ {elapsed:.2f}s</span>
                                <span class="meta-chip">📝 {word_count} words</span>
                                <span class="meta-chip">🔤 {char_count} chars</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

            # Fastest model highlight
            valid = {k: v for k, v in results.items() if v["content"]}
            if valid:
                fastest = min(valid, key=lambda k: valid[k]["time"])
                st.success(f"⚡ **Fastest:** {fastest} — {valid[fastest]['time']:.2f}s")

            # Save history
            st.session_state.history.append({
                "question": question,
                "category": active_cat,
                "results":  results
            })
            st.session_state.total_queries += 1


# ═══════════════════════════════════════════════════════════════
# TAB 2 — CHAT MODE
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Multi-Model Chat — Conversation Mode</div>', unsafe_allow_html=True)
    st.caption("Each model remembers the full conversation. Ask follow-up questions!")

    chat_model_key = st.selectbox(
        "Choose model to chat with:",
        ["⚡ Groq", "🌟 Mistral", "💎 Gemini"],
        label_visibility="visible"
    )

    model_map = {"⚡ Groq": "groq", "🌟 Mistral": "mistral", "💎 Gemini": "gemini"}
    chat_key  = model_map[chat_model_key]
    chat_hist_key = f"chat_{chat_key}"

    # Display chat history
    chat_history = st.session_state[chat_hist_key]
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:8px;">
                    <div class="chat-user">
                        <div class="chat-label" style="color:#a78bfa; text-align:right;">You</div>
                        {msg["content"]}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                badge_map = {"groq": "badge-groq", "mistral": "badge-mistral", "gemini": "badge-gemini"}
                icon_map  = {"groq": "⚡", "mistral": "🌟", "gemini": "💎"}
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div class="chat-ai">
                        <div class="chat-label" style="color:#60a5fa;">{icon_map[chat_key]} {chat_model_key.split()[1]}</div>
                        {msg["content"]}
                        <div style="font-size:0.7rem; color:#3a3a5a; margin-top:4px; font-family:'JetBrains Mono',monospace;">⏱ {msg.get('time', 0):.2f}s</div>
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#3a3a5a; border:1px dashed #2a2a4a; border-radius:12px;">
            Start a conversation — ask anything!
        </div>""", unsafe_allow_html=True)

    # Chat input
    st.markdown("---")
    chat_input = st.text_area("Your message:", height=80, placeholder="Ask a follow-up or start a new topic...", key="chat_input_box")
    c1, c2 = st.columns([4, 1])
    with c1:
        send_btn = st.button("💬 Send Message")
    with c2:
        if st.button("🔄 Reset Chat"):
            st.session_state[chat_hist_key] = []
            st.rerun()

    if send_btn and chat_input.strip():
        # Build message history for LLM
        lc_messages = [SystemMessage(content="You are a helpful, knowledgeable AI assistant. Be clear and concise.")]
        for msg in st.session_state[chat_hist_key]:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
        lc_messages.append(HumanMessage(content=chat_input))

        with st.spinner(f"Thinking..."):
            llm     = build_llm(chat_key, temperature)
            content, elapsed, error = get_response(llm, lc_messages)

        st.session_state[chat_hist_key].append({"role": "user",      "content": chat_input})
        if error:
            st.session_state[chat_hist_key].append({"role": "assistant", "content": f"Error: {error}", "time": 0})
        else:
            st.session_state[chat_hist_key].append({"role": "assistant", "content": content, "time": elapsed})

        st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Query History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding:2rem; color:#3a3a5a; border:1px dashed #2a2a4a; border-radius:12px;">
            No queries yet — go to Compare tab and ask something!
        </div>""", unsafe_allow_html=True)
    else:
        # Filter by category
        all_cats = list(set(h.get("category", "General") for h in st.session_state.history))
        filter_cat = st.selectbox("Filter by category:", ["All"] + all_cats)

        filtered = st.session_state.history if filter_cat == "All" else \
                   [h for h in st.session_state.history if h.get("category") == filter_cat]

        st.markdown(f"Showing **{len(filtered)}** queries")
        st.markdown("---")

        for i, item in enumerate(reversed(filtered)):
            q_num = len(filtered) - i
            cat   = item.get("category", "General")
            cat_icon = CATEGORIES.get(cat, {}).get("icon", "💬")

            with st.expander(f"{cat_icon} [{cat}] Q{q_num}: {item['question'][:65]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown("---")
                res_cols = st.columns(len(item["results"]))
                for j, (model, data) in enumerate(item["results"].items()):
                    with res_cols[j]:
                        if data.get("content"):
                            st.markdown(f"**{model}** — ⏱ {data['time']:.2f}s")
                            st.markdown(f"_{data['content']}_")
                        else:
                            st.markdown(f"**{model}** — ❌ Error")
