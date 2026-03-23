"""
Research Assistant Agent — Streamlit UI
Run: streamlit run app.py
"""

import os
import math
import html
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, body, [class*="css"] { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Layout ── */
.block-container { padding-top: 2rem !important; max-width: 900px !important; }

/* ── Header ── */
.page-title { font-size: 1.6rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; margin-bottom: 0.15rem; }
.page-sub   { font-size: 0.85rem; color: #64748b; margin-bottom: 1.8rem; }

/* ── Input ── */
.stTextInput > div > div > input {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    height: 2.7rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #4f46e5 !important; }

/* ── Step cards ── */
.card {
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 3px solid;
    animation: fadeUp 0.25s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}

.card-question    { background: #1e2a3a; border-color: #6366f1; }
.card-reasoning   { background: #1e1e2e; border-color: #f59e0b; }
.card-observation { background: #0f1e26; border-color: #0ea5e9; }
.card-answer      { background: #0f1e17; border-color: #10b981; }
.card-error       { background: #1e0f0f; border-color: #ef4444; }

.card-tag {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.tag-question    { color: #818cf8; }
.tag-reasoning   { color: #fbbf24; }
.tag-observation { color: #38bdf8; }
.tag-answer      { color: #34d399; }
.tag-error       { color: #f87171; }

.card-body { color: #cbd5e1; font-size: 0.9rem; line-height: 1.65; }

.row       { display: flex; gap: 0.5rem; align-items: flex-start; margin-top: 0.4rem; flex-wrap: wrap; }
.row-label { font-size: 0.78rem; font-weight: 600; color: #94a3b8; white-space: nowrap; padding-top: 2px; }
.row-val   { font-size: 0.88rem; color: #e2e8f0; flex: 1; }

.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: #0f172a;
    color: #94a3b8;
    border-radius: 5px;
    padding: 0.2rem 0.55rem;
    display: inline-block;
}
.obs-pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    background: #0a1220;
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.5rem;
    max-height: 160px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.hint { font-size: 0.75rem; color: #334155; margin-top: 0.5rem; font-style: italic; }

.answer-text { color: #d1fae5; font-size: 0.95rem; line-height: 1.75; }

/* ── Example chips ── */
.stButton[data-testid*="ex_"] > button {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    height: auto !important;
    padding: 0.35rem 0.7rem !important;
}
.stButton[data-testid*="ex_"] > button:hover {
    background: #263245 !important;
    color: #e2e8f0 !important;
    border-color: #6366f1 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e293b; }
.sb-section { margin-bottom: 1.4rem; }
.sb-title   { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.5rem; }
.sb-item    { background: #1e293b; border-radius: 7px; padding: 0.6rem 0.85rem; margin-bottom: 0.35rem; border-left: 2px solid #334155; }
.sb-name    { font-size: 0.85rem; font-weight: 600; color: #a5b4fc; margin-bottom: 0.1rem; }
.sb-desc    { font-size: 0.78rem; color: #64748b; }
.sb-hist    { font-size: 0.8rem; color: #64748b; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1e293b; margin: 1rem 0; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4rem 0; }
.empty-icon  { font-size: 2rem; color: #1e293b; margin-bottom: 1rem; }
.empty-text  { font-size: 0.95rem; color: #334155; }
.empty-sub   { font-size: 0.8rem; color: #1e293b; margin-top: 0.4rem; }

div[data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Agent (cached) ────────────────────────────────────────────────────────────
load_dotenv()

@st.cache_resource
def build_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    llm = ChatGroq(api_key=api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

    @tool
    def wikipedia_search(query: str) -> str:
        """Search Wikipedia for factual information, definitions, historical facts,
        or scientific concepts. Input: a plain search query string."""
        try:
            result = wiki_wrapper.run(query)
            return result if result else "No Wikipedia article found."
        except Exception as e:
            return f"Wikipedia error: {e}"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression. Input MUST be numeric only.
        Examples: '149600000 / 299792.458', '3.7 * 0.05', 'sqrt(144)'.
        Never pass English sentences — only numbers and operators."""
        try:
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed["abs"] = abs
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    system_message = SystemMessage(content=(
        "You are a Research Assistant Agent. "
        "Always use the wikipedia_search tool to find facts before answering. "
        "Use the calculator tool for any numeric computation. "
        "Think step by step and explain your reasoning before each action."
    ))
    return create_react_agent(
        model=llm,
        tools=[wikipedia_search, calculator],
        prompt=system_message,
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Render helpers ────────────────────────────────────────────────────────────
def esc(text):
    """Escape text so it's safe inside HTML."""
    return html.escape(str(text))

def render_question(content):
    st.markdown(f"""
<div class="card card-question">
  <div class="card-tag tag-question">User Question</div>
  <div class="card-body">{esc(content)}</div>
</div>""", unsafe_allow_html=True)

def render_reasoning(step_num, thought, action, inp):
    thought_row = f"""
  <div class="row">
    <span class="row-label">Thought</span>
    <span class="row-val">{esc(thought)}</span>
  </div>""" if thought else ""

    st.markdown(f"""
<div class="card card-reasoning">
  <div class="card-tag tag-reasoning">Step {step_num} &mdash; Agent Reasoning</div>
  {thought_row}
  <div class="row">
    <span class="row-label">Action</span>
    <span class="row-val"><span class="mono">{esc(action)}</span></span>
  </div>
  <div class="row">
    <span class="row-label">Input</span>
    <span class="row-val"><span class="mono">{esc(inp)}</span></span>
  </div>
</div>""", unsafe_allow_html=True)

def render_observation(tool_name, content):
    truncated = content if len(content) <= 600 else content[:600] + "\n... [truncated]"
    st.markdown(f"""
<div class="card card-observation">
  <div class="card-tag tag-observation">Observation &mdash; {esc(tool_name)}</div>
  <div class="obs-pre">{esc(truncated)}</div>
  <div class="hint">Agent will now reason about this result and decide the next step.</div>
</div>""", unsafe_allow_html=True)

def render_answer(content):
    st.markdown(f"""
<div class="card card-answer">
  <div class="card-tag tag-answer">Final Answer</div>
  <div class="answer-text">{esc(content)}</div>
</div>""", unsafe_allow_html=True)

def render_error(content):
    st.markdown(f"""
<div class="card card-error">
  <div class="card-tag tag-error">Error</div>
  <div class="card-body">{esc(content)}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem;color:#475569;">LangGraph ReAct Agent</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Tools</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sb-item">
  <div class="sb-name">Wikipedia Search</div>
  <div class="sb-desc">Retrieves factual information from Wikipedia</div>
</div>
<div class="sb-item">
  <div class="sb-name">Calculator</div>
  <div class="sb-desc">Evaluates numeric expressions</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Model</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sb-item">
  <div class="sb-name">Llama 4 Scout</div>
  <div class="sb-desc">meta-llama/llama-4-scout-17b<br>Groq &mdash; Free tier</div>
</div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="sb-title">Recent Questions</div>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-8:]):
            short = q[:52] + "..." if len(q) > 52 else q
            st.markdown(f'<div class="sb-hist">{esc(short)}</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Research Assistant Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Wikipedia Search + Calculator &nbsp;|&nbsp; LangGraph ReAct &nbsp;|&nbsp; Llama 4 via Groq</div>', unsafe_allow_html=True)

agent = build_agent()
if not agent:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# ── Input row ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input("q", placeholder="Ask a research question...", label_visibility="collapsed", key="q_input")
with col2:
    ask_btn = st.button("Ask", use_container_width=True)

# Example chips
examples = [
    "Who is the PM of India?",
    "Speed of light, time to reach Earth?",
    "What is machine learning?",
    "GDP of India — what is 5% of it?",
]
ex_cols = st.columns(4)
for i, ex in enumerate(examples):
    if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
        question = ex
        ask_btn  = True

st.markdown('<hr>', unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    q = question.strip()
    st.session_state.history.append(q)

    render_question(q)

    step_num = 1
    seen_ids = set()

    try:
        for state in agent.stream(
            {"messages": [{"role": "user", "content": q}]},
            stream_mode="values",
        ):
            msg    = state["messages"][-1]
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tc        = msg.tool_calls[0]
                    tool_name = tc["name"]
                    tool_inp  = tc["args"]
                    thought   = msg.content.strip() if msg.content else ""

                    if not thought:
                        if tool_name == "wikipedia_search":
                            query   = tool_inp.get("query", "")
                            thought = f"I need to find information about '{query}'. Searching Wikipedia to gather relevant facts."
                        elif tool_name == "calculator":
                            expr    = tool_inp.get("expression", "")
                            thought = f"I have the necessary facts. Computing '{expr}' to produce the numeric answer."

                    render_reasoning(step_num, thought, tool_name, str(tool_inp))
                    step_num += 1

                elif msg.content:
                    render_answer(msg.content)

            elif isinstance(msg, ToolMessage):
                render_observation(msg.name, msg.content)

    except Exception as e:
        render_error(str(e))

elif ask_btn:
    st.warning("Please enter a question.")

else:
    st.markdown("""
<div class="empty-state">
  <div class="empty-icon">&#9632;</div>
  <div class="empty-text">Enter a question above to see the agent reason step by step.</div>
  <div class="empty-sub">Thought &rarr; Action &rarr; Observation &rarr; Final Answer</div>
</div>""", unsafe_allow_html=True)