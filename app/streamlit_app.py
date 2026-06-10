import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph.graph_builder import build_graph

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
.big-title { font-size: 2rem; font-weight: 600; margin-bottom: 0; }
.sub { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
.rec-buy  { background:#d4edda; color:#155724; padding:6px 16px; border-radius:6px; font-weight:600; }
.rec-hold { background:#fff3cd; color:#856404; padding:6px 16px; border-radius:6px; font-weight:600; }
.rec-sell { background:#f8d7da; color:#721c24; padding:6px 16px; border-radius:6px; font-weight:600; }
.score-box { background:#f0f2f6; border-radius:8px; padding:10px 16px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">📈 FinSight</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Multi-agent financial intelligence · LangGraph + RAG + Groq</p>', unsafe_allow_html=True)

# Example queries
st.markdown("**Try an example:**")
cols = st.columns(4)
examples = [
    "Is Infosys a good buy?",
    "What are Infosys key risks?",
    "Infosys revenue growth analysis",
    "Should I invest in Infosys?"
]
for i, col in enumerate(cols):
    if col.button(examples[i], key=f"ex_{i}"):
        st.session_state["query"] = examples[i]

query = st.text_input(
    "Ask anything about a stock:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. Is Infosys a good investment right now?"
)

if st.button("🔍 Analyze", type="primary") and query:
    # Agent pipeline status
    st.markdown("---")
    st.markdown("**Agent pipeline**")
    cols = st.columns(5)
    statuses = ["Planner", "PDF RAG", "Live Data", "Synthesizer", "Grader"]
    pills = []
    for i, (col, name) in enumerate(zip(cols, statuses)):
        pills.append(col.empty())
        pills[i].markdown(f"⬜ {name}")

    def update_pill(idx, state):
        icons = {"done": "✅", "running": "🔵", "wait": "⬜"}
        pills[idx].markdown(f"{icons[state]} {statuses[idx]}")

    with st.spinner("Running analysis..."):
        try:
            update_pill(0, "running")
            graph = build_graph()

            # Run full graph
            update_pill(0, "done")
            update_pill(1, "running")

            result = graph.invoke({
                "query": query,
                "retry_count": 0,
                "agents_to_use": [],
                "sub_tasks": {},
                "pdf_answer": None,
                "live_answer": None,
                "final_answer": None,
                "quality_score": None
            })

            update_pill(1, "done")
            update_pill(2, "done")
            update_pill(3, "done")
            update_pill(4, "done")

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Display brief
    st.markdown("---")
    final = result.get("final_answer", "")
    score = result.get("quality_score", 0)

    # Extract and show recommendation badge
    rec = "HOLD"
    if "RECOMMENDATION: BUY" in final:
        rec = "BUY"
    elif "RECOMMENDATION: SELL" in final:
        rec = "SELL"

    rec_class = {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}[rec]
    st.markdown(f'<span class="{rec_class}">{rec}</span>', unsafe_allow_html=True)
    st.markdown(" ")

    # Parse and display sections
    sections = {"OUTLOOK": "", "KEY RISKS": "", "SUMMARY": ""}
    current = None
    for line in final.split("\n"):
        if "OUTLOOK:" in line:
            current = "OUTLOOK"
        elif "KEY RISKS:" in line:
            current = "KEY RISKS"
        elif "SUMMARY:" in line:
            current = "SUMMARY"
        elif current and line.strip() and "RECOMMENDATION:" not in line:
            sections[current] += line + "\n"

    if sections["OUTLOOK"]:
        st.markdown("**📊 Outlook**")
        st.write(sections["OUTLOOK"].strip())

    if sections["KEY RISKS"]:
        st.markdown("**⚠️ Key Risks**")
        st.write(sections["KEY RISKS"].strip())

    if sections["SUMMARY"]:
        st.markdown("**💡 Summary**")
        st.info(sections["SUMMARY"].strip())

    # Sources and score
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Sources:** Annual Report (RAG) · yfinance live")
    c2.markdown(f"**Retries:** {result.get('retry_count', 1)}")
    c3.markdown(f"**Quality score:** `{score}`")

    st.markdown(
        '<div class="score-box">Traced in LangSmith · Powered by Groq llama-3.3-70b · '
        'Embeddings: all-MiniLM-L6-v2</div>',
        unsafe_allow_html=True
    )