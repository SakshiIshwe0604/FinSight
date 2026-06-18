import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from graph.graph_builder import build_graph
import html as html_lib


st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0a0e1a!important;color:#e2e8f0!important}
.stApp{background-color:#0a0e1a!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 4rem!important;max-width:1100px!important;margin:auto}
.stTextInput input{background:#0f1929!important;border:1px solid #1e2d40!important;border-radius:8px!important;color:#e2e8f0!important;font-size:0.95rem!important}
.stTextInput input:focus{border-color:#38bdf8!important;box-shadow:0 0 0 2px rgba(56,189,248,0.1)!important}
.stTextInput label{color:#475569!important;font-size:0.75rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important}
.stButton>button{background:#38bdf8!important;color:#0a0e1a!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:0.55rem 2rem!important}
.stButton>button:hover{background:#7dd3fc!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:1.5rem;border-bottom:1px solid #1e2d40;margin-bottom:2rem">
  <div>
    <div style="font-size:1.6rem;font-weight:600;color:#38bdf8;letter-spacing:-0.5px">◈ FinSight</div>
    <div style="font-size:0.72rem;color:#475569;margin-top:3px;font-family:'JetBrains Mono',monospace">multi-agent financial intelligence platform</div>
  </div>
  <div style="font-size:0.72rem;color:#22c55e;font-family:'JetBrains Mono',monospace;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);padding:6px 14px;border-radius:6px">
    ● LIVE · LangGraph + RAG + Groq
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;font-family:monospace">Quick queries</div>', unsafe_allow_html=True)
examples = ["Is Infosys a good buy?", "What are Infosys key risks?", "Infosys revenue growth", "Should I invest in Infosys?"]
cols = st.columns(4)
for i, col in enumerate(cols):
    with col:
        if st.button(examples[i], key=f"ex_{i}"):
            st.session_state["query"] = examples[i]

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
query = st.text_input("ASK ANYTHING ABOUT A STOCK", value=st.session_state.get("query", ""), placeholder="e.g. Is Infosys a good investment right now?")
st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
analyze = st.button("⚡  Analyze", type="primary")

if analyze and query:

    nodes = ["Planner", "PDF RAG", "Live Data", "Synthesizer", "Grader"]
    pipeline_slot = st.empty()

    def render_pipeline(active=None, done_list=[]):
        def pill(label, state):
            cfg = {
                "wait": ("rgba(71,85,105,0.1)", "#1e2d40", "#475569", "○"),
                "run":  ("rgba(56,189,248,0.12)", "#38bdf8", "#38bdf8", "◉"),
                "done": ("rgba(34,197,94,0.12)", "#22c55e", "#22c55e", "●"),
            }
            bg, border, color, dot = cfg[state]
            return f'<span style="display:inline-flex;align-items:center;gap:5px;padding:5px 14px;border-radius:6px;font-size:0.78rem;font-weight:500;border:1px solid {border};background:{bg};color:{color};font-family:monospace">{dot} {label}</span>'

        arrow = '<span style="color:#1e2d40;margin:0 4px">→</span>'
        html = '<div style="background:#0f1929;border:1px solid #1e2d40;border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0">'
        html += '<div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;font-family:monospace">Agent pipeline</div>'
        html += '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">'
        for i, n in enumerate(nodes):
            state = "done" if n in done_list else ("run" if n == active else "wait")
            html += pill(n, state)
            if i < len(nodes) - 1:
                html += arrow
        html += '</div></div>'
        pipeline_slot.markdown(html, unsafe_allow_html=True)

    render_pipeline(active="Planner", done_list=[])

    try:
        import time
        graph = build_graph()
        time.sleep(0.4)
        render_pipeline(active="PDF RAG", done_list=["Planner"])

        result = graph.invoke({
            "query": query, "retry_count": 0,
            "agents_to_use": [], "sub_tasks": {},
            "pdf_answer": None, "live_answer": None,
            "final_answer": None, "quality_score": None
        })

        render_pipeline(active=None, done_list=nodes)

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    final = result.get("final_answer", "")
     
    score = result.get("quality_score", 0)
    rec = "HOLD"
    if "RECOMMENDATION: BUY" in final:  rec = "BUY"
    elif "RECOMMENDATION: SELL" in final: rec = "SELL"

    sections = {"OUTLOOK": [], "KEY RISKS": [], "SUMMARY": []}
    current = None
    for line in final.split("\n"):
        l = line.strip()
        if not l or "RECOMMENDATION:" in l: continue
        if l.startswith("OUTLOOK:"):
            current = "OUTLOOK"
            rest = l.split("OUTLOOK:", 1)[1].strip()
            if rest: sections[current].append(rest)
            continue
        if l.startswith("KEY RISKS:"):
            current = "KEY RISKS"
            rest = l.split("KEY RISKS:", 1)[1].strip()
            if rest: sections[current].append(rest)
            continue
        if l.startswith("SUMMARY:"):
            current = "SUMMARY"
            rest = l.split("SUMMARY:", 1)[1].strip()
            if rest: sections[current].append(rest)
            continue
        if current: sections[current].append(l)

    rec_cfg = {
        "BUY":  ("rgba(34,197,94,0.12)",  "rgba(34,197,94,0.3)",  "#22c55e"),
        "HOLD": ("rgba(251,191,36,0.12)", "rgba(251,191,36,0.3)", "#fbbf24"),
        "SELL": ("rgba(239,68,68,0.12)",  "rgba(239,68,68,0.3)",  "#ef4444"),
    }
    rec_bg, rec_border, rec_color = rec_cfg[rec]

    import html as html_lib

    outlook_text = html_lib.escape(" ".join(sections["OUTLOOK"]))
    summary_text = html_lib.escape(" ".join(sections["SUMMARY"]))
    risks_html = "".join([
        f'<span style="display:inline-block;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);color:#f87171;font-size:0.75rem;padding:4px 12px;border-radius:4px;margin:3px 5px 3px 0">{html_lib.escape(r.lstrip("-• ").strip())}</span>'
        for r in sections["KEY RISKS"] if r.strip()
    ])

    score_color = "#22c55e" if score >= 0.75 else "#fbbf24"

    # Header
    st.markdown(f"""<div style="background:#0f1929;border:1px solid #1e2d40;border-radius:12px 12px 0 0;overflow:hidden;margin-top:0.5rem;background:#0d1624;padding:1.25rem 1.5rem;border-bottom:1px solid #1e2d40;display:flex;justify-content:space-between;align-items:center"><div><span style="font-size:1.2rem;font-weight:600;color:#f1f5f9;font-family:monospace">Investment Brief</span><span style="font-size:0.75rem;color:#475569;margin-left:12px;font-family:monospace">powered by LangGraph</span></div><span style="background:{rec_bg};border:1px solid {rec_border};color:{rec_color};padding:6px 20px;border-radius:6px;font-weight:600;font-size:0.88rem;font-family:monospace">{rec}</span></div>""", unsafe_allow_html=True)

    # Body wrapper start
    st.markdown('<div style="background:#0f1929;border:1px solid #1e2d40;border-top:none;padding:1.5rem">', unsafe_allow_html=True)

    # Outlook
    st.markdown('<div style="font-size:0.68rem;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;font-family:monospace">● OUTLOOK</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.9rem;color:#94a3b8;line-height:1.75;margin-bottom:1.25rem">{outlook_text}</div>', unsafe_allow_html=True)

    # Key Risks
    st.markdown('<div style="font-size:0.68rem;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;font-family:monospace">● KEY RISKS</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-bottom:1.25rem">{risks_html}</div>', unsafe_allow_html=True)

    # Summary
    st.markdown('<div style="font-size:0.68rem;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;font-family:monospace">● SUMMARY</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.15);border-radius:8px;padding:0.9rem 1.1rem;font-size:0.88rem;color:#7dd3fc;line-height:1.7">{summary_text}</div>', unsafe_allow_html=True)

    # Body wrapper end
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(f"""<div style="padding:0.9rem 1.5rem;border:1px solid #1e2d40;border-top:none;border-radius:0 0 12px 12px;background:#0d1624;display:flex;justify-content:space-between;align-items:center"><div style="font-size:0.72rem;color:#475569;font-family:monospace">SOURCES: Annual Report (RAG) · yfinance live · RETRIES: {result.get('retry_count',1)}</div><div style="font-size:0.72rem;font-family:monospace"><span style="color:#475569">RAGAS SCORE </span><span style="color:{score_color};font-weight:600">{score}</span></div></div>""", unsafe_allow_html=True)