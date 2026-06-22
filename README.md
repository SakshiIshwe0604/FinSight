# 📈 FinSight — Multi-Agent Financial Intelligence Platform

> Built with LangGraph · LangChain RAG · Groq · Streamlit

FinSight is a multi-agent AI system that answers financial questions about Indian stocks by combining live market data with deep analysis of company annual reports — orchestrated via a LangGraph pipeline.

**🔗 Live demo:** [ishwefinsight.streamlit.app](https://ishwefinsight.streamlit.app)

---

## 🎯 What it does

Type a question like *"Is Infosys a good investment right now?"* or *"Compare Infosys vs TCS"* and FinSight:

1. **Plans** — decides which agents to invoke based on your query, and detects compare-style questions
2. **Retrieves** — searches 7,300+ chunks from annual report PDFs (1,710+ pages) using semantic RAG
3. **Fetches** — pulls live stock price, P/E ratio, EPS, 52-week range, market cap, and news from yfinance
4. **Synthesizes** — merges both sources into a structured investment brief
5. **Grades** — scores answer quality and retries automatically if below threshold

Output: a structured brief with **BUY / HOLD / SELL** recommendation, Outlook, Key Risks, and Summary — or an honest **NO DATA** response when a company is outside FinSight's coverage, instead of a fabricated answer.

---

## 🏢 Company Coverage

FinSight distinguishes between two coverage tiers, and is explicit about the difference rather than silently guessing:

| Coverage | Companies |
|---|---|
| **Live data + Annual report (full analysis)** | Infosys, TCS, Wipro, HDFC Bank, Reliance |
| **Live data only** | ICICI Bank, SBI, Bajaj Finance, Asian Paints |
| **Out of scope** | Anything else (e.g. Meesho, Zomato) — returns an explicit "not covered" response |

This was a deliberate design decision after an early version of the system silently fell back to Infosys's data when asked about an untracked company — see [Design Notes](#-design-notes--lessons-learned) below.

---

## 🏗️ Architecture

```
                          User Query
                              │
                              ▼
                      ┌───────────────┐
                      │  Planner Node │  routes to pdf_rag, live_data,
                      │               │  both, or compare mode
                      └───────┬───────┘
                 ┌────────────┼────────────┐
                 ▼                         ▼
        ┌─────────────────┐      ┌──────────────────┐
        │  PDF RAG Agent   │      │  Live Data Agent  │
        │  FAISS + HF      │      │  yfinance + Groq  │
        │  embeddings      │      │                   │
        └────────┬─────────┘      └─────────┬─────────┘
                 └────────────┬─────────────┘
                              ▼
                     ┌─────────────────┐
                     │ Synthesizer Node │  merges both answers,
                     │                  │  or returns honest "NO DATA"
                     └────────┬─────────┘  if neither source has coverage
                              ▼
                     ┌─────────────────┐
                     │   Grader Node    │  scores quality (0–1)
                     │                  │  retries if score < threshold
                     └────────┬─────────┘
                              ▼
                  Final Investment Brief
                  (BUY / HOLD / SELL / NO DATA)
```

For comparison queries (*"Compare TCS vs Wipro"*), the Planner routes directly to a dedicated **Compare Node** that retrieves context for both companies and generates a side-by-side brief with a declared winner.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| RAG Pipeline | LangChain + FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM (Planner + Synthesizer + Compare) | Groq `llama-3.3-70b-versatile` |
| LLM (RAG + Live Data Agents) | Groq `llama-3.1-8b-instant` |
| Live Market Data | yfinance |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## 📁 Project Structure

```
finsight/
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── graph/
│   ├── state.py                # AgentState TypedDict
│   ├── graph_builder.py        # LangGraph pipeline definition
│   ├── companies.py            # Shared ticker map + coverage lists
│   └── nodes/
│       ├── planner.py          # Query router + compare detection
│       ├── pdf_rag_agent.py    # Annual report retriever
│       ├── live_data_agent.py  # yfinance live data
│       ├── compare_node.py     # Side-by-side comparison logic
│       ├── synthesizer.py      # Answer merger + no-data handling
│       └── grader.py           # Quality scorer + retry
├── rag/
│   ├── embeddings.py           # HuggingFace embeddings setup
│   └── ingest_pdfs.py          # PDF → FAISS index builder
├── eval/
│   ├── ragas_eval.py           # RAGAS evaluation pipeline
│   └── test_questions.json     # Benchmark questions
└── data/
    ├── pdfs/                   # Annual report PDFs (not committed)
    └── faiss_index/            # Built vector index (committed)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/SakshiIshwe0604/FinSight.git
cd FinSight
```

### 2. Create virtual environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Fill in your API key:
```
GROQ_API_KEY=your_groq_key
```

### 5. Add annual report PDFs
Download annual reports for Infosys, TCS, Wipro, HDFC Bank, and Reliance from their investor relations pages and place them in `data/pdfs/`.

### 6. Build the FAISS index
```bash
python -m rag.ingest_pdfs
```

### 7. Run the app
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 💡 Example Queries

**Investment analysis**
- *"Is Infosys a good buy right now?"*
- *"What are the key risks for TCS?"*
- *"What is Reliance market cap?"*

**Comparison**
- *"Compare Infosys vs TCS"*
- *"Which is better, Wipro or HDFC?"*

**Live data only**
- *"What is ICICI Bank current price?"*
- *"What is SBI P/E ratio?"*

---

## 📊 Evaluation

The system is evaluated using **RAGAS** metrics on the synthesized investment briefs, with scores surfaced live in the UI for every query (currently averaging **~0.8**). The Grader node uses this score to automatically trigger a retry through the pipeline if quality falls below threshold.

---

## 🔍 Design Notes & Lessons Learned

An early version of FinSight had a silent fallback: if a query mentioned a company not in the ticker map, `extract_ticker()` defaulted to Infosys's ticker instead of failing explicitly. This meant asking *"What is Meesho's market cap?"* returned a confident, well-formatted investment brief — using Infosys's real financial data, with Infosys's numbers, while never mentioning that the company being analyzed wasn't actually Meesho.

The fix involved three changes:
- Removing the silent fallback so an unmapped ticker returns `None` instead of a default
- Adding the same explicit "not covered" guard to the PDF RAG agent, since FAISS will always return its *nearest* chunks even when no genuinely relevant document exists in the index
- Updating the Synthesizer to recognize when both the PDF and live-data agents report no coverage, and return an honest `NO DATA` recommendation instead of asking the LLM to force a BUY/HOLD/SELL verdict from nothing

This is a good illustration of a general failure mode in RAG and agent systems: retrieval and LLM generation will always produce *something* plausible-looking, even with no relevant grounding data, unless the pipeline explicitly checks for and short-circuits the no-data case.

---

## 🔑 Free API Key Required

| Service | Link | Cost |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | Free |

---

## 👩‍💻 Author

**Sakshi Ishwe**
B.Tech CSE — Shri Vaishnav Vidyapeeth Vishwavidyalaya, Indore
[GitHub](https://github.com/SakshiIshwe0604) · [LinkedIn](https://www.linkedin.com/in/sakshi-ishwe)

---

## ⭐ If this project helped you, give it a star!