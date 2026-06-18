import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial query planner. Given a user question about stocks or companies, decide which agents to use:
- "pdf": for questions about annual reports, financials, strategy, risks, management discussion
- "live": for current stock price, P/E ratio, market cap, recent news
- "both": for general investment advice or buy/sell recommendations

Respond ONLY in valid JSON, no extra text:
{{"agents": ["pdf", "live"], "sub_tasks": {{"pdf": "specific question for PDF agent", "live": "specific question for live data agent"}}}}

For single agent, still use array: {{"agents": ["live"], "sub_tasks": {{"live": "question"}}}}"""),
    ("human", "{query}")
])

def planner_node(state: dict) -> dict:
    print(f"\n[Planner] Routing query: {state['query']}")

    # ── detect compare query ──────────────────────────────────
    query_lower = state["query"].lower()
    compare_keywords = ["vs", "versus", "compare", "better", "which is better"]
    companies = ["infosys", "tcs", "wipro", "hdfc", "reliance", "ril"]

    is_compare = any(kw in query_lower for kw in compare_keywords)
    found_companies = [c for c in companies if c in query_lower]

    if is_compare and len(found_companies) >= 2:
        print(f"[Planner] Compare mode: {found_companies[0]} vs {found_companies[1]}")
        return {
            **state,
            "is_compare": True,
            "company1": found_companies[0].upper(),
            "company2": found_companies[1].upper(),
            "agents_to_use": ["pdf"],
            "sub_tasks": {"pdf": state["query"]},
            "retry_count": state.get("retry_count", 0)
        }

    # ── normal routing ────────────────────────────────────────
    chain = PLANNER_PROMPT | llm
    result = chain.invoke({"query": state["query"]})

    try:
        plan = json.loads(result.content)
        agents = plan["agents"]
        if agents == "both" or agents == ["both"]:
            agents = ["pdf", "live"]
        plan["agents"] = agents
        if "pdf" in agents and "pdf" not in plan.get("sub_tasks", {}):
            plan["sub_tasks"]["pdf"] = state["query"]
        if "live" in agents and "live" not in plan.get("sub_tasks", {}):
            plan["sub_tasks"]["live"] = state["query"]
    except (json.JSONDecodeError, KeyError):
        plan = {
            "agents": ["pdf", "live"],
            "sub_tasks": {"pdf": state["query"], "live": state["query"]}
        }

    print(f"[Planner] Using agents: {plan['agents']}")
    return {
        **state,
        "is_compare": False,
        "agents_to_use": plan["agents"],
        "sub_tasks": plan["sub_tasks"],
        "retry_count": state.get("retry_count", 0)
    }