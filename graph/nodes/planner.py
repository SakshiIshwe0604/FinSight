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
    chain = PLANNER_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    
    try:
        plan = json.loads(result.content)
        # handle if LLM returns "both" as a string instead of list
        agents = plan["agents"]
        if agents == "both" or agents == ["both"]:
            agents = ["pdf", "live"]
        plan["agents"] = agents
        # ensure sub_tasks has both keys if both agents needed
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
        "agents_to_use": plan["agents"],
        "sub_tasks": plan["sub_tasks"],
        "retry_count": state.get("retry_count", 0)
    }