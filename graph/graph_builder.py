import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.planner import planner_node
from graph.nodes.pdf_rag_agent import pdf_rag_node
from graph.nodes.live_data_agent import live_data_node
from graph.nodes.synthesizer import synthesizer_node
from graph.nodes.grader import grader_node, should_retry
from graph.nodes.compare_node import compare_node  # ← new

def route_after_planner(state: dict) -> str:
    if state.get("is_compare"):
        return "compare"
    return "pdf_rag"

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("pdf_rag", pdf_rag_node)
    g.add_node("live_data", live_data_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grader", grader_node)
    g.add_node("compare", compare_node)  # ← new

    g.set_entry_point("planner")

    # route after planner — compare or normal
    g.add_conditional_edges("planner", route_after_planner, {
        "compare": "compare",
        "pdf_rag": "pdf_rag"
    })

    # normal pipeline
    g.add_edge("pdf_rag", "live_data")
    g.add_edge("live_data", "synthesizer")
    g.add_edge("synthesizer", "grader")

    # compare goes straight to grader
    g.add_edge("compare", "grader")

    g.add_conditional_edges("grader", should_retry, {
        "end": END,
        "retry": "planner"
    })

    return g.compile()