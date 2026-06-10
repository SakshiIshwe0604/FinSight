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

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("pdf_rag", pdf_rag_node)
    g.add_node("live_data", live_data_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grader", grader_node)

    g.set_entry_point("planner")

    # Sequential: planner → pdf_rag → live_data → synthesizer → grader
    g.add_edge("planner", "pdf_rag")
    g.add_edge("pdf_rag", "live_data")
    g.add_edge("live_data", "synthesizer")
    g.add_edge("synthesizer", "grader")

    g.add_conditional_edges("grader", should_retry, {
        "end": END,
        "retry": "planner"
    })

    return g.compile()