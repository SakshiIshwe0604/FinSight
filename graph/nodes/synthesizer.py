from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def synthesizer_node(state: dict) -> dict:
    print(f"\n[Synthesizer] Merging agent answers...")

    pdf_answer = state.get("pdf_answer", "") or ""
    live_answer = state.get("live_answer", "") or ""

    no_data_markers = [
        "not in finsight's tracked universe",
        "no annual report is indexed"
    ]
    pdf_has_no_data = any(m in pdf_answer.lower() for m in no_data_markers)
    live_has_no_data = any(m in live_answer.lower() for m in no_data_markers)

    # if BOTH sources have no data, don't synthesize a fake verdict
    if pdf_has_no_data and live_has_no_data:
        final = (
            "RECOMMENDATION: NO DATA\n\n"
            "OUTLOOK:\n"
            "This company is outside FinSight's current coverage. "
            "FinSight tracks live data for Infosys, TCS, Wipro, HDFC Bank, Reliance, "
            "ICICI Bank, SBI, Bajaj Finance, and Asian Paints, with annual report "
            "analysis available for Infosys, TCS, Wipro, HDFC Bank, and Reliance.\n\n"
            "KEY RISKS:\n"
            "- No assessment possible without source data\n\n"
            "SUMMARY:\n"
            "No recommendation can be made — this company is not yet covered by FinSight."
        )
        print(f"[Synthesizer] No data from either source — skipping LLM synthesis")
        return {**state, "final_answer": final}

    parts = []
    if pdf_answer and not pdf_has_no_data:
        parts.append(f"[From Annual Report]:\n{pdf_answer}")
    if live_answer and not live_has_no_data:
        parts.append(f"[From Live Market Data]:\n{live_answer}")

    if not parts:
        return {**state, "final_answer": "No data available to generate a brief."}

    combined = "\n\n---\n\n".join(parts)

    prompt = f"""You are a senior financial analyst. Based on the following research, generate a structured investment brief.

Question: {state['query']}

Research:
{combined}

Respond in this exact structure:

RECOMMENDATION: [BUY / HOLD / SELL]

OUTLOOK:
[2-3 sentences on company performance and growth prospects based on the data]

KEY RISKS:
- [Risk 1]
- [Risk 2]
- [Risk 3]

SUMMARY:
[1 sentence final verdict]"""

    answer = llm.invoke(prompt).content
    print(f"[Synthesizer] Brief generated ({len(answer)} chars)")
    return {**state, "final_answer": answer}