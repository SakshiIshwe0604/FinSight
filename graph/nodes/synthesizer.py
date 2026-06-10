from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def synthesizer_node(state: dict) -> dict:
    print(f"\n[Synthesizer] Merging agent answers...")

    parts = []
    if state.get("pdf_answer"):
        parts.append(f"[From Annual Report]:\n{state['pdf_answer']}")
    if state.get("live_answer"):
        parts.append(f"[From Live Market Data]:\n{state['live_answer']}")

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