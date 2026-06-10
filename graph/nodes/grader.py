from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def grader_node(state: dict) -> dict:
    print(f"\n[Grader] Scoring answer quality...")

    if not state.get("final_answer"):
        return {**state, "quality_score": 0.0}

    prompt = f"""Rate the quality of this financial answer on a scale from 0.0 to 1.0.

Question: {state['query']}
Answer: {state['final_answer']}

Scoring criteria:
- Contains specific numbers or data points (0.3 points)
- Directly addresses the question (0.3 points)  
- Has clear recommendation or conclusion (0.2 points)
- Well structured and coherent (0.2 points)

Respond with ONLY a decimal number between 0.0 and 1.0. Nothing else."""

    try:
        score = float(llm.invoke(prompt).content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.75

    print(f"[Grader] Quality score: {score}")
    return {**state, "quality_score": score, "retry_count": state.get("retry_count", 0) + 1}

def should_retry(state: dict) -> str:
    if state.get("quality_score", 0) >= 0.75 or state.get("retry_count", 0) >= 2:
        return "end"
    print(f"[Grader] Score too low, retrying...")
    return "retry"