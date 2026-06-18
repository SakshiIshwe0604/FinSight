from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag.ingest_pdfs import load_index

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def compare_node(state: dict) -> dict:
    if not state.get("is_compare"):
        return state

    company1 = state.get("company1", "Company 1")
    company2 = state.get("company2", "Company 2")
    query = state.get("query", "")

    print(f"\n[Compare] Comparing {company1} vs {company2}")

    # get RAG answers for both companies
    retriever = load_index().as_retriever(search_kwargs={"k": 4})

    q1 = f"{query} for {company1}"
    q2 = f"{query} for {company2}"

    docs1 = retriever.invoke(q1)
    docs2 = retriever.invoke(q2)

    context1 = "\n\n".join(d.page_content for d in docs1)
    context2 = "\n\n".join(d.page_content for d in docs2)

    # generate comparison brief
    prompt = f"""You are a senior financial analyst. Compare {company1} and {company2} based on the research below.

User Question: {query}

{company1} Data:
{context1}

{company2} Data:
{context2}

Respond in this exact structure:

RECOMMENDATION: [which company is better and why in one line]

{company1.upper()} OUTLOOK:
[2-3 sentences on performance and growth]

{company2.upper()} OUTLOOK:
[2-3 sentences on performance and growth]

KEY DIFFERENCES:
- [Difference 1]
- [Difference 2]
- [Difference 3]

WINNER: [company name]
[1 sentence final verdict on which is better investment]"""

    answer = llm.invoke(prompt).content
    print(f"[Compare] Brief generated ({len(answer)} chars)")
    return {**state, "final_answer": answer}