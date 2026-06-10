from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rag.ingest_pdfs import load_index

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst. Answer the question using ONLY the provided context from the company's annual report.
Be specific — mention actual numbers, percentages, and facts from the document.
If the context doesn't contain enough information, say so clearly.

Context:
{context}"""),
    ("human", "{question}")
])

def pdf_rag_node(state: dict) -> dict:
    if "pdf" not in state.get("agents_to_use", []):
        return state

    question = state["sub_tasks"].get("pdf", state["query"])
    print(f"\n[PDF RAG] Question: {question}")

    retriever = load_index().as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    chain = PROMPT | llm
    answer = chain.invoke({"context": context, "question": question})

    print(f"[PDF RAG] Answer length: {len(answer.content)} chars")
    return {**state, "pdf_answer": answer.content}