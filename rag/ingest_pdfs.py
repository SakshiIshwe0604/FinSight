import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings

PDF_DIR = "data/pdfs"
INDEX_PATH = "data/faiss_index"

def build_index():
    all_docs = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/pdfs/ — add annual report PDFs first")
        return

    print(f"Loading {len(pdf_files)} PDF(s)...")
    for fname in pdf_files:
        path = os.path.join(PDF_DIR, fname)
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"  {fname}: {len(docs)} pages")
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    print("Building FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}")

def load_index():
    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

if __name__ == "__main__":
    build_index()
