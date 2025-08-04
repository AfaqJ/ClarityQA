from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_embeddings(chunks, persist_directory: str = "chroma_db"):
    # 1) Prepare documents
    docs = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
            for i, chunk in enumerate(chunks)]

    # 2) Build & persist the vector store
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=persist_directory
    )
    vectordb.persist()

    # 3) Close its connection immediately
    try:
        vectordb._persist_client.close()
    except:
        pass

    # 4) Return only the retriever
    return vectordb.as_retriever()