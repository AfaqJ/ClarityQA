

import streamlit as st
import os
import time
from modules.loader import load_and_split_document
from modules.vector_store import store_embeddings
from modules.llm_chain import get_llm_chain
from langchain_community.vectorstores import Chroma  # 🔧 FIX
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient

TEMP_DOCS_PATH = "temp_docs"
VECTOR_DB_PATH = "chroma_db"
os.makedirs(TEMP_DOCS_PATH, exist_ok=True)

st.title("📄 Chat with Your Document")
st.caption("Accepted formats: PDF, TXT")

# ─── Minimal session state ───────────────────────────────────────────────
for key, default in {
    "vectordb": None,
    "retriever": None,
    "file_name": None,
    "answer": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Placeholder to avoid duplicate UI blocks
if "qa_placeholder" not in st.session_state:
    st.session_state["qa_placeholder"] = st.empty()
# ──────────────────────────────────────────────────────────────────────────

def delete_chroma_collection():
    try:
        client = PersistentClient(path=VECTOR_DB_PATH)
        client.delete_collection("langchain")
    except Exception as e:
        st.warning(f"Could not delete Chroma collection: {e}")

# ─── File uploader on the sidebar ────────────────────────────────────────
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

# ─── Handle new upload ─────────────────────────────────────────────────--
if uploaded_file is not None and uploaded_file.name != st.session_state["file_name"]:
    delete_chroma_collection()
    for fname in os.listdir(TEMP_DOCS_PATH):
        os.remove(os.path.join(TEMP_DOCS_PATH, fname))

    file_path = os.path.join(TEMP_DOCS_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["file_name"] = uploaded_file.name

    status = st.empty()
    status.success(f"✅ File uploaded successfully: {uploaded_file.name}")

    with st.spinner("🔍 Processing and splitting document..."):
        time.sleep(1)
        chunks = load_and_split_document(file_path)
    status.success("✅ Document chunked successfully")

    with st.spinner("📦 Embedding & indexing..."):
        retriever = store_embeddings(chunks, persist_directory=VECTOR_DB_PATH)
        dummy = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state["vectordb"] = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=dummy)
        st.session_state["retriever"] = retriever
    status.success("✅ Embedding & indexing completed")

    st.session_state["answer"] = ""  # reset answer on new upload

# ─── Q&A interface (single placeholder, no duplicates) ───────────────────
if st.session_state["retriever"] is not None:
    # clear old content each run to avoid duplicates
    st.session_state["qa_placeholder"].empty()
    with st.session_state["qa_placeholder"].container():
        st.header("Ask questions about your document")
        with st.form(key="qa_form", clear_on_submit=False):
            query = st.text_input("Your question", key="query_input")
            submitted = st.form_submit_button("Submit 💬")
        if submitted and query:            
            with st.spinner("🤖 Generating answer..."):
                results = st.session_state["retriever"].invoke(query)
                context = "\n".join(doc.page_content for doc in results)
                llmchain = get_llm_chain()
                response = llmchain.invoke({"context": context, "question": query})
                st.session_state["answer"] = response["text"]

        if st.session_state["answer"]:
            st.markdown("### 💬 Answer")
            st.write(st.session_state["answer"])

        if st.button("🗑️ End Session"):
            delete_chroma_collection()
            for fname in os.listdir(TEMP_DOCS_PATH):
                os.remove(os.path.join(TEMP_DOCS_PATH, fname))
            for key in ["vectordb", "retriever", "file_name", "answer"]:
                st.session_state[key] = None
            st.success("✅ Memory cleared. Upload a new file to start again!")
            # clear Q&A block on next run
            st.session_state["qa_placeholder"].empty()
