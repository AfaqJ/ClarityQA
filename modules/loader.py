import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str) -> List[str]:
    # Detect file type
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")

    # Load document as LangChain Document object
    documents = loader.load()
    for doc in documents:
        doc.page_content=doc.page_content.replace("\n"," ")
    # Split into chunks using LangChain's recursive splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    # Return only the text content
    return [doc.page_content for doc in split_docs]
