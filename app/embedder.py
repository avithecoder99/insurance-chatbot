# app/embedder.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

def load_and_chunk_documents(data_path: str):
    all_chunks = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, filename))
            pages = loader.load_and_split()

            for i, page in enumerate(pages):
                # Attach metadata to each page (chunk)
                page.metadata["source"] = filename
                page.metadata["page_number"] = i + 1
                all_chunks.append(page)
    return all_chunks

def chunk_documents(chunks):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60
    )
    return splitter.split_documents(chunks)

def embed_and_store(docs, index_dir="faiss_index"):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")

    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_dir)

if __name__ == "__main__":
    raw_docs = load_and_chunk_documents("data/policy_docs")
    chunked_docs = chunk_documents(raw_docs)
    embed_and_store(chunked_docs)
    print("FAISS index with metadata created and saved.")
