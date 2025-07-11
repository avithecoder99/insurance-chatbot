# app/retriever.py

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from app.prompts import build_prompt
from openai import OpenAI

load_dotenv()

def get_faiss_index(index_dir="faiss_index"):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return FAISS.load_local(index_dir, embeddings,allow_dangerous_deserialization=True)

def retrieve_context(query, k=3):
    vectorstore = get_faiss_index()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

import openai

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt4_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()
