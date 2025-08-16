# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from retriever import retrieve_context, get_gpt4_answer
from prompts import build_prompt

app = FastAPI()

# Define the request schema
class QueryInput(BaseModel):
    query: str

# Query input endpoint (testable)
@app.post("/query")
def handle_query(data: QueryInput):
    context = retrieve_context(data.query)
    prompt = build_prompt(data.query, context)
    answer = get_gpt4_answer(prompt)
    return {
        "query": data.query,
        "answer":answer,
        "source_context":context
    }

# Health check
@app.get("/")
def root():
    return {"message": "GPT Insurance Chatbot Backend is running."}


