# app/prompts.py

def build_prompt(context: str, query: str) -> str:
    return f"""You are a helpful insurance assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
