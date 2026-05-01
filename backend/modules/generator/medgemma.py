import requests

def generate_answer(agent_output):
    query = agent_output["query"]["expanded_query"]
    intent = agent_output["query"].get("intent", "factual")  # 🔥 new
    context = agent_output["context"][0][:800]

    prompt = f"""
You are a medical AI assistant.

Query Type: {intent}

- If factual → be concise
- If comparison → give structured comparison
- If exploratory → explain in detail

Answer ONLY using the given context.
If the answer is not in the context, say "Not enough information".

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
    "model": "medgemma-rag",
    "prompt": prompt,
    "stream": False,
    "options": {
        "keep_alive": "10m"   # 🔥 ADD THIS
    }
}
    )

    return response.json()["response"]