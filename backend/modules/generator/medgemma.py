import requests

def generate_answer(agent_output):
    query = agent_output["query"]["expanded_query"]
    intent = agent_output["query"].get("intent", "factual")  # 🔥 new
    context = agent_output["context"][0][:800]

    prompt = f"""
You are an oncology AI assistant.

Query Type: {intent}

STRICT RULES:
- Answer ONLY using the given context
- DO NOT hallucinate or use outside knowledge
- If answer is not in context → say "Not enough information"
- DO NOT include thinking text, reasoning tags, or <unused*> tokens

USER REQUIREMENT:
- Answer MUST be in bullet points
- Keep points short and clear
- 3–6 points maximum

FORMAT BASED ON QUERY:

If factual:
- point 1
- point 2

If comparison:
- Feature 1: ...
- Feature 2: ...

If exploratory:
- Explanation point 1
- Explanation point 2

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
        "temperature": 0,
        "top_p": 0.9,
        "keep_alive": "10m"   # 🔥 ADD THIS
    }
}
    )

    return response.json()["response"]