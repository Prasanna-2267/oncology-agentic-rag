import requests
import json

def process_query(query):

    prompt = f"""
You are a query analysis system.

Your job is ONLY to:
1. Classify the query into:
   - factual
   - comparison
   - exploratory

2. Extract keywords
3. Rewrite query for better retrieval

IMPORTANT:
- DO NOT reject the query
- DO NOT say "outside scope"
- ALWAYS return JSON
- DO NOT include any extra text

Return ONLY valid JSON:

{{
  "intent": "factual | comparison | exploratory",
  "keywords": ["...", "..."],
  "expanded_query": "..."
}}

Query:
{query}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
    "model": "medgemma-rag",
    "prompt": prompt,
    "stream": False,
    "options": {
        "keep_alive": "10m"   # 🔥 ADD THIS
    }
},
            timeout=120
        )

        data = response.json()

        output = data.get("response", "").strip()

        # 🧠 Extract JSON safely (handles messy LLM output)
        start = output.find("{")
        end = output.rfind("}") + 1

        if start != -1 and end != -1:
            json_str = output[start:end]
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:

        # 🔁 fallback (system should never break)
        parsed = {
            "intent": "factual",
            "keywords": [],
            "expanded_query": query
        }

    # 🔹 Retrieval strategy mapping
    intent = parsed.get("intent", "factual")

    if intent == "exploratory":
        parsed["retrieval_k"] = 6
    elif intent == "comparison":
        parsed["retrieval_k"] = 5
    else:
        parsed["retrieval_k"] = 5

    parsed["original_query"] = query

    # 🔍 Debug output
    print("LAQA PARSED:", parsed)

    return parsed