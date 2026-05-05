import requests
import json
import re

SESSION = requests.Session()


# -------------------------------
# 🔹 Robust JSON Extraction
# -------------------------------
def extract_json(output):
    """
    Extract valid JSON from messy LLM output
    """
    matches = re.findall(r"\{.*?\}", output, re.DOTALL)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    raise ValueError("No valid JSON found")


# -------------------------------
# 🔹 Query Processing (LAQA)
# -------------------------------
def process_query(query):

    query = query.strip()

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
        response = SESSION.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "medgemma-rag",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "keep_alive": "10m"
                }
            },
            timeout=120
        )

        data = response.json()
        output = data.get("response", "").strip()

        print("\n🧠 LAQA RAW OUTPUT:\n", output)

        # 🔥 Use robust extractor
        parsed = extract_json(output)

    except Exception as e:
        print("⚠️ LAQA fallback:", e)

        parsed = {
            "intent": "factual",
            "keywords": [],
            "expanded_query": query
        }

    # -------------------------------
    # 🔹 Normalize + Improve Query
    # -------------------------------
    intent = parsed.get("intent", "factual")
    expanded_query = parsed.get("expanded_query", query)

    # 🔥 Clean query
    expanded_query = expanded_query.lower().strip()

    # 🔥 Improve weak queries (VERY IMPORTANT)
    if len(expanded_query.split()) <= 3:
        expanded_query += " symptoms causes diagnosis treatment"

    # 🔥 Handle spelling issues lightly
    if "symtoms" in expanded_query:
        expanded_query = expanded_query.replace("symtoms", "symptoms")

    parsed["expanded_query"] = expanded_query

    # -------------------------------
    # 🔹 Dynamic retrieval_k
    # -------------------------------
    if intent == "exploratory":
        parsed["retrieval_k"] = 7
    elif intent == "comparison":
        parsed["retrieval_k"] = 6
    else:
        parsed["retrieval_k"] = 5

    parsed["original_query"] = query

    # -------------------------------
    # 🔹 Debug
    # -------------------------------
    print("LAQA PARSED:", parsed)

    return parsed