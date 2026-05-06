import requests
import json
import re

SESSION = requests.Session()

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "phi3:mini"


# -----------------------------------
# 🔹 Robust JSON Extraction
# -----------------------------------
def extract_json(output):
    """
    Extract valid JSON from messy LLM output
    """

    matches = re.findall(
        r"\{.*?\}",
        output,
        re.DOTALL
    )

    for m in matches:

        try:
            return json.loads(m)

        except:
            continue

    raise ValueError("No valid JSON found")


# -----------------------------------
# 🔹 Query Cleaner
# -----------------------------------
def clean_query(text):

    text = text.lower().strip()

    # remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    # basic spelling fixes
    fixes = {
        "symtoms": "symptoms",
        "tretment": "treatment",
        "diagnsis": "diagnosis",
        "chemo therapy": "chemotherapy",
        "immuno therapy": "immunotherapy",
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    return text


# -----------------------------------
# 🔹 Intent-aware Enrichment
# -----------------------------------
def enrich_query(query, intent):

    query = clean_query(query)

    # factual
    if intent == "factual":

        extra = (
            " symptoms signs causes diagnosis clinical features"
        )

    # comparison
    elif intent == "comparison":

        extra = (
            " comparison differences advantages disadvantages effectiveness risks"
        )

    # exploratory
    else:

        extra = (
            " detailed clinical explanation mechanisms pathology treatment latest advances"
        )

    # 🔥 Avoid duplicate expansion
    existing = set(query.split())

    extra_words = [
        w for w in extra.split()
        if w not in existing
    ]

    enriched = (
        query + " " + " ".join(extra_words)
    ).strip()

    return enriched[:350]


# -----------------------------------
# 🔹 Dynamic Retrieval K
# -----------------------------------
def choose_k(intent):

    if intent == "exploratory":
        return 8

    elif intent == "comparison":
        return 6

    return 5


# -----------------------------------
# 🔹 Weak Query Detection
# -----------------------------------
def improve_weak_query(query):

    words = query.split()

    if len(words) <= 3:

        extra = (
            " symptoms causes diagnosis treatment"
        )

        existing = set(words)

        extra_words = [
            w for w in extra.split()
            if w not in existing
        ]

        query += " " + " ".join(extra_words)

    return query.strip()


# -----------------------------------
# 🔹 MAIN QUERY PROCESSOR
# -----------------------------------
def process_query(query):

    query = clean_query(query)

    prompt = f"""
You are a medical query analysis system.

Your tasks:
1. Detect intent:
   - factual
   - comparison
   - exploratory

2. Extract important medical keywords

3. Rewrite the query for medical document retrieval.

STRICT RULES:
- ONLY return valid JSON
- No markdown
- No explanations
- No extra text
- Do NOT reject queries
- Do NOT hallucinate diseases or treatments
- Keep retrieval query medically relevant

JSON FORMAT:

{{
  "intent": "factual | comparison | exploratory",
  "keywords": ["...", "..."],
  "expanded_query": "..."
}}

USER QUERY:
{query}
"""

    try:

        response = SESSION.post(
            OLLAMA_URL,
            json={
                "model": MODEL,

                "prompt": prompt,

                "stream": False,

                "options": {
                    "temperature": 0,

                    "top_p": 0.9,

                    "num_predict": 180,

                    "keep_alive": "30m"
                }
            },
            timeout=60
        )

        data = response.json()

        output = data.get(
            "response",
            ""
        ).strip()

        print("\n🧠 LAQA RAW OUTPUT:\n")
        print(output)

        parsed = extract_json(output)

    except Exception as e:

        print("⚠️ LAQA fallback:", e)

        parsed = {
            "intent": "factual",
            "keywords": [],
            "expanded_query": query
        }

    # -----------------------------------
    # 🔹 Normalize Intent
    # -----------------------------------
    intent = parsed.get(
        "intent",
        "factual"
    ).lower()

    if intent not in [
        "factual",
        "comparison",
        "exploratory"
    ]:
        intent = "factual"

    # -----------------------------------
    # 🔹 Query Expansion
    # -----------------------------------
    expanded_query = parsed.get(
        "expanded_query",
        query
    )

    expanded_query = clean_query(
        expanded_query
    )

    expanded_query = improve_weak_query(
        expanded_query
    )

    expanded_query = enrich_query(
        expanded_query,
        intent
    )

    # -----------------------------------
    # 🔹 Final Output
    # -----------------------------------
    final_output = {

        "intent": intent,

        "keywords": parsed.get(
            "keywords",
            []
        ),

        "expanded_query": expanded_query,

        "retrieval_k": choose_k(intent),

        "original_query": query
    }

    # -----------------------------------
    # 🔹 DEBUG
    # -----------------------------------
    print("\n🧠 LAQA PARSED:")

    print(
        json.dumps(
            final_output,
            indent=2
        )
    )

    return final_output