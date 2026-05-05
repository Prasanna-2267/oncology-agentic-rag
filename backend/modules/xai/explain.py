import re
import requests

SESSION = requests.Session()


# -------------------------------
# 🔹 MAIN EXPLANATION FUNCTION
# -------------------------------
def generate_explanation(answer, docs, eval_result, query):

    explanation = {}

    explanation["supporting_sentences"] = extract_supporting_sentences(answer, docs)

    explanation["confidence"] = float(eval_result.get("confidence", 0.5))

    score = eval_result.get("score", 5)
    explanation["quality"] = (
        "High" if score >= 7 else
        "Medium" if score >= 4 else
        "Low"
    )

    explanation["grounded"] = "not enough information" not in answer.lower()

    # 🔥 FAST PATH (skip LLM)
    if explanation["confidence"] >= 0.8:
        explanation["reasoning"] = "High-confidence answer based on strong document overlap."
        return explanation

    if len(answer) > 150 and explanation["confidence"] >= 0.7:
        explanation["reasoning"] = "Answer is well-supported by retrieved documents."
        return explanation

    explanation["reasoning"] = generate_reasoning(query, answer, docs)

    return explanation


# -------------------------------
# 🔹 SUPPORTING SENTENCES
# -------------------------------
def extract_supporting_sentences(answer, docs):

    answer_words = set(answer.lower().split())
    scored_sentences = []

    for doc in docs[:2]:
        sentences = re.split(r'(?<=[.!?]) +', doc)

        for sent in sentences:
            words = set(sent.lower().split())
            overlap = len(answer_words & words)

            # 🔥 Improved adaptive threshold
            if overlap > 2 and len(sent) > 40:
                scored_sentences.append((overlap, sent.strip()))

    scored_sentences.sort(reverse=True, key=lambda x: x[0])

    final = []
    for _, s in scored_sentences[:3]:
        if len(s) > 120:
            s = s[:120] + "..."
        final.append(s)

    return final


# -------------------------------
# 🔹 LLM REASONING
# -------------------------------
def generate_reasoning(query, answer, docs):

    context = "\n".join(docs[:2])[:1000]

    prompt = f"""
You are verifying a medical answer using context.

STRICT RULES:
- DO NOT hallucinate
- Only use provided context
- Output MUST be bullet points
- Only 2 points

Context:
{context}

Answer:
{answer}
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
            timeout=60
        )

        raw = response.json().get("response", "").strip()

        cleaned = re.sub(r"<.*?>", "", raw)
        cleaned = cleaned.replace("\n", " ").strip()

        return cleaned if cleaned else "Reasoning not available."

    except Exception as e:
        print("⚠️ Reasoning failed:", e)
        return "Reasoning not available."