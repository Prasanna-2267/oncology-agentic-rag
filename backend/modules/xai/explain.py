import re
import requests


# -------------------------------
# 🔹 MAIN EXPLANATION FUNCTION
# -------------------------------
def generate_explanation(answer, docs, eval_result, query):

    explanation = {}

    # 🔹 Supporting sentences
    explanation["supporting_sentences"] = extract_supporting_sentences(answer, docs)

    # 🔹 Confidence
    explanation["confidence"] = float(eval_result.get("confidence", 0.5))

    # 🔹 Quality
    score = eval_result.get("score", 5)
    explanation["quality"] = (
        "High" if score >= 7 else
        "Medium" if score >= 4 else
        "Low"
    )

    # 🔹 Grounding
    explanation["grounded"] = "not enough information" not in answer.lower()

    # 🔥 SMART: Skip reasoning if answer already strong (speed boost)
    if len(answer) > 150 and explanation["confidence"] >= 0.7:
        explanation["reasoning"] = "Answer is well-supported by retrieved documents."
    else:
        explanation["reasoning"] = generate_reasoning(query, answer, docs)

    return explanation


# -------------------------------
# 🔹 SUPPORTING SENTENCES
# -------------------------------
def extract_supporting_sentences(answer, docs):

    answer_words = set(answer.lower().split())
    scored_sentences = []

    for doc in docs[:2]:
        doc_sentences = re.split(r'(?<=[.!?]) +', doc)

        for sent in doc_sentences:
            words = set(sent.lower().split())
            overlap = len(answer_words & words)

            if overlap > 4 and len(sent) > 40:
                scored_sentences.append((overlap, sent.strip()))

    # 🔥 sort by relevance
    scored_sentences.sort(reverse=True, key=lambda x: x[0])

    final = []
    for _, s in scored_sentences[:3]:
        if len(s) > 120:
            s = s[:120] + "..."
        final.append(s)

    return final


# -------------------------------
# 🔹 LLM REASONING (OPTIMIZED)
# -------------------------------
def generate_reasoning(query, answer, docs):

    context = " ".join(docs[:1])[:1000]  # 🔥 smaller context

    prompt = f"""
You are verifying a medical answer using context.

STRICT RULES:
- DO NOT hallucinate
- DO NOT use outside knowledge
- Only use provided context
- Output MUST be bullet points
- 2–4 points only
- No tags, no explanations, no extra text

USER REQUIREMENT:
- Reasoning must be point-wise

FORMAT:

Reasoning:
- point 1
- point 2

strictly follow the format above. only 2 small points are needed.dont exceed more than 2 points.

Context:
{context}

Answer:
{answer}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "medgemma-rag",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "keep_alive": "10m",
                    "temperature": 0
                }
            },
            timeout=60
        )

        raw = response.json().get("response", "").strip()

        # 🔥 CLEAN LLM OUTPUT
        cleaned = re.sub(r"<.*?>", "", raw)  # remove <unused...>
        cleaned = cleaned.replace("\n", " ").strip()

        return cleaned if cleaned else "Reasoning not available."

    except Exception as e:
        print("⚠️ Reasoning failed:", e)
        return "Reasoning not available."