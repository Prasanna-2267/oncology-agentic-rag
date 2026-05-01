def generate_explanation(answer, docs, eval_result, query):

    explanation = {}

    # 🔹 Supporting sentences
    explanation["supporting_sentences"] = extract_supporting_sentences(answer, docs)

    # 🔹 Confidence
    explanation["confidence"] = eval_result.get("confidence", 0.5)

    # 🔹 Quality
    score = eval_result.get("score", 5)
    explanation["quality"] = (
        "High" if score >= 7 else
        "Medium" if score >= 4 else
        "Low"
    )

    # 🔹 Grounding
    explanation["grounded"] = "not enough information" not in answer.lower()

    # 🔹 Reasoning (LLM)
    explanation["reasoning"] = generate_reasoning(query, answer, docs)

    return explanation

import re

def extract_supporting_sentences(answer, docs):
    answer_sentences = re.split(r'(?<=[.!?]) +', answer)

    supporting = []

    for doc in docs:
        doc_sentences = re.split(r'(?<=[.!?]) +', doc)

        for a_sent in answer_sentences:
            for d_sent in doc_sentences:
                # simple overlap check
                common_words = set(a_sent.lower().split()) & set(d_sent.lower().split())

                if len(common_words) > 3:   # threshold
                    supporting.append(d_sent.strip())

    # remove duplicates
    return list(set(supporting))[:5]

import requests

def generate_reasoning(query, answer, docs):

    context = " ".join(docs[:1])[:1200]

    prompt = f"""
You are verifying an answer using given context.

Task:
Explain briefly WHY the answer is supported by the context.

Rules:
- DO NOT say "not enough information"
- DO NOT be uncertain
- ALWAYS find supporting evidence
- Keep it short (2-3 lines)

Context:
{context}

Answer:
{answer}
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

    return response.json().get("response", "").strip()