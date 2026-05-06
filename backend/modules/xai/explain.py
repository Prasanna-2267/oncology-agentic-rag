import re
import requests

SESSION = requests.Session()


# -------------------------------
# 🔹 GROUNDING OVERLAP
# -------------------------------
def grounding_overlap(answer, docs):

    context = " ".join(docs).lower()

    # 🔥 Better token extraction
    answer_words = set(
        re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower())
    )

    if not answer_words:
        return 0

    overlap = sum(
        1 for w in answer_words
        if f" {w} " in f" {context} "
    )

    return overlap / len(answer_words)


# -------------------------------
# 🔹 MAIN EXPLANATION FUNCTION
# -------------------------------
def generate_explanation(answer, docs, eval_result, query):

    explanation = {}

    # -----------------------------------
    # 🔹 Supporting Evidence
    # -----------------------------------
    explanation["supporting_sentences"] = (
        extract_supporting_sentences(
            answer,
            docs
        )
    )

    # -----------------------------------
    # 🔹 Confidence Calibration
    # -----------------------------------
    eval_conf = eval_result.get(
        "confidence",
        0.75
    )

    retrieval_score = eval_result.get(
        "retrieval_score",
        0.5
    )

    grounding_score = grounding_overlap(
        answer,
        docs
    )

    # 🔥 Hybrid calibrated confidence
    confidence = (
        0.5 * eval_conf +
        0.3 * retrieval_score +
        0.2 * grounding_score
    )

    # 🔥 Safer upper bound
    confidence = round(
        min(confidence, 0.92),
        2
    )

    explanation["confidence"] = confidence

    # -----------------------------------
    # 🔹 Quality Estimation
    # -----------------------------------
    score = eval_result.get("score", 5)

    explanation["quality"] = (
        "High" if score >= 7 else
        "Medium" if score >= 4 else
        "Low"
    )

    # -----------------------------------
    # 🔹 Grounding Status
    # -----------------------------------
    explanation["grounded"] = (
        "not enough information"
        not in answer.lower()
    )

    # -----------------------------------
    # 🔹 FAST PATH (skip LLM reasoning)
    # -----------------------------------
    if (
        explanation["confidence"] >= 0.8
        and len(answer.split()) > 20
    ):

        explanation["reasoning"] = (
            "High-confidence answer supported by strong retrieval overlap and grounded medical context."
        )

        return explanation

    if (
        len(answer) > 150
        and explanation["confidence"] >= 0.7
    ):

        explanation["reasoning"] = (
            "The answer is reasonably supported by the retrieved oncology documents."
        )

        return explanation

    # -----------------------------------
    # 🔹 LLM-based Reasoning
    # -----------------------------------
    explanation["reasoning"] = generate_reasoning(
        query,
        answer,
        docs
    )

    return explanation


# -------------------------------
# 🔹 SUPPORTING SENTENCES
# -------------------------------
def extract_supporting_sentences(answer, docs):

    answer_words = set(
        re.findall(
            r'\b[a-zA-Z]{3,}\b',
            answer.lower()
        )
    )

    scored_sentences = []

    for doc in docs[:2]:

        sentences = re.split(
            r'(?<=[.!?]) +',
            doc
        )

        for sent in sentences:

            words = set(
                re.findall(
                    r'\b[a-zA-Z]{3,}\b',
                    sent.lower()
                )
            )

            overlap = len(answer_words & words)

            # 🔥 Better filtering
            if (
                overlap > 2
                and len(sent) > 40
            ):

                scored_sentences.append(
                    (overlap, sent.strip())
                )

    # 🔥 Rank by overlap
    scored_sentences.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    final = []

    for _, sent in scored_sentences[:3]:

        if len(sent) > 140:
            sent = sent[:140] + "..."

        final.append(sent)

    return final


# -------------------------------
# 🔹 LLM REASONING
# -------------------------------
def generate_reasoning(query, answer, docs):

    context = "\n".join(docs[:2])[:1000]

    prompt = f"""
You are verifying whether a medical answer is supported by retrieved context.

STRICT RULES:
- ONLY use provided context
- DO NOT hallucinate
- Output EXACTLY 2 bullet points
- Mention whether the answer matches the retrieved medical evidence
- Keep concise
- No markdown headers
- No XML tags
- No chain-of-thought

CONTEXT:
{context}

ANSWER:
{answer}

VERIFICATION:
"""

    try:

        response = SESSION.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",

                "prompt": prompt,

                "stream": False,

                "options": {
                    "temperature": 0,

                    "num_predict": 80,

                    "keep_alive": "10m"
                }
            },
            timeout=60
        )

        raw = response.json().get(
            "response",
            ""
        ).strip()

        # -----------------------------------
        # 🔹 CLEANUP
        # -----------------------------------
        cleaned = re.sub(
            r"<.*?>",
            "",
            raw
        )

        cleaned = cleaned.replace(
            "\n\n\n",
            "\n"
        )

        cleaned = cleaned.strip()

        if len(cleaned) < 10:
            return (
                "Reasoning could not be generated reliably."
            )

        return cleaned

    except Exception as e:

        print("⚠️ Reasoning failed:", e)

        return (
            "Reasoning could not be generated reliably."
        )