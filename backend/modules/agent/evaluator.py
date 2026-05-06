import requests
import json
import re

PHI3MINI_URL = "http://localhost:11434/api/generate"

EVAL_MODEL = "phi3:mini"


# -----------------------------------
# 🔹 JSON Extraction
# -----------------------------------
def extract_json(output):

    match = re.search(
        r"\{.*?\}",
        output,
        re.DOTALL
    )

    if not match:
        raise ValueError("No JSON found")

    return json.loads(match.group(0))


# -----------------------------------
# 🔹 MAIN EVALUATOR
# -----------------------------------
def evaluate_answer(
    query,
    context,
    answer
):

    prompt = f"""
You are evaluating a medical AI answer.

Your job:
1. Check whether the answer is supported by context
2. Check whether the answer actually answers the question
3. Detect hallucination risk
4. Detect incomplete answers

STRICT RULES:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text

JSON FORMAT:

{{
  "score": 0-10,
  "confidence": 0-1,
  "needs_retry": true/false,
  "answered_question": true/false,
  "answer_relevance": 0-1,
  "hallucination_risk": "low|medium|high",
  "missing_information": true/false
}}

SCORING RULES:
- High score ONLY if:
  - answer is grounded
  - relevant
  - complete
  - medically accurate

- If answer says:
  "not enough information"
  → needs_retry = true

- If answer is generic or partially relevant
  → answered_question = false

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
"""

    try:

        response = requests.post(
            PHI3MINI_URL,
            json={
                "model": EVAL_MODEL,

                "prompt": prompt,

                "stream": False,

                "options": {
                    "temperature": 0,

                    "num_predict": 150,

                    "keep_alive": "30m"
                }
            },
            timeout=60
        )

        data = response.json()

        raw_output = data.get(
            "response",
            ""
        ).strip()

        print("\n🧠 EVAL RAW OUTPUT:\n")
        print(raw_output)

        parsed = extract_json(raw_output)

        # -----------------------------------
        # 🔹 Normalize Values
        # -----------------------------------
        score = int(
            parsed.get("score", 5)
        )

        confidence = float(
            parsed.get("confidence", 0.5)
        )

        needs_retry = bool(
            parsed.get("needs_retry", True)
        )

        answered_question = bool(
            parsed.get("answered_question", True)
        )

        answer_relevance = float(
            parsed.get("answer_relevance", 0.5)
        )

        hallucination_risk = str(
            parsed.get(
                "hallucination_risk",
                "medium"
            )
        ).lower()

        missing_information = bool(
            parsed.get(
                "missing_information",
                False
            )
        )

        # -----------------------------------
        # 🔹 Clamp Values
        # -----------------------------------
        score = max(
            0,
            min(score, 10)
        )

        confidence = max(
            0,
            min(confidence, 1)
        )

        answer_relevance = max(
            0,
            min(answer_relevance, 1)
        )

        if hallucination_risk not in [
            "low",
            "medium",
            "high"
        ]:
            hallucination_risk = "medium"

        # -----------------------------------
        # 🔹 Smart Retry Overrides
        # -----------------------------------
        answer_lower = answer.lower()

        if (
            "not enough information"
            in answer_lower
        ):
            needs_retry = True

        if not answered_question:
            needs_retry = True

        if hallucination_risk == "high":
            needs_retry = True

        if answer_relevance < 0.45:
            needs_retry = True

        return {
            "score": score,

            "confidence": confidence,

            "needs_retry": needs_retry,

            "answered_question": answered_question,

            "answer_relevance": answer_relevance,

            "hallucination_risk": hallucination_risk,

            "missing_information": missing_information
        }

    except Exception as e:

        print("❌ EVAL ERROR:", e)

        # -----------------------------------
        # 🔹 Fallback Heuristics
        # -----------------------------------
        answer_lower = answer.lower()

        if (
            "not enough information"
            in answer_lower
        ):
            return {
                "score": 2,
                "confidence": 0.3,
                "needs_retry": True,
                "answered_question": False,
                "answer_relevance": 0.2,
                "hallucination_risk": "low",
                "missing_information": True
            }

        if len(answer.strip()) < 50:
            return {
                "score": 3,
                "confidence": 0.4,
                "needs_retry": True,
                "answered_question": False,
                "answer_relevance": 0.3,
                "hallucination_risk": "medium",
                "missing_information": True
            }

        return {
            "score": 5,
            "confidence": 0.5,
            "needs_retry": True,
            "answered_question": True,
            "answer_relevance": 0.5,
            "hallucination_risk": "medium",
            "missing_information": False
        }