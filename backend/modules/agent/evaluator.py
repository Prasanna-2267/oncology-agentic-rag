import requests
import json
import re

PHI3MINI_URL = "http://localhost:11434/api/generate"
EVAL_MODEL = "phi3:mini"


def evaluate_answer(query, context, answer):
    """
    Evaluate answer quality using LLM
    Returns:
    {
        score: 0-10,
        confidence: 0-1,
        needs_retry: bool
    }
    """

    prompt = f"""
You are an evaluation system.

STRICT RULES:
- DO NOT explain
- DO NOT think step-by-step
- DO NOT output anything except JSON
- DO NOT include <thought> or extra text

Return EXACTLY in this format:

{{"score": 0-10, "confidence": 0-1, "needs_retry": true/false}}

Evaluation criteria:
- If answer says "not enough information" → needs_retry = true
- If answer is irrelevant → needs_retry = true
- If answer is partial → needs_retry = true
- If answer is correct and complete → needs_retry = false

Question: {query}
Context: {context}
Answer: {answer}
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
                    "num_predict": 80,
                    "keep_alive": "10m"
                }
            },
            timeout=60
        )

        data = response.json()
        raw_output = data.get("response", "").strip()

        print("\n🧠 EVAL RAW OUTPUT:\n", raw_output)

        # -----------------------------
        # 🔹 Extract JSON safely
        # -----------------------------
        match = re.search(r"\{.*?\}", raw_output, re.DOTALL)

        if not match:
            raise ValueError("No JSON found")

        json_str = match.group(0)

        # -----------------------------
        # 🔹 Parse JSON
        # -----------------------------
        parsed = json.loads(json_str)

        score = parsed.get("score", 5)
        confidence = parsed.get("confidence", 0.5)
        needs_retry = parsed.get("needs_retry", True)

        # -----------------------------
        # 🔹 Normalize values
        # -----------------------------
        try:
            score = int(score)
        except:
            score = 5

        try:
            confidence = float(confidence)
        except:
            confidence = 0.5

        needs_retry = bool(needs_retry)

        # clamp values
        score = max(0, min(score, 10))
        confidence = max(0.0, min(confidence, 1.0))

        return {
            "score": score,
            "confidence": confidence,
            "needs_retry": needs_retry
        }

    except Exception as e:
        print("❌ EVAL ERROR:", e)

        # -----------------------------
        # 🔹 Smart fallback
        # -----------------------------
        answer_lower = answer.lower()

        if "not enough information" in answer_lower:
            return {
                "score": 2,
                "confidence": 0.3,
                "needs_retry": True
            }

        if len(answer.strip()) < 50:
            return {
                "score": 3,
                "confidence": 0.4,
                "needs_retry": True
            }

        # medium fallback
        return {
            "score": 5,
            "confidence": 0.5,
            "needs_retry": True
        }