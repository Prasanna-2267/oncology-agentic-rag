import requests
import json
import re

PHI3MINI_URL = "http://localhost:11434/api/generate"
EVAL_MODEL = "phi3:mini"   # 🔥 IMPORTANT: use llama3 (NOT medgemma)


def evaluate_answer(query, context, answer):

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
                    "temperature": 0,     # 🔥 deterministic
                    "num_predict": 80,     # 🔥 short output
                    "keep_alive": "10m"   # 🔥 ADD THIS
                }
            },
            timeout=60
        )

        data = response.json()
        raw_output = data.get("response", "").strip()

        print("\n🧠 EVAL RAW OUTPUT:\n", raw_output)

        # 🔥 Extract JSON safely
        match = re.search(r'\{.*?\}', raw_output, re.DOTALL)

        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)

            # 🛡️ Safety normalization
            score = parsed.get("score")
            confidence = parsed.get("confidence")
            needs_retry = parsed.get("needs_retry")

            # 🔥 Handle nulls safely
            if score is None:
                score = 5
            if confidence is None:
                confidence = 0.5
            if needs_retry is None:
                needs_retry = True

            return {
                "score": int(score),
                "confidence": float(confidence),
                "needs_retry": bool(needs_retry)
            }

            return parsed

        else:
            raise ValueError("No JSON found in output")

    except Exception as e:
        print("❌ EVAL ERROR:", e)

        # 🔥 SMART fallback (not dumb fallback anymore)
        answer_lower = answer.lower()

        if "not enough information" in answer_lower:
            return {
                "score": 2,
                "confidence": 0.3,
                "needs_retry": True
            }

        # fallback assume medium quality
        return {
            "score": 5,
            "confidence": 0.5,
            "needs_retry": True
        }