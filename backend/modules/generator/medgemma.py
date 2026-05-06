import requests

SESSION = requests.Session()


# -----------------------------------
# 🔹 BUILD CONTEXT
# -----------------------------------
def build_context(docs, intent):

    if not docs:
        return ""

    # 🔥 Dynamic context sizing
    if intent == "exploratory":
        max_docs = 3

    elif intent == "comparison":
        max_docs = 2

    else:
        max_docs = 1

    selected_docs = docs[:max_docs]

    # 🔥 Limit noisy long context
    context = "\n\n".join(
        d[:1200] for d in selected_docs
    )

    return context.strip()


# -----------------------------------
# 🔹 ANSWER GENERATION
# -----------------------------------
def generate_answer(agent_output):

    query = agent_output["query"]["expanded_query"]

    intent = agent_output["query"].get(
        "intent",
        "factual"
    )

    docs = agent_output["context"]

    # 🔥 Improved context builder
    context = build_context(docs, intent)

    prompt = f"""
You are an oncology AI assistant.

You MUST answer ONLY using the provided medical context.

STRICT RULES:
- Do NOT use outside knowledge
- Do NOT hallucinate
- Do NOT invent treatments or symptoms
- If answer is missing from context, say:
  "Not enough information in retrieved medical context."

- Avoid generic filler
- Avoid repeating the question
- Avoid unnecessary explanations
- Keep answers medically focused

USER REQUIREMENT:
- Directly answer the question
- Use concise bullet points
- Keep only relevant medical details
- Maximum 3–6 points
- No markdown formatting
- No reasoning tags
- No chain-of-thought
- No XML tags
- No <unused*> tokens
- Important: Do NOT say "Based on the retrieved documents..." or similar phrases.
- The answer should be a straightforward response to the user's question, strictly based on the provided medical context.

FORMAT RULES:

If factual:
- point 1
- point 2

If comparison:
- Feature A: ...
- Feature B: ...

If exploratory:
- Explanation point 1
- Explanation point 2

MEDICAL CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

    try:

        response = SESSION.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "medgemma-rag",
                "prompt": prompt,
                "stream": False,
                "options": {
                    # 🔥 lower hallucination
                    "temperature": 0.1,

                    # 🔥 more focused output
                    "top_p": 0.9,

                    # 🔥 concise answer
                    "num_predict": 220,

                    # 🔥 prevent reload
                    "keep_alive": "30m",

                    # 🔥 reduce repetition
                    "repeat_penalty": 1.1
                }
            },
            timeout=90
        )

        answer = response.json().get(
            "response",
            ""
        ).strip()

        # -----------------------------------
        # 🔹 CLEANUP
        # -----------------------------------

        # remove extra blank lines
        answer = answer.replace("\n\n\n", "\n\n")

        # remove weird tokens
        bad_tokens = [
            "<unused0>",
            "<unused1>",
            "<think>",
            "</think>"
        ]

        for token in bad_tokens:
            answer = answer.replace(token, "")

        answer = answer.strip()

        # 🔥 Empty fallback
        if len(answer) < 15:
            return "Not enough information in retrieved medical context."

        return answer

    except Exception as e:

        print("❌ Generator error:", e)

        return "Not enough information in retrieved medical context."