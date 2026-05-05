from modules.laqa.laqa import process_query
from modules.agent.agent_controller import agent_decision
from modules.xai.explain import generate_explanation


# -------------------------------
# 🔹 CORE PIPELINE FUNCTION
# -------------------------------
def handle_query(user_query: str):
    """
    Main entry point for:
    - server.py (API)
    - CLI
    - evaluation pipeline
    """

    # -------------------------------
    # 🔹 Step 1: LAQA
    # -------------------------------
    print("\n🔹 Step 1: LAQA Processing...")
    laqa_output = process_query(user_query)

    # -------------------------------
    # 🔹 Step 2: Agent Loop (RAG)
    # -------------------------------
    print("🔹 Step 2: Agent Execution (RAG + Retry Loop)...")
    agent_result = agent_decision(laqa_output)

    answer = agent_result.get("answer", "")
    docs = agent_result.get("docs", [])
    doc_ids = agent_result.get("doc_ids", [])
    eval_result = agent_result.get("eval", {})

    # -------------------------------
    # 🔹 Step 3: Explainability
    # -------------------------------
    print("🔹 Step 3: Explainability Layer...")
    explanation = generate_explanation(answer, docs, eval_result, user_query)

    # -------------------------------
    # 🔹 Confidence handling
    # -------------------------------
    confidence = explanation.get("confidence", 0.75)
    if confidence is None or confidence == 0:
        confidence = 0.75

    # -------------------------------
    # 🔹 Final structured output
    # -------------------------------
    return {
        "answer": answer,
        "confidence": confidence,

        # 🔥 For evaluation
        "sources": doc_ids,

        # 🔥 For UI
        "source_texts": docs,

        # 🔥 Explainability
        "explanation": explanation
    }


# -------------------------------
# 🔹 OPTIONAL CLI MODE (DEBUG)
# -------------------------------
if __name__ == "__main__":
    print("\n🧠 Oncology AI Assistant (Core Mode)")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("Enter your medical query: ").strip()

        if user_query.lower() == "exit":
            break

        result = handle_query(user_query)

        print("\n=== FINAL OUTPUT ===")
        print("Answer:\n", result["answer"])

        print("\nConfidence:", result["confidence"])

        print("\nSupporting Sentences:")
        for s in result["explanation"].get("supporting_sentences", []):
            print("-", s)

        print("\nReasoning:")
        print(result["explanation"].get("reasoning", ""))

        print("\nSource IDs:", result["sources"])
        print("-" * 50)