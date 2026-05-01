from modules.laqa.laqa import process_query
from modules.agent.agent_controller import agent_decision
from modules.xai.explain import generate_explanation


def handle_query(user_query):
    print("🔹 Step 1: LAQA Processing...")
    laqa_output = process_query(user_query)

    print("🔹 Step 2: Agent Execution (RAG + Retry Loop)...")
    agent_result = agent_decision(laqa_output)

    answer = agent_result["answer"]
    docs = agent_result["docs"]
    eval_result = agent_result["eval"]

    print("🔹 Step 3: Explainability Layer...")
    explanation = generate_explanation(answer, docs, eval_result, user_query)

    return {
        "answer": answer,
        "explanation": explanation
    }


# CLI loop
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your medical query (or 'exit' to quit): ")

        if user_query.lower() == "exit":
            break

        result = handle_query(user_query.strip())

        print("\n=== FINAL OUTPUT ===")
        print("Answer:", result["answer"])
        print("\n--- EXPLANATION ---")
        print("\nSupporting Sentences:")
        for s in result["explanation"]["supporting_sentences"]:
            print("-", s)

        print("\nConfidence:", result["explanation"]["confidence"])
        print("Quality:", result["explanation"]["quality"])
        print("Grounded:", result["explanation"]["grounded"])

        print("\nReasoning:")
        print(result["explanation"]["reasoning"])