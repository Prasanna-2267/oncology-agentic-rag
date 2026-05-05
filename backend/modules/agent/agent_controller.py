from modules.retrieval.hybrid_retriever import hybrid_search
from modules.generator.medgemma import generate_answer
from modules.agent.evaluator import evaluate_answer
from modules.agent.strategy import choose_strategy
from modules.agent.memory import AgentMemory


def agent_decision(laqa_output):
    """
    Main agent loop:
    - Retrieval
    - Generation
    - Evaluation
    - Strategy (retry / accept)
    """

    memory = AgentMemory()
    max_attempts = 3

    for attempt in range(max_attempts):

        print(f"\n🔁 ATTEMPT {attempt + 1}")

        # -------------------------------
        # 🔹 Retrieval (UPDATED)
        # -------------------------------
        retrieval_result = hybrid_search(laqa_output, None)

        docs = retrieval_result.get("texts", [])
        doc_ids = retrieval_result.get("ids", [])

        if not docs:
            print("⚠️ No documents retrieved")
            return {
                "answer": "I don’t have enough information to answer this.",
                "docs": [],
                "doc_ids": [],
                "eval": {"score": 2, "confidence": 0.3, "needs_retry": True}
            }

        # -------------------------------
        # 🔹 Context Optimization
        # -------------------------------
        context = docs[0][:800] if docs else ""

        # -------------------------------
        # 🔹 Generation
        # -------------------------------
        agent_input = {
            "query": laqa_output,
            "context": docs
        }

        answer = generate_answer(agent_input)

        # -------------------------------
        # 🔹 Evaluation
        # -------------------------------
        eval_result = evaluate_answer(
            laqa_output["expanded_query"],
            context,
            answer
        )

        print("EVAL:", eval_result)

        # -------------------------------
        # 🔹 Memory (optional logging)
        # -------------------------------
        memory.add({
            "attempt": attempt,
            "query": laqa_output["expanded_query"],
            "score": eval_result.get("score"),
            "answer": answer[:150]
        })

        # -------------------------------
        # 🔹 Strategy Decision
        # -------------------------------
        action = choose_strategy(eval_result, attempt)

        print("ACTION:", action)

        # -------------------------------
        # 🔹 ACTION HANDLING
        # -------------------------------
        if action == "accept":
            return {
                "answer": answer,
                "docs": docs,
                "doc_ids": doc_ids,
                "eval": eval_result
            }

        elif action == "expand_query":
            # 🔥 Controlled expansion (no explosion)
            laqa_output["expanded_query"] = (
                laqa_output["expanded_query"] +
                " detailed clinical explanation mechanisms latest treatment"
            )[:300]

        elif action == "increase_k":
            laqa_output["retrieval_k"] = min(
                laqa_output.get("retrieval_k", 5) + 2,
                8
            )

    # -------------------------------
    # 🔹 Fallback (max attempts reached)
    # -------------------------------
    print("\n⚠️ Max attempts reached")

    return {
        "answer": answer,
        "docs": docs,
        "doc_ids": doc_ids,
        "eval": eval_result
    }