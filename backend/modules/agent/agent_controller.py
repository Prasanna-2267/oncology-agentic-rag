from modules.retrieval.hybrid_retriever import hybrid_search
from modules.generator.medgemma import generate_answer
from modules.agent.evaluator import evaluate_answer
from modules.agent.strategy import choose_strategy
from modules.agent.memory import AgentMemory


def agent_decision(laqa_output):

    memory = AgentMemory()
    max_attempts = 3

    for attempt in range(max_attempts):

        print(f"\n🔁 ATTEMPT {attempt + 1}")

        docs = hybrid_search(laqa_output, None)

        agent_output = {
            "query": laqa_output,
            "context": docs
        }

        answer = generate_answer(agent_output)

        eval_result = evaluate_answer(
            laqa_output["expanded_query"],
            " ".join(docs[:1])[:1500],   # ✅ FIXED
            answer
        )

        print("EVAL:", eval_result)

        # 🧠 Store memory
        memory.add({
            "attempt": attempt,
            "query": laqa_output["expanded_query"],
            "score": eval_result.get("score"),
            "answer": answer[:150]
        })

        # 🔥 Strategy selection
        action = choose_strategy(eval_result, attempt)

        print("ACTION:", action)

        if action == "accept":
            return {
    "answer": answer,
    "docs": docs,
    "eval": eval_result
}

        elif action == "expand_query":
            laqa_output["expanded_query"] += " detailed clinical explanation mechanisms latest treatment"

        elif action == "increase_k":
            laqa_output["retrieval_k"] = min(laqa_output["retrieval_k"] + 2, 8)

    print("\n⚠️ Max attempts reached")

    return {
    "answer": answer,
    "docs": docs,
    "eval": eval_result
    }