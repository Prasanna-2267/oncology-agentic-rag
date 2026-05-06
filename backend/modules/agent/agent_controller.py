from modules.retrieval.hybrid_retriever import hybrid_search
from modules.generator.medgemma import generate_answer
from modules.agent.evaluator import evaluate_answer
from modules.agent.strategy import choose_strategy
from modules.agent.memory import AgentMemory


def agent_decision(laqa_output):
    """
    Main Agentic RAG Loop

    Steps:
    1. Retrieval
    2. Generation
    3. Evaluation
    4. Retry Strategy
    """

    memory = AgentMemory()

    # 🔥 Reduced retries (faster + stable)
    max_attempts = 2

    last_answer = ""
    last_docs = []
    last_doc_ids = []
    last_eval = {}

    for attempt in range(max_attempts):

        print(f"\n🔁 ATTEMPT {attempt + 1}")

        # -----------------------------------
        # 🔹 Retrieval
        # -----------------------------------
        retrieval_result = hybrid_search(
            laqa_output,
            None
        )

        docs = retrieval_result.get("texts", [])

        doc_ids = retrieval_result.get("ids", [])

        retrieval_score = retrieval_result.get(
            "retrieval_score",
            0.5
        )

        # -----------------------------------
        # 🔹 Empty Retrieval Handling
        # -----------------------------------
        if not docs:

            print("⚠️ No documents retrieved")

            return {
                "answer": "I don’t have enough medical information to answer this question.",
                "docs": [],
                "doc_ids": [],
                "eval": {
                    "score": 2,
                    "confidence": 0.3,
                    "needs_retry": True,
                    "retrieval_score": 0
                }
            }

        # -----------------------------------
        # 🔹 Context Optimization
        # -----------------------------------
        # 🔥 cleaner eval context
        context = "\n".join(docs[:2])[:1500]

        # -----------------------------------
        # 🔹 Generation
        # -----------------------------------
        agent_input = {
            "query": laqa_output,
            "context": docs
        }

        answer = generate_answer(agent_input)

        # -----------------------------------
        # 🔹 Empty / Weak Answer Handling
        # -----------------------------------
        if len(answer.strip()) < 20:

            answer = (
                "Not enough information in retrieved medical context."
            )

        # -----------------------------------
        # 🔹 Evaluation
        # -----------------------------------
        eval_result = evaluate_answer(
            laqa_output["expanded_query"],
            context,
            answer
        )

        # 🔥 Inject retrieval score
        eval_result["retrieval_score"] = retrieval_score

        print("EVAL:", eval_result)

        # -----------------------------------
        # 🔹 Memory Logging
        # -----------------------------------
        memory.add({
            "attempt": attempt + 1,
            "query": laqa_output["expanded_query"],
            "score": eval_result.get("score"),
            "confidence": eval_result.get("confidence"),
            "retrieval_score": retrieval_score,
            "answer": answer[:200]
        })

        # -----------------------------------
        # 🔹 Save Last Valid State
        # -----------------------------------
        last_answer = answer
        last_docs = docs
        last_doc_ids = doc_ids
        last_eval = eval_result

        # -----------------------------------
        # 🔹 Early Accept (FAST PATH)
        # -----------------------------------
        if (
            eval_result.get("score", 0) >= 8
            and retrieval_score > 0.55
            and len(answer.split()) > 15
        ):

            print("⚡ Early high-quality acceptance")

            return {
                "answer": answer,
                "docs": docs,
                "doc_ids": doc_ids,
                "eval": eval_result
            }

        # -----------------------------------
        # 🔹 Strategy Decision
        # -----------------------------------
        action = choose_strategy(
            eval_result,
            attempt
        )

        print("ACTION:", action)

        # -----------------------------------
        # 🔹 ACTIONS
        # -----------------------------------
        if action == "accept":

            return {
                "answer": answer,
                "docs": docs,
                "doc_ids": doc_ids,
                "eval": eval_result
            }

        # -----------------------------------
        # 🔹 Query Expansion
        # -----------------------------------
        elif action == "expand_query":

            expanded = (
                laqa_output["expanded_query"]
                + " detailed clinical explanation symptoms diagnosis treatment"
            )

            laqa_output["expanded_query"] = expanded[:350]

        # -----------------------------------
        # 🔹 Retrieval Expansion
        # -----------------------------------
        elif action == "increase_k":

            laqa_output["retrieval_k"] = min(
                laqa_output.get("retrieval_k", 5) + 2,
                10
            )

    # -----------------------------------
    # 🔹 Max Attempts Fallback
    # -----------------------------------
    print("\n⚠️ Max attempts reached")

    return {
        "answer": last_answer,
        "docs": last_docs,
        "doc_ids": last_doc_ids,
        "eval": last_eval
    }