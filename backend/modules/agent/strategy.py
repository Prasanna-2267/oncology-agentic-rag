def choose_strategy(eval_result, attempt):
    """
    Decide next action based on evaluation result
    """

    score = eval_result.get("score", 5)
    needs_retry = eval_result.get("needs_retry", True)

    # ✅ If evaluator says it's good → accept immediately
    if not needs_retry:
        return "accept"

    # 🔥 Hard fail → query likely weak
    if score < 4:
        return "expand_query"

    # 🔥 Medium → retrieval issue
    elif score < 8:
        return "increase_k"

    # 🔥 High but flagged retry → still accept
    return "accept"