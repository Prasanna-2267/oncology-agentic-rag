def choose_strategy(eval_result, attempt):

    score = eval_result.get("score", 5)
    needs_retry = eval_result.get("needs_retry", True)

    # 🔥 Highest priority → retry flag
    if not needs_retry:
        return "accept"

    # Otherwise decide strategy
    if score < 4:
        return "expand_query"
    elif score < 7:
        return "increase_k"
    else:
        return "accept"