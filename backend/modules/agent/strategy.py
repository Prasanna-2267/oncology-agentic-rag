def choose_strategy(
    eval_result,
    attempt
):

    score = eval_result.get(
        "score",
        5
    )

    needs_retry = eval_result.get(
        "needs_retry",
        True
    )

    answered_question = eval_result.get(
        "answered_question",
        True
    )

    answer_relevance = eval_result.get(
        "answer_relevance",
        0.5
    )

    hallucination_risk = eval_result.get(
        "hallucination_risk",
        "medium"
    )

    missing_information = eval_result.get(
        "missing_information",
        False
    )

    # -----------------------------------
    # 🔹 Strong Accept
    # -----------------------------------
    if (
        not needs_retry
        and answered_question
        and hallucination_risk == "low"
        and answer_relevance >= 0.7
    ):

        return "accept"

    # -----------------------------------
    # 🔹 Hallucination Recovery
    # -----------------------------------
    if hallucination_risk == "high":

        return "increase_k"

    # -----------------------------------
    # 🔹 Missing Information
    # -----------------------------------
    if missing_information:

        return "expand_query"

    # -----------------------------------
    # 🔹 Weak Relevance
    # -----------------------------------
    if answer_relevance < 0.45:

        return "expand_query"

    # -----------------------------------
    # 🔹 Low Score
    # -----------------------------------
    if score < 4:

        return "expand_query"

    # -----------------------------------
    # 🔹 Medium Quality
    # -----------------------------------
    if score < 8:

        return "increase_k"

    # -----------------------------------
    # 🔹 Retry Limit Protection
    # -----------------------------------
    if attempt >= 1:

        return "accept"

    return "increase_k"