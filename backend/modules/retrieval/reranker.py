from sentence_transformers import CrossEncoder

# 🔥 Load ONCE
print("🔥 Loading reranker model ONCE...")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -------------------------------
# 🔹 Rerank Documents
# -------------------------------
def rerank(query, docs, top_k=3):

    if not docs:
        return docs

    # Create query-doc pairs
    pairs = [[query, doc] for doc in docs]

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Combine docs + scores
    ranked = list(zip(docs, scores))

    # Sort descending
    ranked = sorted(
        ranked,
        key=lambda x: x[1],
        reverse=True
    )

    # Return top docs
    final_docs = [doc for doc, _ in ranked[:top_k]]

    return final_docs