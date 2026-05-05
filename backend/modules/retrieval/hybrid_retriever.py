import faiss
import pickle
import numpy as np

# 🔹 Load once
index = faiss.read_index("backend/database/vector_store/faiss.index")

with open("backend/database/vector_store/docs.pkl", "rb") as f:
    documents = pickle.load(f)

with open("backend/database/vector_store/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("backend/database/vector_store/ids.pkl", "rb") as f:
    ids = pickle.load(f)

with open("backend/database/vector_store/id_to_text.pkl", "rb") as f:
    id_to_text = pickle.load(f)


# -------------------------------
# 🔹 Utility
# -------------------------------
def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    if scores.max() == scores.min():
        return np.ones_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


# -------------------------------
# 🔹 Hybrid Retrieval (Improved)
# -------------------------------
from modules.embeddings.mrl_embeddings import get_dynamic_mrl_embedding


def hybrid_search(laqa_output, _):

    query = laqa_output["expanded_query"]
    k = laqa_output.get("retrieval_k", 5)

    # 🔥 Increase candidate pool (BIG improvement)
    candidate_k = min(max(k * 3, 10), 20)

    # -------------------------------
    # 🔹 Dense Retrieval (FAISS)
    # -------------------------------
    query_vec = get_dynamic_mrl_embedding(
        [query],
        intent=laqa_output.get("intent", "factual")
    )

    D, I = index.search(query_vec, k=candidate_k)

    dense_ids = [ids[i] for i in I[0]]
    dense_scores = normalize_scores(-D[0])  # smaller distance → better

    # -------------------------------
    # 🔹 Sparse Retrieval (BM25)
    # -------------------------------
    import re

    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        return text.split()
    tokenized_query = tokenize(query)

    bm25_scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(bm25_scores)[-candidate_k:]

    sparse_ids = [ids[i] for i in top_idx]
    sparse_scores = normalize_scores([bm25_scores[i] for i in top_idx])

    # -------------------------------
    # 🔹 Fusion (Stable + Weighted)
    # -------------------------------
    doc_score_map = {}

    # Dense contribution
    for doc_id, score in zip(dense_ids, dense_scores):
        doc_score_map[doc_id] = doc_score_map.get(doc_id, 0) + 0.6 * score

    # Sparse contribution
    for doc_id, score in zip(sparse_ids, sparse_scores):
        doc_score_map[doc_id] = doc_score_map.get(doc_id, 0) + 0.4 * score

    # -------------------------------
    # 🔹 Rank + Deduplicate
    # -------------------------------
    ranked_docs = sorted(
        doc_score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 🔥 Final top-k (clean context)
    final_ids = [doc_id for doc_id, _ in ranked_docs[:k]]

    # 🔥 Convert to text
    final_texts = [id_to_text.get(doc_id, "") for doc_id in final_ids]

    # -------------------------------
    # 🔹 Debug (optional)
    # -------------------------------
    print("\n📄 Top Retrieved Docs:")
    for i, d in enumerate(final_texts):
        print(f"{i+1}.", d[:120], "...")

    # -------------------------------
    # 🔹 Return
    # -------------------------------
    return {
        "texts": final_texts,
        "ids": final_ids
    }