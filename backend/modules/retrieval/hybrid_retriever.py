import faiss
import pickle
import numpy as np

# 🔹 Load once (GOOD - keep this)
index = faiss.read_index("backend/database/vector_store/faiss.index")

with open("backend/database/vector_store/docs.pkl", "rb") as f:
    documents = pickle.load(f)

with open("backend/database/vector_store/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)


# -------------------------------
# 🔹 Utility
# -------------------------------
def normalize_scores(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return scores
    return (scores - scores.min()) / (scores.max() - scores.min())


# -------------------------------
# 🔹 Hybrid Retrieval (Optimized)
# -------------------------------
from modules.embeddings.mrl_embeddings import get_dynamic_mrl_embedding

def hybrid_search(laqa_output, _):

    query = laqa_output["expanded_query"]
    k = laqa_output.get("retrieval_k", 5)

    # 🔥 IMPORT HERE (avoid reload issues)


    # -------------------------------
    # 🔹 Dense Retrieval (FAISS)
    # -------------------------------
    query_vec = get_dynamic_mrl_embedding(
        [query],
        intent=laqa_output.get("intent", "factual")
    )

    D, I = index.search(query_vec, k=k)

    dense_docs = [documents[i] for i in I[0]]
    dense_scores = normalize_scores(-D[0])  # smaller distance = better

    # -------------------------------
    # 🔹 Sparse Retrieval (BM25)
    # -------------------------------
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(bm25_scores)[-k:]

    sparse_docs = [documents[i] for i in top_idx]
    sparse_scores = normalize_scores([bm25_scores[i] for i in top_idx])

    # -------------------------------
    # 🔹 Score Fusion (FAST + CLEAN)
    # -------------------------------
    doc_score_map = {}

    for doc, score in zip(dense_docs, dense_scores):
        doc_score_map[doc] = doc_score_map.get(doc, 0) + 0.6 * score

    for doc, score in zip(sparse_docs, sparse_scores):
        doc_score_map[doc] = doc_score_map.get(doc, 0) + 0.4 * score

    # -------------------------------
    # 🔹 Remove duplicates + rank
    # -------------------------------
    ranked_docs = sorted(
        doc_score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 🔥 LIMIT to top 3 (faster + cleaner context)
    final_docs = [doc for doc, _ in ranked_docs[:3]]

    # -------------------------------
    # 🔹 Debug (optional)
    # -------------------------------
    print("\n📄 Top Retrieved Docs:")
    for d in final_docs:
        print("-", d[:120], "...")

    return final_docs