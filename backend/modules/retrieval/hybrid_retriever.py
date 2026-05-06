import faiss
import pickle
import numpy as np
import re

from modules.retrieval.reranker import rerank
from modules.embeddings.mrl_embeddings import (
    get_dynamic_mrl_embedding
)

# -----------------------------------
# 🔹 LOAD VECTOR DATABASE
# -----------------------------------
print("🔥 Loading FAISS + BM25 indexes...")

index = faiss.read_index(
    "backend/database/vector_store/faiss.index"
)

with open(
    "backend/database/vector_store/docs.pkl",
    "rb"
) as f:
    documents = pickle.load(f)

with open(
    "backend/database/vector_store/bm25.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

with open(
    "backend/database/vector_store/ids.pkl",
    "rb"
) as f:
    ids = pickle.load(f)

with open(
    "backend/database/vector_store/id_to_text.pkl",
    "rb"
) as f:
    id_to_text = pickle.load(f)


# -----------------------------------
# 🔹 Adaptive Fusion Weights
# -----------------------------------
def get_fusion_weights(intent):

    if intent == "exploratory":

        return {
            "dense": 0.8,
            "sparse": 0.2
        }

    elif intent == "comparison":

        return {
            "dense": 0.6,
            "sparse": 0.4
        }

    # factual
    return {
        "dense": 0.5,
        "sparse": 0.5
    }


# -----------------------------------
# 🔹 Score Normalization
# -----------------------------------
def normalize_scores(scores):

    scores = np.array(
        scores,
        dtype=np.float32
    )

    if len(scores) == 0:
        return scores

    if scores.max() == scores.min():
        return np.ones_like(scores)

    return (
        (scores - scores.min())
        /
        (scores.max() - scores.min())
    )


# -----------------------------------
# 🔹 Query Tokenizer
# -----------------------------------
def tokenize(text):

    text = text.lower()

    text = re.sub(
        r'[^a-z0-9 ]',
        '',
        text
    )

    return text.split()


# -----------------------------------
# 🔹 Semantic Diversity Filter
# -----------------------------------
def diversify_results(
    texts,
    ids,
    max_docs=3
):

    final_texts = []
    final_ids = []

    seen_prefixes = set()

    for text, doc_id in zip(texts, ids):

        # 🔥 crude semantic dedup
        prefix = text[:120].lower()

        duplicate = False

        for seen in seen_prefixes:

            overlap = len(
                set(prefix.split())
                &
                set(seen.split())
            )

            if overlap > 15:
                duplicate = True
                break

        if duplicate:
            continue

        seen_prefixes.add(prefix)

        final_texts.append(text)
        final_ids.append(doc_id)

        if len(final_texts) >= max_docs:
            break

    return final_texts, final_ids


# -----------------------------------
# 🔹 HYBRID SEARCH
# -----------------------------------
def hybrid_search(
    laqa_output,
    _
):

    query = laqa_output["expanded_query"]

    intent = laqa_output.get(
        "intent",
        "factual"
    )

    k = laqa_output.get(
        "retrieval_k",
        5
    )

    # -----------------------------------
    # 🔹 Adaptive Weights
    # -----------------------------------
    weights = get_fusion_weights(intent)

    dense_weight = weights["dense"]
    sparse_weight = weights["sparse"]

    print(
        f"🔍 Fusion Weights → Dense: {dense_weight} | Sparse: {sparse_weight}"
    )

    # -----------------------------------
    # 🔹 Candidate Pool
    # -----------------------------------
    candidate_k = min(
        max(k * 3, 10),
        20
    )

    # -----------------------------------
    # 🔹 Dense Retrieval (FAISS)
    # -----------------------------------
    query_vec = get_dynamic_mrl_embedding(
        [query],
        intent=intent
    )

    D, I = index.search(
        query_vec,
        k=candidate_k
    )

    dense_ids = [
        ids[i]
        for i in I[0]
    ]

    # smaller distance = better
    dense_scores = normalize_scores(
        -D[0]
    )

    # -----------------------------------
    # 🔹 Sparse Retrieval (BM25)
    # -----------------------------------
    tokenized_query = tokenize(query)

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    top_idx = np.argsort(
        bm25_scores
    )[-candidate_k:]

    sparse_ids = [
        ids[i]
        for i in top_idx
    ]

    sparse_scores = normalize_scores(
        [bm25_scores[i] for i in top_idx]
    )

    # -----------------------------------
    # 🔹 Adaptive Score Fusion
    # -----------------------------------
    doc_score_map = {}

    # Dense contribution
    for doc_id, score in zip(
        dense_ids,
        dense_scores
    ):

        doc_score_map[doc_id] = (
            doc_score_map.get(doc_id, 0)
            +
            dense_weight * score
        )

    # Sparse contribution
    for doc_id, score in zip(
        sparse_ids,
        sparse_scores
    ):

        doc_score_map[doc_id] = (
            doc_score_map.get(doc_id, 0)
            +
            sparse_weight * score
        )

    # -----------------------------------
    # 🔹 Ranking
    # -----------------------------------
    ranked_docs = sorted(
        doc_score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------
    # 🔹 Candidate Selection
    # -----------------------------------
    candidate_ids = [
        doc_id
        for doc_id, _
        in ranked_docs[:10]
    ]

    candidate_texts = [
        id_to_text[doc_id]
        for doc_id in candidate_ids
    ]

    # -----------------------------------
    # 🔹 Cross-Encoder Reranking
    # -----------------------------------
    reranked_texts = rerank(
        query=query,
        docs=candidate_texts,
        top_k=6
    )

    # -----------------------------------
    # 🔹 Recover IDs
    # -----------------------------------
    reranked_ids = []

    for text in reranked_texts:

        for doc_id in candidate_ids:

            if id_to_text[doc_id] == text:

                reranked_ids.append(doc_id)
                break

    # -----------------------------------
    # 🔹 Semantic Diversity Filtering
    # -----------------------------------
    final_texts, final_ids = diversify_results(
        reranked_texts,
        reranked_ids,
        max_docs=3
    )

    # -----------------------------------
    # 🔹 Retrieval Score
    # -----------------------------------
    retrieval_score = float(
        np.mean(
            [
                score
                for _, score
                in ranked_docs[:3]
            ]
        )
    )

    retrieval_score = round(
        retrieval_score,
        3
    )

    # -----------------------------------
    # 🔹 DEBUG
    # -----------------------------------
    print("\n📄 Top Retrieved Docs:")

    for i, d in enumerate(final_texts):

        print(
            f"{i+1}.",
            d[:120],
            "..."
        )

    # -----------------------------------
    # 🔹 RETURN
    # -----------------------------------
    return {
        "texts": final_texts,

        "ids": final_ids,

        "retrieval_score": retrieval_score
    }