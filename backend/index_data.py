import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from modules.embeddings.mrl_embeddings import get_mrl_embedding
from modules.chunking.chunker import load_pdfs, chunk_text

# Paths
DATA_PATH = "backend/data/oncology_docs"
SAVE_PATH = "backend/database/vector_store"

print("🔹 Loading PDFs...")
raw_docs = load_pdfs(DATA_PATH)

print("🔹 Chunking...")
documents = chunk_text(raw_docs)

# 🔥 NEW: extract text + ids (SAFE ADDITION)
texts = [doc["text"] for doc in documents]
ids = [doc["id"] for doc in documents]

print(f"Total chunks: {len(texts)}")

# -----------------------------
# 🔹 Embedding Model
# -----------------------------
print("🔹 Creating MRL embeddings...")
doc_embeddings = get_mrl_embedding(texts, dim=128)

dimension = doc_embeddings.shape[1]

# -----------------------------
# 🔹 FAISS Index
# -----------------------------
print("🔹 Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -----------------------------
# 🔹 BM25
# -----------------------------
print("🔹 Building BM25...")
tokenized_docs = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# 🔹 SAVE EVERYTHING
# -----------------------------
faiss.write_index(index, f"{SAVE_PATH}/faiss.index")

# 🔹 Keep original documents (for UI/debug)
with open(f"{SAVE_PATH}/docs.pkl", "wb") as f:
    pickle.dump(documents, f)

# 🔹 BM25
with open(f"{SAVE_PATH}/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

# 🔥 NEW: save ids
with open(f"{SAVE_PATH}/ids.pkl", "wb") as f:
    pickle.dump(ids, f)

# 🔥 NEW: id → text mapping
id_to_text = {doc["id"]: doc["text"] for doc in documents}
with open(f"{SAVE_PATH}/id_to_text.pkl", "wb") as f:
    pickle.dump(id_to_text, f)

print("✅ Indexing complete!")