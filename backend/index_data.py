import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from modules.chunking.chunker import load_pdfs, chunk_text

# Paths
DATA_PATH = "backend/data/oncology_docs"
SAVE_PATH = "backend/database/vector_store"

print("🔹 Loading PDFs...")
raw_docs = load_pdfs(DATA_PATH)

print("🔹 Chunking...")
documents = chunk_text(raw_docs)
print(f"Total chunks: {len(documents)}")

# -----------------------------
# 🔹 Embedding Model
# -----------------------------
from modules.embeddings.mrl_embeddings import get_mrl_embedding

print("🔹 Creating MRL embeddings...")
doc_embeddings = get_mrl_embedding(documents, dim=128)

dimension = doc_embeddings.shape[1]
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
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# -----------------------------
# 🔹 SAVE EVERYTHING
# -----------------------------
faiss.write_index(index, f"{SAVE_PATH}/faiss.index")

with open(f"{SAVE_PATH}/docs.pkl", "wb") as f:
    pickle.dump(documents, f)

with open(f"{SAVE_PATH}/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("✅ Indexing complete!")