from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        print("🔥 Loading embedding model ONCE...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_mrl_embedding(texts, dim=128):
    model = get_model()

    # 🔥 Added batching for performance
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )

    return emb[:, :dim]


def get_dynamic_mrl_embedding(texts, intent="factual"):
    # 🔥 keep same behavior but cleaner
    return get_mrl_embedding(texts, dim=128)