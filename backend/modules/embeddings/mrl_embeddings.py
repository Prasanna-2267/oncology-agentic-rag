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
    emb = model.encode(texts)
    return emb[:, :dim]


def get_dynamic_mrl_embedding(texts, intent="factual"):
    model = get_model()
    dim = 128
    emb = model.encode(texts)
    return emb[:, :dim]