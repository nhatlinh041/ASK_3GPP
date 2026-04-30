"""
Shared model loading — embedding + cross-encoder.
Loaded once at startup to avoid per-request overhead.
"""
from sentence_transformers import SentenceTransformer, CrossEncoder

# Embedding model (e5-base-v2, same as current RAG system)
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
# Cross-encoder for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using e5 model."""
    model = get_embedding_model()
    # e5 requires "query: " / "passage: " prefix
    return model.encode(texts, normalize_embeddings=True).tolist()
