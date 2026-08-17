
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from app.config import settings

# Lazily initialized so importing this module never requires live credentials
# or a downloaded model (keeps unit tests and `alembic` runs import-safe).
_pc = None
_index = None
_model = None


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = _pc.Index(settings.pinecone_index)
    return _index


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_and_store_chunks(chunks: list[str], namespace: str) -> None:
    """Embed chunks and upsert them into Pinecone under a per-document namespace."""
    model = _get_model()
    index = _get_index()
    embeddings = model.encode(chunks)
    vectors = [
        {"id": f"{namespace}-chunk-{i}", "values": emb.tolist(), "metadata": {"text": chunk}}
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors=vectors, namespace=namespace)


def query_top_chunks(query: str, namespace: str, top_k: int = 3) -> str:
    """Retrieve the top-k most similar chunks for a question, scoped to one document."""
    model = _get_model()
    index = _get_index()
    query_vec = model.encode([query])[0].tolist()
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True, namespace=namespace)
    return "\n".join(match["metadata"]["text"] for match in res["matches"])
