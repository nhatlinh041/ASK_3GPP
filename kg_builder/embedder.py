"""
Embedder — computes e5-base-v2 embeddings for all Chunk nodes
and creates a Neo4j vector index for cosine similarity search.
"""
import os

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/e5-base-v2"
VECTOR_INDEX_NAME = "chunk_embeddings"
EMBEDDING_DIM = 768
BATCH_SIZE = 64


class Embedder:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._model: SentenceTransformer | None = None

    def close(self) -> None:
        self._driver.close()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[embed] Loading {EMBEDDING_MODEL}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def create_vector_index(self) -> None:
        """Create Neo4j vector index if not exists."""
        with self._driver.session() as s:
            s.run(
                f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {EMBEDDING_DIM},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        print(f"[embed] Vector index '{VECTOR_INDEX_NAME}' ready.")

    def embed_all_chunks(self) -> int:
        """Embed every Chunk node that has no embedding yet. Returns count embedded."""
        model = self._get_model()

        # Fetch chunks without embeddings
        with self._driver.session() as s:
            rows = s.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.chunk_id AS id, c.content AS content"
            ).data()

        if not rows:
            print("[embed] All chunks already have embeddings.")
            return 0

        print(f"[embed] Embedding {len(rows)} chunks in batches of {BATCH_SIZE}...")
        total = 0

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            texts = [f"passage: {r['content']}" for r in batch]
            embeddings = model.encode(texts, normalize_embeddings=True).tolist()

            with self._driver.session() as s:
                for row, emb in zip(batch, embeddings):
                    s.run(
                        "MATCH (c:Chunk {chunk_id: $id}) SET c.embedding = $emb",
                        id=row["id"], emb=emb,
                    )

            total += len(batch)
            pct = total / len(rows) * 100
            print(f"[embed] {total}/{len(rows)} ({pct:.0f}%)", end="\r")

        print(f"\n[embed] Done — {total} chunks embedded.")
        return total
