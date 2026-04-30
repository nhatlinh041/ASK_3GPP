"""
Tests for kg_builder.embedder.Embedder — verifies vector index creation
and that Chunk nodes get embedding vectors of the right dimension.
"""
import pytest

from tests.conftest import requires_neo4j

EMBEDDING_DIM = 768  # e5-base-v2


@requires_neo4j
def test_create_vector_index_idempotent(embedder):
    """Should not fail when index already exists."""
    embedder.create_vector_index()
    embedder.create_vector_index()

    with embedder._driver.session() as s:
        rows = s.run(
            "SHOW INDEXES YIELD name WHERE name = 'chunk_embeddings' RETURN name"
        ).data()
    assert len(rows) == 1


@requires_neo4j
def test_embed_chunks_writes_vectors(kg_builder, embedder, fixture_dir):
    """After embedding, every test Chunk should have a 768-dim embedding."""
    kg_builder.setup_schema()
    kg_builder.load_json_dir(fixture_dir)

    embedder.create_vector_index()
    embedder.embed_all_chunks()

    with embedder._driver.session() as s:
        rows = s.run(
            """
            MATCH (c:Chunk) WHERE c.spec_id STARTS WITH 'ts_99_'
            RETURN c.chunk_id AS id, size(c.embedding) AS dim
            """
        ).data()

    assert len(rows) == 5
    for row in rows:
        assert row["dim"] == EMBEDDING_DIM, f"Wrong dim for {row['id']}: {row['dim']}"


@requires_neo4j
def test_embed_skips_already_embedded(kg_builder, embedder, fixture_dir):
    """Re-running embed should be a no-op when all chunks already have embeddings."""
    kg_builder.setup_schema()
    kg_builder.load_json_dir(fixture_dir)
    embedder.create_vector_index()

    first = embedder.embed_all_chunks()
    second = embedder.embed_all_chunks()

    assert first == 5
    assert second == 0


@requires_neo4j
def test_vector_search_finds_relevant_chunk(kg_builder, embedder, fixture_dir):
    """Embed fixtures, then query — closest match for 'AMF access management' should be chunk 2."""
    from sentence_transformers import SentenceTransformer

    kg_builder.setup_schema()
    kg_builder.load_json_dir(fixture_dir)
    embedder.create_vector_index()
    embedder.embed_all_chunks()

    # Encode a query about access management
    model = SentenceTransformer("intfloat/e5-base-v2")
    query_emb = model.encode(
        ["query: AMF access management"], normalize_embeddings=True
    )[0].tolist()

    with embedder._driver.session() as s:
        rows = s.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', 5, $emb)
            YIELD node, score
            WHERE node.spec_id STARTS WITH 'ts_99_'
            RETURN node.chunk_id AS id, score
            ORDER BY score DESC
            """,
            emb=query_emb,
        ).data()

    assert len(rows) > 0
    # Top result should be the access-management chunk
    assert rows[0]["id"] == "ts_99_001_2"
