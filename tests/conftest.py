"""
Pytest config — adds demo/ to sys.path and provides Neo4j availability check.
"""
import os
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def neo4j_available() -> bool:
    """Quick connection check — skip integration tests if Neo4j unavailable."""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# Module-level skip marker for all integration tests
requires_neo4j = pytest.mark.skipif(
    not neo4j_available(),
    reason="Neo4j not available — start with: npm run dev (or docker run neo4j)",
)


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def kg_builder():
    """Fresh KGBuilder, cleans up after test."""
    from kg_builder import KGBuilder
    builder = KGBuilder()
    yield builder
    # Cleanup: remove test data (ts_99_xxx) — Documents, Chunks, và Terms chỉ
    # định nghĩa trong test specs. Subject nodes giữ lại (singleton taxonomy).
    with builder._driver.session() as s:
        s.run("MATCH (n) WHERE n.spec_id STARTS WITH 'ts_99_' OR n.chunk_id STARTS WITH 'ts_99_' DETACH DELETE n")
        s.run("""
            MATCH (t:Term)
            WHERE t.source_specs IS NOT NULL
              AND size(t.source_specs) > 0
              AND all(spec IN t.source_specs WHERE spec STARTS WITH 'ts_99_')
            DETACH DELETE t
        """)
    builder.close()


@pytest.fixture
def embedder():
    from kg_builder import Embedder
    e = Embedder()
    yield e
    e.close()


@pytest.fixture
def neo4j_driver():
    """Read-only Neo4j driver — không ghi, không cleanup."""
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv(DEMO_DIR / ".env")
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    yield driver
    driver.close()
