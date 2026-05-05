"""
FastAPI RAG Engine — entry point.
POST /api/query → SSE stream các stage: intent, retrieval_vector, retrieval_graph, rerank, answer, sources.
POST /api/cypher → execute read-only Cypher against Neo4j (for the Cypher Tester demo page).
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator

# Load demo/.env before pipeline imports so OLLAMA_URL/NEO4J_* are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path as GraphPath
from neo4j.time import Date, DateTime, Time, Duration
from pydantic import BaseModel

from pipeline.orchestrator import RAGOrchestrator

app = FastAPI(title="3GPP RAG Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RAGOrchestrator()


class QueryRequest(BaseModel):
    question: str
    mode: str = "fixed"  # "fixed" | "react_agent"
    model: str = "qwen3:14b"
    # When False, reasoning models skip the <think> phase (faster, no chain-of-thought stream)
    think: bool = True


# Heartbeat interval for the SSE stream. Cloudflare tunnels (and many reverse
# proxies) close idle HTTP responses after ~30-100s; long ReAct runs có khoảng
# im lặng (chờ Neo4j, rerank, lúc model warm-up) đủ để chạm ngưỡng. Ta gửi SSE
# comment line `: ping\n\n` mỗi HEARTBEAT_INTERVAL_S giây — client EventSource
# bỏ qua, chỉ tunnel/proxy thấy có byte chảy nên không đóng connection.
HEARTBEAT_INTERVAL_S = 10.0


async def event_stream(request: QueryRequest) -> AsyncIterator[str]:
    """Convert orchestrator events to SSE format with periodic heartbeats so
    long-running streams survive reverse-proxy idle timeouts. Producer task
    chạy độc lập để có thể chen heartbeat vào những khoảng im lặng (Neo4j,
    rerank, model load) — `asyncio.wait_for` trên iter trực tiếp không an
    toàn vì sẽ huỷ generator giữa chừng khi timeout."""
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    # Producer: drain orchestrator into queue; cuối cùng push sentinel hoặc exception
    async def producer() -> None:
        try:
            async for event in orchestrator.query(
                request.question, request.mode, request.model, think=request.think
            ):
                await queue.put(event)
        except Exception as exc:  # noqa: BLE001 — surface to consumer below
            await queue.put(exc)
        else:
            await queue.put(sentinel)

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                # Im lặng quá lâu → gửi comment giữ kết nối; client bỏ qua dòng này
                yield ": ping\n\n"
                continue
            if item is sentinel:
                break
            if isinstance(item, Exception):
                # Surface lỗi orchestrator dưới dạng SSE event để frontend hiển thị
                yield f"data: {json.dumps({'stage': 'error', 'data': str(item)})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0)
    finally:
        # Client ngắt giữa chừng (cloudflared cancel, browser close) → huỷ producer
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@app.post("/api/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """Main RAG query endpoint — returns SSE stream."""
    return StreamingResponse(
        event_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----- Cypher Tester endpoint -------------------------------------------------
# Shared driver for the tester page (kept separate from the orchestrator's driver
# so its lifecycle is independent and a long-running query won't stall the RAG path)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
_cypher_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Block any clause that mutates the graph — the tester is read-only by design
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|"
    r"CREATE\s+CONSTRAINT|CREATE\s+INDEX|DROP\s+CONSTRAINT|DROP\s+INDEX|"
    r"LOAD\s+CSV|CALL\s+\{[^}]*\b(CREATE|MERGE|DELETE|SET|REMOVE)\b)",
    re.IGNORECASE,
)
DEFAULT_ROW_LIMIT = 200


class CypherRequest(BaseModel):
    query: str
    # Cap rows returned to avoid flooding the UI when the query has no LIMIT
    limit: int = DEFAULT_ROW_LIMIT
    # Cypher parameters bound by name (e.g. {"top_k": 10, "term": "AMF"})
    params: dict[str, Any] = {}


def _serialize_value(value: Any) -> Any:
    """Convert Neo4j-native types to JSON-friendly shapes for the UI."""
    if isinstance(value, Node):
        return {
            "_type": "node",
            "id": value.element_id,
            "labels": list(value.labels),
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    if isinstance(value, Relationship):
        return {
            "_type": "relationship",
            "id": value.element_id,
            "type": value.type,
            "start": value.start_node.element_id if value.start_node else None,
            "end": value.end_node.element_id if value.end_node else None,
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    if isinstance(value, GraphPath):
        return {
            "_type": "path",
            "nodes": [_serialize_value(n) for n in value.nodes],
            "relationships": [_serialize_value(r) for r in value.relationships],
        }
    if isinstance(value, (Date, DateTime, Time, Duration)):
        return str(value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


@app.post("/api/cypher")
async def run_cypher(request: CypherRequest) -> dict:
    """Execute a read-only Cypher query and return columns + rows."""
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Reject any write/DDL clause before sending the query to Neo4j
    if _WRITE_CLAUSES.search(query):
        raise HTTPException(
            status_code=400,
            detail="Only read-only queries are allowed (no CREATE/MERGE/DELETE/SET/REMOVE/DROP/LOAD CSV)",
        )

    limit = max(1, min(request.limit, 1000))
    started = time.perf_counter()
    try:
        with _cypher_driver.session() as session:
            result = session.run(query, **(request.params or {}))
            columns = list(result.keys()) if result.keys() else []
            rows: list[dict] = []
            for record in result:
                if len(rows) >= limit:
                    break
                rows.append({k: _serialize_value(record[k]) for k in columns})
            # Drain so consume() reports accurate counters
            summary = result.consume()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cypher error: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    counters = summary.counters
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= limit,
        "elapsed_ms": elapsed_ms,
        "stats": {
            "nodes_created": counters.nodes_created,
            "relationships_created": counters.relationships_created,
            "properties_set": counters.properties_set,
            "labels_added": counters.labels_added,
            "contains_updates": counters.contains_updates,
        },
    }


@app.get("/api/cypher/schema")
async def cypher_schema() -> dict:
    """Return KG schema summary (node labels, relationship types, sample counts)."""
    try:
        with _cypher_driver.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")]
            rel_types = [
                r["relationshipType"]
                for r in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
                )
            ]
            label_counts = {}
            for label in labels:
                row = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
                label_counts[label] = row["c"] if row else 0
            rel_counts = {}
            for rel in rel_types:
                row = session.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c").single()
                rel_counts[rel] = row["c"] if row else 0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Schema error: {exc}") from exc

    return {
        "labels": labels,
        "label_counts": label_counts,
        "relationship_types": rel_types,
        "relationship_counts": rel_counts,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
