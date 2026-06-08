"""
RAG pipeline orchestrator — coordinates retrieval stages and yields SSE events.

Two modes (selected per request):
  - 'fixed':       intent → vector → graph (LLM Cypher) → rrf fusion → rerank → answer
  - 'react_agent': intent → ADAPTIVE ReAct (LLM picks vector/cypher/expand_term/
                            inspect_chunk/finish each iter) → rerank → answer
"""
import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any


# Wrap sync iterator (Ollama stream, graph search) thành async để event loop được
# free giữa mỗi `next()` — uvicorn cần tick event loop để flush SSE bytes ra socket.
# Không dùng cái này: orchestrator block ~15-30s và TOÀN BỘ stream chỉ tới client
# 1 lần ở cuối (uvicorn không flush được khi event loop bận).
#
# Quan trọng: KHÔNG raise StopIteration qua asyncio Future (Python cấm —
# `TypeError: StopIteration interacts badly with generators`). Dùng sentinel
# trả từ helper sync để tránh.
_ITER_DONE = object()


def _next_or_done(it: Iterator[Any]) -> Any:
    try:
        return next(it)
    except StopIteration:
        return _ITER_DONE


async def _async_iter(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
    while True:
        ev = await asyncio.to_thread(_next_or_done, sync_iter)
        if ev is _ITER_DONE:
            break
        yield ev

from neo4j import GraphDatabase

from retrieval import (
    VectorSearcher,
    GraphSearcher,
    AdaptiveHopSearcher,
    rrf_fusion,
    rerank,
    rerank_per_gap,
    LLMCypherGenerator,
)
from llm import OllamaClient, build_prompt
from pipeline.intent_classifier import IntentClassifier
from pipeline.term_first import TermFirstStrategy
from pipeline.term_index import build_term_index

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# Model used by graph-search Cypher generation AND the adaptive planner.
CYPHER_MODEL = os.getenv("CYPHER_MODEL", "qwen3:14b")

# Adaptive ReAct retrieval (react_agent mode only). Quality > latency per project lead.
ADAPTIVE_HOP_MAX_ITER = int(os.getenv("ADAPTIVE_HOP_MAX_ITER", "5"))
# 0 (or any non-positive value) disables the wall-clock budget — only iter cap applies.
ADAPTIVE_HOP_BUDGET_MS = int(os.getenv("ADAPTIVE_HOP_BUDGET_MS", "0"))
ADAPTIVE_HOP_PLANNER_MODEL = os.getenv("ADAPTIVE_HOP_PLANNER_MODEL", CYPHER_MODEL)

TOP_K_VECTOR = 10
TOP_K_GRAPH = 8
TOP_K_FINAL = 6


def _chunks_to_context(chunks: list[dict], max_chars: int = 20000) -> str:
    """Format chunks into LLM context string with citations."""
    parts = []
    total = 0
    for c in chunks:
        citation = f"[{c.get('spec_id', '?')} §{c.get('section', '?')}]"
        block = f"{citation}\n{c.get('content', '')}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


# Compact per-chunk preview list for the trail UI
def _chunks_to_preview(
    chunks: list[dict], n: int = 3, preview_chars: int = 100, score_key: str = "score"
) -> list[dict]:
    previews = []
    for c in chunks[:n]:
        content = (c.get("content") or "").strip().replace("\n", " ")
        if len(content) > preview_chars:
            content = content[:preview_chars].rstrip() + "…"
        previews.append({
            "spec_id": c.get("spec_id", "?"),
            "section": c.get("section", "?"),
            "score": c.get(score_key),
            "preview": content,
        })
    return previews


class RAGOrchestrator:
    def __init__(self):
        # Single shared Neo4j driver — used by the TermIndex builder, graph searcher,
        # and schema introspector.
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # In-memory snapshot of all Term nodes (~31k). Replaces hard-coded
        # NETWORK_FUNCTIONS + per-query Neo4j round-trips. Built once here,
        # shared with TermFirstStrategy and AdaptiveHopSearcher.
        self._term_index = build_term_index(self._driver)
        self._term_first = TermFirstStrategy(index=self._term_index)

        # Vector searcher: used as a deterministic step in fixed mode, AND as a tool
        # the planner can call in react_agent mode.
        self._vector = VectorSearcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        self._llm = OllamaClient()
        # LLM-first intent classifier (regex fallback). Dùng cùng OllamaClient
        # để Ollama giữ model warm; classify gọi với format=json + think=False
        self._intent_clf = IntentClassifier(self._llm)
        self._cypher_gen = LLMCypherGenerator(self._llm, model=CYPHER_MODEL)
        # Graph search (LLM Cypher single-shot) — fixed mode only.
        self._graph = GraphSearcher(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, generator=self._cypher_gen
        )

        # Adaptive ReAct retrieval — react_agent mode only. Picks one tool per
        # iteration (cypher_query / vector_search / expand_term / inspect_chunk /
        # finish) and decides on its own when there's enough evidence to stop.
        self._adaptive_hop = AdaptiveHopSearcher(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            llm=self._llm,
            planner_model=ADAPTIVE_HOP_PLANNER_MODEL,
            vector_searcher=self._vector,
            cypher_generator=self._cypher_gen,
            term_index=self._term_index,
            max_iter=ADAPTIVE_HOP_MAX_ITER,
            budget_ms=ADAPTIVE_HOP_BUDGET_MS,
        )

    async def query(
        self, question: str, mode: str = "fixed", model: str = "qwen3:14b", think: bool = True
    ) -> AsyncIterator[dict]:
        """
        Yield SSE event dicts: {stage, data}.
        Each non-token stage emits {input, output, ...flat fields} so the UI
        can show input/output of every tool when the user clicks Details.
        """
        # Step 1: combined intent + term extraction in ONE LLM call (format=json,
        # think=False). LLM returns intent + abbreviations + full_names + spec_refs;
        # TermFirstStrategy then HARD-VALIDATES each candidate against the live
        # KG TermIndex so hallucinations (e.g. "TELL") never reach the Cypher gen.
        # If the LLM fails (timeout, bad JSON), fall back to (regex intent +
        # deterministic TermIndex extraction) — still KG-backed, no hard-coded list.
        intent_terms = await asyncio.to_thread(
            self._intent_clf.classify_with_terms, question, model
        )
        if intent_terms is not None:
            intent = intent_terms["intent"]
            terms = self._term_first.extract_with_llm_terms(
                question,
                intent_terms["abbreviations"],
                intent_terms["full_names"],
                intent_terms["spec_refs"],
            )
        else:
            intent = await asyncio.to_thread(self._intent_clf.classify, question, model)
            terms = self._term_first.extract_fallback(question)
        resolved = terms.get("resolved", {})
        yield {"stage": "intent", "data": {
            "input": {"question": question},
            "output": {"intent": intent, "terms": terms},
            "intent": intent,
            "terms": terms,
        }}

        # Step 2 — RETRIEVAL: branch on mode.
        seeds = terms["network_functions"] or [terms["primary_term"]]
        # Captured from hop_research_done — fed into per-gap rerank below so each
        # sub-question gets its own slot in the final top_k (avoids "compound query
        # bias" where the cross-encoder favours chunks that mention many topics).
        research_gaps: list[str] = []
        if mode == "react_agent":
            candidates: list[dict] = []
            # Adaptive ReAct: planner LLM picks each tool (vector_search / cypher_query
            # / expand_term / inspect_chunk / finish) and decides when to stop.
            adaptive_iter = self._adaptive_hop.search_streaming(
                question=question,
                intent=intent,
                seeds=seeds,
                resolved_terms=resolved,
                prior_chunks=None,
                think=think,
                model=model,
            )
            async for ev in _async_iter(adaptive_iter):
                if ev.get("stage") == "hop_research_done":
                    research_gaps = (ev.get("data") or {}).get("gaps") or []
                if ev.get("stage") == "hop_finish":
                    candidates = (ev.get("data") or {}).pop("chunks", []) or []
                yield ev
            yield {"stage": "retrieval_adaptive", "data": {
                "input": {"seeds": seeds, "mode": "react_agent"},
                "output": {
                    "count": len(candidates),
                    "top": _chunks_to_preview(candidates),
                },
                "count": len(candidates),
                "seeds": seeds,
                "top": _chunks_to_preview(candidates),
            }}
        else:
            # Fixed pipeline: deterministic vector + LLM-Cypher graph search,
            # merged via Reciprocal Rank Fusion. No adaptive loop.
            # Vector search (sentence-transformers encode + Neo4j query) là sync blocking
            # → to_thread để event loop tick được giữa intent event và retrieval_vector event
            vec_results = await asyncio.to_thread(self._vector.search, question, top_k=TOP_K_VECTOR)
            yield {"stage": "retrieval_vector", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR},
                "output": {
                    "count": len(vec_results),
                    "top": _chunks_to_preview(vec_results),
                },
                "count": len(vec_results),
                "top": _chunks_to_preview(vec_results),
            }}
            # Pass top-3 vector hits as anchors for the Cypher generator —
            # only title + spec_id, NOT content (keeps graph scoring independent
            # of vector scoring; mitigates Pattern C over-firing when vector
            # already found a real section anchor).
            vector_hints = [
                {"section": v.get("section", "?"), "spec_id": v.get("spec_id", "?")}
                for v in vec_results[:3]
            ]
            graph_chunks: list[dict] = []
            # Sync iterator (Ollama iter_lines) → wrap _async_iter để event loop free
            # giữa mỗi token. Không có wrap này thì cypher token stream "im lặng" cho
            # đến khi cả graph search xong rồi mới flush hết.
            graph_iter = self._graph.search_streaming(
                question,
                intent=intent,
                term=terms["primary_term"],
                resolved_terms=resolved,
                top_k=TOP_K_GRAPH,
                think=think,
                vector_hints=vector_hints,
                # Single-model run: Cypher gen uses the SAME model as the answer
                # stage so Ollama keeps one set of weights resident the whole
                # query. Avoids 15-30s load/unload thrash per question.
                model=model,
            )
            async for ev in _async_iter(graph_iter):
                if ev.get("stage") == "retrieval_graph":
                    data = ev.get("data") or {}
                    graph_chunks = data.pop("_chunks", []) or []
                yield ev
            candidates = rrf_fusion(
                [vec_results, graph_chunks],
                weights=[1.0, 1.0],
                top_k=TOP_K_FINAL * 2,
            )

        # Step 3: cross-encoder rerank picks the final top_k for the answer prompt.
        # When research_gaps is non-empty (react_agent mode), use per-gap reranking
        # so each sub-question gets its own slot — this fixes the compound-query
        # bias where chunks that vaguely match many topics outrank chunks that
        # squarely answer one specific gap.
        # Rerank dùng cross-encoder (sentence-transformers, sync) — to_thread để
        # event loop tick được trước khi yield rerank event
        if mode == "react_agent" and len(research_gaps) > 1:
            reranked = await asyncio.to_thread(
                rerank_per_gap,
                question=question,
                gaps=research_gaps,
                chunks=candidates,
                total_top_k=TOP_K_FINAL,
                resolved_terms=resolved,
            )
        else:
            reranked = await asyncio.to_thread(
                rerank, question, candidates, top_k=TOP_K_FINAL,
                resolved_terms=resolved,
            )
        yield {"stage": "rerank", "data": {
            "input": {
                "query": question,
                "candidate_count": len(candidates),
                "top_k_final": TOP_K_FINAL,
                "gap_count": len(research_gaps),
                "rerank_mode": "per_gap" if (mode == "react_agent" and len(research_gaps) > 1) else "single",
            },
            "output": {
                "count": len(reranked),
                "top": _chunks_to_preview(reranked, score_key="final_score"),
            },
            "count": len(reranked),
            "top": _chunks_to_preview(reranked, score_key="final_score"),
        }}

        # Step 4: build prompt + stream LLM answer
        context = _chunks_to_context(reranked)
        prompt = build_prompt(intent, context, question)

        yield {"stage": "answer_start", "data": {
            "input": {
                "model": model,
                "prompt_chars": len(prompt),
                "prompt": prompt,
            },
        }}

        # Stream thinking + response tokens. Reasoning models (deepseek-r1, qwen3)
        # emit thinking first when think=True; otherwise (or for non-reasoning models)
        # we go straight to the response phase.
        # Sync iter_lines từ Ollama → wrap _async_iter để mỗi token được flush
        # ngay (không phải đợi cả answer xong mới thấy gì)
        answer_tokens: list[str] = []
        answer_iter = self._llm.generate_stream_full(prompt, model=model, think=think)
        async for ev in _async_iter(answer_iter):
            if ev["kind"] == "thinking":
                yield {"stage": "thinking", "data": ev["token"]}
            else:
                answer_tokens.append(ev["token"])
                yield {"stage": "answer", "data": ev["token"]}

        # Step 5: sources summary. `picked_for_gap` is set by rerank_per_gap so
        # the UI can show which sub-question each chunk was selected for.
        sources = [
            {
                "spec_id": c.get("spec_id", "?"),
                "section": c.get("section", "?"),
                "chunk_id": c.get("chunk_id"),
                "content": c.get("content", ""),
                # Surface both the blended final_score (sort key) and the raw
                # cross-encoder logit (for debugging) so trail UI can show either.
                "score": c.get("final_score"),
                "rerank_score": c.get("rerank_score"),
                "picked_for_gap": c.get("picked_for_gap"),
            }
            for c in reranked
        ]
        yield {"stage": "sources", "data": sources}
