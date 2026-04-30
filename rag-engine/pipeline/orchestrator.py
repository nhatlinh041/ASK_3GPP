"""
RAG pipeline orchestrator — coordinates retrieval stages and yields SSE events.

Two modes (selected per request):
  - 'fixed':       intent → vector → graph (LLM Cypher) → rrf fusion → rerank → answer
  - 'react_agent': intent → ADAPTIVE ReAct (LLM picks vector/cypher/expand_term/
                            inspect_chunk/finish each iter) → rerank → answer
"""
import os
from collections.abc import AsyncIterator

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
from pipeline.term_first import TermFirstStrategy, TermResolver

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
        # Single shared Neo4j driver — used by the term resolver, graph searcher,
        # and schema introspector
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        self._intent_clf = IntentClassifier()
        # Term-First with KG-backed resolver for authoritative full names + spec sources
        self._term_resolver = TermResolver(self._driver)
        self._term_first = TermFirstStrategy(resolver=self._term_resolver)

        # Vector searcher: used as a deterministic step in fixed mode, AND as a tool
        # the planner can call in react_agent mode.
        self._vector = VectorSearcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        self._llm = OllamaClient()
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
        # Step 1: intent + term extraction (with KG-backed term resolution).
        # Cheap, deterministic — feeds the planner with seeds + authoritative term info.
        intent = self._intent_clf.classify(question)
        terms = self._term_first.extract(question)
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
            for ev in self._adaptive_hop.search_streaming(
                question=question,
                intent=intent,
                seeds=seeds,
                resolved_terms=resolved,
                prior_chunks=None,
                think=think,
            ):
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
            vec_results = self._vector.search(question, top_k=TOP_K_VECTOR)
            yield {"stage": "retrieval_vector", "data": {
                "input": {"query": question, "top_k": TOP_K_VECTOR},
                "output": {
                    "count": len(vec_results),
                    "top": _chunks_to_preview(vec_results),
                },
                "count": len(vec_results),
                "top": _chunks_to_preview(vec_results),
            }}
            graph_chunks: list[dict] = []
            for ev in self._graph.search_streaming(
                question,
                intent=intent,
                term=terms["primary_term"],
                resolved_terms=resolved,
                top_k=TOP_K_GRAPH,
                think=think,
            ):
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
        if mode == "react_agent" and len(research_gaps) > 1:
            reranked = rerank_per_gap(
                question=question,
                gaps=research_gaps,
                chunks=candidates,
                total_top_k=TOP_K_FINAL,
            )
        else:
            reranked = rerank(question, candidates, top_k=TOP_K_FINAL)
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
                "top": _chunks_to_preview(reranked, score_key="rerank_score"),
            },
            "count": len(reranked),
            "top": _chunks_to_preview(reranked, score_key="rerank_score"),
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
        answer_tokens: list[str] = []
        for ev in self._llm.generate_stream_full(prompt, model=model, think=think):
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
                "score": c.get("rerank_score"),
                "picked_for_gap": c.get("picked_for_gap"),
            }
            for c in reranked
        ]
        yield {"stage": "sources", "data": sources}
