"""Streaming parser for benchmark debug.log artifacts.

Each question block in debug.log has the shape:

    ================================================================================
    [  N] (OK|WRONG|ERR)  Category: <cat>  |  Subject: <subj>
    ================================================================================

    PROMPT:
    <multi-line prompt>

    PIPELINE STAGES:
    -- stage: <name> --
    {... json body ...}

    -- stage: <streaming_name> (coalesced × N) --
    <raw text body>

    GOLD: <idx> = '<text>'
    PRED: <idx>  |  CORRECT: <bool>  |  TIME: <ms>ms  |  SOURCES: <n>
    [ERROR: ...]

This parser walks lines once and yields one ``QuestionTrace`` per block.
It deliberately discards the ``graph_cypher_token`` body (a 40-50 KB
streaming dump of token-by-token Cypher generation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# Regex anchors — kept tight to avoid mis-firing inside JSON bodies.
_HEADER_RE = re.compile(
    r"^\[\s*(?P<idx>\d+)\]\s+(?P<mark>OK|WRONG|ERR)\s+Category:\s+(?P<cat>.+?)\s+\|\s+Subject:\s+(?P<subj>.+?)\s*$"
)
_STAGE_RE = re.compile(
    r"^--\s*stage:\s*(?P<name>[a-z_]+)(?:\s*\(coalesced\s*×\s*(?P<count>\d+)\))?\s*--\s*$"
)
_GOLD_RE = re.compile(r"^GOLD:\s+(?P<idx>-?\d+)\s+=\s*(?P<rest>.*)$")
_PRED_RE = re.compile(
    r"^PRED:\s+(?P<pred>None|-?\d+)\s+\|\s+CORRECT:\s+(?P<correct>True|False)\s+\|\s+TIME:\s+(?P<ms>\d+)ms\s+\|\s+SOURCES:\s+(?P<src>\d+)\s*$"
)
_ERROR_RE = re.compile(r"^ERROR:\s*(?P<msg>.*)$")
_SEPARATOR = "=" * 80

# Stages that emit raw text (one streamed token per event in the original SSE
# stream); body should NOT be JSON-parsed.
_STREAMING_STAGES = {"graph_cypher_token", "answer"}


@dataclass
class QuestionTrace:
    """Per-question signals extracted from debug.log."""

    idx: int
    mark: str  # OK | WRONG | ERR
    category: str
    subject: str

    # intent stage
    intent: Optional[str] = None
    resolved_terms: list[str] = field(default_factory=list)
    primary_term: Optional[str] = None

    # retrieval_vector stage
    vector_count: int = 0
    vector_top_scores: list[float] = field(default_factory=list)

    # retrieval_graph stage
    cypher: Optional[str] = None
    cypher_pattern: str = "unknown"  # A1 | A2 | B | C | unknown
    graph_count: int = 0
    graph_error: Optional[str] = None

    # rerank stage
    rerank_candidate_count: int = 0
    rerank_output_count: int = 0
    rerank_top_scores: list[float] = field(default_factory=list)

    # answer stage
    answer_text: str = ""
    answer_prompt_chars: int = 0

    # sources stage
    sources: list[dict] = field(default_factory=list)

    # footer
    gold: int = -1
    pred: Optional[int] = None
    correct: bool = False
    latency_ms: int = 0
    sources_count: int = 0
    error: Optional[str] = None


# Map a Cypher query string to one of the four canonical patterns documented in
# rag-engine/retrieval/cypher_generator.py. Defaults to ``unknown`` so callers
# can flag the LLM going off-script.
def classify_cypher_pattern(cypher: str) -> str:
    if not cypher:
        return "unknown"
    text = cypher.strip()
    if re.search(r"WHERE\s+false\b", text, re.IGNORECASE):
        return "C"
    if re.search(r"MATCH\s*\(\s*t\s*:\s*Term\s*\{\s*abbreviation\s*:", text, re.IGNORECASE):
        return "A1"
    if re.search(r"MATCH\s*\(\s*t\s*:\s*Term\s*\)\s*WHERE\s+t\.abbreviation\s+IN", text, re.IGNORECASE):
        return "A2"
    if re.search(r"section_title\s*=~\s*['\"]", text):
        return "B"
    return "unknown"


def _parse_json_body(lines: list[str]) -> Optional[dict]:
    """Parse a JSON body collected between two stage markers.

    Returns ``None`` when the body is empty or not valid JSON. We do NOT raise
    because some streaming stages have raw-text bodies and the caller already
    decides which stages to JSON-parse.
    """
    if not lines:
        return None
    blob = "\n".join(lines).strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


# Pull top-3 final scores out of a rerank/sources structure. Both blocks store
# the blended score under ``score``; rerank_score (raw cross-encoder) is also
# present in ``sources`` but not in ``rerank.output.top``.
def _extract_top_scores(top_list: list, key: str = "score") -> list[float]:
    out: list[float] = []
    if not isinstance(top_list, list):
        return out
    for item in top_list[:3]:
        if isinstance(item, dict) and isinstance(item.get(key), (int, float)):
            out.append(float(item[key]))
    return out


# Each question block is ingested incrementally — call ``finish_stage`` on
# every transition and ``finalize`` at the footer line.
class _BlockBuilder:
    def __init__(self, header: re.Match) -> None:
        self.trace = QuestionTrace(
            idx=int(header["idx"]),
            mark=header["mark"],
            category=header["cat"],
            subject=header["subj"],
        )
        self._current_stage: Optional[str] = None
        self._current_streaming: bool = False
        self._buffer: list[str] = []

    def begin_stage(self, name: str, streaming: bool) -> None:
        self.finish_stage()
        self._current_stage = name
        self._current_streaming = streaming
        self._buffer = []

    def append(self, line: str) -> None:
        if self._current_stage is None:
            return
        self._buffer.append(line)

    def finish_stage(self) -> None:
        if self._current_stage is None:
            return
        name = self._current_stage
        if self._current_streaming:
            # Only the answer stream carries useful text; skip cypher tokens.
            if name == "answer":
                self.trace.answer_text = "\n".join(self._buffer).strip()
        else:
            data = _parse_json_body(self._buffer)
            if data is not None:
                self._absorb_stage(name, data)
        self._current_stage = None
        self._buffer = []

    def _absorb_stage(self, name: str, data: dict) -> None:
        t = self.trace
        if name == "intent":
            out = data.get("output") or data
            t.intent = out.get("intent") if isinstance(out, dict) else None
            terms = out.get("terms") if isinstance(out, dict) else None
            if isinstance(terms, dict):
                resolved = terms.get("resolved") or {}
                if isinstance(resolved, dict):
                    t.resolved_terms = sorted(resolved.keys())
                t.primary_term = terms.get("primary_term")
        elif name == "retrieval_vector":
            t.vector_count = int(data.get("count") or 0)
            t.vector_top_scores = _extract_top_scores(data.get("top") or [], "score")
        elif name == "retrieval_graph":
            t.cypher = data.get("cypher")
            t.cypher_pattern = classify_cypher_pattern(t.cypher or "")
            out = data.get("output") if isinstance(data.get("output"), dict) else data
            t.graph_count = int(out.get("count") or 0) if isinstance(out, dict) else 0
            t.graph_error = (data.get("error") or (out.get("error") if isinstance(out, dict) else None))
        elif name == "rerank":
            inp = data.get("input") or {}
            t.rerank_candidate_count = int(inp.get("candidate_count") or 0)
            out = data.get("output") if isinstance(data.get("output"), dict) else data
            t.rerank_output_count = int(out.get("count") or 0) if isinstance(out, dict) else 0
            t.rerank_top_scores = _extract_top_scores(
                (out.get("top") if isinstance(out, dict) else None) or [], "score"
            )
        elif name == "answer_start":
            inp = data.get("input") or {}
            t.answer_prompt_chars = int(inp.get("prompt_chars") or 0)
        elif name == "sources":
            # ``sources`` is a JSON list, but _parse_json_body returns dict only.
            # Re-parse from the same buffer here.
            pass

    # The sources stage is a JSON *array*, which _parse_json_body rejects.
    # Hook it directly so we can keep the rest of the pipeline dict-typed.
    def absorb_sources(self, blob: str) -> None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return
        if isinstance(data, list):
            self.trace.sources = data

    def absorb_footer_gold(self, m: re.Match) -> None:
        try:
            self.trace.gold = int(m["idx"])
        except ValueError:
            self.trace.gold = -1

    def absorb_footer_pred(self, m: re.Match) -> None:
        pred_raw = m["pred"]
        self.trace.pred = None if pred_raw == "None" else int(pred_raw)
        self.trace.correct = m["correct"] == "True"
        self.trace.latency_ms = int(m["ms"])
        self.trace.sources_count = int(m["src"])

    def absorb_error(self, m: re.Match) -> None:
        self.trace.error = m["msg"].strip()


# Walk a debug.log file once, yielding one QuestionTrace per question block.
def parse_debug_log(path: Path) -> Iterator[QuestionTrace]:
    builder: Optional[_BlockBuilder] = None
    in_pipeline = False
    sources_buffer: Optional[list[str]] = None
    in_sources = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Question header: 3-line opener (===, [N] mark Cat | Subj, ===).
            # We trigger on the [N] line; the surrounding `=` lines are no-ops.
            header_match = _HEADER_RE.match(line)
            if header_match:
                if builder is not None:
                    builder.finish_stage()
                    yield builder.trace
                builder = _BlockBuilder(header_match)
                in_pipeline = False
                in_sources = False
                sources_buffer = None
                continue

            if builder is None:
                continue

            if line.strip() == "PIPELINE STAGES:":
                in_pipeline = True
                continue

            stage_match = _STAGE_RE.match(line)
            if stage_match and in_pipeline:
                # If we were buffering ``sources`` (a JSON array, not handled
                # by the dict-only stage absorber), flush it now.
                if in_sources and sources_buffer is not None:
                    builder.absorb_sources("\n".join(sources_buffer))
                    in_sources = False
                    sources_buffer = None

                name = stage_match["name"]
                streaming = bool(stage_match["count"]) or name in _STREAMING_STAGES

                if name == "sources":
                    builder.finish_stage()
                    in_sources = True
                    sources_buffer = []
                else:
                    builder.begin_stage(name, streaming)
                continue

            # Footer lines end the question block.
            gold_m = _GOLD_RE.match(line)
            if gold_m:
                if in_sources and sources_buffer is not None:
                    builder.absorb_sources("\n".join(sources_buffer))
                    in_sources = False
                    sources_buffer = None
                builder.finish_stage()
                builder.absorb_footer_gold(gold_m)
                continue

            pred_m = _PRED_RE.match(line)
            if pred_m:
                builder.absorb_footer_pred(pred_m)
                continue

            err_m = _ERROR_RE.match(line)
            if err_m and builder.trace.gold != -1:
                # Only treat as footer error AFTER GOLD/PRED have been seen,
                # so we do not eat ``ERROR: ...`` mentions inside a JSON body.
                builder.absorb_error(err_m)
                continue

            # Body lines: route to either the stage buffer or sources buffer.
            if in_sources and sources_buffer is not None:
                sources_buffer.append(line)
            elif in_pipeline:
                builder.append(line)

    if builder is not None:
        if in_sources and sources_buffer is not None:
            builder.absorb_sources("\n".join(sources_buffer))
        builder.finish_stage()
        yield builder.trace
