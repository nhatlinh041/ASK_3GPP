"""
Unit tests for the LLM Cypher generator's static helpers (_strip_mcq, _clean,
_validate, _build_prompt). These don't need Neo4j or Ollama — pure text logic.

Targets the regressions we hit in benchmark/origin_rerank_100_question_thinking:
  - Class A/B (LLM leaks 'Answer: <letter>' prose before the Cypher)
  - Class E (query ends on bare WITH instead of RETURN)
  - Multi-term + Pattern C examples must reach the prompt
  - MCQ choices in the question must NOT reach the LLM
"""
import sys
from pathlib import Path

import pytest

# Add rag-engine to sys.path so `from llm import ...` and `from retrieval...` resolve.
RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from retrieval.cypher_generator import (  # noqa: E402
    CypherValidationError,
    LLMCypherGenerator,
)


# ---------- _strip_mcq ------------------------------------------------------


class TestStripMCQ:
    """Strip multiple-choice tail (A./B./C./...) so the question stem alone
    reaches the Cypher generator."""

    def test_strips_letter_period_choices(self):
        q = (
            "What is the role of the AMF?\n"
            "A. Routing user-plane packets\n"
            "B. Mobility management\n"
            "C. Charging\n"
            "Begin your reply with `Answer: <letter>`"
        )
        assert LLMCypherGenerator._strip_mcq(q) == "What is the role of the AMF?"

    def test_strips_letter_paren_choices(self):
        q = (
            "Which interface connects UPF to DN?\n"
            "A) N3\n"
            "B) N6\n"
            "C) N9"
        )
        assert LLMCypherGenerator._strip_mcq(q) == "Which interface connects UPF to DN?"

    def test_passthrough_when_no_mcq(self):
        q = "What is network slicing in 5G?"
        assert LLMCypherGenerator._strip_mcq(q) == q

    def test_handles_empty(self):
        assert LLMCypherGenerator._strip_mcq("") == ""
        assert LLMCypherGenerator._strip_mcq(None) == ""  # type: ignore[arg-type]


# ---------- _clean ----------------------------------------------------------


class TestClean:
    """Skip-to-keyword cleaner: strip any prose preceding the first Cypher head
    keyword (MATCH / OPTIONAL MATCH / WITH / UNWIND)."""

    def test_strips_answer_prefix_with_justification(self):
        # Class B regression: model leaks 'Answer: B\nJustification: ...' before MATCH
        raw = (
            "Answer: B\n"
            "Justification: A switch is the networking device responsible for...\n"
            "MATCH (n) RETURN n"
        )
        assert LLMCypherGenerator._clean(raw) == "MATCH (n) RETURN n"

    def test_strips_markdown_fence(self):
        raw = "```cypher\nMATCH (c:Chunk) RETURN c LIMIT 5\n```"
        assert LLMCypherGenerator._clean(raw) == "MATCH (c:Chunk) RETURN c LIMIT 5"

    def test_passthrough_clean_query(self):
        raw = "MATCH (n) RETURN n"
        assert LLMCypherGenerator._clean(raw) == "MATCH (n) RETURN n"

    def test_strips_trailing_semicolon(self):
        raw = "MATCH (n) RETURN n;"
        assert LLMCypherGenerator._clean(raw) == "MATCH (n) RETURN n"

    def test_keeps_optional_match(self):
        raw = "Some prose before.\nOPTIONAL MATCH (c:Chunk) RETURN c"
        assert LLMCypherGenerator._clean(raw).startswith("OPTIONAL MATCH")

    def test_starts_with_with(self):
        raw = "Answer: A\nWITH 1 AS x MATCH (c:Chunk) RETURN c LIMIT 5"
        cleaned = LLMCypherGenerator._clean(raw)
        assert cleaned.startswith("WITH ")

    def test_handles_empty(self):
        assert LLMCypherGenerator._clean("") == ""


# ---------- _validate -------------------------------------------------------


class TestValidate:
    def test_accepts_simple_match_return(self):
        # Should not raise
        LLMCypherGenerator._validate("MATCH (n) RETURN n LIMIT 5")

    def test_rejects_empty(self):
        with pytest.raises(CypherValidationError, match="empty"):
            LLMCypherGenerator._validate("")

    def test_rejects_multiple_statements(self):
        with pytest.raises(CypherValidationError, match="Multiple"):
            LLMCypherGenerator._validate("MATCH (n) RETURN n; MATCH (m) RETURN m")

    def test_rejects_write_keywords(self):
        with pytest.raises(CypherValidationError, match="Forbidden"):
            LLMCypherGenerator._validate("MATCH (n) SET n.x=1 RETURN n")

    def test_rejects_no_match_unwind_return(self):
        with pytest.raises(CypherValidationError, match="MATCH/UNWIND/RETURN"):
            LLMCypherGenerator._validate("CALL db.labels()")

    def test_rejects_query_ending_in_with(self):
        # Class E regression: 'Query cannot conclude with WITH' from Neo4j.
        # Either rejection reason is acceptable — the missing-RETURN check
        # fires first when the query has no RETURN at all.
        cypher = "MATCH (n) WITH n LIMIT 5"
        with pytest.raises(CypherValidationError, match="RETURN|WITH"):
            LLMCypherGenerator._validate(cypher)

    def test_rejects_query_with_return_then_trailing_with(self):
        # Pathological: model wraps extra WITH after RETURN (still invalid Cypher).
        cypher = "MATCH (n) RETURN n.x AS x UNION MATCH (m) WITH m"
        with pytest.raises(CypherValidationError, match="end with WITH"):
            LLMCypherGenerator._validate(cypher)

    def test_accepts_with_then_return(self):
        # WITH in the middle is fine — only end-on-WITH is rejected.
        cypher = "MATCH (n) WITH n LIMIT 5 RETURN n"
        LLMCypherGenerator._validate(cypher)


# ---------- _build_prompt ---------------------------------------------------


class TestBuildPrompt:
    """The prompt must include the new pattern guidance and DYNAMIC context
    sections — and must NEVER include the MCQ choices verbatim."""

    @staticmethod
    def _prompt(**overrides) -> str:
        defaults = dict(
            question="What is AMF?",
            schema_text="(unused — left for legacy)",
            intent="definition",
            resolved_terms={"AMF": {"full_name": "Access and Mobility Management Function", "specs": ["ts_23_501"]}},
            primary_term="AMF",
            vector_hints=None,
        )
        defaults.update(overrides)
        return LLMCypherGenerator._build_prompt(**defaults)

    def test_includes_purpose_statement(self):
        p = self._prompt()
        assert "graph branch BACKS UP a parallel vector branch" in p

    def test_includes_pattern_a2_multi_term(self):
        p = self._prompt()
        assert "Pattern A2" in p
        assert "MATCH (t:Term) WHERE t.abbreviation IN" in p
        # Must NOT teach the broken `MATCH (t1) OR (t2)` form
        assert "NEVER write `MATCH (t1) OR (t2)`" in p

    def test_includes_pattern_c_sentinel(self):
        p = self._prompt()
        assert "MATCH (c:Chunk) WHERE false" in p

    def test_includes_version_filter_warning(self):
        p = self._prompt()
        assert "DO NOT filter on `d.version`" in p

    def test_includes_strict_output_rules(self):
        p = self._prompt()
        assert "Output format — STRICT" in p
        assert "IGNORE all of that" in p

    def test_strips_mcq_from_question(self):
        q_with_choices = (
            "What is the role of the AMF?\n"
            "A. Routing\n"
            "B. Mobility\n"
            "Begin your reply with `Answer: <letter>`"
        )
        p = self._prompt(question=q_with_choices)
        # The stem stays
        assert "What is the role of the AMF?" in p
        # Inspect ONLY the user-question section (stripped from the rules block,
        # which intentionally mentions 'Begin your reply' as something to ignore).
        marker = "# User question (stem only — choices and meta-instructions stripped)"
        assert marker in p
        question_section = p.split(marker, 1)[1]
        assert "A. Routing" not in question_section
        assert "B. Mobility" not in question_section
        assert "Begin your reply" not in question_section

    def test_vector_hints_block_renders(self):
        hints = [
            {"section": "AMF discovery and selection", "spec_id": "ts_23_501"},
            {"section": "Network Function Service framework", "spec_id": "ts_29_500"},
        ]
        p = self._prompt(vector_hints=hints)
        assert "AMF discovery and selection" in p
        assert "ts_23_501" in p

    def test_vector_hints_empty_renders_placeholder(self):
        p = self._prompt(vector_hints=None)
        assert "Vector hints:" in p

    def test_no_resolved_terms_renders_pattern_c_hint(self):
        p = self._prompt(resolved_terms=None, primary_term=None)
        assert "Pattern C" in p
