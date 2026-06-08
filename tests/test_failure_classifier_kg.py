"""Unit tests for the OUT_OF_DOMAIN rule in tests/benchmark/_failure_classifier.

Uses a fake `kg_lookup` callable so the tests don't need Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root needs to be on sys.path so `tests.benchmark.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.benchmark._debug_log_parser import QuestionTrace  # noqa: E402
from tests.benchmark._failure_classifier import (  # noqa: E402
    classify,
    classify_all,
)


def _make_trace(idx: int = 1, **overrides) -> QuestionTrace:
    """Build a minimally-valid wrong-answer QuestionTrace."""
    defaults = dict(
        idx=idx,
        mark="WRONG",
        category="Lexicon",
        subject="Lexicon",
        intent="definition",
        resolved_terms=[],
        primary_term=None,
        vector_count=10,
        vector_top_scores=[0.7, 0.6, 0.5],
        cypher="MATCH (c:Chunk) WHERE false RETURN ...",
        cypher_pattern="C",
        graph_count=0,
        graph_error=None,
        rerank_candidate_count=10,
        rerank_output_count=0,
        rerank_top_scores=[],
        answer_text="Answer: B  ...",
        answer_prompt_chars=200,
        sources=[],
        gold=0,
        pred=1,
        correct=False,
        latency_ms=5000,
        sources_count=0,
        error=None,
    )
    defaults.update(overrides)
    return QuestionTrace(**defaults)


# A fake KG that returns True iff a sentinel substring appears in the gold.
def _fake_kg(sentinel: str):
    def _lookup(text: str) -> bool:
        return sentinel.lower() in (text or "").lower()
    return _lookup


class TestOutOfDomainRule:
    def test_kg_lookup_none_skips_rule(self):
        # Default offline mode — never marks anything OOD even if gold is gibberish.
        trace = _make_trace()
        rules = classify(trace, {"expected_answer": "Bitcoin keeps users anonymous"})
        assert "OUT_OF_DOMAIN" not in rules

    def test_kg_hit_does_not_flag_ood(self):
        # Gold text matches KG (fake reports True for "AMF").
        trace = _make_trace(cypher_pattern="A1")
        rules = classify(
            trace,
            {"expected_answer": "AMF receives the registration request"},
            kg_lookup=_fake_kg("AMF"),
        )
        assert "OUT_OF_DOMAIN" not in rules

    def test_kg_miss_without_topic_check_legacy_ood(self):
        # Backward compat: if only kg_lookup is provided (no kg_topic_check),
        # a gold-miss falls into the legacy OOD bucket. The new
        # GOLD_PARAPHRASED rule only fires when the topic probe is wired up.
        trace = _make_trace()
        rules = classify(
            trace,
            {"expected_answer": "Bitcoin keeps users anonymous"},
            kg_lookup=_fake_kg("AMF"),
        )
        assert "GOLD_PARAPHRASED" in rules
        assert "OUT_OF_DOMAIN" not in rules

    def test_gold_miss_topic_in_kg_flags_paraphrased(self):
        # Gold text not in KG, BUT question topic exists → GOLD_PARAPHRASED.
        trace = _make_trace()
        rules = classify(
            trace,
            {
                "expected_answer": "the lowest of all QoS traffic classes",
                "question": "What is best effort QoS?",
            },
            kg_lookup=_fake_kg("xxxxx"),       # gold absent
            kg_topic_check=_fake_kg("QoS"),    # topic present
        )
        assert "GOLD_PARAPHRASED" in rules
        assert "OUT_OF_DOMAIN" not in rules

    def test_gold_miss_topic_miss_flags_truly_ood(self):
        # Both gold and topic missing → strict OUT_OF_DOMAIN.
        trace = _make_trace()
        rules = classify(
            trace,
            {
                "expected_answer": "Bitcoin keeps public keys anonymous",
                "question": "How does Bitcoin achieve privacy for its users?",
            },
            kg_lookup=_fake_kg("AMF"),
            kg_topic_check=_fake_kg("AMF"),
        )
        assert "OUT_OF_DOMAIN" in rules
        assert "GOLD_PARAPHRASED" not in rules

    def test_paraphrased_suppresses_retrieval_no_gold_text(self):
        # When GOLD_PARAPHRASED fires (gold missing but topic present), the
        # RETRIEVAL_NO_GOLD_TEXT rule is also suppressed — same reason as
        # OUT_OF_DOMAIN: "wrong spec selected" is meaningless if the gold
        # phrase isn't anywhere in the KG.
        trace = _make_trace(
            sources=[
                {"content": "some related but wrong chunk content", "spec_id": "ts_x", "section": "S"}
            ],
            rerank_top_scores=[0.7],
            rerank_output_count=1,
        )
        rules = classify(
            trace,
            {
                "expected_answer": "the lowest of all QoS traffic classes",
                "question": "What is best effort QoS?",
            },
            kg_lookup=_fake_kg("xxxxx"),    # gold absent
            kg_topic_check=_fake_kg("QoS"),  # topic present
        )
        assert "GOLD_PARAPHRASED" in rules
        assert "RETRIEVAL_NO_GOLD_TEXT" not in rules

    def test_strict_ood_suppresses_retrieval_no_gold_text(self):
        # Same suppression for OUT_OF_DOMAIN.
        trace = _make_trace(
            sources=[
                {"content": "some related chunk content"}
            ],
            rerank_top_scores=[0.7],
            rerank_output_count=1,
        )
        rules = classify(
            trace,
            {
                "expected_answer": "Bitcoin keeps public keys anonymous",
                "question": "How does Bitcoin achieve privacy for its users?",
            },
            kg_lookup=_fake_kg("AMF"),
            kg_topic_check=_fake_kg("AMF"),
        )
        assert "OUT_OF_DOMAIN" in rules
        assert "RETRIEVAL_NO_GOLD_TEXT" not in rules

    def test_in_domain_with_wrong_sources_still_flags_no_gold_text(self):
        # When KG has the gold text but the retrieved sources don't,
        # RETRIEVAL_NO_GOLD_TEXT still fires (and OOD does not).
        trace = _make_trace(
            sources=[
                {"content": "off-topic content about something unrelated"}
            ],
            rerank_top_scores=[0.7],
            rerank_output_count=1,
        )
        rules = classify(
            trace,
            {"expected_answer": "AMF handles registration"},
            kg_lookup=_fake_kg("AMF"),  # gold IS in KG
        )
        assert "OUT_OF_DOMAIN" not in rules
        assert "RETRIEVAL_NO_GOLD_TEXT" in rules

    def test_kg_lookup_exception_does_not_break_classification(self):
        # If the KG probe raises (network blip), OOD silently skipped — we
        # don't want a transient driver error to mark every wrong answer OOD.
        def _angry_kg(_text: str) -> bool:
            raise RuntimeError("connection reset")

        trace = _make_trace()
        rules = classify(
            trace,
            {"expected_answer": "AMF handles registration"},
            kg_lookup=_angry_kg,
        )
        assert "OUT_OF_DOMAIN" not in rules
        # Other rules still fire normally.
        assert rules  # at least one rule


class TestClassifyAllPropagatesKGLookup:
    def test_classify_all_with_two_stage_kg(self):
        # Three wrong questions:
        #   #1 — gold absent, topic absent → OUT_OF_DOMAIN
        #   #2 — correct, no rules expected
        #   #3 — gold absent, topic present → GOLD_PARAPHRASED
        traces = [
            _make_trace(idx=1),
            _make_trace(idx=2, correct=True),
            _make_trace(idx=3),
        ]
        results = [
            {"expected_answer": "Bitcoin",
             "question": "How does Bitcoin work?",
             "model_response": ""},
            {"expected_answer": "AMF",
             "question": "What is AMF?",
             "model_response": ""},
            {"expected_answer": "the lowest of all QoS traffic classes",
             "question": "What is best effort QoS?",
             "model_response": ""},
        ]
        labels = classify_all(
            traces,
            results,
            kg_lookup=_fake_kg("AMF"),       # only "AMF" is in KG
            kg_topic_check=_fake_kg("QoS"),  # only "QoS" topic is in KG
        )
        assert "OUT_OF_DOMAIN" in labels[1]
        assert labels[2] == []
        assert "GOLD_PARAPHRASED" in labels[3]
        assert "OUT_OF_DOMAIN" not in labels[3]

    def test_classify_all_offline_default(self):
        traces = [_make_trace(idx=1)]
        results = [{"expected_answer": "anything", "model_response": ""}]
        labels = classify_all(traces, results)  # no probes
        assert "OUT_OF_DOMAIN" not in labels[1]
        assert "GOLD_PARAPHRASED" not in labels[1]
