"""
Unit tests for TermFirstStrategy — both LLM-validated and deterministic
fallback paths. Uses an in-memory TermIndex so no Neo4j required.
"""
import sys
from pathlib import Path

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from pipeline.term_first import TermFirstStrategy  # noqa: E402
from pipeline.term_index import build_from_records  # noqa: E402


@pytest.fixture
def index():
    return build_from_records([
        {
            "abbreviation": "SCP",
            "full_name": "Service Communication Proxy",
            "primary_spec": "ts_23_501",
            "source_specs": ["ts_23_501", "ts_29_500"],
        },
        {
            "abbreviation": "AMF",
            "full_name": "Access and Mobility Management Function",
            "primary_spec": "ts_23_501",
            "source_specs": ["ts_23_501"],
        },
        {
            "abbreviation": "SMF",
            "full_name": "Session Management Function",
            "primary_spec": "ts_23_501",
            "source_specs": ["ts_23_501"],
        },
    ])


@pytest.fixture
def strategy(index):
    return TermFirstStrategy(index=index)


# ---- LLM-validated path (extract_with_llm_terms) -------------------------


def test_llm_path_resolves_full_name(strategy):
    out = strategy.extract_with_llm_terms(
        "tell me about Service Communication Proxy",
        llm_abbreviations=[],
        llm_full_names=["Service Communication Proxy"],
        llm_spec_refs=[],
    )
    assert out["primary_term"] == "SCP"
    assert "SCP" in out["network_functions"]
    assert out["resolved"]["SCP"]["full_name"] == "Service Communication Proxy"
    assert out["resolved"]["SCP"]["specs"] == ["ts_23_501", "ts_29_500"]


def test_llm_path_resolves_abbrev_lowercase(strategy):
    out = strategy.extract_with_llm_terms(
        "what is scp",
        llm_abbreviations=["scp"],  # LLM emitted lowercase
        llm_full_names=[],
        llm_spec_refs=[],
    )
    assert out["primary_term"] == "SCP"
    assert "SCP" in out["network_functions"]


def test_llm_path_drops_hallucinated_abbrev(strategy):
    out = strategy.extract_with_llm_terms(
        "tell me about XYZ",
        llm_abbreviations=["XYZ"],  # not in KG
        llm_full_names=[],
        llm_spec_refs=[],
    )
    assert out["network_functions"] == []
    assert out["resolved"] == {}
    # primary_term falls back to first word in the query
    assert out["primary_term"] == "tell"


def test_llm_path_drops_hallucinated_full_name(strategy):
    out = strategy.extract_with_llm_terms(
        "explain something",
        llm_abbreviations=[],
        llm_full_names=["Made Up Function"],
        llm_spec_refs=[],
    )
    assert out["network_functions"] == []
    assert out["resolved"] == {}


def test_llm_path_full_name_wins_primary(strategy):
    # Both full_name and abbrev provided — full_name wins as primary because
    # it's the strongest signal (verbatim mention in query).
    out = strategy.extract_with_llm_terms(
        "compare AMF and Service Communication Proxy",
        llm_abbreviations=["AMF"],
        llm_full_names=["Service Communication Proxy"],
        llm_spec_refs=[],
    )
    assert out["primary_term"] == "SCP"
    assert set(out["network_functions"]) == {"SCP", "AMF"}


def test_llm_path_normalises_spec_refs(strategy):
    out = strategy.extract_with_llm_terms(
        "what's in TS 23.501",
        llm_abbreviations=[],
        llm_full_names=[],
        llm_spec_refs=["ts 23.501", "TR 38.300", "garbage"],
    )
    assert out["spec_refs"] == ["TS 23.501", "TR 38.300"]


def test_llm_path_dedup_resolved(strategy):
    out = strategy.extract_with_llm_terms(
        "scp again scp",
        llm_abbreviations=["SCP", "SCP"],
        llm_full_names=["Service Communication Proxy"],
        llm_spec_refs=[],
    )
    assert out["network_functions"] == ["SCP"]


# ---- Deterministic fallback path (extract_fallback) ----------------------


def test_fallback_resolves_full_name(strategy):
    out = strategy.extract_fallback("tell me about Service Communication Proxy")
    assert out["primary_term"] == "SCP"
    assert "SCP" in out["network_functions"]


def test_fallback_resolves_lowercase_full_name(strategy):
    out = strategy.extract_fallback("tell me about service communication proxy")
    assert out["primary_term"] == "SCP"


def test_fallback_resolves_abbrev_lowercase(strategy):
    out = strategy.extract_fallback("what is scp")
    assert out["primary_term"] == "SCP"
    assert "SCP" in out["network_functions"]


def test_fallback_no_match_falls_to_first_word(strategy):
    out = strategy.extract_fallback("can you help me")
    assert out["primary_term"] == "can"
    assert out["network_functions"] == []


def test_fallback_extracts_spec_refs(strategy):
    out = strategy.extract_fallback("section in TS 23.501 and TR 38.300")
    assert "TS 23.501" in out["spec_refs"]
    assert "TR 38.300" in out["spec_refs"]


def test_fallback_full_name_wins_over_abbrev(strategy):
    # AMF is also a token in the query, but the longer full-name match
    # determines primary_term.
    out = strategy.extract_fallback(
        "compare AMF with Service Communication Proxy"
    )
    assert out["primary_term"] == "SCP"
    assert set(out["network_functions"]) == {"SCP", "AMF"}


def test_fallback_does_not_match_common_words(strategy):
    # "tell", "me", "about", "what" are not in the KG — must not produce
    # spurious abbreviations.
    out = strategy.extract_fallback("tell me what about a story")
    assert out["network_functions"] == []
    assert out["resolved"] == {}
