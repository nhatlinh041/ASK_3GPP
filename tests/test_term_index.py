"""
Unit tests for TermIndex — covers case-insensitive lookup, longest-first
full_name matching, false-positive avoidance, search substring, and the
graceful-empty fallback. No Neo4j live dependency for the unit tests; one
integration test (requires_neo4j) verifies a real KG load.
"""
import sys
from pathlib import Path

import pytest

# Match the import pattern used by test_cypher_generator.py — rag-engine/
# isn't on sys.path by default since pytest's rootdir is the repo root.
RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from pipeline.term_index import (  # noqa: E402
    TermIndex,
    build_from_records,
    build_term_index,
)
from tests.conftest import requires_neo4j  # noqa: E402


# Synthetic KG snapshot covering: NF abbreviation (SCP), full-name with
# "Service" prefix collision (Service Communication Proxy vs Service Function
# Chaining — to test longest-first), a duplicate abbrev with shorter source_specs
# to verify dedup keeps the richer entry.
@pytest.fixture
def sample_records() -> list[dict]:
    return [
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
            "source_specs": ["ts_23_501", "ts_23_502"],
        },
        {
            "abbreviation": "SFC",
            "full_name": "Service Function Chaining",
            "primary_spec": "ts_22_261",
            "source_specs": ["ts_22_261"],
        },
        # Duplicate abbreviation with fewer source_specs — should be displaced
        # by the entry above (build_from_records prefers the richer record).
        {
            "abbreviation": "SMF",
            "full_name": "Session Management Function",
            "primary_spec": None,
            "source_specs": ["ts_23_501"],
        },
    ]


@pytest.fixture
def index(sample_records) -> TermIndex:
    return build_from_records(sample_records)


def test_build_from_records_populates_maps(index: TermIndex):
    assert "SCP" in index.abbrev_map
    assert "AMF" in index.abbrev_map
    assert "service communication proxy" in index.full_name_map
    assert index.full_name_map["service communication proxy"] == "SCP"


def test_dedup_keeps_richer_source_specs(index: TermIndex):
    # Record SMF appears twice; the one with 2 source_specs must win.
    assert index.abbrev_map["SMF"]["source_specs"] == ["ts_23_501", "ts_23_502"]


def test_lookup_abbrev_case_insensitive(index: TermIndex):
    assert index.lookup_abbrev("SCP")["full_name"] == "Service Communication Proxy"
    assert index.lookup_abbrev("scp")["full_name"] == "Service Communication Proxy"
    assert index.lookup_abbrev("Scp")["full_name"] == "Service Communication Proxy"
    assert index.lookup_abbrev(" sCp ")["full_name"] == "Service Communication Proxy"


def test_lookup_abbrev_unknown_returns_none(index: TermIndex):
    assert index.lookup_abbrev("TELL") is None
    assert index.lookup_abbrev("WHAT") is None
    assert index.lookup_abbrev("") is None


def test_lookup_full_name_case_insensitive(index: TermIndex):
    rec = index.lookup_full_name("Service Communication Proxy")
    assert rec is not None and rec["abbreviation"] == "SCP"
    rec = index.lookup_full_name("service communication proxy")
    assert rec is not None and rec["abbreviation"] == "SCP"


def test_find_full_name_longest_first(index: TermIndex):
    # Both "Service Communication Proxy" (longer) and "Service Function Chaining"
    # exist in the index. The longer one must be matched, not a shorter
    # substring of either.
    matches = index.find_full_name_matches(
        "tell me about Service Communication Proxy please"
    )
    assert len(matches) == 1
    assert matches[0][0] == "SCP"
    assert "Service Communication Proxy" in matches[0][1]


def test_find_full_name_lowercase_query(index: TermIndex):
    matches = index.find_full_name_matches(
        "tell me about service communication proxy"
    )
    assert len(matches) == 1
    assert matches[0][0] == "SCP"


def test_find_full_name_multiple_matches(index: TermIndex):
    matches = index.find_full_name_matches(
        "compare Service Communication Proxy and Session Management Function"
    )
    abbrevs = {m[0] for m in matches}
    assert abbrevs == {"SCP", "SMF"}


def test_find_full_name_no_false_positive_for_substring(index: TermIndex):
    # "Service" alone shouldn't match — only full names get added to the regex,
    # word-bounded. None of "Service" / "Function" / "Management" exist as
    # standalone full_name entries.
    assert index.find_full_name_matches("the Service alone is generic") == []


def test_search_substring_abbrev(index: TermIndex):
    rows = index.search("scp", limit=5)
    assert any(r["abbreviation"] == "SCP" for r in rows)


def test_search_substring_full_name(index: TermIndex):
    rows = index.search("Communication", limit=5)
    assert any(r["abbreviation"] == "SCP" for r in rows)


def test_search_respects_limit(index: TermIndex):
    rows = index.search("Function", limit=2)
    assert len(rows) <= 2


def test_resolve_returns_legacy_shape(index: TermIndex):
    out = index.resolve(["SCP", "amf", "TELL"])
    # TELL hallucination should be dropped; SCP and AMF resolve.
    assert set(out.keys()) == {"SCP", "amf"}
    assert out["SCP"]["full_name"] == "Service Communication Proxy"
    assert out["SCP"]["specs"] == ["ts_23_501", "ts_29_500"]
    assert out["SCP"]["matched_property"] == "abbreviation"


def test_empty_index_graceful():
    idx = TermIndex.empty()
    assert idx.lookup_abbrev("SCP") is None
    assert idx.lookup_full_name("Service Communication Proxy") is None
    assert idx.find_full_name_matches("anything goes here") == []
    assert idx.search("anything", limit=10) == []
    assert idx.resolve(["SCP"]) == {}


def test_skip_empty_abbreviation():
    records = [
        {"abbreviation": "", "full_name": "Empty One", "source_specs": []},
        {"abbreviation": None, "full_name": "Null Two", "source_specs": []},
        {"abbreviation": "OK", "full_name": "Valid", "source_specs": ["ts_99_001"]},
    ]
    idx = build_from_records(records)
    assert "OK" in idx.abbrev_map
    assert len(idx.abbrev_map) == 1


def test_handles_special_regex_chars_in_full_name():
    # Full names may contain parentheses / plus signs / dots; re.escape must
    # ensure they don't blow up regex compile or match.
    records = [
        {
            "abbreviation": "FOO",
            "full_name": "Foo (Plus+Bar) v1.0",
            "source_specs": ["ts_99_001"],
        },
    ]
    idx = build_from_records(records)
    matches = idx.find_full_name_matches("the Foo (Plus+Bar) v1.0 is here")
    assert len(matches) == 1
    assert matches[0][0] == "FOO"


# ---- Integration test (live KG) -----------------------------------------


@requires_neo4j
def test_build_from_live_kg(neo4j_driver):
    idx = build_term_index(neo4j_driver)
    # Sanity: KG has thousands of Terms; SCP must be present and resolve.
    assert len(idx.abbrev_map) > 100
    rec = idx.lookup_abbrev("SCP")
    assert rec is not None
    assert rec["full_name"] == "Service Communication Proxy"
    # Lowercase query must hit too.
    assert idx.lookup_abbrev("scp")["full_name"] == "Service Communication Proxy"
    # Full-name match must win over per-token search for the SCP query.
    matches = idx.find_full_name_matches("tell me about Service Communication Proxy")
    assert ("SCP" in [m[0] for m in matches])
