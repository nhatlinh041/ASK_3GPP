"""
Unit tests for the cross-encoder rerank input augmentation:

Step 1 (_augment_for_rerank) — prepends section_title to chunk content.
Step 2 (_expand_query_for_rerank) — appends KG-validated abbreviation +
        full_name from resolved_terms when missing from the query.

These pure-text helpers don't require Neo4j or the cross-encoder model.
"""
import sys
from pathlib import Path

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from retrieval.fusion import (  # noqa: E402
    _augment_for_rerank,
    _expand_query_for_rerank,
)


# ---- Step 1 — content augmentation --------------------------------------


def test_augment_prepends_section_title():
    chunk = {
        "section": "Requirements on the Service Communication Proxy (SCP)",
        "content": "The SCP has interfaces with Network Functions (NF).",
    }
    out = _augment_for_rerank(chunk)
    assert out.startswith("Requirements on the Service Communication Proxy (SCP)")
    assert "The SCP has interfaces" in out


def test_augment_accepts_section_title_alias():
    # Vector path uses `section`, graph path uses `section_title`. Both work.
    chunk = {
        "section_title": "Definitions",
        "content": "For the purposes of this document...",
    }
    out = _augment_for_rerank(chunk)
    assert out.startswith("Definitions")


def test_augment_skips_when_title_already_in_content():
    # Don't double-feed the same phrase to the cross-encoder.
    chunk = {
        "section": "SCP Definition",
        "content": "SCP Definition: The SCP is a network function...",
    }
    assert _augment_for_rerank(chunk) == "SCP Definition: The SCP is a network function..."


def test_augment_handles_missing_title():
    chunk = {"content": "Body only."}
    assert _augment_for_rerank(chunk) == "Body only."


def test_augment_handles_missing_content():
    # Title-only chunk: still emit the title so the cross-encoder has
    # something semantic to work with rather than an empty string.
    chunk = {"section": "Just a title"}
    out = _augment_for_rerank(chunk)
    assert out.startswith("Just a title")


def test_augment_handles_both_missing():
    assert _augment_for_rerank({}) == ""


# ---- Step 2 — query expansion -------------------------------------------


def test_expand_appends_full_name_when_query_has_only_abbrev():
    resolved = {"SCP": {"full_name": "Service Communication Proxy", "specs": []}}
    out = _expand_query_for_rerank("tell me about SCP", resolved)
    assert "Service Communication Proxy" in out
    # Should keep the original query verbatim
    assert out.startswith("tell me about SCP")


def test_expand_appends_abbrev_when_query_has_only_full_name():
    resolved = {"SCP": {"full_name": "Service Communication Proxy", "specs": []}}
    out = _expand_query_for_rerank("tell me about Service Communication Proxy", resolved)
    assert "SCP" in out


def test_expand_skips_when_both_already_present():
    # User already wrote both forms — nothing to add.
    resolved = {"SCP": {"full_name": "Service Communication Proxy", "specs": []}}
    out = _expand_query_for_rerank("SCP Service Communication Proxy details", resolved)
    assert out == "SCP Service Communication Proxy details"


def test_expand_case_insensitive_match():
    # Lowercase "scp" in query should be detected as already present.
    resolved = {"SCP": {"full_name": "Service Communication Proxy", "specs": []}}
    out = _expand_query_for_rerank("what is scp", resolved)
    # "SCP" not appended (lowercase form already there); full_name appended
    assert out.lower().count("scp") == 1
    assert "Service Communication Proxy" in out


def test_expand_handles_multiple_resolved_terms():
    resolved = {
        "SCP": {"full_name": "Service Communication Proxy", "specs": []},
        "AMF": {"full_name": "Access and Mobility Management Function", "specs": []},
    }
    out = _expand_query_for_rerank("compare SCP and AMF", resolved)
    assert "Service Communication Proxy" in out
    assert "Access and Mobility Management Function" in out


def test_expand_no_resolved_returns_query_unchanged():
    assert _expand_query_for_rerank("hello world", None) == "hello world"
    assert _expand_query_for_rerank("hello world", {}) == "hello world"


def test_expand_handles_invalid_entry_shape():
    # Defensive: if resolved has a non-dict value, it must be silently skipped.
    resolved = {"BAD": "not-a-dict", "SCP": {"full_name": "Service Communication Proxy"}}
    out = _expand_query_for_rerank("tell me about SCP", resolved)
    assert "Service Communication Proxy" in out


def test_expand_skips_empty_full_name():
    resolved = {"SCP": {"full_name": "", "specs": []}}
    out = _expand_query_for_rerank("tell me about SCP", resolved)
    # "SCP" already in query; nothing to add → unchanged
    assert out == "tell me about SCP"


def test_expand_v3_bug_repro():
    # Exact case from json_debug/tell me about Service Communication Proxy_v3.json:
    # query was "tell me about 5G SCP" — needs full_name appended so the
    # cross-encoder can match SCP-content chunks that use the full term.
    resolved = {"SCP": {"full_name": "Service Communication Proxy", "specs": []}}
    out = _expand_query_for_rerank("tell me about 5G SCP", resolved)
    assert out == "tell me about 5G SCP Service Communication Proxy"
