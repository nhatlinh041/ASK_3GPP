"""
Unit tests for IntentClassifier.classify_with_terms — verifies the JSON
parsing, intent validation, defensive list coercion, and failure modes
(timeout / invalid JSON). Uses a stub OllamaClient so no live LLM is needed.
"""
import json
import sys
from pathlib import Path

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from pipeline.intent_classifier import IntentClassifier  # noqa: E402


class StubLLM:
    """Mimics the OllamaClient.generate() interface used by IntentClassifier."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise = raise_exc
        self.calls: list[dict] = []

    def generate(self, prompt, *, model, format=None, think=None, timeout=None):
        self.calls.append({
            "prompt": prompt, "model": model, "format": format,
            "think": think, "timeout": timeout,
        })
        if self._raise is not None:
            raise self._raise
        return self._response


def _payload(intent="definition", abbrevs=None, fulls=None, refs=None) -> str:
    return json.dumps({
        "intent": intent,
        "abbreviations": abbrevs or [],
        "full_names": fulls or [],
        "spec_refs": refs or [],
    })


def test_classify_with_terms_full_name_extraction():
    llm = StubLLM(response=_payload(
        intent="definition",
        fulls=["Service Communication Proxy"],
        abbrevs=["SCP"],
    ))
    clf = IntentClassifier(llm=llm)
    out = clf.classify_with_terms("tell me about Service Communication Proxy", model="qwen3:14b")

    assert out is not None
    assert out["intent"] == "definition"
    assert out["abbreviations"] == ["SCP"]
    assert out["full_names"] == ["Service Communication Proxy"]
    assert out["spec_refs"] == []


def test_classify_with_terms_uppercases_lowercase_abbrev():
    # LLM might emit lowercase despite prompt instructions — classifier
    # normalises to UPPER so downstream lookup_abbrev matches the KG.
    llm = StubLLM(response=_payload(intent="definition", abbrevs=["scp", "amf"]))
    clf = IntentClassifier(llm=llm)
    out = clf.classify_with_terms("what is scp and amf", model="qwen3:14b")
    assert out["abbreviations"] == ["SCP", "AMF"]


def test_classify_with_terms_no_llm_returns_none():
    clf = IntentClassifier(llm=None)
    assert clf.classify_with_terms("anything", model="qwen3:14b") is None


def test_classify_with_terms_no_model_returns_none():
    clf = IntentClassifier(llm=StubLLM(response=_payload()))
    assert clf.classify_with_terms("anything", model=None) is None


def test_classify_with_terms_llm_raises_returns_none():
    llm = StubLLM(raise_exc=TimeoutError("ollama down"))
    clf = IntentClassifier(llm=llm)
    assert clf.classify_with_terms("anything", model="qwen3:14b") is None


def test_classify_with_terms_invalid_json_returns_none():
    llm = StubLLM(response="not json at all")
    clf = IntentClassifier(llm=llm)
    assert clf.classify_with_terms("anything", model="qwen3:14b") is None


def test_classify_with_terms_invalid_intent_returns_none():
    llm = StubLLM(response=json.dumps({
        "intent": "made_up_intent",
        "abbreviations": ["SCP"],
        "full_names": [],
        "spec_refs": [],
    }))
    clf = IntentClassifier(llm=llm)
    assert clf.classify_with_terms("anything", model="qwen3:14b") is None


def test_classify_with_terms_missing_keys_coerced_empty():
    # Only intent provided — abbreviations / full_names / spec_refs missing.
    # Defensive parser should coerce them to [].
    llm = StubLLM(response=json.dumps({"intent": "general"}))
    clf = IntentClassifier(llm=llm)
    out = clf.classify_with_terms("anything", model="qwen3:14b")
    assert out["intent"] == "general"
    assert out["abbreviations"] == []
    assert out["full_names"] == []
    assert out["spec_refs"] == []


def test_classify_with_terms_drops_non_string_items():
    llm = StubLLM(response=json.dumps({
        "intent": "definition",
        "abbreviations": ["SCP", 123, None, "AMF"],
        "full_names": ["Service Communication Proxy", {"x": 1}],
        "spec_refs": [],
    }))
    clf = IntentClassifier(llm=llm)
    out = clf.classify_with_terms("anything", model="qwen3:14b")
    assert out["abbreviations"] == ["SCP", "AMF"]
    assert out["full_names"] == ["Service Communication Proxy"]


def test_classify_with_terms_filters_empty_strings():
    llm = StubLLM(response=json.dumps({
        "intent": "definition",
        "abbreviations": ["", "  ", "SCP"],
        "full_names": [""],
        "spec_refs": [],
    }))
    clf = IntentClassifier(llm=llm)
    out = clf.classify_with_terms("anything", model="qwen3:14b")
    assert out["abbreviations"] == ["SCP"]
    assert out["full_names"] == []


def test_classify_with_terms_uses_format_json_and_think_false():
    llm = StubLLM(response=_payload())
    clf = IntentClassifier(llm=llm)
    clf.classify_with_terms("hello", model="qwen3:14b")
    assert llm.calls[0]["format"] == "json"
    assert llm.calls[0]["think"] is False
