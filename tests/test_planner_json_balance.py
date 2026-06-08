"""
Unit tests for the JSON brace-balancing recovery in adaptive_hop_prompts.
Regression: qwen3:14b sometimes truncates planner output after closing the
last array, dropping the outer `}`. The parser must still succeed instead
of bailing out with a parse error and aborting the react_agent run.
"""
import sys
from pathlib import Path

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

from retrieval.adaptive_hop_prompts import (  # noqa: E402
    PlannerParseError,
    _balance_json,
    parse_planner_action,
)


def test_balance_json_already_balanced():
    s = '{"a": 1, "b": [2, 3]}'
    assert _balance_json(s) == s


def test_balance_json_missing_outer_brace():
    s = '{"a": 1, "b": [2, 3]'
    assert _balance_json(s) == s + "}"


def test_balance_json_missing_array_and_brace():
    s = '{"a": 1, "b": [2, 3'
    assert _balance_json(s) == s + "]}"


def test_balance_json_braces_inside_string_ignored():
    # Cypher-like value with `{abbreviation: 'SCP'}` inside a string.
    # Inner braces must NOT pop the stack.
    s = '{"cypher": "MATCH (t:Term {abbreviation: \'SCP\'})", "k": 1'
    assert _balance_json(s) == s + "}"


def test_balance_json_escaped_quote_does_not_toggle_string():
    # The `\"` is an escaped quote inside a JSON string, so the string
    # continues. The trailing `}` should be appended at the very end.
    s = '{"msg": "she said \\"hi\\" to him", "n": 5'
    assert _balance_json(s) == s + "}"


def test_parse_planner_action_recovers_truncated_output():
    # Real shape produced by qwen3:14b in the v2 debug capture: closes
    # remaining_gaps array with `]` then forgets the outer `}`.
    raw = (
        '{"thought": "go now",\n'
        '  "tool": "cypher_query",\n'
        '  "args": {"cypher": "MATCH (t:Term {abbreviation: \'SCP\'}) RETURN t LIMIT $top_k", "top_k": 6, "purpose": "definition"},\n'
        '  "remaining_gaps": ["What is SCP?", "Use cases?"]'  # <-- no closing }
    )
    action = parse_planner_action(raw)
    assert action.tool == "cypher_query"
    assert action.args["cypher"].startswith("MATCH")
    assert action.remaining_gaps == ["What is SCP?", "Use cases?"]


def test_parse_planner_action_still_rejects_truly_invalid():
    # Garbage that can't be balanced — extra closing brace with no opener.
    with pytest.raises(PlannerParseError):
        parse_planner_action('}')
