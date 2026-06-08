"""Multi-label rule engine for benchmark wrong-answer triage.

Each rule inspects a paired ``(QuestionTrace, result_dict)`` — the trace gives
us pipeline stage signals (intent, Cypher pattern, top reranked scores …) and
the result dict (an entry from ``results.json``) gives the textual model
response and gold answer text.

Rules are deliberately additive: one wrong answer can fire several at once
(e.g. ``GRAPH_PATTERN_C`` + ``WEAK_RERANK``). The downstream report then
counts each rule independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ._debug_log_parser import QuestionTrace

# Type aliases for the optional KG probes — see _kg_lookup.KGContentLookup.
# ``KGLookup`` checks if the gold answer text appears anywhere in the KG.
# ``KGTopicLookup`` checks if any topic noun phrase from the question stem
# appears anywhere in the KG. Combined, they let us split "OUT_OF_DOMAIN"
# (neither gold nor topic in KG) from "GOLD_PARAPHRASED" (topic in KG but
# gold answer phrased differently than the spec).
KGLookup = Callable[[str], bool]
KGTopicLookup = Callable[[str], bool]

# Same regex the runner uses to detect a "I don't know" reply (run_teleqna.py).
_REFUSAL_PATTERNS = (
    re.compile(r"\bcontext\s+does\s+not\s+(?:cover|specify|include|provide|address|mention|contain)\b", re.IGNORECASE),
    re.compile(r"\bnone\s+of\s+the\s+(?:options|choices|answers)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+(?:be\s+)?(?:determined|determine)\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:context|information)\b", re.IGNORECASE),
)

# Letter-prefixed answer that the runner's regex steps 1-2 would catch.
_LETTER_ANSWER = re.compile(r"^\s*(?:the\s+)?answer[:\s]+(?:\(?\d+\)?|[A-Za-z])\b", re.IGNORECASE)

_WEAK_RERANK_THRESHOLD = 0.3
_STRONG_RERANK_THRESHOLD = 0.5
_RETRIEVAL_OK_MIN_OVERLAP_CHARS = 6


@dataclass
class FailureLabels:
    idx: int
    rules: list[str]


# A rule that consistently fires explains a class of failures; per-question
# rule lists let the report show worked examples.
RULE_DESCRIPTIONS = {
    "REFUSAL": "LLM từ chối trả lời ('context does not cover/specify/...')",
    "EXTRACT_FUZZY": "Phải dùng fuzzy match để đoán letter — model không output 'Answer: <letter>' rõ",
    "GRAPH_ERROR": "Cypher chạy lỗi runtime (graph_error non-null)",
    "GRAPH_PATTERN_C": "Cypher generator phát Pattern C sentinel — không có anchor term/section",
    "GRAPH_ZERO_ROW": "Cypher chạy được nhưng trả 0 row (term/regex không khớp KG)",
    "WEAK_RERANK": "Top reranked score < 0.3 — không có chunk nào liên quan thực sự",
    "RETRIEVAL_OK_LLM_WRONG": "Retrieval ổn (top score ≥0.5) và gold text nằm trong context, nhưng LLM vẫn chọn sai",
    "RETRIEVAL_NO_GOLD_TEXT": "Sources non-empty nhưng gold text không nằm trong chunk nào — chọn nhầm spec/section",
    "EMPTY_CONTEXT_GUESS": "Cả vector lẫn graph rỗng sau rerank — câu trả lời thuần dựa training",
    "GOLD_PARAPHRASED": "Topic câu hỏi có trong KG nhưng gold answer được paraphrase khác spec — fixable bằng retrieval/rerank tốt hơn (cần --kg-lookup)",
    "OUT_OF_DOMAIN": "Cả gold text lẫn topic câu hỏi đều không có trong KG — câu hỏi thực sự nằm ngoài phạm vi 3GPP (cần --kg-lookup)",
    "UNKNOWN": "Không khớp rule nào",
}


def _matches_refusal(text: str) -> bool:
    if not text:
        return False
    head = text.split("\n", 1)[0]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


# Same word-boundary substring check the answer evaluator runs against
# choice text, but here we test whether the gold *content* is reachable from
# any of the reranked sources. We lower-case both sides.
def _expected_in_sources(expected: str, sources: list[dict]) -> bool:
    if not expected or len(expected) < _RETRIEVAL_OK_MIN_OVERLAP_CHARS:
        return False
    needle = expected.strip().lower()
    if len(needle) < _RETRIEVAL_OK_MIN_OVERLAP_CHARS:
        return False
    # The gold text is often longer than chunk content; loosen by trying the
    # first significant n-gram (5-6 words) instead of full match.
    head = " ".join(needle.split()[:6])
    head = head if len(head) >= _RETRIEVAL_OK_MIN_OVERLAP_CHARS else needle
    for src in sources:
        content = (src.get("content") or "").lower()
        if not content:
            continue
        if head and head in content:
            return True
        if needle in content:
            return True
    return False


# Return all rule codes that fire for this question. Empty list means the
# question was answered correctly (callers usually filter on incorrect ones,
# but we return [] for correct rather than UNKNOWN).
#
# Two-stage KG probe (both optional):
#   * ``kg_lookup(gold_text)`` — does the gold answer string exist somewhere
#     in any chunk?
#   * ``kg_topic_check(question)`` — does the question stem reference a topic
#     that exists in the KG?
#
# Decision tree when both are available:
#   gold ∈ KG                               → ordinary retrieval rules apply
#   gold ∉ KG, topic ∈ KG                   → GOLD_PARAPHRASED (in-domain)
#   gold ∉ KG, topic ∉ KG                   → OUT_OF_DOMAIN (truly outside 3GPP)
#
# Both rules suppress RETRIEVAL_NO_GOLD_TEXT — the latter only makes sense
# when the gold is reachable from somewhere in the KG.
def classify(
    trace: QuestionTrace,
    result: dict,
    kg_lookup: Optional[KGLookup] = None,
    kg_topic_check: Optional[KGTopicLookup] = None,
) -> list[str]:
    if trace.correct:
        return []

    rules: list[str] = []
    response = (result or {}).get("model_response") or ""
    expected = (result or {}).get("expected_answer") or ""
    question = (result or {}).get("question") or ""

    out_of_domain = False
    gold_paraphrased = False
    if kg_lookup is not None and expected:
        try:
            gold_in_kg = kg_lookup(expected)
        except Exception:
            # Transient KG error — bail out of the whole probe so we don't
            # wrongly mark everything as OOD.
            gold_in_kg = True
        if not gold_in_kg:
            # Gold isn't anywhere — ask the topic probe whether the question
            # at least *names* something the KG covers. Without the topic
            # probe we conservatively keep the legacy OOD label.
            topic_in_kg = True
            if kg_topic_check is not None and question:
                try:
                    topic_in_kg = kg_topic_check(question)
                except Exception:
                    topic_in_kg = True
            if topic_in_kg:
                gold_paraphrased = True
            else:
                out_of_domain = True
    if out_of_domain:
        rules.append("OUT_OF_DOMAIN")
    elif gold_paraphrased:
        rules.append("GOLD_PARAPHRASED")

    if trace.pred is None and _matches_refusal(response):
        rules.append("REFUSAL")
    elif trace.pred is not None and not _LETTER_ANSWER.match(response.lstrip()):
        # Runner extracted a letter via fuzzy fallback (steps 5-7). Surfaces
        # answers where the model's headline did not say "Answer: X" but the
        # extractor picked something anyway — high false-positive risk.
        rules.append("EXTRACT_FUZZY")

    if trace.graph_error:
        rules.append("GRAPH_ERROR")
    if trace.cypher_pattern == "C":
        rules.append("GRAPH_PATTERN_C")
    elif (
        trace.graph_count == 0
        and trace.cypher_pattern in {"A1", "A2", "B"}
        and not trace.graph_error
    ):
        rules.append("GRAPH_ZERO_ROW")

    top_score = max(trace.rerank_top_scores) if trace.rerank_top_scores else 0.0

    if not trace.sources and trace.rerank_output_count == 0:
        rules.append("EMPTY_CONTEXT_GUESS")
    elif top_score < _WEAK_RERANK_THRESHOLD and trace.sources:
        rules.append("WEAK_RERANK")

    if trace.sources:
        if _expected_in_sources(expected, trace.sources):
            if top_score >= _STRONG_RERANK_THRESHOLD:
                rules.append("RETRIEVAL_OK_LLM_WRONG")
        elif not (out_of_domain or gold_paraphrased):
            # Sources exist but gold text not present — retrieval surfaced
            # related-but-wrong chunks (often the spec disagrees with the
            # benchmark's expected answer text). Suppressed when the gold
            # isn't reachable from the KG at all (OUT_OF_DOMAIN/GOLD_PARAPHRASED):
            # in that case "wrong spec selected" is meaningless.
            rules.append("RETRIEVAL_NO_GOLD_TEXT")

    if not rules:
        rules.append("UNKNOWN")
    return rules


def classify_all(
    traces: Iterable[QuestionTrace],
    results: list[dict],
    kg_lookup: Optional[KGLookup] = None,
    kg_topic_check: Optional[KGTopicLookup] = None,
) -> dict[int, list[str]]:
    """Build a {idx → rule list} mapping. Aligns trace.idx (1-based) with
    results list position (0-based). Pass ``kg_lookup`` + ``kg_topic_check``
    to enable the OUT_OF_DOMAIN / GOLD_PARAPHRASED split (live Neo4j probes
    per wrong question)."""
    by_idx: dict[int, list[str]] = {}
    for trace in traces:
        result = results[trace.idx - 1] if 0 <= trace.idx - 1 < len(results) else {}
        by_idx[trace.idx] = classify(
            trace,
            result,
            kg_lookup=kg_lookup,
            kg_topic_check=kg_topic_check,
        )
    return by_idx
