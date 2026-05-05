"""
Intent classifier — gọi LLM (cùng model user chọn) để map câu hỏi vào 1 trong 7
intent + trả JSON. Có regex fallback để giữ pipeline luôn chạy được khi LLM fail
(timeout, parse error, model trả intent ngoài enum).
"""
import json
import re
from typing import Optional

from llm import OllamaClient


# Intent enum — phải khớp với những gì cypher_generator + orchestrator hiểu
ALLOWED_INTENTS = (
    "definition",
    "procedure",
    "comparison",
    "network_function",
    "reference",
    "relationship",
    "general",
)


# Regex pattern fallback — dùng khi LLM không khả dụng hoặc trả invalid output.
# Giữ nguyên priority cũ để hành vi predictable trong test offline.
_REGEX_PATTERNS: dict[str, list[str]] = {
    "definition": [
        r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bmeaning of\b",
        r"\bstand for\b", r"\bacronym\b",
    ],
    "procedure": [
        r"\bhow does\b", r"\bhow to\b", r"\bsteps\b", r"\bprocess\b",
        r"\bprocedure\b", r"\bflow\b", r"\bsequence\b",
    ],
    "comparison": [
        r"\bdifference\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bsimilar\b", r"\bdistinguish\b",
    ],
    "network_function": [
        r"\bamf\b", r"\bsmf\b", r"\bupf\b", r"\bnrf\b", r"\bausf\b",
        r"\budm\b", r"\bpcf\b", r"\bnssf\b", r"\baf\b", r"\bnef\b",
        r"\bsepp\b", r"\bscp\b",
    ],
    "reference": [
        r"\bts\s+\d{2}\.\d{3}\b", r"\bspec\b", r"\bspecification\b",
        r"\bdocument\b", r"\bclause\b", r"\bsection\b",
    ],
    "relationship": [
        r"\brelationship\b", r"\binterface\b", r"\binteract\b",
        r"\bconnect\b", r"\blink\b", r"\bbetween\b.*\band\b",
    ],
}


# LLM prompt — ngắn, có disambiguation rule cho cases ambiguous (vd "What is AMF?"
# trùng cả definition lẫn network_function), Ollama format=json đảm bảo parse OK.
def _build_prompt(query: str) -> str:
    return f"""Classify this 3GPP technical question into exactly ONE intent.

Intents:
- definition: "What is X?", "Define X", "Meaning of X" — asking for a concept's meaning.
- procedure: "How does X work?", "Steps to do X" — asking for a process or sequence.
- comparison: "Difference between X and Y", "Compare X vs Y" — asking how things differ.
- network_function: questions whose PRIMARY focus is listing/attributes of a 5G NF (AMF/SMF/UPF/NRF/...) where the question is NOT a what/how/compare/relationship form.
- reference: questions about which spec/clause/document defines something.
- relationship: "How is X related to Y?", "What NFs interact with X?", "Interface between X and Y".
- general: anything else.

Disambiguation rules (apply in order):
1. If the question matches "What is X?" / "Define X" — choose `definition`, even if X is a network function.
2. If the question is "List/which NFs interact with X" or "How does X talk to Y" — choose `relationship`.
3. If the question is purely about a NF without what/how/compare/relationship verbs — choose `network_function`.

Question: {query}

Respond with STRICT JSON, no prose, no markdown fences:
{{"intent": "<one of: definition|procedure|comparison|network_function|reference|relationship|general>"}}
"""


class IntentClassifier:
    """LLM-first classifier với regex fallback. Constructor nhận OllamaClient
    để classify gọi cùng instance LLM của orchestrator (giữ Ollama warm)."""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self._llm = llm

    def classify(self, query: str, model: Optional[str] = None) -> str:
        """LLM path nếu có llm + model; fallback regex khi fail bất kỳ chỗ nào.
        Trả về string intent (luôn nằm trong ALLOWED_INTENTS)."""
        if self._llm is not None and model:
            llm_intent = self._classify_llm(query, model)
            if llm_intent is not None:
                return llm_intent
        return self._classify_regex(query)

    # Gọi LLM với format=json + think=False (intent là task nhỏ, không cần CoT)
    # Trả intent string nếu hợp lệ, None nếu fail (để caller fallback regex).
    def _classify_llm(self, query: str, model: str) -> Optional[str]:
        try:
            raw = self._llm.generate(
                _build_prompt(query),
                model=model,
                format="json",
                think=False,
                # Timeout 90s đủ cho cold-start qwen3:14b (~45s); subsequent
                # calls khi model warm sẽ chỉ mất 1-3s.
                timeout=90,
            )
        except Exception:
            return None
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        intent = str(obj.get("intent", "")).strip().lower()
        return intent if intent in ALLOWED_INTENTS else None

    # Regex classifier nguyên bản — giữ làm safety net
    def _classify_regex(self, query: str) -> str:
        q = query.lower()

        # Network function check first (high precision) — match cũ
        for pattern in _REGEX_PATTERNS["network_function"]:
            if re.search(pattern, q):
                return "network_function"

        # Other intents by match count
        scores: dict[str, int] = {}
        for intent, patterns in _REGEX_PATTERNS.items():
            if intent == "network_function":
                continue
            scores[intent] = sum(1 for p in patterns if re.search(p, q))

        if not scores or max(scores.values()) == 0:
            return "general"

        return max(scores, key=lambda i: scores[i])
