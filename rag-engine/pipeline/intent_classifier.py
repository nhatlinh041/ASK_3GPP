"""
Intent classifier — maps user question to one of 7 intents using keyword heuristics.
"""
import re

INTENT_PATTERNS: dict[str, list[str]] = {
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


class IntentClassifier:
    def classify(self, query: str) -> str:
        """
        Return intent string. Checks patterns in priority order.
        Falls back to 'general' when no pattern matches.
        """
        q = query.lower()

        # Network function check first (high precision)
        for pattern in INTENT_PATTERNS["network_function"]:
            if re.search(pattern, q):
                return "network_function"

        # Other intents by match count
        scores: dict[str, int] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            if intent == "network_function":
                continue
            scores[intent] = sum(1 for p in patterns if re.search(p, q))

        if not scores or max(scores.values()) == 0:
            return "general"

        return max(scores, key=lambda i: scores[i])
