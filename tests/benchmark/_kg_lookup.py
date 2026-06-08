"""Online Neo4j helper to test if a gold-answer phrase exists anywhere in
the KG. Used by the failure analyzer to distinguish "fixable retrieval miss"
from "out-of-domain question (gold text not in any 3GPP chunk)".

Connects via the standard ``.env`` credentials
(``NEO4J_URI``/``NEO4J_USER``/``NEO4J_PASSWORD``) — same pattern as
``rag-engine/retrieval/graph_search.py:78``. Only emits a single read-only
``MATCH (c:Chunk) WHERE toLower(c.content) CONTAINS toLower($needle)
RETURN c.chunk_id LIMIT 1`` per unique needle and caches the result.
"""

from __future__ import annotations

import re
from typing import Optional


# Stopwords + structural words that produce false positives if used alone as
# a needle ("a", "the", "system" appear in tens of thousands of chunks).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "as", "from",
    "that", "which", "this", "these", "those", "it", "its", "such", "any",
    "all", "no", "not", "if", "then", "else", "than", "into", "out", "up",
    "down", "over", "under", "via",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Question-stem words that look like uppercase abbreviations but are actually
# English wh-words. Without this set, "WHAT/WHICH/HOW" would always match KG
# because they appear in section titles like "What's New".
_QUESTION_STOPLIST = {
    "WHAT", "WHICH", "HOW", "WHY", "WHEN", "WHERE", "WHO",
    "DOES", "ARE", "DOES", "RIT", "SRIT",
}

# Patterns to extract topic candidates from a question stem. Order matters:
# more specific patterns first so we prefer "the User Equipment (UE)" over the
# bare token "User".
_TOPIC_PATTERNS = [
    re.compile(r"\b([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){1,3})\b"),  # CamelCase n-gram
    re.compile(r"\b([A-Z]{3,8}(?:-[A-Z0-9]+)?)\b"),              # bare abbreviations
    # Lowercase noun phrase after wh-anchors / prepositions
    re.compile(r"\b(?:of|is|are|about|for) (?:a |an |the )?([a-z][\w\s\-]{4,40}?)(?:\?|\.|,|\sin\s|\sto\s|\sfor\s|\sthat\s|\(|$)"),
]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# Lookup helper. Holds a long-lived bolt session and an in-process cache so
# repeated calls with the same needle are free.
class KGContentLookup:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        ngram_words: int = 6,
    ):
        # Lazy import keeps the analyzer importable on machines that don't
        # have the neo4j driver installed (offline classifier still works).
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._cache: dict[str, bool] = {}
        self._ngram_words = max(2, ngram_words)

    # Cypher: indexed labels, content scan, LIMIT 1 → bails on first hit.
    # toLower applied on both sides so the match is case-insensitive.
    _Q = (
        "MATCH (c:Chunk) "
        "WHERE toLower(c.content) CONTAINS toLower($needle) "
        "RETURN c.chunk_id LIMIT 1"
    )

    def _lookup_raw(self, needle: str) -> bool:
        key = needle.strip().lower()
        if not key:
            return False
        if key in self._cache:
            return self._cache[key]
        with self._driver.session() as s:
            rec = s.run(self._Q, needle=key).single()
        hit = rec is not None
        self._cache[key] = hit
        return hit

    # Build a list of needle candidates from the gold text, in priority order:
    # 1. The full text (best when gold is a short phrase like "Basic Service Area").
    # 2. The first N content tokens (skip stopwords) joined as a phrase — covers
    #    long descriptive golds like "A telecommunication service that uses short
    #    messages...".
    # 3. The longest 4-word window made of non-stopword tokens — last-ditch.
    def _candidates(self, gold_text: str) -> list[str]:
        out: list[str] = []
        full = (gold_text or "").strip()
        if not full:
            return []
        out.append(full)

        toks = _tokens(full)
        content_toks = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
        if len(content_toks) >= 2:
            head = " ".join(content_toks[: self._ngram_words])
            if head and head != full.lower():
                out.append(head)
        # Sliding 4-window of content tokens — surfaces the most distinctive
        # multiword span in long descriptive golds.
        if len(content_toks) > 4:
            window = " ".join(content_toks[:4])
            if window not in out:
                out.append(window)

        return out

    # Public predicate: True ⇔ at least one candidate needle matches a chunk.
    def has_content(self, gold_text: str) -> bool:
        for needle in self._candidates(gold_text):
            if self._lookup_raw(needle):
                return True
        return False

    # Pull out plausible topic spans from the question stem: capitalized
    # phrases (e.g. "User Equipment", "RIS"), bare uppercase abbreviations,
    # and noun phrases following "of/is/about/for". WH-words are excluded so
    # we don't probe ``WHAT``.
    @staticmethod
    def extract_topics(question: str) -> list[str]:
        if not question:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for pat in _TOPIC_PATTERNS:
            for m in pat.finditer(question):
                cand = m.group(1).strip(' \'"?.,()').strip()
                if not cand:
                    continue
                upper = cand.upper()
                if upper in _QUESTION_STOPLIST:
                    continue
                if len(cand) < 3 or len(cand) > 60:
                    continue
                key = cand.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        return out[:8]

    # Returns True if any extracted topic from the question stem is found in
    # at least one chunk. Used by the failure classifier to distinguish
    # "in-domain but gold paraphrased" (topic present, gold absent) from
    # "out-of-domain" (both absent).
    def has_topic_in_kg(self, question: str) -> bool:
        for topic in self.extract_topics(question):
            if self._lookup_raw(topic):
                return True
        return False

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


# Tiny .env parser used when python-dotenv isn't installed. Handles the simple
# `KEY=VALUE` lines we have in the repo's .env (no expansion, no quoting tricks).
def _read_dotenv_into_environ(path) -> None:
    import os

    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Don't clobber values the shell already set.
        os.environ.setdefault(key, val)


# Convenience constructor: read credentials from the repo `.env`. Returns
# ``None`` if Neo4j is unreachable so callers can degrade to offline mode.
def from_env() -> Optional["KGContentLookup"]:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    try:
        from dotenv import load_dotenv  # preferred when available
        load_dotenv(env_path, override=False)
    except ImportError:
        _read_dotenv_into_environ(env_path)

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        return None
    try:
        kg = KGContentLookup(uri, user, password)
        # Warm-up query catches "Neo4j down" before the analyzer fires
        # hundreds of misleading misses.
        kg._lookup_raw("network")
        return kg
    except Exception as e:
        import sys
        print(f"  [warn] KG lookup disabled — could not connect ({e})", file=sys.stderr)
        return None
