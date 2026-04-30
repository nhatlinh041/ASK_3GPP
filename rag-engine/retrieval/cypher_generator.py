"""
LLM-driven Cypher generator. Given the live KG schema and the user's question
(plus resolved term context), asks an Ollama model to write a single read-only
Cypher query that returns chunk results.

Safety: rejects any query containing write keywords or multiple statements.
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from llm import OllamaClient


# Cypher write/admin keywords that must never appear in a generated query.
# Matched as whole words (case-insensitive) to allow them inside string literals
# that the validator will reject anyway via the semicolon/multi-statement check.
FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
    "LOAD CSV", "USING PERIODIC", "FOREACH",
    "CALL { ", "CALL{",  # block subqueries that may write
    "CALL DBMS.", "CALL APOC.CREATE", "CALL APOC.MERGE", "CALL APOC.LOAD",
    "CALL APOC.PERIODIC", "CALL APOC.CYPHER.RUN", "CALL APOC.DO.",
)

# Cypher fence patterns the LLM may emit; we strip them
FENCE_PATTERN = re.compile(r"^```(?:cypher|sql|neo4j)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class CypherGeneration:
    cypher: str            # the validated Cypher to run
    raw_response: str      # the LLM's raw output (for debugging / showing in trail)
    model: str
    prompt: str            # full prompt that was sent (so the UI can display it)


class CypherValidationError(ValueError):
    pass


class LLMCypherGenerator:
    def __init__(self, llm: OllamaClient, model: str):
        self._llm = llm
        self._model = model

    def generate(
        self,
        question: str,
        schema_text: str,
        intent: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        primary_term: Optional[str] = None,
    ) -> CypherGeneration:
        prompt = self._build_prompt(
            question=question,
            schema_text=schema_text,
            intent=intent,
            resolved_terms=resolved_terms,
            primary_term=primary_term,
        )
        raw = self._llm.generate(prompt, model=self._model)
        cypher = self._clean(raw)
        self._validate(cypher)
        return CypherGeneration(cypher=cypher, raw_response=raw, model=self._model, prompt=prompt)

    def generate_stream(
        self,
        question: str,
        schema_text: str,
        intent: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        primary_term: Optional[str] = None,
        think: bool = True,
    ) -> Iterator[dict]:
        """
        Stream Cypher generation token-by-token so the UI can show progress live.

        Yields events:
          {kind: 'prompt', prompt: str, model: str}                       — once at start
          {kind: 'thinking', token: str, accumulated: str}                — per thinking token (reasoning models)
          {kind: 'token', token: str, accumulated: str}                   — per Cypher token from LLM
          {kind: 'done', cypher: str, raw: str}                           — once at end (after validation)
          {kind: 'error', error: str, raw: str}                           — if validation fails
        """
        prompt = self._build_prompt(
            question=question,
            schema_text=schema_text,
            intent=intent,
            resolved_terms=resolved_terms,
            primary_term=primary_term,
        )
        yield {"kind": "prompt", "prompt": prompt, "model": self._model}

        raw_parts: list[str] = []
        thinking_parts: list[str] = []
        try:
            # Reasoning models think first, then emit the Cypher. We forward both
            # phases so the UI can show progress during the (often long) thinking phase.
            # When think=False, the model jumps straight to the Cypher output.
            for ev in self._llm.generate_stream_full(prompt, model=self._model, think=think):
                if ev["kind"] == "thinking":
                    thinking_parts.append(ev["token"])
                    yield {
                        "kind": "thinking",
                        "token": ev["token"],
                        "accumulated": "".join(thinking_parts),
                    }
                else:
                    raw_parts.append(ev["token"])
                    yield {
                        "kind": "token",
                        "token": ev["token"],
                        "accumulated": "".join(raw_parts),
                    }
        except Exception as e:
            yield {"kind": "error", "error": f"llm_error: {type(e).__name__}: {e}", "raw": "".join(raw_parts)}
            return

        raw = "".join(raw_parts)
        cypher = self._clean(raw)
        try:
            self._validate(cypher)
        except CypherValidationError as e:
            yield {"kind": "error", "error": f"validation: {e}", "raw": raw}
            return
        yield {"kind": "done", "cypher": cypher, "raw": raw}

    # ----- internal -----

    @staticmethod
    def _build_prompt(
        question: str,
        schema_text: str,
        intent: Optional[str],
        resolved_terms: Optional[dict],
        primary_term: Optional[str],
    ) -> str:
        # Format resolved-term hints — gives the LLM authoritative full names + spec sources
        terms_block = ""
        if resolved_terms:
            lines = []
            for abbr, info in resolved_terms.items():
                full = info.get("full_name", "?")
                specs = info.get("specs") or []
                spec_str = f" (defined in {', '.join(specs)})" if specs else ""
                lines.append(f"  - {abbr} = {full}{spec_str}")
            terms_block = "Resolved terms (authoritative from KG):\n" + "\n".join(lines)
        elif primary_term:
            terms_block = f"Primary term hint: {primary_term}"

        intent_block = f"Intent classification: {intent}" if intent else ""

        return f"""You write a single Cypher query for a Neo4j knowledge graph of 3GPP technical specifications.
The query retrieves the most relevant Chunk nodes for the user's question.

# Live graph schema
{schema_text}

# User question
{question}

{intent_block}

{terms_block}

# Real KG schema (verified against live database — TRUST THIS over any schema_text)

Node properties:
- Term:           abbreviation, full_name, primary_spec, source_specs, term_type
- Chunk:          chunk_id, spec_id, section_id, section_title, content,
                  chunk_type, subject, key_terms, is_parent_section
- Document:       spec_id, version, title, total_chunks
- Subject:        name, priority, description (generic categories — NOT term-specific)

Relationships (verified counts after rebuild with new schema):
- (Chunk)-[:HAS_SUBJECT]->(Subject)        # ~197k — Subject is a GENERIC category, NOT a Term. DO NOT use to find chunks for a term.
- (Chunk)-[:REFERENCES_SPEC]->(Document)   # ~165k — chunk-to-doc link (use for "any chunk citing TS X" queries)
- (Chunk)-[:REFERENCES_CHUNK]->(Chunk)     # internal section refs + cross-spec section refs.
                                           #   Edge property `is_external`: false=same-spec, true=cross-spec.
                                           #   Other properties: ref_type ('clause'), ref_id, confidence.
                                           #   USE for chunk-level cross-ref. For cross-spec query, prefer this over
                                           #   REFERENCES_SPEC when section-precise targeting matters.
- (Document)-[:CONTAINS]->(Chunk)          # ~195k — every Chunk has incoming CONTAINS from its Document. SAFE to use for doc→chunks traversal.
- (Term)-[:DEFINED_IN]->(Document)         # ~5k — Term defined in Document. SAFE but NOTE: gold pattern for "chunks mentioning a term"
                                           #   is still `c.spec_id IN t.source_specs` (property lookup, faster than traversal).
                                           #   Use DEFINED_IN only for "list all terms defined in spec X" type queries.
- (Chunk)-[:PARENT_SECTION]->(Chunk)       # ~100k — section hierarchy (child → nearest parent, e.g. 4.2.1 → 4.2).
                                           #   Property `Chunk.is_parent_section` (bool) marks chunks that have children.
                                           #   USE for "show overview of section X" queries via [:PARENT_SECTION*] traversal.

# KG quirks — CRITICAL
- There is NO direct `(Chunk)->Term` relationship. To find chunks that mention a
  term, use `c.spec_id IN t.source_specs` (a list on the Term node) — this is
  the gold path.
- `Term.full_name` (NOT `Term.name` — that property does not exist).
- `Term.primary_spec` is **section-level** (e.g. `'ts_29.500_3.2'` = the section
  where the abbreviation is listed). It is NOT the document where the concept
  is canonically defined. DO NOT use it for `STARTS WITH d.spec_id + '_'` —
  that pattern finds chunks in the same document as the abbreviation listing,
  but the canonical definition is often in a DIFFERENT spec.
- `Term.source_specs` is the list of section-level spec_ids where the term
  appears in an Abbreviations / Definitions section. Use UNWIND or `IN`.
- spec_ids exist in DUPLICATE FORMATS (`ts_29.500` AND `ts_29_500`) — same
  content, different nodes. Live with it; do not try to dedup at query time.
- DO NOT use `chunk.content CONTAINS ...` — duplicates vector search, full
  scan, low precision.

# Performance rules — query MUST run <500ms (197k Chunks, 18k Terms)

1. **Indexed properties** (RANGE index): `Term.abbreviation`, `Document.spec_id`,
   `Chunk.spec_id`, `Chunk.chunk_id`, `Chunk.chunk_type`. Match them with `=`
   (NOT `toLower(...)`). The app has case-normalized the term to upper case.
2. **Never wrap an indexed property in `toLower()` in WHERE** — bypasses the
   index. `Term.abbreviation` is canonical upper case already.
3. **Pipeline with `WITH` between MATCH clauses** to avoid cartesian products.
4. **Carry `full_name` forward** with `WITH t, t.full_name AS full_name LIMIT 1`
   so subsequent CASE / WHERE clauses can reference it without re-fetching.

# Intent → chunk_type mapping (verified empirically against KG)
- definition / what_is / abbreviation / network_function:
    chunk_type IN ['definition', 'abbreviation']
- procedure / how_does:
    chunk_type IN ['procedure', 'requirement']
- general / overview:
    chunk_type IN ['definition', 'general']
NOTE: chunk_type='interface' in source data is mislabeled (4909 chunks
mostly named "Service requirements" / "WEB and FTP services"). DO NOT
filter on chunk_type='interface'. For interface/api/signaling questions
involving an identifier (N1, N6, S1...), use Pattern B below.

# Three retrieval patterns — pick exactly ONE based on the question

## Pattern A — Term-anchored (use when resolved_terms is non-empty)
Single MATCH, OR-filter, CASE for tiered scoring. Recalls chunks both
from the term's defining specs AND from sections whose title matches the
full_name (the canonical definition section, often in a DIFFERENT spec
from `primary_spec`):

  MATCH (t:Term {{abbreviation: '<TERM_UPPER>'}})
  WITH t, t.full_name AS full_name LIMIT 1
  MATCH (c:Chunk)
  WHERE (c.spec_id IN t.source_specs AND c.chunk_type IN [<INTENT_TYPES>])
     OR (full_name IS NOT NULL AND c.section_title CONTAINS full_name)
  WITH c, full_name,
    CASE
      WHEN full_name IS NOT NULL AND c.section_title CONTAINS full_name THEN 1.0
      WHEN c.chunk_type = 'definition'   THEN 0.9
      WHEN c.chunk_type = 'abbreviation' THEN 0.85
      WHEN c.chunk_type = 'procedure'    THEN 0.85
      WHEN c.chunk_type = 'requirement'  THEN 0.8
      ELSE 0.6
    END AS score
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, score
  ORDER BY score DESC LIMIT $top_k

Substitute `<TERM_UPPER>` (e.g. `'SCP'`) and `<INTENT_TYPES>`
(e.g. `['definition','abbreviation']`) with literals from the resolved-terms
and intent blocks.

For multiple resolved terms, use `t.abbreviation IN ['<ABBR1>','<ABBR2>']`.

## Pattern B — Section-identifier word-boundary regex
Use when the question references an interface/reference-point identifier
matching `^[A-Z]\\d+[a-z]*$` (N1, N2, N6, S1, S5, Xn, X2) or known
identifiers (Uu). These are NOT stored as standalone Term abbreviations
in the KG — Term lookup returns 0 rows. Anchor on section_title regex:

  MATCH (c:Chunk)
  WHERE c.section_title =~ '(?i).*\\\\b<IDENT>\\\\b.*'
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Doubled backslashes are required to escape `\\b` inside a Cypher string.
For multiple identifiers, OR them: `c.section_title =~ '(?i).*\\\\b(N6|S1)\\\\b.*'`.

## Pattern C — No structural anchor (sentinel empty)
If neither resolved_terms nor section identifiers fit (purely conceptual
question like "key benefit of network slicing"), emit exactly:

  MATCH (c:Chunk) WHERE false
  RETURN c.chunk_id AS chunk_id, c.content AS content,
         c.spec_id AS spec_id, c.section_title AS section, 0.0 AS score
  LIMIT $top_k

This signals "vector search should carry this query" and avoids polluting
RRF with hallucinated matches. NEVER match on Subject nodes — Subject is
a generic category, not a topic anchor.

# Rules — failure to follow ANY rule will be rejected
1. READ-ONLY only. Allowed keywords: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, SKIP, UNION, UNWIND, CASE, AS, DISTINCT.
2. NEVER use: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, FOREACH, CALL with write side-effects.
3. EXACTLY ONE statement. No semicolons.
4. Return EXACTLY these columns (alias if needed):
     chunk_id, content, spec_id, section, score
   `score` is a float you assign as a relevance heuristic (1.0 = canonical definition, 0.75 = related spec, 0.5 = weak signal).
5. Use the parameter `$top_k` for the LIMIT.
6. Use only the labels and relationship types listed in the schema above. Do not invent any.
7. Keep the query compact (under ~15 lines).
8. FORBIDDEN: `chunk.content CONTAINS ...` clauses (see KG quirks above). Filter on `section_title` or graph structure only.
9. FORBIDDEN: bare `MENTIONS` traversal without a `section_title` filter (see KG quirks above).
10. If the question has no resolvable Term and no section identifier, use Pattern C (sentinel empty `MATCH (c:Chunk) WHERE false`). NEVER invent Subject node names, generic Chunk filters without anchor, or `chunk.content CONTAINS` substring matches.

# Output format
Output ONLY the Cypher query. No prose. No comments. No markdown fences.
"""

    @staticmethod
    def _clean(raw: str) -> str:
        text = raw.strip()
        # Strip fenced blocks if present
        text = FENCE_PATTERN.sub("", text).strip()
        # Some local models prepend a label like "Cypher:" — strip it
        for prefix in ("cypher:", "query:", "answer:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].lstrip()
        # Drop trailing semicolons
        text = text.rstrip(";").strip()
        return text

    @staticmethod
    def _validate(cypher: str) -> None:
        if not cypher:
            raise CypherValidationError("LLM produced empty Cypher")
        # Reject multi-statement
        if ";" in cypher:
            raise CypherValidationError("Multiple Cypher statements are not allowed")
        upper = cypher.upper()
        for kw in FORBIDDEN_KEYWORDS:
            if kw in upper:
                raise CypherValidationError(f"Forbidden keyword/clause in generated Cypher: {kw!r}")
        # Must contain at least one MATCH or UNWIND or RETURN — otherwise it's not a useful read query
        if not any(k in upper for k in ("MATCH", "UNWIND", "RETURN")):
            raise CypherValidationError("Generated Cypher does not contain MATCH/UNWIND/RETURN")
