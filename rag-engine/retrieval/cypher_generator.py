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

# First Cypher head keyword in the LLM output — used to skip prose prefixes
# (e.g. "Answer: B\nJustification: ...") that some reasoning models leak in
# front of the actual query when the question contains MCQ choices.
CYPHER_HEAD_PATTERN = re.compile(r"\b(MATCH|OPTIONAL\s+MATCH|UNWIND|WITH)\b", re.IGNORECASE)

# Detects multiple-choice tail "\nA. ...", "\nA) ...", "\nB. ...", etc. used by
# TeleQnA-style benchmarks. We strip from the first such marker to keep only
# the question stem so the Cypher generator never sees "Begin your reply with
# Answer: <letter>" instructions meant for a different stage.
MCQ_PATTERN = re.compile(r"\n\s*[A-E][.)]\s")


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
        vector_hints: Optional[list[dict]] = None,
        model: Optional[str] = None,
    ) -> CypherGeneration:
        prompt = self._build_prompt(
            question=question,
            schema_text=schema_text,
            intent=intent,
            resolved_terms=resolved_terms,
            primary_term=primary_term,
            vector_hints=vector_hints,
        )
        # Per-call override so the orchestrator can route every LLM hit through
        # the same model the user picked (avoids Ollama load/unload thrashing
        # when the answer-stage model differs from the construction-time one).
        chosen_model = model or self._model
        raw = self._llm.generate(prompt, model=chosen_model)
        cypher = self._clean(raw)
        self._validate(cypher)
        return CypherGeneration(cypher=cypher, raw_response=raw, model=chosen_model, prompt=prompt)

    def generate_stream(
        self,
        question: str,
        schema_text: str,
        intent: Optional[str] = None,
        resolved_terms: Optional[dict] = None,
        primary_term: Optional[str] = None,
        think: bool = True,
        vector_hints: Optional[list[dict]] = None,
        model: Optional[str] = None,
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
            vector_hints=vector_hints,
        )
        # Per-call override so the orchestrator can keep both Cypher and answer
        # stages on the same model (no Ollama load/unload thrash). Falls back to
        # the construction-time default for callers that don't specify.
        chosen_model = model or self._model
        yield {"kind": "prompt", "prompt": prompt, "model": chosen_model}

        raw_parts: list[str] = []
        thinking_parts: list[str] = []
        try:
            # Reasoning models think first, then emit the Cypher. We forward both
            # phases so the UI can show progress during the (often long) thinking phase.
            # When think=False, the model jumps straight to the Cypher output.
            for ev in self._llm.generate_stream_full(prompt, model=chosen_model, think=think):
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
            yield {"kind": "error", "error": f"validation: {e}", "raw": raw, "model": chosen_model}
            return
        yield {"kind": "done", "cypher": cypher, "raw": raw, "model": chosen_model}

    # ----- internal -----

    @staticmethod
    def _strip_mcq(query: str) -> str:
        """Remove multiple-choice tail (A./B./C./...) and any trailing meta-instruction
        like 'Begin your reply with Answer: <letter>' from the question text. Returns
        the question stem only — keeps the Cypher generator from copying answer-stage
        instructions into its output.
        """
        if not query:
            return ""
        m = MCQ_PATTERN.search(query)
        return query[: m.start()].strip() if m else query.strip()

    @staticmethod
    def _build_prompt(
        question: str,
        schema_text: str,
        intent: Optional[str],
        resolved_terms: Optional[dict],
        primary_term: Optional[str],
        vector_hints: Optional[list[dict]] = None,
    ) -> str:
        # ---- DYNAMIC block (per-query — placed at end so STABLE prefix can be cached) ----
        question_stem = LLMCypherGenerator._strip_mcq(question)

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
        else:
            terms_block = "Resolved terms: (none) — likely Pattern C unless an interface identifier matches Pattern B."

        intent_block = f"Intent classification: {intent}" if intent else "Intent classification: (unknown)"

        # Vector hints: top section_titles vector retrieval found. Title + spec_id only,
        # no content — keeps Cypher generator independent of vector scoring while still
        # giving it real anchors instead of forcing it to invent section names.
        if vector_hints:
            hint_lines = []
            for h in vector_hints[:5]:
                section = (h.get("section") or h.get("section_title") or "?").strip()
                spec = (h.get("spec_id") or "?").strip()
                if section and section != "?":
                    hint_lines.append(f"  - \"{section}\"  ({spec})")
            vector_hints_block = (
                "Vector hints (top section_titles vector found — use as anchors when relevant; do NOT invent):\n"
                + "\n".join(hint_lines)
            ) if hint_lines else "Vector hints: (none useful)"
        else:
            vector_hints_block = "Vector hints: (not provided)"

        # ---- STABLE block (role + schema + rules + examples — cacheable across queries) ----
        return f"""You write a single Cypher query for a Neo4j knowledge graph of 3GPP technical specifications.

# Purpose
The graph branch BACKS UP a parallel vector branch via Reciprocal Rank Fusion.
Your job is to retrieve Chunk nodes anchored on canonical KG structure (Term
abbreviations, section_title regexes). Do NOT try to answer the user's
question. Do NOT include prose, explanations, or letter answers.

# Real KG schema (verified against live database)

Node properties:
- Term:           abbreviation, full_name, primary_spec, source_specs, term_type
- Chunk:          chunk_id, spec_id, section_id, section_title, content,
                  chunk_type, subject, key_terms, is_parent_section
- Document:       spec_id, version, title, total_chunks
- Subject:        name, priority, description (generic categories — NOT term-specific)

Relationships (verified counts after rebuild):
- (Chunk)-[:HAS_SUBJECT]->(Subject)        # ~197k — Subject is a GENERIC category, NOT a Term. DO NOT use to find chunks for a term.
- (Chunk)-[:REFERENCES_SPEC]->(Document)   # ~165k — chunk-to-doc link (use for "any chunk citing TS X" queries)
- (Chunk)-[:REFERENCES_CHUNK]->(Chunk)     # internal section refs + cross-spec section refs.
                                           #   Edge property `is_external`: false=same-spec, true=cross-spec.
                                           #   Other properties: ref_type ('clause'), ref_id, confidence.
- (Document)-[:CONTAINS]->(Chunk)          # ~195k — every Chunk has incoming CONTAINS from its Document.
- (Term)-[:DEFINED_IN]->(Document)         # ~5k — Term defined in Document. NOTE: gold pattern for "chunks mentioning a term"
                                           #   is still `c.spec_id IN t.source_specs` (faster than traversal).
- (Chunk)-[:PARENT_SECTION]->(Chunk)       # ~100k — section hierarchy (child → nearest parent).

# KG quirks — CRITICAL
- There is NO direct `(Chunk)->Term` relationship. To find chunks that mention a
  term, use `c.spec_id IN t.source_specs` (a list on the Term node).
- `Term.full_name` (NOT `Term.name` — that property does not exist).
- `Term.primary_spec` is **section-level** (e.g. `'ts_29.500_3.2'`). Do NOT use
  `STARTS WITH d.spec_id + '_'` against it.
- `Term.source_specs` is the list of section-level spec_ids where the term
  appears. Use `IN` against `c.spec_id`.
- spec_ids exist in DUPLICATE FORMATS (`ts_29.500` AND `ts_29_500`) — same
  content, different nodes. Live with it; do not try to dedup at query time.
- DO NOT use `chunk.content CONTAINS ...` — duplicates vector search, full
  scan, low precision.
- 3GPP Release / version (Rel-17, Rel-18, R18) is NOT reliably populated on
  `Document.version` — DO NOT filter on `d.version` or other version fields.
  If the user asks about a release, ignore the release filter and rely on
  spec_id structure.
- Property access on anonymous node literals is INVALID Cypher
  (`(:Document {{spec_id: 'x'}}).version` parses as a syntax error). Always
  bind a variable: `MATCH (d:Document {{spec_id: 'x'}}) RETURN d.version`.

# Performance rules — query MUST run <500ms (197k Chunks, 18k Terms)

1. **Indexed properties** (RANGE index): `Term.abbreviation`, `Document.spec_id`,
   `Chunk.spec_id`, `Chunk.chunk_id`, `Chunk.chunk_type`. Match them with `=`
   (NOT `toLower(...)`).
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
NOTE: chunk_type='interface' in source data is mislabeled — DO NOT filter on it.
For interface/api/signaling questions involving an identifier (N1, N6, S1...),
use Pattern B below.

# Three retrieval patterns — pick exactly ONE based on the question

## Pattern A1 — Single term anchor (use when resolved_terms has exactly one entry)

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

## Pattern A2 — Multi-term anchor (use when resolved_terms has ≥2 entries)
NEVER write `MATCH (t1) OR (t2)` — that is invalid Cypher. Collect terms in
ONE pass, then traverse Chunks once. Carry `names` and `all_specs` forward.

  MATCH (t:Term) WHERE t.abbreviation IN ['<ABBR1>', '<ABBR2>']
  WITH collect(t.full_name) AS names,
       reduce(acc = [], s IN collect(t.source_specs) | acc + s) AS all_specs
  MATCH (c:Chunk)
  WHERE c.spec_id IN all_specs
     OR ANY(n IN names WHERE n IS NOT NULL AND c.section_title CONTAINS n)
  WITH c, names,
    CASE
      WHEN ANY(n IN names WHERE n IS NOT NULL AND c.section_title CONTAINS n) THEN 1.0
      WHEN c.chunk_type = 'definition'   THEN 0.9
      WHEN c.chunk_type = 'abbreviation' THEN 0.85
      ELSE 0.6
    END AS score
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, score
  ORDER BY score DESC LIMIT $top_k

## Pattern B — Section-identifier word-boundary regex
Use when the question references an interface/reference-point identifier
matching `^[A-Z]\\d+[a-z]*$` (N1, N2, N6, S1, S5, Xn, X2) or known
identifiers (Uu). These are NOT stored as standalone Term abbreviations.

  MATCH (c:Chunk)
  WHERE c.section_title =~ '(?i).*\\\\b<IDENT>\\\\b.*'
  RETURN c.chunk_id AS chunk_id, c.content AS content, c.spec_id AS spec_id,
         c.section_title AS section, 1.0 AS score
  ORDER BY score DESC LIMIT $top_k

Doubled backslashes are required to escape `\\b` inside a Cypher string.
For multiple identifiers, OR them: `c.section_title =~ '(?i).*\\\\b(N6|S1)\\\\b.*'`.

## Pattern C — No structural anchor (sentinel empty)
If neither resolved_terms nor section identifiers fit (purely conceptual
question like "key benefit of network slicing"), emit EXACTLY:

  MATCH (c:Chunk) WHERE false
  RETURN c.chunk_id AS chunk_id, c.content AS content,
         c.spec_id AS spec_id, c.section_title AS section, 0.0 AS score
  LIMIT $top_k

This signals "vector search should carry this query" and avoids polluting
RRF with hallucinated matches. NEVER match on Subject nodes.

# Rules — failure to follow ANY rule will be rejected
1. READ-ONLY only. Allowed keywords: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, SKIP, UNION, UNWIND, CASE, AS, DISTINCT.
2. NEVER use: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, FOREACH, CALL with write side-effects.
3. EXACTLY ONE statement. No semicolons.
4. Return EXACTLY these columns (alias if needed):
     chunk_id, content, spec_id, section, score
5. Use the parameter `$top_k` for the LIMIT.
6. Use only the labels and relationship types listed in the schema above.
7. Keep the query compact (under ~15 lines).
8. FORBIDDEN: `chunk.content CONTAINS ...`. Filter on `section_title` only.
9. FORBIDDEN: filtering on `Document.version` / Release / `d.version`.
10. The query MUST end with `RETURN ...`. Never end with bare `WITH`.
11. If the question has no resolvable Term and no section identifier, use Pattern C.

# Output format — STRICT
Your output MUST start with one of these tokens:
  MATCH | OPTIONAL MATCH | WITH | UNWIND
The very first character cannot be a letter A–E, the word "Answer", "Cypher",
"Query", or any prose. The user's question may include "Begin your reply with
`Answer: <letter>`" or A./B./C./D./E. choices — IGNORE all of that. You are
NOT answering the question. You are writing a Neo4j Cypher query that
retrieves chunks helpful for someone else to answer it. Do NOT include any
text outside the Cypher query. No prose. No comments. No markdown fences.

# ----- DYNAMIC context (per-query) -----
{intent_block}

{terms_block}

{vector_hints_block}

# User question (stem only — choices and meta-instructions stripped)
{question_stem}
"""

    @staticmethod
    def _clean(raw: str) -> str:
        text = (raw or "").strip()
        # Strip fenced blocks if present
        text = FENCE_PATTERN.sub("", text).strip()
        # Skip everything before the first Cypher head keyword. Catches prose
        # leaks like "Answer: B\nJustification: ...\nMATCH (n) RETURN n" —
        # which the previous prefix-stripper missed once the leak grew past
        # a single label word.
        m = CYPHER_HEAD_PATTERN.search(text)
        if m:
            text = text[m.start():]
        # Drop trailing semicolons
        return text.rstrip(";").strip()

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
        # Must end with a RETURN clause (possibly followed by ORDER BY / LIMIT / SKIP).
        # A query ending on bare WITH is a Neo4j syntax error ("Query cannot conclude with WITH").
        if "RETURN" not in upper:
            raise CypherValidationError("Generated Cypher must contain a RETURN clause")
        # Find the last RETURN; only ORDER BY / LIMIT / SKIP / DESC / ASC / commas / identifiers / params allowed after.
        last_return_idx = upper.rfind("RETURN")
        last_with_idx = upper.rfind("WITH")
        if last_with_idx > last_return_idx:
            raise CypherValidationError("Generated Cypher cannot end with WITH — must end with RETURN ...")
