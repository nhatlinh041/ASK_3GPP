# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Polyglot monorepo for a 3GPP specification QA system. Documentation is in Vietnamese (`docs/`).

**Data flow** (left to right is build-time; reverse is query-time):

```
DOCX (.docx)  →  JSON              →  Neo4j KG          →  FastAPI  →  React
document_processing/  3GPP_JSON_DOC/   kg_builder/          rag-engine/  frontend/
                      processed_json_v4
```

The current canonical JSON directory is `processed_json_v4` (1800 files). `scripts/rebuild-kg.sh` still defaults to `processed_json_v3`, so always set `JSON_DIR=3GPP_JSON_DOC/processed_json_v4` when rebuilding (or pass it explicitly).

1. **`document_processing/download_and_process_3gpp.py`** — single-file pipeline (download → extract → chunk → JSON). Subcommands: `download | extract | process | process-local | init-kg | all | status`. The `process-local --input <dir> [--output <dir>]` form skips download and re-chunks already-extracted DOCX files. **`spec_id` format is canonical underscore** (e.g. `ts_23_501`, `ts_38_508-1`); cross-references emit underscored `target_spec`. Background and recent fixes: [`docs/spec_id_format_investigation.md`](docs/spec_id_format_investigation.md).

2. **`kg_builder/`** — `KGBuilder` (`builder.py`) loads JSON dirs into Neo4j; `Embedder` (`embedder.py`) creates `e5-base-v2` vectors + vector index. Schema is 4 nodes (`Document`, `Chunk`, `Term`, `Subject`) + 6 edges (`CONTAINS`, `REFERENCES_SPEC`, `REFERENCES_CHUNK`, `DEFINED_IN`, `HAS_SUBJECT`, `PARENT_SECTION`); see [`docs/kg_schema.md`](docs/kg_schema.md). All `Document.spec_id` and `Chunk.spec_id` are underscore form — never mix dot form when MATCHing.

3. **`rag-engine/`** — FastAPI on `:8000` exposing `POST /api/query` (SSE stream of stages: `intent → retrieval_vector → retrieval_graph → rerank → answer → sources`) and `POST /api/cypher` (read-only Cypher tester, blocks write clauses). Two retrieval modes via request body `mode`:
   - `fixed`: vector + LLM-generated Cypher → RRF fusion → cross-encoder rerank
   - `react_agent`: adaptive ReAct loop, planner LLM picks tool each iteration (`vector | cypher | expand_term | inspect_chunk | finish`)
   Models loaded once at startup in `models.py`. LLM calls go through `llm/ollama_client.py` to local Ollama.

   **Cypher generator picks one of 3 patterns** ([`rag-engine/retrieval/cypher_generator.py`](rag-engine/retrieval/cypher_generator.py) `_build_prompt`):
   - **Pattern A (term-anchored)**: `MATCH (t:Term {abbreviation: 'X'}) ... c.spec_id IN t.source_specs OR c.section_title CONTAINS full_name`. Use when `resolved_terms` is non-empty.
   - **Pattern B (section-identifier regex)**: `c.section_title =~ '(?i).*\\bN6\\b.*'` for interface/reference-point identifiers (N1, N6, S1, Uu, Xn). These are NOT stored as standalone Term abbreviations.
   - **Pattern C (sentinel empty)**: `MATCH (c:Chunk) WHERE false RETURN ... LIMIT $top_k` when neither anchor fits. Lets vector branch carry the query without polluting RRF.
   The intent→chunk_type mapping intentionally omits `chunk_type='interface'` because that label in source data is mislabeled (4909 chunks named "Service requirements" / "WEB and FTP services", not actual interfaces).

4. **`frontend/` (Vite + React, port 3000)** — chat UI + Cypher tester page (`cypher.html`). Gọi thẳng FastAPI `/api/query`, `/api/cypher`, `/api/cypher/schema` qua Vite dev proxy (`proxy: { '/api': 'http://localhost:8000' }` trong `frontend/vite.config.ts`). Lịch sử chat lưu ở `localStorage` theo key `chat-history-${sessionId}` — không có server-side session store.

**External services** (started by `scripts/start-deps.sh`):
- Neo4j in Docker container `neo4j-server` (ports 7474/7687).
- Ollama at `OLLAMA_URL` (must already be running externally; script only checks reachability).

## Common commands

Run from repo root unless noted.

**Setup**:
```bash
npm run install:all       # installs root + frontend node_modules
# Python deps go into a venv — rebuild-kg.sh expects ../.venv but a local ./venv also works
# if VIRTUAL_ENV is already activated:
source venv/bin/activate
pip install -r document_processing/requirements.txt fastapi uvicorn python-dotenv \
    sentence-transformers neo4j tqdm requests
```

**Run all services** (kills all on any failure, color-coded logs):
```bash
npm run dev               # → predev: start-deps.sh; then rag (8000), frontend (3000), cloudflared tunnel
npm run stop              # tear down via scripts/stop.sh
```

Individual services: `npm run dev:rag | dev:frontend | dev:tunnel`.

**Rebuild Knowledge Graph** (always invoked after re-chunking):
```bash
npm run rebuild-kg                    # full: clean + KG + embeddings
npm run rebuild-kg:kg-only            # graph only (skip embeddings)
npm run rebuild-kg:embed-only         # just (re)embed existing chunks
JSON_DIR=/path/to/processed_json_v4 bash scripts/rebuild-kg.sh full   # custom JSON source
SKIP_RESTART=1 bash scripts/rebuild-kg.sh                              # keep zombie label/type tokens (debug)
```
The script does an opinionated `clean_all` then **restarts the Neo4j container** to flush zombie label/type metadata before building. Skipping that step (`SKIP_RESTART=1`) is rarely correct.

**Re-chunk DOCX → fresh JSON** (no re-download; uses already-extracted DOCX):
```bash
python document_processing/download_and_process_3gpp.py process-local \
  --input  document_processing/data/rel18_extracted \
  --output 3GPP_JSON_DOC/processed_json_v4
```
`_process_single` swallows exceptions silently — if "0 success / N failed", import the parser and call `process_document()` directly with traceback enabled to see the real error.

**Backfill missing Term nodes without rebuilding the whole KG**:
```bash
venv/bin/python -m kg_builder.enrich_terms --json-dir 3GPP_JSON_DOC/processed_json_v4
```
Reads JSON, runs the same `TermExtractor` + `_merge_terms` as the build pipeline, then `MERGE`s Term nodes + DEFINED_IN edges. Unlike `KGBuilder._create_terms` it surfaces write errors (the original `except Exception: pass` at builder.py:691-692 is why AMF/SMF/NRF were silently absent in earlier builds). Touches only Term/DEFINED_IN — Chunk/Document/Subject/REFERENCES_*/HAS_SUBJECT/PARENT_SECTION are untouched.

**Tests** (pytest config in [`pytest.ini`](pytest.ini), `testpaths=tests`):
```bash
npm test                                          # all
npm run test:kg                                   # KG builder only
npm run test:embed                                # embedder only
pytest tests/test_kg_builder.py::test_xyz -v      # single test
```
Integration tests use the `requires_neo4j` marker from [`tests/conftest.py`](tests/conftest.py) — they auto-skip when Neo4j is unreachable. Test fixtures use synthetic spec ids `ts_99_001` / `ts_99_002`.

## Project conventions worth knowing

- **Vietnamese is the documentation language** in `docs/` and most code comments. Keep that style when adding to existing files; the user prefers Vietnamese for explanatory docs.
- **`spec_id` is underscore-canonical everywhere downstream of chunking**. If you see dot form (`ts_23.501`) outside legacy data, treat it as a bug — see [`docs/spec_id_format_investigation.md`](docs/spec_id_format_investigation.md) for the full root-cause analysis and the three regex/format fixes already applied to `download_and_process_3gpp.py`.
- **Neo4j schema is enforced via constraints** in `kg_builder/builder.py:CYPHER_CONSTRAINTS`. Adding a new node label or relationship type means updating the schema list, the `validate()` checks, and the rebuild script's clean phase — otherwise `clean_all` will leave zombie tokens.
- **`rag-engine` SSE event names are part of the contract** with the frontend trail UI. Don't rename `intent / retrieval_vector / retrieval_graph / rerank / answer / sources` without updating the frontend.
- **The Cypher tester endpoint is read-only by design** (`_WRITE_CLAUSES` regex in `rag-engine/main.py`). Don't loosen it; users have direct Bolt access if they need writes.
- **`.env` at repo root** is the single source of truth for `NEO4J_*`, `OLLAMA_URL`, `RAG_ENGINE_URL`. `rag-engine/main.py` loads it explicitly before importing the pipeline so module-level `os.getenv` calls see the values.

## KG quirks worth knowing (verified against live data)

- **No `Chunk → Term` edge exists.** `HAS_SUBJECT` goes `Chunk → Subject` (5 generic categories: `name`, `priority`, `description`), NOT to Term. To find chunks for a term, use `c.spec_id IN t.source_specs` (property lookup) — this is the gold path. For chunks mentioning a term in their key list, `'AMF' IN c.key_terms` works too.
- **Term has `abbreviation`, `full_name`, `term_type`, `source_specs`, `primary_spec` — NOT `name`.** Many older sample queries used `t.name`; that property doesn't exist and silently returns 0 rows. Up-to-date sample set lives in [`frontend/src/components/CypherTester.tsx`](frontend/src/components/CypherTester.tsx).
- **Interface / reference-point identifiers (N1, N6, S1, Uu, Xn) are NOT standalone Term abbreviations.** They only appear inside compound full_names ("N6 PDU session") or as substrings in `Chunk.section_title` ("N6 Reference Point"). Use Cypher Pattern B for them.
- **`Term.primary_spec` is section-level**, not document-level (e.g. `'ts_29.500_3.2'`). Don't `STARTS WITH t.primary_spec + '_'` — the canonical definition section is often in a different spec entirely. Use `t.source_specs` (list of section-level spec_ids) for "chunks defining this term" queries.
- **spec_ids exist in DUPLICATE FORMATS** (`ts_29.500` AND `ts_29_500`) for the same content — different nodes. Live with it; do not try to dedup at query time.
- **Never `chunk.content CONTAINS ...` in graph search.** Duplicates the vector branch, full table scan, low precision. Filter on `section_title` or graph structure only.
