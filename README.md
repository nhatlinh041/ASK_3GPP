# ASK_3GPP

Hệ thống Hỏi-Đáp (RAG + Knowledge Graph) trên tài liệu đặc tả 3GPP.

Pipeline kết hợp **vector search** (embedding `e5-base-v2`), **graph
search** (Neo4j Cypher do LLM sinh) và **cross-encoder rerank** để trả
lời câu hỏi kỹ thuật về 5G/4G specs với citation chính xác tới
`spec_id` và `section`.

## Kiến trúc

Build-time đi từ trái sang phải; query-time đi ngược lại:

```
DOCX (.docx)  →  JSON                    →  Neo4j KG       →  FastAPI    →  React
document_processing/   3GPP_JSON_DOC/         kg_builder/       rag-engine/    frontend/
                       processed_json_v4
```

- **`document_processing/`** — pipeline 1 file: download → extract DOCX
  → chunk → JSON. Subcommand `download | extract | process | process-local | init-kg | all | status`.
- **`kg_builder/`** — `KGBuilder` load JSON vào Neo4j, `Embedder` sinh
  vector + index. Schema: 4 node (`Document`, `Chunk`, `Term`,
  `Subject`) + 6 edge (`CONTAINS`, `REFERENCES_SPEC`,
  `REFERENCES_CHUNK`, `DEFINED_IN`, `HAS_SUBJECT`, `PARENT_SECTION`).
  Chi tiết: [docs/kg_schema.md](docs/kg_schema.md).
- **`rag-engine/`** — FastAPI port `:8000`, expose:
  - `POST /api/query` — SSE stream qua các stage `intent → retrieval_vector → retrieval_graph → rerank → answer → sources`.
  - `POST /api/cypher` — Cypher tester read-only (chặn write clauses).
- **`frontend/`** (Vite + React, port `:3000`) — chat UI + Cypher tester
  page (`cypher.html`). Lịch sử chat lưu `localStorage`, không có
  server-side session.

### Hai chế độ retrieval (`mode` trong request body `/api/query`)

- `fixed` — vector + LLM-generated Cypher → RRF fusion → cross-encoder rerank.
- `react_agent` — adaptive ReAct loop, planner LLM mỗi vòng chọn tool
  trong `vector | cypher | expand_term | inspect_chunk | finish`.

## External services

Khởi động bằng `scripts/start-deps.sh`:

- **Neo4j** — Docker container `neo4j-server`, ports `7474`/`7687`.
- **Ollama** — phải đang chạy trước; script chỉ kiểm tra reachability tại `OLLAMA_URL`.

`.env` ở repo root là single source of truth cho `NEO4J_*`,
`OLLAMA_URL`, `RAG_ENGINE_URL`.

## Setup

```bash
# Node deps cho root + frontend
npm run install:all

# Python venv (rebuild-kg.sh expect ../.venv hoặc local ./venv)
python -m venv venv
source venv/bin/activate
pip install -r document_processing/requirements.txt fastapi uvicorn python-dotenv \
    sentence-transformers neo4j tqdm requests
```

## Chạy hệ thống

```bash
npm run dev      # predev: start-deps.sh; rồi rag (8000) + frontend (3000) + cloudflared tunnel
npm run stop     # dừng toàn bộ
```

Service riêng:

```bash
npm run dev:rag        # uvicorn FastAPI :8000
npm run dev:frontend   # Vite :3000
npm run dev:tunnel     # cloudflared
```

## Build / rebuild Knowledge Graph

```bash
npm run rebuild-kg                  # full: clean + KG + embeddings
npm run rebuild-kg:kg-only          # graph only
npm run rebuild-kg:embed-only       # chỉ embed lại
JSON_DIR=3GPP_JSON_DOC/processed_json_v4 bash scripts/rebuild-kg.sh full   # custom JSON source
SKIP_RESTART=1 bash scripts/rebuild-kg.sh                                  # giữ zombie label/type (debug)
```

JSON canonical hiện tại: `3GPP_JSON_DOC/processed_json_v4` (1800 file).
Script mặc định trỏ `processed_json_v3` — luôn set `JSON_DIR` khi rebuild.

### Re-chunk DOCX → JSON mới (không re-download)

```bash
python document_processing/download_and_process_3gpp.py process-local \
  --input  document_processing/data/rel18_extracted \
  --output 3GPP_JSON_DOC/processed_json_v4
```

### Backfill Term node mà không rebuild toàn KG

```bash
venv/bin/python -m kg_builder.enrich_terms --json-dir 3GPP_JSON_DOC/processed_json_v4
```

Chỉ động Term + DEFINED_IN, không đụng Chunk/Document/Subject.

## Test

Cấu hình ở [pytest.ini](pytest.ini), `testpaths=tests`.

```bash
npm test                                          # toàn bộ
npm run test:kg                                   # chỉ KG builder
npm run test:embed                                # chỉ embedder
pytest tests/test_kg_builder.py::test_xyz -v      # 1 test
```

Integration test mark `requires_neo4j` (xem [tests/conftest.py](tests/conftest.py))
— tự skip nếu Neo4j unreachable. Fixture dùng synthetic spec id
`ts_99_001` / `ts_99_002`.

## Quy ước project

- **`spec_id` underscore-canonical** ở mọi nơi sau bước chunking
  (`ts_23_501`, `ts_38_508-1`). Nếu thấy dot form (`ts_23.501`) trong
  data mới — coi như bug. Chi tiết:
  [docs/spec_id_format_investigation.md](docs/spec_id_format_investigation.md).
- **Schema enforce qua constraint** ở `kg_builder/builder.py:CYPHER_CONSTRAINTS`.
  Thêm node label / edge type → cập nhật cả `validate()` và rebuild
  script's clean phase.
- **SSE event names là contract** với frontend — không đổi tên
  `intent / retrieval_vector / retrieval_graph / rerank / answer / sources`.
- **Cypher tester read-only by design** (`_WRITE_CLAUSES` trong
  [rag-engine/main.py](rag-engine/main.py)). Đừng nới lỏng — user có Bolt access nếu cần write.
- **Tài liệu giải thích viết tiếng Việt** (`docs/`, comment giải nghĩa).

## KG quirks (verify trên live data)

- **Không có edge `Chunk → Term`.** `HAS_SUBJECT` đi từ Chunk → Subject
  (5 generic category: `name`, `priority`, `description`). Để tìm
  chunk theo term, dùng `c.spec_id IN t.source_specs`.
- **Term có `abbreviation`, `full_name`, `term_type`, `source_specs`,
  `primary_spec`** — KHÔNG có `name`. Sample query mẫu cập nhật ở
  [frontend/src/components/CypherTester.tsx](frontend/src/components/CypherTester.tsx).
- **Interface / reference-point identifier** (N1, N6, S1, Uu, Xn)
  KHÔNG phải standalone Term — chỉ xuất hiện trong compound `full_name`
  hoặc `Chunk.section_title`. Dùng pattern regex `=~ '(?i).*\\bN6\\b.*'`.
- **`Term.primary_spec` là section-level** (`'ts_29.500_3.2'`), không
  phải document-level. Đừng `STARTS WITH t.primary_spec + '_'` — dùng
  `t.source_specs` thay thế.
- **spec_id tồn tại 2 format duplicate** (`ts_29.500` AND `ts_29_500`)
  cho cùng nội dung — node khác nhau. Không cần dedup.
- **Không bao giờ `chunk.content CONTAINS ...`** trong graph search —
  duplicate vector branch, full scan, low precision.

## Thư mục chính

```
ASK_3GPP/
├── 3GPP_JSON_DOC/          # JSON đã chunk (canonical: processed_json_v4)
├── docs/                   # Documentation tiếng Việt
├── document_processing/    # DOCX → JSON pipeline
├── kg_builder/             # JSON → Neo4j
├── rag-engine/             # FastAPI + retrieval pipeline
│   ├── llm/                # Ollama client
│   ├── retrieval/          # vector_search, graph_search, cypher_generator, fusion
│   └── pipeline/           # orchestrator
├── frontend/               # Vite + React UI
├── tests/                  # Unit + integration test (requires_neo4j marker)
├── tele_qna/               # Benchmark TeleQnA
├── scripts/                # start-deps, stop, rebuild-kg
└── CLAUDE.md               # Hướng dẫn cho Claude Code
```

## Tham khảo

- [CLAUDE.md](CLAUDE.md) — hướng dẫn chi tiết về convention + command.
- [docs/kg_schema.md](docs/kg_schema.md) — schema Neo4j.
- [docs/spec_id_format_investigation.md](docs/spec_id_format_investigation.md) — phân tích format `spec_id`.
- [docs/portable_structure.md](docs/portable_structure.md) — hướng dẫn copy project sang máy khác.
