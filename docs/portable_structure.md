# Demo folder — cấu trúc portable

Tài liệu mô tả layout của `demo/` sau khi copy `document_processing/` + data.
Mục tiêu: demo/ self-contained, copy đến đâu chạy đó (chỉ cần cài deps + Neo4j).

## Layout sau copy

```
demo/
├── document_processing/             ← code + data chunking (NEW)
│   ├── download_and_process_3gpp.py # main script (chưa fix bug, giữ nguyên)
│   ├── document_processing_v2.ipynb # notebook tham khảo
│   ├── requirements.txt
│   ├── 23501-i90.docx               # test file
│   ├── 29500-i80.docx               # test file
│   ├── ts_29.500.json               # sample output cũ
│   └── data/                        # 11GB DOCX raw + extracted (NEW)
│       ├── rel18_download/          # 2.9GB ZIP files Rel-18
│       ├── rel18_extracted/         # 2.3GB DOCX Rel-18
│       ├── rel15_download/          # ZIP Rel-15 (16K)
│       ├── rel15_extracted/         # DOCX Rel-15 (4K)
│       ├── 23_series/               # 742M raw Rel-15 23.x
│       ├── 29_series/               # 217M raw Rel-15 29.x
│       ├── extracted/               # 3.1G all extracted
│       ├── extracted_doc/           # 1G .doc files
│       ├── converted_docx/          # 373M converted
│       ├── processed_json/          # 72M legacy JSON
│       └── single_specs/            # 2M individual specs
│
├── 3GPP_JSON_DOC/                   ← processed JSON (NEW)
│   └── processed_json_v3/           # 795M, 1796 files
│
├── kg_builder/                      ← Neo4j build pipeline
│   ├── builder.py                   # 4 nodes + 6 edges + PARENT_SECTION
│   └── embedder.py                  # e5-base-v2 vectors
│
├── rag-engine/                      ← FastAPI RAG service
│   ├── main.py
│   ├── pipeline/
│   ├── retrieval/
│   └── llm/
│
├── frontend/                        ← React + Vite SPA (gọi thẳng FastAPI qua Vite proxy)
├── scripts/
│   ├── rebuild-kg.sh                # auto-detects local vs parent JSON
│   ├── start-deps.sh
│   └── stop.sh
├── tests/
├── docs/                            ← documentation (Vietnamese)
├── evaluation/
├── .env, .env.example
├── package.json, pytest.ini
└── cloudflared-demo.yml
```

## Tổng kích thước

| Thành phần | Size |
|---|---:|
| Code (non-binary) | ~50 MB |
| `document_processing/data/` | ~11 GB |
| `3GPP_JSON_DOC/processed_json_v3/` | ~795 MB |
| `node_modules/` (sau npm install) | ~500 MB |
| Neo4j database (sau rebuild) | ~5 GB |
| **Tổng (chưa Neo4j)** | **~12.5 GB** |
| **Tổng (đầy đủ)** | **~17 GB** |

## Path resolution logic

### `rebuild-kg.sh` auto-detect JSON dir

Script tự động detect 2 location:

```bash
if [ -d "$DEMO_DIR/3GPP_JSON_DOC/processed_json_v3" ]; then
  JSON_DIR_DEFAULT="$DEMO_DIR/3GPP_JSON_DOC/processed_json_v3"  # Portable mode
else
  JSON_DIR_DEFAULT="$DEMO_DIR/../3GPP_JSON_DOC/processed_json_v3"  # Legacy mode
fi
```

→ Ưu tiên `demo/3GPP_JSON_DOC` nếu có, fallback `../3GPP_JSON_DOC` (link cũ).

### `download_and_process_3gpp.py` paths

Script này dùng:
```python
SCRIPT_DIR = Path(__file__).parent              # demo/document_processing/
PROJECT_ROOT = SCRIPT_DIR.parent                # demo/
self.download_dir = SCRIPT_DIR / "data" / f"rel{release}_download"
self.data_dir = SCRIPT_DIR / "data" / f"rel{release}_extracted"
self.output_dir = PROJECT_ROOT / "3GPP_JSON_DOC" / "processed_json_v3"
```

→ Khi copy vào `demo/document_processing/`, các path tự động resolve về:
- `demo/document_processing/data/rel18_extracted/` ✓
- `demo/3GPP_JSON_DOC/processed_json_v3/` ✓

→ **Logic giữ nguyên, không cần sửa.**

## Cách dùng (post-copy)

### Chạy chunking từ demo/

```bash
cd demo/document_processing

# Process tất cả docs từ data/rel18_extracted/ → output ../3GPP_JSON_DOC/processed_json_v3/
python download_and_process_3gpp.py process

# Hoặc process từ thư mục riêng → output v4
python download_and_process_3gpp.py process-local \
  --input data/rel18_extracted \
  --output ../3GPP_JSON_DOC/processed_json_v4
```

### Rebuild KG từ demo/

```bash
cd demo
bash scripts/rebuild-kg.sh full     # auto-detect demo/3GPP_JSON_DOC/
```

## Move sang folder mới (cách triển khai)

Demo đã self-contained, copy nguyên xi:

```bash
# Option 1 — copy đến vị trí mới
cp -r /home/linguyen/3GPP/demo /path/to/new_location

# Option 2 — tar archive (transfer giữa machines)
tar czf demo.tar.gz -C /home/linguyen/3GPP demo
scp demo.tar.gz user@remote:/path/
ssh user@remote "tar xzf /path/demo.tar.gz -C /path/"
```

Sau khi move:

```bash
cd /path/to/new_location/demo

# 1. Cài deps
cd frontend && npm install && cd ..
python -m venv .venv
source .venv/bin/activate
pip install -r document_processing/requirements.txt

# 2. Chỉnh .env
cp .env.example .env
# Edit NEO4J_URI, OLLAMA_URL nếu khác

# 3. Khởi động Neo4j (Docker)
bash scripts/start-deps.sh

# 4. Rebuild KG (~30 phút)
bash scripts/rebuild-kg.sh full
```

## Trạng thái 3 bug cần fix (CHƯA APPLY)

Bug đã được phát hiện trong [`document_processing/download_and_process_3gpp.py`](../document_processing/download_and_process_3gpp.py):

| Line | Bug | Status |
|---|---|---|
| 557 | `f"ts_{series}.{spec_num}"` (DOT) khi fallback no-part | ⏳ Chưa fix |
| 676 | `target_spec` không normalize từ regex | ⏳ Chưa fix |
| 693 | Standalone `target_spec` không normalize | ⏳ Chưa fix |

Chi tiết: [`spec_id_format_investigation.md`](spec_id_format_investigation.md).

→ **Khi fix, áp dụng đồng thời ở demo copy + parent copy** (hoặc chỉ demo nếu parent đã deprecated).

## Files giữ ở parent folder (không động)

```
/home/linguyen/3GPP/
├── document_processing/        ← bản gốc (chưa fix), giữ làm reference
├── 3GPP_JSON_DOC/              ← bản gốc, giữ
├── kg_initializer.py           ← parent KG builder (orchestrator dùng)
├── orchestrator.py
├── KG_builder.ipynb
└── ...
```

→ Parent folder vẫn chạy bình thường nếu user muốn dùng pipeline cũ qua `python orchestrator.py init-kg`.

## Checklist sau copy

- [ ] `demo/document_processing/data/` có 11GB
- [ ] `demo/3GPP_JSON_DOC/processed_json_v3/` có 1,796 files
- [ ] Test chunking: `cd demo/document_processing && python download_and_process_3gpp.py status`
- [ ] Test rebuild: `cd demo && bash scripts/rebuild-kg.sh kg-only`
- [ ] Verify path resolution: script log phải show `JSON dir: <DEMO_DIR>/3GPP_JSON_DOC/...`

## Liên quan

- Schema: [`kg_schema.md`](kg_schema.md)
- Bug fix detail: [`spec_id_format_investigation.md`](spec_id_format_investigation.md)
- Schema audit: [`schema_compatibility_audit.md`](schema_compatibility_audit.md)
- Cross-spec chunk-level ref: [`references_external_chunk_evaluation.md`](references_external_chunk_evaluation.md)
