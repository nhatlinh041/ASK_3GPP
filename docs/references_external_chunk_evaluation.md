# Đánh giá: `REFERENCES_EXTERNAL_CHUNK` — chunk-level cross-spec citation

**Bối cảnh:** Schema hiện tại có `REFERENCES_SPEC` (Chunk→Document, doc-level) nhưng JSON `cross_references.external` chứa thêm `ref_id` chỉ rõ section đích. ~47% external refs có `ref_id` đang bị bỏ phí.

**Câu hỏi nghiên cứu:** Có nên tạo edge chunk-to-chunk xuyên spec để giữ thông tin đó không?

**TL;DR:** 🟢 Đáng làm — nhưng phải **chạy feasibility test trước** (đo match rate). Khuyến nghị Option B (reuse `REFERENCES_CHUNK` + property `is_external`) để giữ schema tối thiểu.

---

## Mục lục

- [1. Tình trạng hiện tại](#1-tình-trạng-hiện-tại)
- [2. Giá trị retrieval](#2-giá-trị-retrieval)
- [3. Concerns về data quality](#3-concerns-về-data-quality)
- [4. Filter để giảm nhiễu](#4-filter-để-giảm-nhiễu)
- [5. Design choices](#5-design-choices)
- [6. Implementation complexity](#6-implementation-complexity)
- [7. Rủi ro & side effects](#7-rủi-ro--side-effects)
- [8. Verdict + roadmap](#8-verdict--roadmap)
- [9. Pattern Cypher mẫu](#9-pattern-cypher-mẫu-sau-khi-có-option-b)

---

## 1. Tình trạng hiện tại

### Schema hiện tại link cross-spec ở **doc-level**

```cypher
MATCH (src:Chunk {chunk_id: $source_id})
MATCH (dst:Document {spec_id: $target_spec})
MERGE (src)-[r:REFERENCES_SPEC]->(dst)
SET r.ref_id = $ref_id, r.ref_type = $ref_type, ...
```

→ `ref_id` (vd `"4.2.1"`) được lưu làm **property** của edge nhưng không link đến chunk cụ thể trong target spec.

### Data thô có sẵn

```json
{"target_spec": "ts_23.221", "ref_type": "clause", "ref_id": "4.2.1", "confidence": 0.9}
//                                                  ^^^^^^^^^^^^^^^ section đích cụ thể!
```

Mẫu 50 file: **15,759 external refs**, trong đó **7,356 có `ref_id`** (~47%).

→ Projection toàn KG (1,796 specs): **~80,000–250,000** external refs có ref_id.

---

## 2. Giá trị retrieval

### Use cases unique mà vector search KHÔNG làm được

| Use case | Hiện tại | Sau khi có edge |
|---|---|---|
| *"AMF reg procedure phụ thuộc gì cross-spec?"* | Vector + LLM ghép cứng | Multi-hop traversal trực tiếp |
| *"All chunks citing TS 23.501 clause 6.3.2"* | Grep content (chậm, fuzzy) | `MATCH ()-[:REFERENCES_CHUNK {is_external:true}]->(:Chunk {section_id:'6.3.2', spec_id:'ts_23_501'})` |
| Citation graph (PageRank, hub) | Doc-level (coarse) | Chunk-level (granular) |

### Retrieval impact estimation

| Question type | % benchmark | Lợi từ edge mới |
|---|---:|---|
| Definition (Term-First đã gold) | ~30% | 0% |
| Same-spec procedure (intra REFERENCES_CHUNK đã cover) | ~25% | 0% |
| **Cross-spec procedure** | **~15%** | **+5-15%** |
| **Multi-hop reasoning** | **~10%** | **+3-10%** |
| Comparison, MCQ | ~20% | +1-2% |

→ Tổng kỳ vọng accuracy: **+1-3%** trên benchmark hiện tại (87% → 88-90%).

→ Lớn hơn nếu benchmark có nhiều câu cross-spec.

---

## 3. Concerns về data quality

### Match rate chưa đo

**Internal REFERENCES_CHUNK đã verify**: 40% match rate (3-tier matching trên ts_23_009).

**External có thể thấp hơn** vì:
- Target spec có thể version cũ — `section_id` đã đổi qua các Release
- `ref_id` có thể trỏ figure/table (không phải chunk)
- Format inconsistency giữa specs (e.g. spec dùng "clause 4.2" vs "4.2.1.1")
- Spec_id mismatch: external ref dùng `"ts_23.501"` nhưng Document trong KG là `"ts_23_501"`

### Feasibility test cần chạy TRƯỚC

```python
# Pseudo-code
import json, glob
from collections import defaultdict

# 1. Build section_index per spec
section_index = defaultdict(dict)
for f in glob.glob("3GPP_JSON_DOC/processed_json_v3/*.json"):
    data = json.load(open(f))
    spec_id = data["metadata"]["specification_id"]
    for chunk in data["chunks"]:
        section_index[spec_id][chunk["section_id"]] = chunk["chunk_id"]

# 2. Sample 1000 external refs có ref_id
samples, matched = 0, 0
breakdown = defaultdict(lambda: {"total": 0, "matched": 0})

for f in glob.glob("3GPP_JSON_DOC/processed_json_v3/*.json"):
    data = json.load(open(f))
    for chunk in data["chunks"]:
        for ref in chunk.get("cross_references", {}).get("external", []):
            if not ref.get("ref_id"):
                continue
            samples += 1
            ref_type = ref.get("ref_type", "unknown")
            breakdown[ref_type]["total"] += 1

            target_spec = ref["target_spec"]
            # Try variants of spec_id format
            for variant in [target_spec, target_spec.replace(".", "_")]:
                if variant in section_index:
                    if match_3tier(ref["ref_id"], section_index[variant]):
                        matched += 1
                        breakdown[ref_type]["matched"] += 1
                        break
            if samples >= 1000: break

# 3. Report
print(f"Match rate: {matched}/{samples} = {matched/samples*100:.1f}%")
for rt, stats in breakdown.items():
    print(f"  {rt}: {stats['matched']}/{stats['total']}")
```

**Decision criteria:**
- Match rate **≥ 25%** → đáng implement
- **15–25%** → implement nhưng chỉ lọc `ref_type='clause'`
- **< 15%** → bỏ qua, không xứng công

---

## 4. Filter để giảm nhiễu

Không phải mọi external ref đều đáng tạo edge:

| ref_type | Tạo edge? | Lý do |
|---|---|---|
| `clause`, `section`, `subclause` | ✓ | Semantic mạnh — trỏ đến nội dung cụ thể |
| `figure`, `table` | ✗ | Cấu trúc, ref_id là số figure không phải section_id |
| `reference`, `annex` | ✗ | Meta-ref, không phải nội dung |
| `spec` (no ref_id) | ✗ | Đã có REFERENCES_SPEC cover |
| `confidence < 0.7` | ✗ | Extraction không tin cậy |

→ Sau filter ước lượng còn **~20-30%** external refs với ref_id → **~30k-75k edges** toàn KG.

---

## 5. Design choices

### Option A — Edge type mới `REFERENCES_EXTERNAL_CHUNK`

```cypher
(c1:Chunk)-[:REFERENCES_EXTERNAL_CHUNK]->(c2:Chunk)
```

| ✅ Pros | ❌ Cons |
|---|---|
| Semantic rõ, dễ query | Schema bloat — giờ có 3 edge cross-ref: SPEC, CHUNK, EXTERNAL_CHUNK |
| LLM dễ hiểu trong prompt | Vi phạm nguyên tắc minimal schema |

### Option B — Reuse `REFERENCES_CHUNK` + property `is_external` ⭐ **khuyến nghị**

```cypher
MERGE (c1:Chunk)-[r:REFERENCES_CHUNK]->(c2:Chunk)
SET r.is_external = (c1.spec_id <> c2.spec_id),
    r.ref_type = $ref_type,
    r.ref_id = $ref_id,
    r.confidence = $confidence
```

| ✅ Pros | ❌ Cons |
|---|---|
| Không thêm edge type — schema vẫn 6 edges | Cần property check khi muốn phân biệt |
| Multi-hop pattern thống nhất: `[:REFERENCES_CHUNK*1..3]` traverse cả internal + external | Query phức tạp hơn 1 chút khi muốn filter |
| Filter linh hoạt: `WHERE r.is_external = true/false` | |

### Option C — Property trên REFERENCES_SPEC + kèm chunk-level

Tạo cả 2 song song: REFERENCES_SPEC (cũ, doc-level) + REFERENCES_CHUNK is_external=true (mới)

| ✅ Pros | ❌ Cons |
|---|---|
| Backward compat 100% | Redundant — mọi cross-spec ref có 2 edges |
| | LLM khó biết chọn pattern nào |

### So sánh 3 phương án

| Tiêu chí | A | **B** ⭐ | C |
|---|:---:|:---:|:---:|
| Schema cleanliness | 🟡 | 🟢 | 🔴 |
| Query simplicity | 🟢 | 🟡 | 🔴 |
| Multi-hop pattern unity | 🟡 | 🟢 | 🔴 |
| Backward compat | 🟢 | 🟢 | 🟢 |
| LLM prompt complexity | 🟢 | 🟡 | 🔴 |

→ **Khuyến nghị Option B**.

---

## 6. Implementation complexity

### Code change ước lượng (~30-50 dòng trong [`kg_builder/builder.py`](../kg_builder/builder.py))

```python
# Option B implementation
ALLOWED_TYPES = {'clause', 'section', 'subclause'}

def _create_references_spec_edges(self, chunks: List[dict]) -> None:
    # Bước 1: Build section_index_per_spec (1 lần, cached in-memory)
    section_index_per_spec: Dict[str, Dict[str, str]] = defaultdict(dict)
    for c in chunks:
        section_index_per_spec[c["_spec_id"]][c["section_id"]] = c["chunk_id"]

    with self._driver.session() as s:
        for chunk in tqdm(chunks, desc="[kg] REFERENCES_SPEC + EXT_CHUNK"):
            source_id = chunk["chunk_id"]
            for ref in chunk.get("cross_references", {}).get("external", []):
                target_spec = ref.get("target_spec", "")
                if not target_spec:
                    continue

                # ── Existing REFERENCES_SPEC (giữ nguyên) ────────────
                ref_uid = hashlib.md5(...).hexdigest()[:10]
                s.run("""... REFERENCES_SPEC ...""", ...)

                # ── NEW: REFERENCES_CHUNK external nếu có ref_id ────
                ref_id = ref.get("ref_id", "")
                ref_type = ref.get("ref_type", "")
                conf = ref.get("confidence", 0.0)

                if (ref_id and ref_type in ALLOWED_TYPES and conf >= 0.7
                        and target_spec in section_index_per_spec):
                    target_chunk_id = self._match_ref_to_chunk(
                        ref_id, source_id, section_index_per_spec[target_spec]
                    )
                    if target_chunk_id:
                        s.run("""
                            MATCH (src:Chunk {chunk_id: $sid})
                            MATCH (tgt:Chunk {chunk_id: $tid})
                            MERGE (src)-[r:REFERENCES_CHUNK]->(tgt)
                            SET r.is_external = true,
                                r.ref_type = $rtype,
                                r.ref_id = $rid,
                                r.confidence = $conf
                        """, sid=source_id, tid=target_chunk_id,
                            rtype=ref_type, rid=ref_id, conf=conf)
```

### Build time impact

- Build section_index_per_spec: O(N_chunks) trong RAM, ~1-2s
- Match per ref: O(1) lookup
- Tạo edges: ~30-75k UNWIND batches → +5-10 phút

### Storage impact

- ~30k-75k edges thêm × ~100 bytes/edge ≈ **3-7 MB** thêm trong Neo4j

---

## 7. Rủi ro & side effects

| Rủi ro | Khả năng | Impact | Mitigation |
|---|---|---|---|
| Match sai do section_id collision (e.g. "4.2" trong nhiều spec) | Trung bình | Edge mislead | `target_spec` filter ngăn chặn |
| Hub spam — section "1" của mọi spec bị reference rất nhiều | Cao | PageRank skew | Filter `section_id` không quá generic; `confidence` weight |
| Phá kết quả benchmark hiện tại | Thấp | Regression | Chỉ thêm edges, không xoá; query cũ vẫn chạy |
| LLM bị confused giữa internal vs external | Thấp | Cypher sai | `is_external` property + cập nhật prompt |
| spec_id format mismatch (`ts_23.501` vs `ts_23_501`) | **Cao** | 0 match | Try cả 2 variants khi lookup section_index |
| Build time tăng | Thấp | UX | +5-10 phút chấp nhận được |

---

## 8. Verdict + roadmap

### Đánh giá tổng

| Tiêu chí | Điểm |
|---|---|
| **Unique value** (graph capability vector không có) | ⭐⭐⭐⭐ |
| **Implementation cost** | ⭐⭐⭐ (vừa phải) |
| **Risk** | ⭐⭐ (thấp) |
| **Match rate certainty** | ⭐⭐ (cần test trước) |
| **Schema cleanliness** (Option B) | ⭐⭐⭐⭐ |

### Roadmap khuyến nghị

```
1. ⏸ TRƯỚC — Run feasibility test (sample 1000 external refs có ref_id, đo match rate)
   ├─ Nếu < 15% match → DỪNG, không đáng làm
   ├─ Nếu 15–25% → implement Option B với filter ref_type='clause' only
   └─ Nếu ≥ 25% → implement Option B đầy đủ

2. ⏳ SAU — Implement trong builder.py
   ├─ Pre-build section_index_per_spec
   ├─ Filter ref_type ∈ {clause, section, subclause}, confidence ≥ 0.7
   └─ MERGE REFERENCES_CHUNK với is_external=true

3. ⏳ POST-BUILD — Update prompt schema doc cho LLM
   ├─ adaptive_hop_prompts.py: thêm pattern multi-hop cross-spec
   └─ cypher_generator.py: thêm filter is_external trong gold pattern

4. ⏳ FINAL — Benchmark before/after
   ├─ Chia benchmark theo question type (cross-spec vs same-spec)
   ├─ Measure accuracy gain trên cross-spec subset
   └─ Update thesis nếu gain ≥ 1%
```

---

## 9. Pattern Cypher mẫu (sau khi có Option B)

### 1. Multi-hop cross-spec procedure traversal

```cypher
MATCH (start:Chunk {chunk_id: 'ts_23_502_4.2.1'})
MATCH path = (start)-[:REFERENCES_CHUNK*1..3]->(end:Chunk)
WHERE any(r IN relationships(path) WHERE r.is_external = true)
RETURN end, length(path) AS hops
ORDER BY hops ASC
LIMIT 10
```

### 2. Tìm chunk được trích dẫn nhiều nhất từ specs khác

```cypher
MATCH (src:Chunk)-[r:REFERENCES_CHUNK {is_external: true}]->(tgt:Chunk)
RETURN tgt.spec_id + '/' + tgt.section_id AS target,
       tgt.section_title,
       count(*) AS citations
ORDER BY citations DESC
LIMIT 20
```

### 3. Citation graph quality (high-confidence cross-spec)

```cypher
MATCH (c1:Chunk)-[r:REFERENCES_CHUNK]->(c2:Chunk)
WHERE r.is_external = true
  AND r.ref_type = 'clause'
  AND r.confidence >= 0.8
RETURN c1.spec_id + '/' + c1.section_id AS source,
       c2.spec_id + '/' + c2.section_id AS target,
       r.confidence,
       r.ref_id
ORDER BY r.confidence DESC
LIMIT 50
```

### 4. Hybrid Vector + Graph cross-spec boost

```cypher
// Vector search seed → expand cross-spec via REFERENCES_CHUNK
CALL db.index.vector.queryNodes('chunk_embeddings', 10, $query_vector)
YIELD node AS seed, score
OPTIONAL MATCH (seed)-[r:REFERENCES_CHUNK]->(neighbor:Chunk)
WHERE r.is_external = true AND r.confidence >= 0.7
RETURN seed.chunk_id, score AS vector_score,
       collect(distinct neighbor.chunk_id)[0..3] AS cross_spec_neighbors
ORDER BY vector_score DESC
LIMIT 5
```

---

## 10. Liên quan

- Schema hiện tại: [`kg_schema.md`](kg_schema.md)
- Audit tương thích: [`schema_compatibility_audit.md`](schema_compatibility_audit.md)
- Code build: [`kg_builder/builder.py`](../kg_builder/builder.py) — function `_create_references_spec_edges` (line ~357)
- 3-tier matching: [`kg_builder/builder.py`](../kg_builder/builder.py) — function `_match_ref_to_chunk` (line ~478)

---

## 11. Câu hỏi mở

1. **Xử lý spec_id format mismatch** (`ts_23.501` vs `ts_23_501`) như thế nào? Try cả 2 variants? Hay normalize trước khi MATCH?
2. **Có nên giữ REFERENCES_SPEC** khi đã có REFERENCES_CHUNK external? Doc-level vẫn hữu ích cho query "all specs that cite X" — **nên giữ song song**.
3. **PageRank** chạy trên `REFERENCES_CHUNK` (gồm cả internal + external) hay chỉ external? → Nên có 2 metric riêng.
4. **Loop phòng ngừa**: nếu chunk A cite B, B cite A → variable-length traversal có thể infinite. Cần cap depth (`*1..3`).

---

## 12. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-27 | Đề xuất evaluation | User hỏi về cross-spec chunk-level reference |
| TBD | Run feasibility test | Cần đo match rate trước khi commit |
| TBD | Implement Option B (nếu match ≥ 25%) | Schema minimalism + multi-hop unity |
| TBD | Update LLM prompts | Để tận dụng edge mới |
| TBD | Benchmark before/after | Verify gain |
