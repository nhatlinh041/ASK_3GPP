# Schema Compatibility Audit — KG mới vs rag-engine hiện tại

**Mục đích:** Verify schema mới (4 nodes + 6 edges, gồm `PARENT_SECTION`) trong [`kg_builder/builder.py`](../kg_builder/builder.py) **không phá** code retrieval đang chạy.

**Phương pháp:** Grep toàn bộ Cypher pattern trong `rag-engine/` đối chiếu schema mới.

**Kết luận nhanh:**
- ✅ **Pipeline chính (term-first, adaptive-hop, cypher-generator) tương thích** vì đã dùng pattern parent (`Term.abbreviation`, `Term.source_specs`, `c.spec_id IN t.source_specs`).
- 🔴 **`multihop_search.py` BROKEN** với schema mới (dùng `Term.name`, `DEFINED_IN→Chunk` — schema cũ của project khác). May là **không file nào import nó**.
- 🟡 **Prompts cho LLM (cypher_generator.py, adaptive_hop_prompts.py)** chứa thông tin **lỗi thời** về schema (count cũ, "DO NOT use" sai). Cần update để LLM không bị mislead.

---

## 1. Audit chi tiết theo file

### ✅ `pipeline/term_first.py` — TƯƠNG THÍCH

[term_first.py:158-160](../rag-engine/pipeline/term_first.py#L158-L160):
```cypher
MATCH (t:Term { <prop>: $val })
OPTIONAL MATCH (t)-[:DEFINED_IN]->(d:Document)
RETURN coalesce(t.full_name, t.fullName, t.definition, t.name) AS full_name, ...
```

| Element | Schema mới | Status |
|---|---|---|
| `Term.<prop>` | tries `abbreviation` first | ✓ khớp |
| `(t)-[:DEFINED_IN]->(d:Document)` | edge tồn tại trong schema mới | ✓ |
| `coalesce(... t.name)` | fallback có dùng `t.name` | ⚠️ harmless — chỉ fallback, không cần thiết nhưng không gây lỗi |

**Verdict:** Hoạt động bình thường.

---

### ✅ `retrieval/adaptive_hop.py` — TƯƠNG THÍCH

Các Cypher quan trọng:

[adaptive_hop.py:534-538](../rag-engine/retrieval/adaptive_hop.py#L534-L538) — match Term theo abbreviation:
```cypher
MATCH (t:Term)
WHERE toLower(t.full_name) CONTAINS toLower($kw)
   OR toLower(t.abbreviation) CONTAINS toLower($kw)
RETURN t.abbreviation, t.full_name, t.primary_spec
```

[adaptive_hop.py:692-694](../rag-engine/retrieval/adaptive_hop.py#L692-L694):
```cypher
MATCH (t:Term {abbreviation: $abbr})
RETURN t.abbreviation, t.full_name, t.source_specs, t.primary_spec
```

| Element | Schema mới | Status |
|---|---|---|
| `Term.abbreviation` (UNIQUE) | ✓ |  |
| `Term.full_name` | ✓ |  |
| `Term.source_specs` (list) | ✓ |  |
| `Term.primary_spec` | ✓ |  |

**Verdict:** Hoạt động bình thường.

---

### ✅ `retrieval/cypher_generator.py` — TƯƠNG THÍCH (gold pattern)

[cypher_generator.py:231-234](../rag-engine/retrieval/cypher_generator.py#L231-L234):
```cypher
MATCH (t:Term {abbreviation: '<TERM_UPPER>'})
WITH t LIMIT 1
MATCH (c:Chunk)
WHERE (c.spec_id IN t.source_specs AND c.chunk_type IN [<INTENT_TYPES>])
```

→ Đây là **gold pattern**, dùng property `Term.source_specs` thay vì traverse `DEFINED_IN`. Schema mới giữ nguyên thuộc tính này → **OK**.

**Verdict:** Hoạt động bình thường.

---

### 🔴 `retrieval/multihop_search.py` — BROKEN (5/5 patterns)

**Tin tốt:** File này declared trong [`retrieval/__init__.py`](../rag-engine/retrieval/__init__.py) nhưng **không file nào trong demo import `MultiHopSearcher`** (grep confirm — chỉ `__init__.py` re-export). → Hiện tại **dead code**, không ảnh hưởng runtime.

**Vấn đề (nếu tương lai có ai gọi):**

| Pattern | Cypher có vấn đề | Schema cũ → Schema mới |
|---|---|---|
| 1, 3, 5 | `(t:Term {name: $seed})` | `Term.name` → **`Term.abbreviation`** |
| 1, 3, 5 | `(t)-[:DEFINED_IN]->(c:Chunk)` | DEFINED_IN trỏ Chunk → **trỏ Document** |
| 2 | `(c1)-[:REFERENCES_SPEC]->(d)<-[:CONTAINS]-(c2)` | OK về structure (CONTAINS giờ đầy đủ) |
| 4 | `(c1)-[:HAS_SUBJECT]->(s:Subject)<-[:HAS_SUBJECT]-(c2)` | ✓ tương thích |

**Pattern fix mẫu (chỉ tham khảo, KHÔNG sửa code):**

```cypher
-- Pattern 1 cũ (BROKEN):
MATCH (t:Term {name: $seed})-[:DEFINED_IN]->(c:Chunk)-[:REFERENCES_CHUNK]->(related:Chunk)

-- Pattern 1 mới (compatible):
MATCH (t:Term {abbreviation: $seed})
MATCH (c:Chunk) WHERE c.spec_id IN t.source_specs
MATCH (c)-[:REFERENCES_CHUNK]->(related:Chunk)

-- Pattern 3 mới: chain qua source_specs thay vì DEFINED_IN→Chunk
MATCH (t1:Term {abbreviation: $seed})
MATCH (c1:Chunk) WHERE c1.spec_id IN t1.source_specs
MATCH (c1)-[:REFERENCES_CHUNK]->(c2:Chunk)
MATCH (t2:Term) WHERE c2.spec_id IN t2.source_specs AND t2.abbreviation <> $seed
RETURN c2, t2.abbreviation
```

**Verdict:** Cần update nếu muốn dùng. **Không khẩn cấp** vì dead code.

---

### 🟡 `retrieval/adaptive_hop_prompts.py` — Schema info **lỗi thời**

LLM Planner đọc schema description này để generate Cypher. Hiện tại nó nói **sai** về schema mới:

[adaptive_hop_prompts.py:130-137](../rag-engine/retrieval/adaptive_hop_prompts.py#L130-L137):

| Statement trong prompt | Sau rebuild với schema mới |
|---|---|
| `(Chunk)-[:MENTIONS]->(NetworkFunction)   ~486` | ❌ **KHÔNG còn** — builder mới không tạo |
| `(Term)-[:DEFINED_IN]->(Document)         (1 row only)` | ✅ **vài nghìn rows** — populate đầy đủ, dùng được |
| `(Chunk)-[:REFERENCES_CHUNK]->(Chunk)     (0 rows)` | ✅ **vài chục nghìn rows** — populate qua 3-tier matching |
| `(Document)-[:CONTAINS]->(Chunk)          (195 rows — broken)` | ✅ **~197k rows** — đầy đủ |
| Không nhắc | 🆕 `(Chunk)-[:PARENT_SECTION]->(Chunk)` + `Chunk.is_parent_section` |

**Hậu quả:** LLM Planner sẽ **không sử dụng** REFERENCES_CHUNK / DEFINED_IN / CONTAINS dù chúng đã populate đầy đủ → mất khả năng multi-hop, traverse hierarchy.

---

### 🟡 `retrieval/cypher_generator.py` — Schema doc lỗi thời

[cypher_generator.py:179-183](../rag-engine/retrieval/cypher_generator.py#L179-L183):
```
- (Chunk)-[:HAS_SUBJECT]->(Subject)        # 197k
- (Chunk)-[:REFERENCES_SPEC]->(Document)   # 165k
- (Document)-[:CONTAINS]->(Chunk)          # 195 — sparse, broken, DO NOT use   ← SAI
- (Term)-[:DEFINED_IN]->(Document)         # 1   — broken, DO NOT use            ← SAI
- (Chunk)-[:MENTIONS]->(NetworkFunction)   # 486 — sparse + noisy                 ← KHÔNG còn
```

[cypher_generator.py:175-178](../rag-engine/retrieval/cypher_generator.py#L175-L178):
```
- NetworkFunction: name, full_name           ← KHÔNG còn label này
```

**Hậu quả:** Cypher LLM generate sẽ né `CONTAINS`, `DEFINED_IN` dù chúng giờ đã hoạt động — pattern bị giới hạn không cần thiết.

---

## 2. Tổng kết tác động

| Component | Trạng thái sau rebuild | Hành động |
|---|---|---|
| `term_first.py` | ✅ Hoạt động | Không cần sửa |
| `adaptive_hop.py` | ✅ Hoạt động | Không cần sửa |
| `cypher_generator.py` (Cypher) | ✅ Hoạt động (gold pattern OK) | Không cần sửa |
| `cypher_generator.py` (prompt schema doc) | 🟡 Lỗi thời | **Nên** update để LLM tận dụng `CONTAINS`, `DEFINED_IN`, `REFERENCES_CHUNK` |
| `adaptive_hop_prompts.py` (prompt schema doc) | 🟡 Lỗi thời | **Nên** update + thêm `PARENT_SECTION` |
| `multihop_search.py` | 🔴 Broken patterns | **Không gấp** — dead code (no callers) |

## 3. Đề xuất ưu tiên

### 🔴 P0 — Bắt buộc nếu muốn LLM tận dụng schema mới

Update 2 file prompt:
- `retrieval/cypher_generator.py` (lines 175-200)
- `retrieval/adaptive_hop_prompts.py` (lines 120-150)

Thay đổi cần làm:
1. **Bỏ** `NetworkFunction` khỏi Node properties
2. **Bỏ** `MENTIONS` khỏi Edges
3. **Move** `CONTAINS`, `DEFINED_IN`, `REFERENCES_CHUNK` từ "BROKEN" sang "SAFE to use"
4. **Add** `PARENT_SECTION` (Chunk→Chunk) và property `Chunk.is_parent_section` vào schema description
5. Update count thực tế sau rebuild

### 🟡 P1 — Nice to have

Update `multihop_search.py`:
- Đổi `Term {name: ...}` → `Term {abbreviation: ...}`
- Đổi `(t)-[:DEFINED_IN]->(c:Chunk)` → `MATCH (c:Chunk) WHERE c.spec_id IN t.source_specs`
- Hoặc nếu không dùng: **delete** (move to trash/) để tránh nhầm lẫn

### 🟢 P2 — Optional

Thêm pattern khai thác `PARENT_SECTION`:
- "Show overview of section X" → traverse xuống children
- Boost retrieval khi parent + child cùng match

## 4. Test recommendation sau rebuild

Verify nhanh sau khi `rebuild-kg.sh` xong, **trước khi serve queries**:

```bash
/home/linguyen/3GPP/.venv/bin/python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j','password'))
with d.session() as s:
    print('=== Edges (kỳ vọng tất cả > 0) ===')
    for r in s.run('MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC'):
        print(f'  {r[\"t\"]:<22} {r[\"c\"]:>10}')
    # Sanity: gold pattern hoạt động không?
    n = s.run('''
        MATCH (t:Term {abbreviation: 'AMF'})
        MATCH (c:Chunk) WHERE c.spec_id IN t.source_specs
        RETURN count(c) AS c
    ''').single()['c']
    print(f'\\n[gold pattern] AMF chunks via source_specs: {n}')
d.close()
"
```

Nếu output có 6 edges đầy đủ + gold pattern trả về > 0 → schema mới sẵn sàng phục vụ rag-engine.

## 5. Risk Matrix

| Risk | Khả năng | Impact | Mitigation |
|---|---|---|---|
| `multihop_search.py` bị gọi nhầm | Thấp (dead code) | Crash query | Delete hoặc update sau |
| LLM né dùng REFERENCES_CHUNK vì prompt cũ | **Cao** | Giảm chất lượng multi-hop | Update prompts (P0) |
| LLM gọi MENTIONS không tồn tại | Trung bình | Cypher error → fallback vector | Update prompts (P0) |
| `Chunk.subject` denormalized không match | Thấp | OK — builder mới vẫn SET property này | Không cần sửa |

## 6. Files liên quan

- Schema mới: [`kg_builder/builder.py`](../kg_builder/builder.py)
- Schema doc: [`docs/kg_schema.md`](kg_schema.md)
- Schema review: [`/.md/research/kg_schema_review.md`](../../.md/research/kg_schema_review.md)
- Files cần update sau (khi user duyệt):
  - [`rag-engine/retrieval/cypher_generator.py`](../rag-engine/retrieval/cypher_generator.py) (prompt schema doc)
  - [`rag-engine/retrieval/adaptive_hop_prompts.py`](../rag-engine/retrieval/adaptive_hop_prompts.py) (prompt schema doc)
  - [`rag-engine/retrieval/multihop_search.py`](../rag-engine/retrieval/multihop_search.py) (Cypher patterns)
