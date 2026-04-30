# Điều tra: spec_id format mismatch → REFERENCES_SPEC underpopulated

**Ngày:** 2026-04-27
**Ảnh hưởng:** REFERENCES_SPEC = 28,808 thay vì kỳ vọng ~130k–165k (giảm ~80% edges)
**Scope:** Bug nguồn gốc trong [`document_processing/download_and_process_3gpp.py`](../../document_processing/download_and_process_3gpp.py)

---

## TL;DR

3 chỗ trong code chunking sinh `spec_id` với **2 format khác nhau**:
- 81% docs: underscore `ts_23_501`
- 19% docs: dot `ts_23.501`
- 100% cross-refs trong các docs: DOT `ts_23.501`

→ Khi builder MATCH `Document {spec_id: $target_spec}`, **80% refs miss** vì format không khớp.

**Solution nhanh:** normalize trong builder (Solution B). Long-term: fix code chunking.

---

## 1. Đo lường thực tế

### 1.1. Document spec_id distribution (1,796 docs trong KG)

```
Format dot     'ts_22.983' →   337 (18.8%)
Format us      'ts_22_001' → 1,459 (81.2%)
```

### 1.2. Cross-refs target_spec distribution

```
Total external refs trong JSON:  237,895
Distinct target_spec values:      5,637
Top targets:
  ts_38.508  15,862
  ts_36.508  10,236
  ts_29.500   6,154
  ts_23.501   6,028
  ts_23.502   5,736
```
→ **Tất cả** target_spec dùng DOT format.

### 1.3. Match rate

| Method | Match | % |
|---|---:|---:|
| Exact match (current builder) | 28,897 / 237,895 | **12.1%** |
| If try cả 2 variants (dot + underscore) | 165,973 / 237,895 (projected) | **~70%** |

→ **Recover được 5x edges nếu fix.**

---

## 2. Root cause — 3 bug code

### 2.1. Bug #1 — Document.spec_id có 2 path inconsistent

[`download_and_process_3gpp.py:531-558`](../../document_processing/download_and_process_3gpp.py#L531-L558):

```python
# Path A — Đọc TS number từ DOCX content
ts_version_match = re.search(r'3GPP\s+TS\s+(\d+\.\d+(?:-\d+)?)\s+V(\d+\.\d+\.\d+)', text)
if ts_version_match:
    ts_number = ts_version_match.group(1)
    specification_id = f"ts_{ts_number.replace('.', '_')}"   # ✓ underscore

# Path B — Fallback khi không đọc được DOCX header
else:
    filename_match = re.search(r'(\d{2})(\d{3})(?:-(\d+))?', filename)
    if filename_match:
        series = filename_match.group(1)
        spec_num = filename_match.group(2)
        part = filename_match.group(3)
        if part:
            specification_id = f"ts_{series}_{spec_num}-{part}"   # ✓ underscore
        else:
            specification_id = f"ts_{series}.{spec_num}"          # ❌ DOT — BUG
```

**Khi nào DOCX header không đọc được?**
- DOCX hỏng / encrypted
- DOCX không có header chuẩn `"3GPP TS XX.XXX VYY.Y.Y"`
- DOCX là file phụ (annex, separate part)

→ 337 / 1,796 docs (~19%) đi qua Path B no-part.

### 2.2. Bug #2 — Cross-ref target_spec luôn DOT

[`download_and_process_3gpp.py:672-680`](../../document_processing/download_and_process_3gpp.py#L672-L680):

```python
ts_match = re.search(r'(?:3GPP\s+)?(?:TS|TR)\s+(\d+\.\d+)', context, re.IGNORECASE)
#                                                ^^^^^^^^ regex DOT format

if ts_match and ts_match.group(1) != spec_num:
    external_refs.append({
        "target_spec": f"ts_{ts_match.group(1)}",    # ❌ DOT giữ nguyên — BUG
        "ref_type": ref_type,
        "ref_id": ref_num,
        "confidence": 0.9
    })
```

→ Mọi clause/table/figure cross-ref → DOT format target_spec.

### 2.3. Bug #3 — Standalone spec ref cũng DOT

[`download_and_process_3gpp.py:687-700`](../../document_processing/download_and_process_3gpp.py#L687-L700):

```python
for pattern in self.external_patterns:
    for match in pattern.finditer(content):
        target_num = groups[0]                    # vd "23.501"
        target_spec = f"ts_{target_num}"          # ❌ DOT — BUG
        if target_num != spec_num:
            external_refs.append({
                "target_spec": target_spec,
                ...
            })
```

→ Mọi standalone spec ref → DOT format.

---

## 3. Giải thích pattern lỗi quan sát được

### 3.1. Match rate 28,808 = (chunks of 337 dot-docs) × ref count

Documents dot format (337 docs) **chỉ match được** với refs DOT (100%) → tạo edges.
Documents underscore format (1,459 docs) **không match** ref DOT bao giờ → 0 edges từ nhóm này.

→ 28,808 ≈ tổng external refs **của 337 docs dot format**.

### 3.2. Top missing targets all are popular specs

```
ts_38.508 → 15,862 refs missing (file là ts_38_508-1.json + ts_38_508-2.json)
ts_29.500 →  6,154 refs missing (Document trong KG là ts_29_500)
ts_23.501 →  6,028 refs missing (Document trong KG là ts_23_501)
ts_23.502 →  5,736 refs missing (Document trong KG là ts_23_502)
```

Tất cả đều có Document tương ứng trong KG (chỉ khác format) → toàn bộ recover được nếu fix.

---

## 4. Solutions

### Solution A — Fix gốc (`download_and_process_3gpp.py`)

**3 sửa đổi:**

```python
# Line 557 (Path B no-part):
- specification_id = f"ts_{series}.{spec_num}"
+ specification_id = f"ts_{series}_{spec_num}"

# Line 676 (cross-ref clause/table/figure):
- "target_spec": f"ts_{ts_match.group(1)}",
+ "target_spec": f"ts_{ts_match.group(1).replace('.', '_')}",

# Line 693 (standalone spec ref):
- target_spec = f"ts_{target_num}"
+ target_spec = f"ts_{target_num.replace('.', '_')}"
```

**Tác động:**
- ✅ Triệt để, nguồn gốc
- 🔴 Cần rerun chunking pipeline trên 1,796 DOCX files (~vài giờ)
- 🔴 Cần DOCX files gốc

**Áp dụng khi:** có thời gian rerun, muốn dataset sạch lâu dài.

### Solution B — Normalize tại builder (KHUYẾN NGHỊ)

Trong [`kg_builder/builder.py`](../kg_builder/builder.py), thêm helper + áp dụng tại 2 chỗ:

```python
@staticmethod
def _normalize_spec_id(s: str) -> str:
    """Canonical form: underscore between series and number.

    Examples:
        ts_23.501       → ts_23_501
        ts_23_501       → ts_23_501  (no-op)
        ts_38_508-1     → ts_38_508-1  (no-op, suffix giữ nguyên)
        ts_29.500-2     → ts_29_500-2
    """
    if not s:
        return s
    # Tách suffix (-1, -2) trước, normalize phần chính, ghép lại
    if "-" in s:
        base, suffix = s.split("-", 1)
        return f"{base.replace('.', '_')}-{suffix}"
    return s.replace(".", "_")
```

**Áp dụng tại:**

```python
# 1. _create_documents — đảm bảo Documents đều underscore
def _create_documents(self, documents):
    for spec_id, data in tqdm(documents.items(), desc="[kg] Documents"):
        normalized = self._normalize_spec_id(spec_id)
        s.run("""MERGE (d:Document {spec_id: $spec_id})...""", spec_id=normalized, ...)

# 2. _create_chunks — Chunk.spec_id cũng normalize
def _create_chunks(self, chunks):
    for chunk in tqdm(chunks):
        normalized = self._normalize_spec_id(chunk["_spec_id"])
        s.run("""MERGE (c:Chunk {chunk_id: $chunk_id})
                 SET c.spec_id = $spec_id, ...""",
              spec_id=normalized, ...)

# 3. _create_references_spec_edges — normalize target_spec trước MATCH
def _create_references_spec_edges(self, chunks):
    for chunk in tqdm(chunks):
        for ref in chunk.get("cross_references", {}).get("external", []):
            target = self._normalize_spec_id(ref.get("target_spec", ""))
            # MATCH với target đã normalize
            ...
```

**Tác động:**
- ✅ Đơn giản (~10 dòng code)
- ✅ Không phụ thuộc rerun chunking
- ✅ Match rate ~70% (recover hầu hết)
- ⚠️ Cần rebuild KG (chạy lại `bash scripts/rebuild-kg.sh`)
- ⚠️ Solution dataset gốc vẫn lệch — chỉ patch trong KG

**Áp dụng khi:** muốn fix nhanh, không cần dataset sạch.

### Solution C — One-time normalize script

Viết script sửa toàn bộ 1,796 JSON files:

```python
import json, glob

for path in glob.glob("3GPP_JSON_DOC/processed_json_v3/*.json"):
    with open(path) as f:
        data = json.load(f)

    # Normalize specification_id
    sid = data["metadata"]["specification_id"]
    data["metadata"]["specification_id"] = normalize_spec_id(sid)

    # Normalize all target_spec
    for chunk in data.get("chunks", []):
        for ref in chunk.get("cross_references", {}).get("external", []):
            ref["target_spec"] = normalize_spec_id(ref["target_spec"])

    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

**Tác động:**
- ✅ Sửa data, builder không cần thay đổi
- ✅ Sạch hơn Solution B (data trên disk consistent)
- 🟡 Một lần chạy ~vài phút
- 🔴 Mất original format, không revert được dễ

**Áp dụng khi:** muốn sửa data on-disk one-time, không động code build.

---

## 5. So sánh 3 solutions

| Tiêu chí | A (fix code chunking) | **B (normalize trong builder)** ⭐ | C (sửa JSON files) |
|---|:---:|:---:|:---:|
| Triệt để | ✓✓✓ | ✓ | ✓✓ |
| Implementation cost | 🔴 Cao | 🟢 Thấp | 🟡 Vừa |
| Need rerun chunking | ✓ | ✗ | ✗ |
| Match rate sau fix | ~95% | ~70% | ~70% |
| Backward compat | 🟡 | ✓ | 🟡 |
| Risk | Vừa | Thấp | Thấp |

---

## 6. Roadmap khuyến nghị

```
1. ⏳ NOW — Implement Solution B trong builder.py (~10 dòng)
   ├─ _normalize_spec_id() helper
   ├─ Apply trong _create_documents, _create_chunks, _create_references_spec_edges
   └─ Test với fixtures + rebuild

2. ⏳ NEXT — Rebuild KG
   └─ bash scripts/rebuild-kg.sh
   Kỳ vọng: REFERENCES_SPEC nhảy từ 28k → 130-165k

3. ⏳ LATER — Solution A (long-term)
   ├─ Fix bug document_processing/download_and_process_3gpp.py (3 places)
   ├─ Document fix trong release notes
   └─ Rerun chunking khi có thời gian

4. ⏳ FUTURE — Validate
   ├─ Chạy benchmark before/after để measure accuracy gain
   └─ Update thesis nếu cross-spec questions tăng accuracy
```

---

## 7. Câu hỏi mở

1. **Tại sao 337 docs đi qua Path B (fallback)?**
   Investigate những file nào có format DOT trong KG. Nguyên nhân:
   - DOCX không có header `"3GPP TS XX.XXX V..."` (thường multi-part hoặc spec phụ)
   - DOCX format khác chuẩn
   - File extraction sai

2. **Solution B normalize có break existing tests không?**
   Tests hiện tại dùng `ts_99_001` (đã underscore) → no-op của normalize → không break.

3. **`Term.source_specs` cũng cần normalize?**
   Yes — Term.source_specs lưu spec_id từ chunk["_spec_id"]. Nếu Chunk.spec_id normalize, source_specs cũng tự normalize theo.

4. **Có nên giữ original spec_id ở property khác (e.g. `spec_id_original`)?**
   Optional. Nếu cần audit nguồn gốc data, giữ. Nếu không cần, bỏ để minimize schema.

---

## 8. Liên quan

- Schema mới: [`kg_schema.md`](kg_schema.md)
- Audit tương thích: [`schema_compatibility_audit.md`](schema_compatibility_audit.md)
- Cross-spec chunk-level reference (P2): [`references_external_chunk_evaluation.md`](references_external_chunk_evaluation.md)
- Code chunking: [`/document_processing/download_and_process_3gpp.py`](../../document_processing/download_and_process_3gpp.py)
- Code KG build: [`/demo/kg_builder/builder.py`](../kg_builder/builder.py)

---

## 9. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-27 | Phát hiện format mismatch | REFERENCES_SPEC = 28k bất thường so với 165k cũ |
| 2026-04-27 | Trace root cause | 3 bugs trong download_and_process_3gpp.py |
| TBD | Implement Solution B | Quick win, không cần rerun chunking |
| TBD | Solution A long-term | Khi có thời gian rerun toàn bộ pipeline |
