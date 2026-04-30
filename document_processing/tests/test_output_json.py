"""
Verify output JSON sau khi chạy `download_and_process_3gpp.py process[-local]`.

Trọng tâm: kiểm tra các bug đã fix trong spec_id_format_investigation.md không tái phát:
  - spec_id luôn dùng underscore (Path A + Path B no-part đồng nhất)
  - target_spec trong cross_references luôn underscore (không còn dot)
  - Không bị truncate (vd "ts_38_33" thay vì "ts_38_331" do regex backtracking)
  - Pattern "clause X of TS Y" capture đúng spec, không lấy clause làm target_spec

Chạy:
    pytest document_processing/tests -v
    pytest document_processing/tests --json-dir=/path/to/processed_json_v4 -v
    JSON_DIR=/path/to/dir pytest document_processing/tests
"""
import re
from pathlib import Path
from typing import List

import pytest

# Spec id canonical: ts_<series>_<number>[-<part>]
# Series 2 chữ số, number 3 chữ số (chuẩn 3GPP), part là số nguyên optional
SPEC_ID_RE = re.compile(r"^ts_\d{2}_\d{3}(?:-\d+)?$")

# Cùng pattern nhưng cho phép edge case (vd. số khác chuẩn) — dùng cho warning soft check
SPEC_ID_LOOSE_RE = re.compile(r"^ts_\d+_\d+(?:-\w+)?$")


# ── Document-level checks ───────────────────────────────────────────────────


def test_all_files_loaded(all_docs):
    # Sanity: phải có ít nhất 1 doc và mỗi doc có metadata.specification_id
    assert len(all_docs) > 0
    for doc in all_docs:
        assert "metadata" in doc, f"{doc['_path'].name}: thiếu metadata"
        assert "specification_id" in doc["metadata"], f"{doc['_path'].name}: thiếu specification_id"


def test_specification_id_underscore_format(all_docs):
    # specification_id phải underscore (không còn dot do Path B fallback)
    bad = []
    for doc in all_docs:
        sid = doc["metadata"]["specification_id"]
        if "." in sid:
            bad.append((doc["_path"].name, sid))
    assert not bad, f"specification_id có dot format ({len(bad)} files): {bad[:5]}"


def test_specification_id_canonical(all_docs):
    # Canonical: ts_NN_NNN[-P]. Liệt kê file lệch chuẩn để debug nguồn gốc.
    nonconforming = []
    for doc in all_docs:
        sid = doc["metadata"]["specification_id"]
        if not SPEC_ID_RE.match(sid):
            nonconforming.append((doc["_path"].name, sid))
    if nonconforming:
        # Soft fail: in ra để trace nhưng không break (có thể có spec đặc biệt)
        pytest.fail(
            f"{len(nonconforming)} specification_id không khớp ts_NN_NNN[-P]: "
            f"{nonconforming[:10]}"
        )


def test_filename_matches_spec_id(all_docs):
    # File phải tên = spec_id.json (do save_to_json đặt theo specification_id)
    mismatches = []
    for doc in all_docs:
        expected = f"{doc['metadata']['specification_id']}.json"
        if doc["_path"].name != expected:
            mismatches.append((doc["_path"].name, expected))
    assert not mismatches, f"Filename ≠ spec_id ({len(mismatches)}): {mismatches[:5]}"


# ── Chunk-level checks ──────────────────────────────────────────────────────


def test_chunks_exist(all_docs):
    # Mỗi doc phải có ít nhất 1 chunk (nếu 0 thì pipeline fail silently)
    empty = [doc["_path"].name for doc in all_docs if not doc.get("chunks")]
    assert not empty, f"{len(empty)} files không có chunks: {empty[:5]}"


def test_chunk_id_starts_with_spec_id(all_docs):
    # chunk_id format: {spec_id}_{section_id}
    bad = []
    for doc in all_docs:
        sid = doc["metadata"]["specification_id"]
        prefix = f"{sid}_"
        for c in doc.get("chunks", []):
            cid = c.get("chunk_id", "")
            if not cid.startswith(prefix):
                bad.append((doc["_path"].name, cid))
                break
    assert not bad, f"chunk_id không bắt đầu bằng spec_id ({len(bad)}): {bad[:5]}"


def test_chunk_ids_unique_within_doc(all_docs):
    # Cùng 1 file không được trùng chunk_id (Neo4j sẽ MERGE đè)
    duplicates = []
    for doc in all_docs:
        seen = set()
        for c in doc.get("chunks", []):
            cid = c["chunk_id"]
            if cid in seen:
                duplicates.append((doc["_path"].name, cid))
            seen.add(cid)
    assert not duplicates, f"chunk_id trùng ({len(duplicates)}): {duplicates[:5]}"


def test_required_chunk_fields(all_docs):
    # Field tối thiểu cho KG builder
    required = {"chunk_id", "section_id", "section_title", "content", "chunk_type", "cross_references"}
    missing = []
    for doc in all_docs:
        for c in doc.get("chunks", []):
            absent = required - c.keys()
            if absent:
                missing.append((doc["_path"].name, c.get("chunk_id"), absent))
                break
    assert not missing, f"Chunk thiếu field bắt buộc ({len(missing)}): {missing[:3]}"


# ── Cross-reference checks (trọng tâm bug đã fix) ───────────────────────────


def test_target_spec_underscore_format(all_docs):
    # Bug #2 + #3: target_spec phải underscore. Dot format = chunking cũ chưa rerun.
    dot_refs = []
    for doc in all_docs:
        for c in doc.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                target = ref.get("target_spec", "")
                if "." in target:
                    dot_refs.append((doc["_path"].name, c["chunk_id"], target))
    assert not dot_refs, (
        f"{len(dot_refs)} target_spec còn dùng dot format. "
        f"Sample: {dot_refs[:5]}"
    )


def test_target_spec_no_truncation(all_docs):
    # Bug regex backtracking: target_spec dạng ts_38_33 (truncated từ 38.331)
    # Heuristic: số sau underscore cuối phải >= 3 chữ số (chuẩn 3GPP)
    suspicious = []
    for doc in all_docs:
        for c in doc.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                target = ref.get("target_spec", "")
                m = re.match(r"^ts_\d+_(\d+)", target)
                if m and len(m.group(1)) < 3:
                    suspicious.append((doc["_path"].name, c["chunk_id"], target))
    assert not suspicious, (
        f"{len(suspicious)} target_spec có vẻ bị truncate (<3 chữ số): "
        f"{suspicious[:5]}"
    )


def test_target_spec_not_clause_number(all_docs):
    # Pattern 2 bug cũ: "clause X of TS Y" lấy nhầm X làm target_spec.
    # Sau fix named groups, target_spec phải khớp shape spec số 3GPP.
    bad_shape = []
    for doc in all_docs:
        for c in doc.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                target = ref.get("target_spec", "")
                if not SPEC_ID_LOOSE_RE.match(target):
                    bad_shape.append((doc["_path"].name, c["chunk_id"], target))
    assert not bad_shape, (
        f"{len(bad_shape)} target_spec sai shape (có thể là clause number): "
        f"{bad_shape[:5]}"
    )


def test_no_self_external_reference(all_docs):
    # Self-ref phải nằm ở internal[], không phải external[]
    self_refs = []
    for doc in all_docs:
        sid = doc["metadata"]["specification_id"]
        for c in doc.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                if ref.get("target_spec") == sid:
                    self_refs.append((doc["_path"].name, c["chunk_id"]))
                    break
    assert not self_refs, (
        f"{len(self_refs)} chunk có self-ref bị phân loại external: "
        f"{self_refs[:5]}"
    )


# ── Aggregate stats (in summary để human inspection) ────────────────────────


def test_print_summary(all_docs, capsys):
    # Không assert — chỉ in stats. Chạy với -s để xem.
    total_docs = len(all_docs)
    total_chunks = sum(len(d.get("chunks", [])) for d in all_docs)
    total_ext = 0
    distinct_targets = set()
    for d in all_docs:
        sid = d["metadata"]["specification_id"]
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                total_ext += 1
                distinct_targets.add(ref.get("target_spec", ""))

    with capsys.disabled():
        print(f"\n=== Output JSON Summary ===")
        print(f"  Documents:           {total_docs}")
        print(f"  Chunks:              {total_chunks}")
        print(f"  External refs:       {total_ext}")
        print(f"  Distinct targets:    {len(distinct_targets)}")
        # Cảnh báo nếu có target trỏ đến spec không tồn tại trong corpus
        all_sids = {d["metadata"]["specification_id"] for d in all_docs}
        orphan_targets = distinct_targets - all_sids
        # Bỏ multi-part collapse: nếu target = ts_38_508 thì có thể có ts_38_508-1 ... -7
        orphan_after_part_collapse = {
            t for t in orphan_targets
            if not any(s.startswith(f"{t}-") for s in all_sids)
        }
        print(f"  Orphan targets:      {len(orphan_after_part_collapse)} (không có Document tương ứng)")
        if orphan_after_part_collapse:
            sample = sorted(orphan_after_part_collapse)[:10]
            print(f"    Sample: {sample}")
