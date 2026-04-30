#!/usr/bin/env python3
"""
Standalone verifier — chạy các check tương tự test_output_json.py
nhưng không cần pytest. Phù hợp để chạy ngay sau khi pipeline xong.

Usage:
    python document_processing/tests/verify.py [JSON_DIR]

Default JSON_DIR: 3GPP_JSON_DOC/processed_json_v4

Exit code: 0 nếu all pass, 1 nếu có lỗi.
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIR = REPO_ROOT / "3GPP_JSON_DOC" / "processed_json_v4"

SPEC_ID_RE = re.compile(r"^ts_\d{2}_\d{3}(?:-\d+)?$")
SPEC_ID_LOOSE_RE = re.compile(r"^ts_\d+_\d+(?:-\w+)?$")


# Định dạng output có màu cho terminal
class C:
    R = "\033[31m"
    G = "\033[32m"
    Y = "\033[33m"
    B = "\033[36m"
    X = "\033[0m"


def fail(msg: str) -> None:
    print(f"{C.R}✗{C.X} {msg}")


def ok(msg: str) -> None:
    print(f"{C.G}✓{C.X} {msg}")


def warn(msg: str) -> None:
    print(f"{C.Y}!{C.X} {msg}")


def info(msg: str) -> None:
    print(f"{C.B}→{C.X} {msg}")


def load_all(json_dir: Path) -> List[dict]:
    # Load JSON files vào list dict, kèm _path để báo lỗi
    docs = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                docs.append({"_path": p, **json.load(f)})
        except Exception as e:
            fail(f"Không parse được {p.name}: {e}")
    return docs


def run_checks(docs: List[dict]) -> Tuple[int, int]:
    """Chạy các check. Trả về (passed, failed) count."""
    passed, failed = 0, 0

    def check(name: str, violations: List, sample_n: int = 5) -> None:
        nonlocal passed, failed
        if violations:
            failed += 1
            fail(f"{name} — {len(violations)} vi phạm")
            for v in violations[:sample_n]:
                print(f"     {v}")
            if len(violations) > sample_n:
                print(f"     ... ({len(violations) - sample_n} more)")
        else:
            passed += 1
            ok(name)

    # 1. specification_id underscore format
    bad = [
        (d["_path"].name, d["metadata"]["specification_id"])
        for d in docs
        if "." in d["metadata"]["specification_id"]
    ]
    check("specification_id không còn dot format", bad)

    # 2. specification_id canonical shape
    nonconform = [
        (d["_path"].name, d["metadata"]["specification_id"])
        for d in docs
        if not SPEC_ID_RE.match(d["metadata"]["specification_id"])
    ]
    check("specification_id khớp ts_NN_NNN[-P]", nonconform)

    # 3. Filename = spec_id.json
    name_mismatch = [
        (d["_path"].name, f"{d['metadata']['specification_id']}.json")
        for d in docs
        if d["_path"].name != f"{d['metadata']['specification_id']}.json"
    ]
    check("filename khớp specification_id", name_mismatch)

    # 4. Chunks tồn tại
    empty = [d["_path"].name for d in docs if not d.get("chunks")]
    check("mọi doc có chunks", empty)

    # 5. chunk_id prefix
    cid_bad = []
    for d in docs:
        sid = d["metadata"]["specification_id"]
        for c in d.get("chunks", []):
            if not c.get("chunk_id", "").startswith(f"{sid}_"):
                cid_bad.append((d["_path"].name, c.get("chunk_id")))
                break
    check("chunk_id bắt đầu bằng spec_id", cid_bad)

    # 6. chunk_id unique trong file
    cid_dup = []
    for d in docs:
        seen = set()
        for c in d.get("chunks", []):
            cid = c["chunk_id"]
            if cid in seen:
                cid_dup.append((d["_path"].name, cid))
            seen.add(cid)
    check("chunk_id unique trong từng doc", cid_dup)

    # 7. target_spec underscore (Bug #2 + #3)
    dot_refs = []
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                if "." in t:
                    dot_refs.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec không còn dot format", dot_refs)

    # 8. target_spec không truncated
    trunc = []
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                m = re.match(r"^ts_\d+_(\d+)", t)
                if m and len(m.group(1)) < 3:
                    trunc.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec không truncated (>=3 digit)", trunc)

    # 9. target_spec shape hợp lệ (Pattern 2 bug)
    bad_shape = []
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                t = ref.get("target_spec", "")
                if not SPEC_ID_LOOSE_RE.match(t):
                    bad_shape.append((d["_path"].name, c["chunk_id"], t))
    check("target_spec đúng shape ts_X_Y", bad_shape)

    # 10. Self-ref không leak vào external
    self_refs = []
    for d in docs:
        sid = d["metadata"]["specification_id"]
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                if ref.get("target_spec") == sid:
                    self_refs.append((d["_path"].name, c["chunk_id"]))
                    break
    check("self-ref không leak vào external", self_refs)

    return passed, failed


def print_summary(docs: List[dict]) -> None:
    # Aggregate stats để human đọc
    total_chunks = sum(len(d.get("chunks", [])) for d in docs)
    total_ext = 0
    distinct = set()
    for d in docs:
        for c in d.get("chunks", []):
            for ref in c.get("cross_references", {}).get("external", []):
                total_ext += 1
                distinct.add(ref.get("target_spec", ""))

    all_sids = {d["metadata"]["specification_id"] for d in docs}
    # Bỏ multi-part collapse khi check orphan
    orphan = {
        t for t in distinct
        if t not in all_sids and not any(s.startswith(f"{t}-") for s in all_sids)
    }

    print()
    print(f"{C.B}=== Stats ==={C.X}")
    print(f"  Documents:        {len(docs)}")
    print(f"  Chunks:           {total_chunks}")
    print(f"  External refs:    {total_ext}")
    print(f"  Distinct targets: {len(distinct)}")
    print(f"  Orphan targets:   {len(orphan)} (không có Document tương ứng trong corpus)")
    if orphan:
        sample = sorted(orphan)[:10]
        print(f"    Sample: {sample}")


def main() -> int:
    # Resolve JSON dir từ CLI hoặc default
    json_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON_DIR
    if not json_dir.is_dir():
        fail(f"JSON dir không tồn tại: {json_dir}")
        return 1

    info(f"Verify: {json_dir}")
    docs = load_all(json_dir)
    if not docs:
        fail(f"Không có file JSON trong {json_dir}")
        return 1
    info(f"Loaded {len(docs)} files")
    print()

    passed, failed = run_checks(docs)
    print_summary(docs)

    print()
    if failed:
        fail(f"{failed} check fail / {passed + failed} total")
        return 1
    ok(f"All {passed} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
