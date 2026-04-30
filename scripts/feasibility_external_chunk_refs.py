"""
Feasibility test cho Option B (REFERENCES_CHUNK external).

Đo match-rate của external refs có `ref_id` qua hàm `_match_ref_to_chunk`
của KGBuilder, KHÔNG cần Neo4j — chỉ load JSON và chạy matching in-memory.

Decision criteria (theo docs/references_external_chunk_evaluation.md):
  ≥25%  → implement đầy đủ (filter ref_type ∈ {clause, section, subclause}, conf ≥ 0.7)
  15–25% → implement với filter ref_type='clause' only
  <15%  → dừng, không đáng làm

Usage:
  python scripts/feasibility_external_chunk_refs.py [json_dir]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# Import _match_ref_to_chunk từ builder (bypass kg_builder/__init__.py để tránh load Embedder)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kg_builder"))
from builder import KGBuilder  # type: ignore


def load_specs(json_dir: Path) -> Dict[str, List[dict]]:
    """Load mọi JSON → dict[spec_id] = list[chunk]."""
    spec_chunks: Dict[str, List[dict]] = defaultdict(list)
    files = sorted(json_dir.glob("*.json"))
    print(f"[feasibility] Loading {len(files)} JSON files...")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        spec_id = data["metadata"]["specification_id"]
        for c in data.get("chunks", []):
            c["_spec_id"] = spec_id
            spec_chunks[spec_id].append(c)
    return spec_chunks


def build_section_index(spec_chunks: Dict[str, List[dict]]) -> Dict[str, Dict[str, str]]:
    """{spec_id: {section_id: chunk_id}}."""
    index: Dict[str, Dict[str, str]] = defaultdict(dict)
    for spec_id, chunks in spec_chunks.items():
        for c in chunks:
            index[spec_id][c.get("section_id", "")] = c["chunk_id"]
    return index


def measure_match_rate(spec_chunks: Dict[str, List[dict]],
                       section_index: Dict[str, Dict[str, str]]) -> dict:
    """Đếm external refs có ref_id, đo match-rate tổng + breakdown theo ref_type/confidence."""
    total = 0
    matched = 0
    by_type_total: Counter = Counter()
    by_type_match: Counter = Counter()
    by_conf_bucket_total: Counter = Counter()
    by_conf_bucket_match: Counter = Counter()
    target_spec_missing = 0

    for spec_id, chunks in spec_chunks.items():
        for chunk in chunks:
            for ref in chunk.get("cross_references", {}).get("external", []):
                ref_id = ref.get("ref_id", "")
                target_spec = ref.get("target_spec", "")
                ref_type = ref.get("ref_type", "") or "(none)"
                conf = float(ref.get("confidence", 0.0))

                if not ref_id or not target_spec:
                    continue

                total += 1
                by_type_total[ref_type] += 1
                bucket = "≥0.9" if conf >= 0.9 else "≥0.7" if conf >= 0.7 else "≥0.5" if conf >= 0.5 else "<0.5"
                by_conf_bucket_total[bucket] += 1

                if target_spec not in section_index:
                    target_spec_missing += 1
                    continue

                target_id = KGBuilder._match_ref_to_chunk(
                    ref_id, chunk["chunk_id"], section_index[target_spec]
                )
                if target_id:
                    matched += 1
                    by_type_match[ref_type] += 1
                    by_conf_bucket_match[bucket] += 1

    return {
        "total": total,
        "matched": matched,
        "target_spec_missing": target_spec_missing,
        "by_type_total": by_type_total,
        "by_type_match": by_type_match,
        "by_conf_bucket_total": by_conf_bucket_total,
        "by_conf_bucket_match": by_conf_bucket_match,
    }


def report(stats: dict) -> None:
    total = stats["total"]
    matched = stats["matched"]
    rate = matched / total * 100 if total else 0.0

    print()
    print("=" * 70)
    print(f"  EXTERNAL REF MATCH-RATE FEASIBILITY")
    print("=" * 70)
    print(f"  Total external refs có ref_id : {total:,}")
    print(f"  Matched (3-tier matching)     : {matched:,}  ({rate:.1f}%)")
    print(f"  Target spec missing in KG     : {stats['target_spec_missing']:,}")
    print()

    # Breakdown theo ref_type
    print("  Breakdown theo ref_type:")
    print(f"    {'type':<15} {'total':>10} {'matched':>10} {'rate':>8}")
    for t, n_tot in sorted(stats["by_type_total"].items(), key=lambda x: -x[1]):
        n_mat = stats["by_type_match"].get(t, 0)
        r = n_mat / n_tot * 100 if n_tot else 0.0
        print(f"    {t:<15} {n_tot:>10,} {n_mat:>10,} {r:>7.1f}%")
    print()

    # Breakdown theo confidence
    print("  Breakdown theo confidence bucket:")
    print(f"    {'bucket':<10} {'total':>10} {'matched':>10} {'rate':>8}")
    for b in ["≥0.9", "≥0.7", "≥0.5", "<0.5"]:
        n_tot = stats["by_conf_bucket_total"].get(b, 0)
        n_mat = stats["by_conf_bucket_match"].get(b, 0)
        if n_tot:
            r = n_mat / n_tot * 100
            print(f"    {b:<10} {n_tot:>10,} {n_mat:>10,} {r:>7.1f}%")
    print()

    # Decision
    print("  DECISION:")
    if rate >= 25:
        print(f"    ✓ {rate:.1f}% ≥ 25% → IMPLEMENT đầy đủ")
        print(f"    Filter: ref_type ∈ {{clause, section, subclause}}, confidence ≥ 0.7")
    elif rate >= 15:
        print(f"    ⚠ {rate:.1f}% ∈ [15, 25) → IMPLEMENT với filter ref_type='clause' only")
    else:
        print(f"    ✗ {rate:.1f}% < 15% → DỪNG, không đáng làm")
    print("=" * 70)


def main():
    json_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "3GPP_JSON_DOC" / "processed_json_v3"
    if not json_dir.exists():
        print(f"[feasibility] {json_dir} không tồn tại")
        sys.exit(1)

    spec_chunks = load_specs(json_dir)
    print(f"[feasibility] Loaded {len(spec_chunks)} specs, "
          f"{sum(len(v) for v in spec_chunks.values()):,} chunks")

    section_index = build_section_index(spec_chunks)
    stats = measure_match_rate(spec_chunks, section_index)
    report(stats)


if __name__ == "__main__":
    main()
