"""
Enrichment: đọc JSON đã processed, extract Term từ abbreviation/definition
sections, MERGE vào KG hiện tại.

Không xóa, không rebuild — chỉ thêm/cập nhật Term nodes và DEFINED_IN edges.
Khác với `_create_terms` của KGBuilder ở chỗ in lỗi cụ thể thay vì swallow,
giúp lộ ra các write conflict (vd. type mismatch trên `source_specs`) đang
âm thầm làm thiếu Term trong KG hiện tại.

Usage:
    python kg_builder/enrich_terms.py --json-dir 3GPP_JSON_DOC/processed_json_v4
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm

# Reuse existing extractor + merge logic — không duplicate
from kg_builder.builder import KGBuilder, TermExtractor


def _load_env_file(env_path: Path) -> None:
    """Parser tối giản cho .env, không cần python-dotenv. Skip line trống/comment."""
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env_file(Path(__file__).resolve().parent.parent / ".env")


def collect_terms_from_json(json_dir: Path) -> Dict[str, dict]:
    """Đọc tất cả *.json trong json_dir, extract terms, consolidate qua
    `KGBuilder._merge_terms`. Trả về term_dict tương đương output của
    `_create_terms` trước khi write."""
    extractor = TermExtractor()
    term_dict: Dict[str, dict] = {}
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {json_dir}")

    for jf in tqdm(files, desc="[enrich] Reading JSON"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            spec_id = data["metadata"]["specification_id"]
        except Exception as e:
            print(f"[skip] {jf.name}: {e}")
            continue

        for chunk in data.get("chunks", []):
            section_title = (chunk.get("section_title") or "").lower()
            content = chunk.get("content") or ""
            if "abbreviation" in section_title:
                terms = extractor.extract_abbreviations(content, spec_id)
                KGBuilder._merge_terms(term_dict, terms)
            elif "definition" in section_title:
                terms = extractor.extract_definitions(content, spec_id)
                KGBuilder._merge_terms(term_dict, terms)
    return term_dict


def write_terms_to_neo4j(driver, term_dict: Dict[str, dict]) -> Tuple[int, int, int]:
    """MERGE Term nodes + DEFINED_IN edges. Trả về (new, updated, failed)."""
    new_count = 0
    updated_count = 0
    failed_count = 0
    with driver.session() as s:
        for abbr, td in tqdm(term_dict.items(), desc="[enrich] Writing Terms"):
            # Check existence để phân biệt new vs updated trong stat
            existed = s.run(
                "MATCH (t:Term {abbreviation: $abbr}) RETURN count(t) AS n",
                abbr=abbr,
            ).single()["n"]
            try:
                s.run(
                    """
                    MERGE (t:Term {abbreviation: $abbr})
                    SET t.full_name = $full_name,
                        t.term_type = $term_type,
                        t.source_specs = $source_specs,
                        t.primary_spec = $primary_spec
                    """,
                    abbr=abbr,
                    full_name=td["full_name"],
                    term_type=td["term_type"],
                    source_specs=td["source_specs"],
                    primary_spec=td["primary_spec"],
                )
                # Tạo DEFINED_IN edges (skip nếu Document chưa có — không lỗi)
                for spec_id in td["source_specs"]:
                    s.run(
                        """
                        MATCH (t:Term {abbreviation: $abbr})
                        MATCH (d:Document {spec_id: $spec_id})
                        MERGE (t)-[:DEFINED_IN]->(d)
                        """,
                        abbr=abbr,
                        spec_id=spec_id,
                    )
                if existed:
                    updated_count += 1
                else:
                    new_count += 1
            except Exception as e:
                # In lỗi cụ thể thay vì swallow (anti-pattern của builder.py:691-692)
                print(f"[warn] {abbr}: {type(e).__name__}: {e}", file=sys.stderr)
                failed_count += 1
    return new_count, updated_count, failed_count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json-dir",
        required=True,
        help="Path to processed JSON directory (e.g. 3GPP_JSON_DOC/processed_json_v4)",
    )
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        raise SystemExit(f"Not a directory: {json_dir}")

    print(f"[enrich] Reading JSON from {json_dir}")
    term_dict = collect_terms_from_json(json_dir)
    print(f"[enrich] Collected {len(term_dict)} unique terms")

    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        new, upd, fail = write_terms_to_neo4j(driver, term_dict)
        print(f"[enrich] Done. New={new}, Updated={upd}, Failed={fail}")
        if fail > 0:
            print(
                "[enrich] Some Terms failed to write. Check stderr for "
                "specific abbreviations + error types — typically property "
                "type conflicts on existing nodes from older builds."
            )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
