"""
KG Builder — port từ kg_initializer.py + term_extractor.py + subject_classifier.py
của project cha, gộp vào 1 file self-contained cho demo.

Pipeline khi gọi load_json_dir(path):
  1. Load JSON files → documents + chunks (inject _spec_id từ metadata)
  2. Create constraints (Document.spec_id, Chunk.chunk_id, Term.abbreviation, Subject.name UNIQUE)
  3. Create Document nodes
  4. Create Chunk nodes (key_terms, word_count, complexity_score, ...)
  5. CONTAINS edges (Document → Chunk) — match theo spec_id
  6. REFERENCES_SPEC edges (Chunk → Document) — từ cross_references.external (doc-level)
  7. REFERENCES_CHUNK edges (Chunk → Chunk) — 3-tier matching trên section_id;
     gồm cả internal (cùng spec) và external (cross-spec, filter clause + conf≥0.7).
     Property `is_external` phân biệt 2 nhóm.
  8. Term nodes + DEFINED_IN edges (Term → Document) — từ section abbreviation/definition
  9. Subject nodes + HAS_SUBJECT edges (Chunk → Subject)
  10. PARENT_SECTION edges (Chunk → Chunk) + Chunk.is_parent_section property

Schema (4 nodes + 6 edges):
  Nodes:
    Document  (spec_id, version, title, total_chunks)
    Chunk     (chunk_id, spec_id, section_id, section_title, content,
               chunk_type, word_count, complexity_score, key_terms,
               subject, subject_confidence, is_parent_section)
    REFERENCES_CHUNK edge properties:
      is_external (bool), ref_type (str), ref_id (str), confidence (float)
    Term      (abbreviation, full_name, term_type, source_specs, primary_spec)
    Subject   (name, priority, description)

  Edges:
    (Document)-[:CONTAINS]->(Chunk)
    (Chunk)-[:REFERENCES_SPEC]->(Document)
    (Chunk)-[:REFERENCES_CHUNK]->(Chunk)
    (Term)-[:DEFINED_IN]->(Document)
    (Chunk)-[:HAS_SUBJECT]->(Subject)
    (Chunk)-[:PARENT_SECTION]->(Chunk)         # NEW — section hierarchy
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# 5G-related spec series (dùng cho conflict resolution khi merge Term)
_5G_SPEC_PREFIXES = (
    'ts_23_5', 'ts_29_5', 'ts_23_4', 'ts_29_2',
    'ts_33_5', 'ts_38_', 'ts_24_5', 'ts_26_5'
)

CYPHER_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.spec_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.abbreviation IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subject) REQUIRE s.name IS UNIQUE",
]

CYPHER_INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.spec_id)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.section_id)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Term Extraction (port từ term_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedTerm:
    """Một thuật ngữ trích xuất được (abbreviation hoặc definition)."""
    abbreviation: str
    full_name: str
    term_type: str  # 'abbreviation' hoặc 'definition'
    source_spec: str


class TermExtractor:
    """Trích xuất abbreviations + definitions từ chunk content."""

    def __init__(self):
        # Pattern: ABBR<spaces>Full Name (alternative khi không có tab)
        self._abbr_space_pattern = re.compile(
            r'^([A-Z][A-Z0-9/-]{1,15})\s{2,}([A-Z][A-Za-z0-9\s\-/()]+)$',
            re.MULTILINE
        )

    def extract_abbreviations(self, content: str, spec_id: str) -> List[ExtractedTerm]:
        """Pattern: ABBR<tab>Full Name hoặc ABBR<spaces>Full Name."""
        terms: List[ExtractedTerm] = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip introductory sentences
            lower = line.lower()
            if lower.startswith('for the purposes') or 'apply' in lower[:50]:
                continue
            if 'tr 21.905' in lower or 'precedence' in lower:
                continue

            # Tab-separated format
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    abbr, full_name = parts[0].strip(), parts[1].strip()
                    if self._is_valid_abbreviation(abbr) and full_name:
                        terms.append(ExtractedTerm(abbr, full_name, 'abbreviation', spec_id))
                        continue

            # Space-separated format
            m = self._abbr_space_pattern.match(line)
            if m:
                abbr, full_name = m.group(1).strip(), m.group(2).strip()
                if self._is_valid_abbreviation(abbr) and full_name:
                    terms.append(ExtractedTerm(abbr, full_name, 'abbreviation', spec_id))
        return terms

    def extract_definitions(self, content: str, spec_id: str) -> List[ExtractedTerm]:
        """Pattern: 'Term: definition text'."""
        terms: List[ExtractedTerm] = []
        for line in content.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            lower = line.lower()
            if lower.startswith('for the purposes') or 'apply' in lower[:50]:
                continue
            if 'tr 21.905' in lower or 'ts 23.501' in lower:
                continue
            term, definition = line.split(':', 1)
            term, definition = term.strip(), definition.strip()
            if self._is_valid_definition_term(term) and len(definition) > 10:
                terms.append(ExtractedTerm(term, definition, 'definition', spec_id))
        return terms

    @staticmethod
    def _is_valid_abbreviation(abbr: str) -> bool:
        if not abbr or len(abbr) < 2:
            return False
        if not (abbr[0].isupper() or abbr[0].isdigit()):
            return False
        valid_chars = sum(1 for c in abbr if c.isupper() or c.isdigit() or c in '-/')
        return valid_chars >= len(abbr) * 0.6

    @staticmethod
    def _is_valid_definition_term(term: str) -> bool:
        if not term or len(term) < 2 or len(term) > 100:
            return False
        has_letters = any(c.isalpha() for c in term)
        is_reference = term.startswith('[') and term.endswith(']')
        return has_letters and not is_reference


# ─────────────────────────────────────────────────────────────────────────────
# Subject Classification (port từ subject_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

class Subject(Enum):
    """5 chủ đề khớp với benchmark TeleQnA."""
    STANDARDS_SPECIFICATIONS = "Standards specifications"
    STANDARDS_OVERVIEW = "Standards overview"
    LEXICON = "Lexicon"
    RESEARCH_PUBLICATIONS = "Research publications"
    RESEARCH_OVERVIEW = "Research overview"


@dataclass
class SubjectClassification:
    subject: Subject
    confidence: float
    reason: str


class SubjectClassifier:
    """Phân loại chunk vào 1 trong 5 Subject."""

    STANDARDS_SPEC_KEYWORDS = [
        'procedure', 'ie ', 'information element', 'message', 'timer',
        'state machine', 'nas ', 'rrc ', 'ngap ', 'xnap ', 'f1ap ',
        'pdcp', 'rlc', 'mac ', 'phy ', 'harq', 'drb', 'srb',
        'service operation', 'qos flow', 'pdu session'
    ]
    STANDARDS_OVERVIEW_KEYWORDS = [
        'overview', 'architecture', 'introduction', 'general',
        'reference model', 'functional', 'deployment', 'use case',
        'service', 'feature', 'capability', 'scenario'
    ]
    LEXICON_KEYWORDS = [
        'abbreviation', 'definition', 'terminology', 'acronym',
        'vocabulary', 'glossary'
    ]
    RESEARCH_PUB_KEYWORDS = [
        'algorithm', 'optimization', 'machine learning', 'deep learning',
        'neural network', 'theorem', 'proof', 'simulation', 'experimental',
        'performance analysis', 'complexity', 'convergence'
    ]
    RESEARCH_OVERVIEW_KEYWORDS = [
        'survey', 'review', 'state of the art', 'trend', 'challenge',
        'future', 'evolution', 'comparison', 'taxonomy'
    ]
    SPEC_PATTERN = re.compile(
        r'TS[_\s]*\d+[\._]\d+|TR[_\s]*\d+[\._]\d+|3GPP\s+Release\s+\d+',
        re.IGNORECASE
    )

    def classify_chunk(self, chunk: dict) -> SubjectClassification:
        section_title = chunk.get('section_title', '').lower()
        content = chunk.get('content', '').lower()
        chunk_type = chunk.get('chunk_type', '').lower()
        spec_id = chunk.get('spec_id', '') or chunk.get('_spec_id', '')

        # Lexicon (highest confidence cho abbreviation/definition)
        if chunk_type in ['abbreviation', 'definition']:
            return SubjectClassification(Subject.LEXICON, 0.95, f"chunk_type={chunk_type}")
        if any(kw in section_title for kw in self.LEXICON_KEYWORDS):
            return SubjectClassification(Subject.LEXICON, 0.9, "section title contains lexicon keyword")

        # Standards specifications
        spec_score = sum(
            1 for kw in self.STANDARDS_SPEC_KEYWORDS
            if kw in section_title or kw in content[:500]
        )
        if spec_score >= 2 or any(kw in section_title for kw in ['procedure', 'ie ', 'message']):
            return SubjectClassification(
                Subject.STANDARDS_SPECIFICATIONS,
                min(0.7 + spec_score * 0.05, 0.95),
                f"matched {spec_score} standards spec keywords"
            )

        # Standards overview
        overview_score = sum(1 for kw in self.STANDARDS_OVERVIEW_KEYWORDS if kw in section_title)
        if overview_score >= 1:
            return SubjectClassification(
                Subject.STANDARDS_OVERVIEW,
                0.7 + overview_score * 0.1,
                "section title contains overview keyword"
            )

        # Research publications
        research_score = sum(1 for kw in self.RESEARCH_PUB_KEYWORDS if kw in content[:1000])
        if research_score >= 2:
            return SubjectClassification(
                Subject.RESEARCH_PUBLICATIONS,
                min(0.6 + research_score * 0.1, 0.9),
                f"matched {research_score} research keywords"
            )

        # Research overview
        if any(kw in section_title or kw in content[:500] for kw in self.RESEARCH_OVERVIEW_KEYWORDS):
            return SubjectClassification(Subject.RESEARCH_OVERVIEW, 0.7, "matched research overview keyword")

        # Default
        if self.SPEC_PATTERN.search(spec_id):
            return SubjectClassification(Subject.STANDARDS_SPECIFICATIONS, 0.6, "default for 3GPP spec")
        return SubjectClassification(Subject.RESEARCH_OVERVIEW, 0.5, "default classification")


# 5 Subject taxonomy (constant)
_SUBJECT_TAXONOMY = [
    ('Standards specifications', 1, 'Specific 3GPP procedures, IEs, messages'),
    ('Standards overview',       2, 'Architecture, overview, introduction'),
    ('Lexicon',                  3, 'Abbreviations, definitions, terminology'),
    ('Research publications',    4, 'Algorithms, techniques, methods'),
    ('Research overview',        5, 'General concepts, surveys'),
]


# ─────────────────────────────────────────────────────────────────────────────
# KG Builder — main entry
# ─────────────────────────────────────────────────────────────────────────────

class KGBuilder:
    """Build Knowledge Graph từ processed JSON files vào Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._term_extractor = TermExtractor()
        self._subject_classifier = SubjectClassifier()

    def close(self) -> None:
        self._driver.close()

    def verify_connection(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"[kg] Neo4j connection failed: {e}")
            return False

    def setup_schema(self) -> None:
        """Tạo constraints + indexes (idempotent)."""
        with self._driver.session() as s:
            for cypher in CYPHER_CONSTRAINTS + CYPHER_INDEXES:
                try:
                    s.run(cypher)
                except Exception:
                    # Ignore conflict với constraint cũ
                    pass
        print("[kg] Schema ready.")

    def clear(self, batch_size: int = 10000) -> None:
        """Xoá toàn bộ nodes + relationships theo batch để tránh OOM/timeout
        trên KG lớn (Neo4j single-transaction DETACH DELETE thường fail >100k nodes)."""
        total_deleted = 0
        with self._driver.session() as s:
            while True:
                result = s.run(
                    f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(*) AS deleted
                    """
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
                total_deleted += deleted
                print(f"[kg] Deleted {total_deleted:,} nodes...", end="\r")

            # Verify — phải về 0 mới in "cleared"
            remaining = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]

        if remaining == 0:
            print(f"\n[kg] Graph cleared ({total_deleted:,} nodes deleted).")
        else:
            raise RuntimeError(
                f"[kg] Clear failed: {remaining:,} nodes remain after deleting {total_deleted:,}"
            )

    def drop_all_schema(self) -> None:
        """Drop tất cả constraints + indexes (kể cả vector + legacy).
        Setup_schema() sau đó sẽ tạo lại đúng những cái cần."""
        dropped_c, dropped_i = 0, 0
        with self._driver.session() as s:
            # Drop ALL constraints
            constraints = list(s.run("SHOW CONSTRAINTS YIELD name RETURN name"))
            for c in constraints:
                try:
                    s.run(f"DROP CONSTRAINT {c['name']} IF EXISTS")
                    dropped_c += 1
                except Exception as e:
                    print(f"[kg] ⚠ Cannot drop constraint {c['name']}: {e}")

            # Drop ALL indexes (gồm vector index + legacy)
            indexes = list(s.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"))
            for i in indexes:
                try:
                    s.run(f"DROP INDEX {i['name']} IF EXISTS")
                    dropped_i += 1
                except Exception as e:
                    print(f"[kg] ⚠ Cannot drop index {i['name']}: {e}")
        print(f"[kg] Dropped {dropped_c} constraints, {dropped_i} indexes.")

    def clean_all(self, batch_size: int = 10000) -> None:
        """Wipe toàn diện: drop schema + delete data.
        KHÔNG xoá zombie label/type tokens (cần restart container).
        Use restart_neo4j_container() để dọn 100%."""
        print("[kg] === FULL CLEAN ===")
        self.drop_all_schema()
        self.clear(batch_size=batch_size)

    def load_json_dir(self, json_dir: Path) -> int:
        """Pipeline đầy đủ: load JSON → tạo nodes + edges → classify subject → parent_section.
        Trả về tổng số chunks."""
        json_dir = Path(json_dir)
        files = sorted(json_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {json_dir}")

        # Step 1: Load tất cả JSON, inject _spec_id
        documents, chunks = self._load_json_files(files)
        print(f"[kg] Loaded {len(documents)} documents, {len(chunks)} chunks")

        # Step 2-9: Pipeline build KG
        self._create_documents(documents)
        self._create_chunks(chunks)
        self._create_contains_edges()
        self._create_references_spec_edges(chunks)
        n_ref_chunk = self._create_references_chunk_edges(chunks)
        print(f"[kg] Created {n_ref_chunk} REFERENCES_CHUNK edges")
        n_terms = self._create_terms(chunks)
        print(f"[kg] Created {n_terms} Term nodes")
        self._create_subjects(chunks)
        n_parent = self._create_parent_section_edges()
        print(f"[kg] Created {n_parent} PARENT_SECTION edges")

        return len(chunks)

    # ── Step 1: Load JSON ────────────────────────────────────────────────────

    def _load_json_files(self, files: List[Path]) -> Tuple[Dict[str, dict], List[dict]]:
        documents: Dict[str, dict] = {}
        chunks: List[dict] = []
        for f in tqdm(files, desc="[kg] Loading JSON"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[kg] Skip {f.name}: {e}")
                continue
            spec_id = data["metadata"]["specification_id"]
            documents[spec_id] = data
            for chunk in data.get("chunks", []):
                chunk["_spec_id"] = spec_id
                chunks.append(chunk)
        return documents, chunks

    # ── Step 2: Documents ────────────────────────────────────────────────────

    def _create_documents(self, documents: Dict[str, dict]) -> None:
        with self._driver.session() as s:
            for spec_id, data in tqdm(documents.items(), desc="[kg] Documents"):
                meta = data.get("metadata", {})
                export = data.get("export_info", {})
                s.run(
                    """
                    MERGE (d:Document {spec_id: $spec_id})
                    SET d.title = $title,
                        d.version = $version,
                        d.total_chunks = $total_chunks
                    """,
                    spec_id=spec_id,
                    title=meta.get("title", spec_id),
                    version=meta.get("version", ""),
                    total_chunks=export.get("total_chunks", 0),
                )

    # ── Step 3: Chunks ───────────────────────────────────────────────────────

    def _create_chunks(self, chunks: List[dict]) -> None:
        with self._driver.session() as s:
            for chunk in tqdm(chunks, desc="[kg] Chunks"):
                content_meta = chunk.get("content_metadata", {})
                s.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.spec_id = $spec_id,
                        c.section_id = $section_id,
                        c.section_title = $section_title,
                        c.content = $content,
                        c.chunk_type = $chunk_type,
                        c.word_count = $word_count,
                        c.complexity_score = $complexity_score,
                        c.key_terms = $key_terms
                    """,
                    chunk_id=chunk["chunk_id"],
                    spec_id=chunk["_spec_id"],
                    section_id=chunk.get("section_id", ""),
                    section_title=chunk.get("section_title", ""),
                    content=chunk.get("content", ""),
                    chunk_type=chunk.get("chunk_type", "general"),
                    word_count=content_meta.get("word_count", 0),
                    complexity_score=content_meta.get("complexity_score", 0.0),
                    key_terms=content_meta.get("key_terms", []),
                )

    # ── Step 4: CONTAINS edges ───────────────────────────────────────────────

    def _create_contains_edges(self) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MATCH (d:Document), (c:Chunk)
                WHERE d.spec_id = c.spec_id
                MERGE (d)-[:CONTAINS]->(c)
                """
            )

    # ── Step 5: REFERENCES_SPEC edges ────────────────────────────────────────

    def _create_references_spec_edges(self, chunks: List[dict]) -> None:
        with self._driver.session() as s:
            for chunk in tqdm(chunks, desc="[kg] REFERENCES_SPEC"):
                source_id = chunk["chunk_id"]
                for ref in chunk.get("cross_references", {}).get("external", []):
                    target_spec = ref.get("target_spec", "")
                    if not target_spec:
                        continue
                    ref_uid = hashlib.md5(
                        f"{source_id}_{target_spec}_{ref.get('ref_id', '')}".encode()
                    ).hexdigest()[:10]

                    # Chỉ tạo edge nếu target document tồn tại
                    s.run(
                        """
                        MATCH (src:Chunk {chunk_id: $source_id})
                        MATCH (dst:Document {spec_id: $target_spec})
                        MERGE (src)-[r:REFERENCES_SPEC {ref_uid: $ref_uid}]->(dst)
                        SET r.ref_id = $ref_id,
                            r.ref_type = $ref_type,
                            r.confidence = $confidence
                        """,
                        source_id=source_id,
                        target_spec=target_spec,
                        ref_uid=ref_uid,
                        ref_id=ref.get("ref_id", ""),
                        ref_type=ref.get("ref_type", ""),
                        confidence=ref.get("confidence", 0.0),
                    )

    # ── Step 6: REFERENCES_CHUNK edges (internal + external, 3-tier matching) ─

    # Filter cho external refs: chỉ giữ ref_type='clause' (theo feasibility test
    # tháng 4/2026 — 15.7% gross match rate rơi vào bucket conservative).
    _EXT_ALLOWED_REF_TYPES = {"clause"}
    _EXT_CONFIDENCE_THRESHOLD = 0.7

    def _create_references_chunk_edges(self, chunks: List[dict]) -> int:
        """Tạo Chunk→Chunk edges từ cross_references.internal + .external.

        - Internal: match trong cùng spec, set r.is_external = false.
        - External: match cross-spec với filter ref_type ∈ _EXT_ALLOWED_REF_TYPES
          và confidence ≥ _EXT_CONFIDENCE_THRESHOLD, set r.is_external = true.

        3-tier matching: exact section_id → prefix → parent (strip suffix '-N')."""
        # Build section index cho TẤT CẢ specs (1 lần) — phục vụ cả internal + external
        section_index_per_spec: Dict[str, Dict[str, str]] = defaultdict(dict)
        for c in chunks:
            section_index_per_spec[c["_spec_id"]][c.get("section_id", "")] = c["chunk_id"]

        # Group chunks theo spec để giữ structure log per-spec
        spec_chunks: Dict[str, List[dict]] = defaultdict(list)
        for chunk in chunks:
            spec_chunks[chunk["_spec_id"]].append(chunk)

        total_created = 0
        for spec_id, spec_chunk_list in tqdm(spec_chunks.items(), desc="[kg] REFERENCES_CHUNK"):
            same_spec_index = section_index_per_spec[spec_id]
            refs_to_create: List[dict] = []

            for chunk in spec_chunk_list:
                source_id = chunk["chunk_id"]
                cross_refs = chunk.get("cross_references", {})

                # Internal refs (cùng spec)
                for ref in cross_refs.get("internal", []):
                    ref_id = ref.get("ref_id", "")
                    if not ref_id:
                        continue
                    target_id = self._match_ref_to_chunk(ref_id, source_id, same_spec_index)
                    if target_id:
                        refs_to_create.append({
                            "source": source_id,
                            "target": target_id,
                            "is_external": False,
                            "ref_type": ref.get("ref_type", "clause"),
                            "ref_id": ref_id,
                            "confidence": float(ref.get("confidence", 1.0)),
                        })

                # External refs (cross-spec, có filter)
                for ref in cross_refs.get("external", []):
                    ref_id = ref.get("ref_id", "")
                    target_spec = ref.get("target_spec", "")
                    ref_type = ref.get("ref_type", "")
                    conf = float(ref.get("confidence", 0.0))

                    if not ref_id or not target_spec:
                        continue
                    if ref_type not in self._EXT_ALLOWED_REF_TYPES:
                        continue
                    if conf < self._EXT_CONFIDENCE_THRESHOLD:
                        continue

                    target_index = section_index_per_spec.get(target_spec)
                    if not target_index:
                        # Target spec chưa load vào KG → bỏ qua (REFERENCES_SPEC vẫn cover doc-level)
                        continue

                    target_id = self._match_ref_to_chunk(ref_id, source_id, target_index)
                    if target_id:
                        refs_to_create.append({
                            "source": source_id,
                            "target": target_id,
                            "is_external": True,
                            "ref_type": ref_type,
                            "ref_id": ref_id,
                            "confidence": conf,
                        })

            if refs_to_create:
                with self._driver.session() as s:
                    s.run(
                        """
                        UNWIND $refs AS ref
                        MATCH (src:Chunk {chunk_id: ref.source})
                        MATCH (tgt:Chunk {chunk_id: ref.target})
                        MERGE (src)-[r:REFERENCES_CHUNK]->(tgt)
                        SET r.is_external = ref.is_external,
                            r.ref_type = ref.ref_type,
                            r.ref_id = ref.ref_id,
                            r.confidence = ref.confidence
                        """,
                        refs=refs_to_create,
                    )
                total_created += len(refs_to_create)
        return total_created

    @staticmethod
    def _match_ref_to_chunk(
        ref_id: str, source_chunk_id: str, section_index: Dict[str, str]
    ) -> Optional[str]:
        """3-tier matching: exact → prefix → parent (strip '-N' suffix)."""
        # Tier 1: exact match
        if ref_id in section_index:
            target = section_index[ref_id]
            if target != source_chunk_id:
                return target
        # Tier 2: prefix match — ref "5.2" match section "5.2.1"
        for sid, cid in section_index.items():
            if sid.startswith(ref_id + ".") and cid != source_chunk_id:
                return cid
        # Tier 3: parent match — ref "5.2.3-1" → strip "-1" → "5.2.3"
        if "-" in ref_id:
            parent_ref = ref_id.split("-")[0]
            if parent_ref in section_index:
                target = section_index[parent_ref]
                if target != source_chunk_id:
                    return target
        return None

    # ── Step 7: Term nodes + DEFINED_IN edges ────────────────────────────────

    def _create_terms(self, chunks: List[dict]) -> int:
        """Trích Term từ section abbreviation/definition, dedupe + merge across specs."""
        term_dict: Dict[str, dict] = {}

        for chunk in chunks:
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("content", "")
            spec_id = chunk["_spec_id"]

            if 'abbreviation' in section_title:
                terms = self._term_extractor.extract_abbreviations(content, spec_id)
                self._merge_terms(term_dict, terms)
            elif 'definition' in section_title:
                terms = self._term_extractor.extract_definitions(content, spec_id)
                self._merge_terms(term_dict, terms)

        # Write to Neo4j
        created = 0
        with self._driver.session() as s:
            for abbr, td in tqdm(term_dict.items(), desc="[kg] Term nodes"):
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
                        full_name=td['full_name'],
                        term_type=td['term_type'],
                        source_specs=td['source_specs'],
                        primary_spec=td['primary_spec'],
                    )
                    # Tạo DEFINED_IN edges đến mọi spec đã định nghĩa term này
                    for spec_id in td['source_specs']:
                        s.run(
                            """
                            MATCH (t:Term {abbreviation: $abbr})
                            MATCH (d:Document {spec_id: $spec_id})
                            MERGE (t)-[:DEFINED_IN]->(d)
                            """,
                            abbr=abbr, spec_id=spec_id,
                        )
                    created += 1
                # Lộ lỗi cụ thể thay vì swallow — đã từng làm AMF/SMF/NRF
                # biến mất âm thầm. Vẫn không re-raise để 1 term hỏng không
                # phá toàn bộ build batch 30k+ term.
                except Exception as e:
                    print(
                        f"[kg] Term '{abbr}' write failed: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
        return created

    @staticmethod
    def _merge_terms(term_dict: Dict[str, dict], terms: List[ExtractedTerm]) -> None:
        """Merge terms với 5G priority resolution.
        Conflict order:
          1. 5G beats legacy
          2. Both 5G & differ → most 5G citations (tie: lexically smallest spec)
          3. Otherwise first-write-wins
        """
        def is_5g(spec: str) -> bool:
            return any(spec.startswith(p) for p in _5G_SPEC_PREFIXES)

        for term in terms:
            abbr = term.abbreviation
            if abbr not in term_dict:
                term_dict[abbr] = {
                    'abbreviation': abbr,
                    'full_name': term.full_name,
                    'term_type': term.term_type,
                    'source_specs': [term.source_spec],
                    'primary_spec': term.source_spec,
                    'definitions': {term.full_name: [term.source_spec]},
                }
                continue

            entry = term_dict[abbr]
            if term.source_spec not in entry['source_specs']:
                entry['source_specs'].append(term.source_spec)

            defs = entry.setdefault(
                'definitions', {entry['full_name']: [entry['primary_spec']]}
            )
            defs.setdefault(term.full_name, []).append(term.source_spec)

            existing_5g = is_5g(entry['primary_spec'])
            new_5g = is_5g(term.source_spec)

            if new_5g and not existing_5g:
                entry['full_name'] = term.full_name
                entry['primary_spec'] = term.source_spec
            elif new_5g and existing_5g and term.full_name != entry['full_name']:
                new_5g_cites = sum(1 for s in defs[term.full_name] if is_5g(s))
                exist_5g_cites = sum(1 for s in defs[entry['full_name']] if is_5g(s))
                if (new_5g_cites > exist_5g_cites or
                        (new_5g_cites == exist_5g_cites
                         and term.source_spec < entry['primary_spec'])):
                    entry['full_name'] = term.full_name
                    entry['primary_spec'] = term.source_spec

    # ── Step 8: Subject nodes + HAS_SUBJECT edges ────────────────────────────

    def _create_subjects(self, chunks: List[dict]) -> int:
        """Tạo 5 Subject nodes + classify mọi chunk + tạo HAS_SUBJECT edges."""
        with self._driver.session() as s:
            # Tạo 5 Subject nodes
            for name, priority, description in _SUBJECT_TAXONOMY:
                s.run(
                    """
                    MERGE (s:Subject {name: $name})
                    SET s.priority = $priority, s.description = $description
                    """,
                    name=name, priority=priority, description=description,
                )

            # Classify từng chunk + set property
            classified = 0
            for chunk in tqdm(chunks, desc="[kg] HAS_SUBJECT"):
                cls = self._subject_classifier.classify_chunk(chunk)
                s.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    SET c.subject = $subject, c.subject_confidence = $confidence
                    """,
                    chunk_id=chunk["chunk_id"],
                    subject=cls.subject.value,
                    confidence=cls.confidence,
                )
                classified += 1

            # Bulk create HAS_SUBJECT edges (1 query)
            s.run(
                """
                MATCH (c:Chunk), (s:Subject)
                WHERE c.subject = s.name
                MERGE (c)-[:HAS_SUBJECT]->(s)
                """
            )
        return classified

    # ── Step 9: PARENT_SECTION edges + is_parent_section property (NEW) ──────

    def _create_parent_section_edges(self) -> int:
        """Mark chunk là parent section + tạo edge child→nearest_parent.
        Quan hệ dựa trên prefix section_id (e.g., '6.3.1.1' → parent '6.3.1')."""
        with self._driver.session() as s:
            # Đánh dấu chunk có ít nhất 1 con
            s.run(
                """
                MATCH (parent:Chunk)
                WHERE EXISTS {
                    MATCH (child:Chunk)
                    WHERE child.spec_id = parent.spec_id
                      AND child.section_id STARTS WITH parent.section_id + '.'
                }
                SET parent.is_parent_section = true
                """
            )

            # Tạo edge đến nearest parent (không có chunk trung gian)
            result = s.run(
                """
                MATCH (child:Chunk), (parent:Chunk)
                WHERE child.spec_id = parent.spec_id
                  AND child.section_id <> parent.section_id
                  AND child.section_id STARTS WITH parent.section_id + '.'
                  AND NOT EXISTS {
                    MATCH (mid:Chunk)
                    WHERE mid.spec_id = child.spec_id
                      AND child.section_id STARTS WITH mid.section_id + '.'
                      AND mid.section_id STARTS WITH parent.section_id + '.'
                      AND mid.section_id <> child.section_id
                      AND mid.section_id <> parent.section_id
                  }
                MERGE (child)-[:PARENT_SECTION]->(parent)
                RETURN count(*) AS created
                """
            )
            return result.single()["created"]

    # ── Reporting ────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        with self._driver.session() as s:
            print("[kg] Schema stats:")
            for label in ["Document", "Chunk", "Term", "Subject"]:
                count = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                print(f"[kg]   :{label:<10} {count:>10}")
            for rel in ["CONTAINS", "REFERENCES_SPEC", "REFERENCES_CHUNK",
                        "DEFINED_IN", "HAS_SUBJECT", "PARENT_SECTION"]:
                count = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
                print(f"[kg]   :{rel:<18} {count:>10}")

    def validate(self) -> Dict[str, int]:
        """Verify mọi relationship type đều populate. Trả về dict count."""
        rels = ["CONTAINS", "REFERENCES_SPEC", "REFERENCES_CHUNK",
                "DEFINED_IN", "HAS_SUBJECT", "PARENT_SECTION"]
        counts: Dict[str, int] = {}
        with self._driver.session() as s:
            for rel in rels:
                counts[rel] = s.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                ).single()["c"]

        missing = [r for r, c in counts.items() if c == 0]
        if missing:
            print(f"[kg] ⚠ Empty relationships: {', '.join(missing)}")
        else:
            print("[kg] ✓ All relationship types populated")
        return counts
