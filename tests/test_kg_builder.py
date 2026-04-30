"""
Tests cho kg_builder.builder.KGBuilder — verify schema mới (4 nodes + 6 edges).
Khớp với schema parent project: Term có abbreviation (không phải name),
DEFINED_IN trỏ Term→Document (không phải Term→Chunk).
"""
import pytest

from tests.conftest import requires_neo4j


# ── Pure-Python tests (no Neo4j) ─────────────────────────────────────────────

def test_kg_builder_loads_fixture_file(fixture_dir):
    """Sanity check fixture JSON structure."""
    import json
    data = json.loads((fixture_dir / "ts_99_001.json").read_text())
    assert data["metadata"]["specification_id"] == "ts_99_001"
    assert len(data["chunks"]) == 4
    # Chunk 2 là abbreviations section, dùng để extract Term
    assert data["chunks"][1]["section_title"] == "Abbreviations"


def test_term_extractor_parses_tab_separated():
    """TermExtractor.extract_abbreviations parse format ABBR<tab>Full Name."""
    from kg_builder.builder import TermExtractor
    content = "AMF\tAccess and Mobility Management Function\nSMF\tSession Management Function\n"
    terms = TermExtractor().extract_abbreviations(content, "ts_test")
    assert len(terms) == 2
    assert terms[0].abbreviation == "AMF"
    assert terms[0].full_name == "Access and Mobility Management Function"
    assert terms[0].term_type == "abbreviation"


def test_subject_classifier_lexicon_for_abbreviation_chunk():
    """chunk_type='abbreviation' → Subject.LEXICON với confidence cao."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {"chunk_type": "abbreviation", "section_title": "Abbreviations",
             "content": "", "spec_id": "ts_99_001"}
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.LEXICON
    assert result.confidence >= 0.9


def test_match_ref_to_chunk_three_tier():
    """3-tier matching: exact → prefix → parent (strip suffix)."""
    from kg_builder.builder import KGBuilder
    section_index = {"3": "c3", "3.1": "c3_1", "5.2.3": "c5_2_3"}
    # Tier 1: exact
    assert KGBuilder._match_ref_to_chunk("3", "src", section_index) == "c3"
    # Tier 2: prefix (ref "5.2" matches "5.2.3")
    assert KGBuilder._match_ref_to_chunk("5.2", "src", section_index) == "c5_2_3"
    # Tier 3: parent (ref "5.2.3-1" → "5.2.3")
    assert KGBuilder._match_ref_to_chunk("5.2.3-1", "src", section_index) == "c5_2_3"
    # Self-reference fall-through: ref "3" từ source "c3" → bỏ exact, fallback prefix → "c3_1"
    assert KGBuilder._match_ref_to_chunk("3", "c3", section_index) == "c3_1"


def test_match_ref_to_chunk_returns_none_when_no_match():
    """Không match được tier nào → None."""
    from kg_builder.builder import KGBuilder
    section_index = {"3": "c3", "3.1": "c3_1"}
    # ref_id hoàn toàn không tồn tại
    assert KGBuilder._match_ref_to_chunk("99.99", "src", section_index) is None
    # ref_id "7-2" → parent "7" cũng không có
    assert KGBuilder._match_ref_to_chunk("7-2", "src", section_index) is None


def test_match_ref_to_chunk_empty_index():
    """Section index rỗng → luôn None."""
    from kg_builder.builder import KGBuilder
    assert KGBuilder._match_ref_to_chunk("3", "src", {}) is None


# ── TermExtractor — additional unit tests ────────────────────────────────────

def test_term_extractor_parses_space_separated():
    """Space-separated format: ABBR<spaces>Full Name."""
    from kg_builder.builder import TermExtractor
    # Cần >=2 spaces giữa abbr và full_name
    content = "AMF  Access and Mobility Management Function\n"
    terms = TermExtractor().extract_abbreviations(content, "ts_test")
    assert len(terms) == 1
    assert terms[0].abbreviation == "AMF"


def test_term_extractor_skips_intro_lines():
    """Bỏ qua 'For the purposes...' và các câu giới thiệu boilerplate."""
    from kg_builder.builder import TermExtractor
    content = (
        "For the purposes of the present document, the following abbreviations apply.\n"
        "TR 21.905 references apply.\n"
        "AMF\tAccess and Mobility Management Function\n"
    )
    terms = TermExtractor().extract_abbreviations(content, "ts_test")
    # Chỉ AMF được giữ lại
    assert len(terms) == 1
    assert terms[0].abbreviation == "AMF"


def test_term_extractor_extracts_definitions():
    """Pattern 'Term: long definition text'."""
    from kg_builder.builder import TermExtractor
    content = (
        "5G System: a 3GPP system consisting of 5GC and NG-RAN.\n"
        "Network Slice: logical network providing specific capabilities.\n"
    )
    terms = TermExtractor().extract_definitions(content, "ts_test")
    assert len(terms) == 2
    assert terms[0].abbreviation == "5G System"
    assert terms[0].term_type == "definition"
    assert "3GPP system" in terms[0].full_name


def test_term_extractor_skips_short_definitions():
    """Definition dưới 10 ký tự bị bỏ."""
    from kg_builder.builder import TermExtractor
    content = "AMF: short\n5G System: a 3GPP system consisting of 5GC and NG-RAN.\n"
    terms = TermExtractor().extract_definitions(content, "ts_test")
    # Chỉ definition dài đủ được giữ
    assert len(terms) == 1
    assert terms[0].abbreviation == "5G System"


def test_is_valid_abbreviation_edge_cases():
    """Validate độ dài, ký tự đầu, tỷ lệ uppercase/digits."""
    from kg_builder.builder import TermExtractor
    valid = TermExtractor._is_valid_abbreviation
    # Hợp lệ
    assert valid("AMF") is True
    assert valid("5G") is True
    assert valid("NG-RAN") is True
    # Không hợp lệ: rỗng / quá ngắn
    assert valid("") is False
    assert valid("A") is False
    # Không hợp lệ: ký tự đầu lowercase
    assert valid("amf") is False


def test_is_valid_definition_term_edge_cases():
    """Definition term phải có chữ và không phải reference [...]"""
    from kg_builder.builder import TermExtractor
    valid = TermExtractor._is_valid_definition_term
    assert valid("5G System") is True
    # Không hợp lệ: rỗng / quá ngắn
    assert valid("") is False
    assert valid("A") is False
    # Không hợp lệ: reference dạng [TS 23.501]
    assert valid("[TS 23.501]") is False
    # Không hợp lệ: chỉ ký tự số/đặc biệt
    assert valid("123") is False
    # Không hợp lệ: quá dài
    assert valid("x" * 101) is False


# ── SubjectClassifier — additional unit tests ────────────────────────────────

def test_subject_classifier_standards_specifications():
    """Section title chứa 'procedure' → STANDARDS_SPECIFICATIONS."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "procedure",
        "section_title": "Registration procedure",
        "content": "UE sends NAS message to AMF.",
        "spec_id": "ts_23_501",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.STANDARDS_SPECIFICATIONS


def test_subject_classifier_standards_overview():
    """Section title chứa 'architecture' → STANDARDS_OVERVIEW."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "System Architecture",
        "content": "The 5G system architecture is service-based.",
        "spec_id": "ts_23_501",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.STANDARDS_OVERVIEW


def test_subject_classifier_lexicon_via_section_title():
    """Section title chứa từ khoá lexicon → LEXICON (qua keyword fallback)."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "Glossary of terminology",
        "content": "",
        "spec_id": "ts_99_001",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.LEXICON


def test_subject_classifier_research_publications():
    """Content chứa nhiều research keywords → RESEARCH_PUBLICATIONS."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "Method",
        "content": "We propose a deep learning algorithm with neural network for optimization.",
        "spec_id": "research_paper",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.RESEARCH_PUBLICATIONS


def test_subject_classifier_research_overview():
    """Section title 'survey' → RESEARCH_OVERVIEW."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "Survey of 5G",
        "content": "A taxonomy of approaches.",
        "spec_id": "research_paper",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.RESEARCH_OVERVIEW


def test_subject_classifier_default_3gpp_spec():
    """spec_id match SPEC_PATTERN, không match keyword → STANDARDS_SPECIFICATIONS default."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "Random title",
        "content": "Plain text without keywords.",
        "spec_id": "TS 23.501",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.STANDARDS_SPECIFICATIONS
    assert result.confidence == 0.6


def test_subject_classifier_default_fallback():
    """Không match gì → RESEARCH_OVERVIEW default conf 0.5."""
    from kg_builder.builder import SubjectClassifier, Subject
    chunk = {
        "chunk_type": "general",
        "section_title": "Random",
        "content": "Plain content.",
        "spec_id": "no_pattern",
    }
    result = SubjectClassifier().classify_chunk(chunk)
    assert result.subject == Subject.RESEARCH_OVERVIEW
    assert result.confidence == 0.5


def test_subject_enum_has_five_values():
    """Đúng 5 Subject khớp benchmark TeleQnA."""
    from kg_builder.builder import Subject
    assert len(list(Subject)) == 5


# ── KGBuilder._merge_terms — 5G priority resolution ──────────────────────────

def test_merge_terms_first_write():
    """Term lần đầu xuất hiện → tạo entry mới."""
    from kg_builder.builder import KGBuilder, ExtractedTerm
    term_dict = {}
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Access and Mobility Management Function",
                       "abbreviation", "ts_23_501")],
    )
    assert "AMF" in term_dict
    assert term_dict["AMF"]["primary_spec"] == "ts_23_501"
    assert term_dict["AMF"]["source_specs"] == ["ts_23_501"]


def test_merge_terms_appends_source_specs():
    """Cùng abbr, spec khác → append vào source_specs."""
    from kg_builder.builder import KGBuilder, ExtractedTerm
    term_dict = {}
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Access and Mobility Management Function",
                       "abbreviation", "ts_23_501")],
    )
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Access and Mobility Management Function",
                       "abbreviation", "ts_29_500")],
    )
    assert set(term_dict["AMF"]["source_specs"]) == {"ts_23_501", "ts_29_500"}


def test_merge_terms_5g_beats_legacy():
    """Term từ legacy spec trước, 5G spec sau → primary_spec đổi sang 5G."""
    from kg_builder.builder import KGBuilder, ExtractedTerm
    term_dict = {}
    # Legacy spec (không thuộc 5G prefix)
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Old definition", "abbreviation", "ts_22_101")],
    )
    assert term_dict["AMF"]["primary_spec"] == "ts_22_101"

    # 5G spec đến sau, full_name khác → ghi đè
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Access and Mobility Management Function",
                       "abbreviation", "ts_23_501")],
    )
    assert term_dict["AMF"]["primary_spec"] == "ts_23_501"
    assert term_dict["AMF"]["full_name"] == "Access and Mobility Management Function"


def test_merge_terms_5g_vs_5g_tie_lexical():
    """Hai 5G spec cùng số citations → spec lexically smallest thắng."""
    from kg_builder.builder import KGBuilder, ExtractedTerm
    term_dict = {}
    # Spec lớn vào trước
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Definition B", "abbreviation", "ts_29_500")],
    )
    # Spec lexically nhỏ hơn vào sau với definition khác → thắng
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Definition A", "abbreviation", "ts_23_501")],
    )
    assert term_dict["AMF"]["primary_spec"] == "ts_23_501"
    assert term_dict["AMF"]["full_name"] == "Definition A"


def test_merge_terms_5g_more_citations_wins():
    """5G definition có nhiều citation hơn → primary chuyển sang nó."""
    from kg_builder.builder import KGBuilder, ExtractedTerm
    term_dict = {}
    # Definition A từ 1 spec 5G
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Definition A", "abbreviation", "ts_29_500")],
    )
    # Definition B từ 2 specs 5G khác nhau → nhiều citation hơn
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Definition B", "abbreviation", "ts_38_300")],
    )
    KGBuilder._merge_terms(
        term_dict,
        [ExtractedTerm("AMF", "Definition B", "abbreviation", "ts_24_501")],
    )
    assert term_dict["AMF"]["full_name"] == "Definition B"


# ── Integration tests — read-only queries trên KG thật ───────────────────────
# Tất cả tests dưới đây chỉ đọc, không ghi bất kỳ node/edge nào.

@requires_neo4j
def test_verify_connection(neo4j_driver):
    """Kiểm tra kết nối Neo4j."""
    with neo4j_driver.session() as s:
        result = s.run("RETURN 1 AS n").single()["n"]
    assert result == 1


@requires_neo4j
def test_node_counts(neo4j_driver):
    """KG phải có đủ 4 loại node với số lượng hợp lý."""
    with neo4j_driver.session() as s:
        docs    = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
        chunks  = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        terms   = s.run("MATCH (t:Term) RETURN count(t) AS c").single()["c"]
        subjects = s.run("MATCH (s:Subject) RETURN count(s) AS c").single()["c"]
    assert docs >= 1000,    f"Quá ít Document: {docs}"
    assert chunks >= 100000, f"Quá ít Chunk: {chunks}"
    assert terms >= 1000,   f"Quá ít Term: {terms}"
    assert subjects == 5,   f"Subject phải đúng 5, got {subjects}"


@requires_neo4j
def test_relationship_counts(neo4j_driver):
    """6 loại edge phải tồn tại với số lượng hợp lý."""
    with neo4j_driver.session() as s:
        def count(rel):
            return s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
        contains         = count("CONTAINS")
        parent_section   = count("PARENT_SECTION")
        has_subject      = count("HAS_SUBJECT")
        references_spec  = count("REFERENCES_SPEC")
        references_chunk = count("REFERENCES_CHUNK")
        defined_in       = count("DEFINED_IN")
    assert contains >= 100000,       f"CONTAINS quá ít: {contains}"
    assert parent_section >= 10000,  f"PARENT_SECTION quá ít: {parent_section}"
    assert has_subject >= 100000,    f"HAS_SUBJECT quá ít: {has_subject}"
    assert references_spec >= 1000,  f"REFERENCES_SPEC quá ít: {references_spec}"
    assert references_chunk >= 1000, f"REFERENCES_CHUNK quá ít: {references_chunk}"
    assert defined_in >= 100,        f"DEFINED_IN quá ít: {defined_in}"


@requires_neo4j
def test_all_chunks_embedded(neo4j_driver):
    """Mọi Chunk phải có embedding (Phase 3 embed hoàn thành)."""
    with neo4j_driver.session() as s:
        total    = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        embedded = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS c").single()["c"]
    assert total > 0
    assert total == embedded, f"Còn {total - embedded} chunk chưa có embedding"


@requires_neo4j
def test_chunk_properties_schema(neo4j_driver):
    """Spot-check 1 Chunk thật có đủ required properties."""
    with neo4j_driver.session() as s:
        row = s.run(
            "MATCH (c:Chunk) WHERE c.spec_id IS NOT NULL RETURN c LIMIT 1"
        ).single()
    assert row is not None
    chunk = row["c"]
    assert chunk.get("spec_id") is not None
    assert chunk.get("section_id") is not None
    assert chunk.get("content") is not None
    assert chunk.get("chunk_type") is not None


@requires_neo4j
def test_spec_id_no_dot_form(neo4j_driver):
    """Không có spec_id dạng dot (ts_23.501) — chỉ underscore (ts_23_501)."""
    with neo4j_driver.session() as s:
        count = s.run(
            "MATCH (n) WHERE n.spec_id =~ '.*\\.\\d+.*' RETURN count(n) AS c"
        ).single()["c"]
    assert count == 0, f"Tìm thấy {count} node có spec_id dạng dot"


@requires_neo4j
def test_references_chunk_has_is_external_property(neo4j_driver):
    """Mọi REFERENCES_CHUNK edge phải có property is_external (Option B)."""
    with neo4j_driver.session() as s:
        missing = s.run(
            "MATCH ()-[r:REFERENCES_CHUNK]->() WHERE r.is_external IS NULL RETURN count(r) AS c"
        ).single()["c"]
    assert missing == 0, f"{missing} REFERENCES_CHUNK thiếu is_external"


@requires_neo4j
def test_external_references_chunk_exist(neo4j_driver):
    """Phải có ít nhất 1 cross-spec REFERENCES_CHUNK (is_external=true)."""
    with neo4j_driver.session() as s:
        count = s.run(
            "MATCH ()-[r:REFERENCES_CHUNK {is_external: true}]->() RETURN count(r) AS c"
        ).single()["c"]
    assert count > 0, "Không có external REFERENCES_CHUNK nào — Option B build lỗi"


@requires_neo4j
def test_defined_in_points_to_document(neo4j_driver):
    """DEFINED_IN phải trỏ Term → Document (không phải Chunk)."""
    with neo4j_driver.session() as s:
        wrong = s.run(
            "MATCH (t:Term)-[:DEFINED_IN]->(c:Chunk) RETURN count(*) AS c"
        ).single()["c"]
        correct = s.run(
            "MATCH (t:Term)-[:DEFINED_IN]->(d:Document) RETURN count(*) AS c"
        ).single()["c"]
    assert wrong == 0,   f"DEFINED_IN trỏ vào Chunk: {wrong} edges sai schema"
    assert correct > 0,  "Không có DEFINED_IN → Document nào"


# ── REFERENCES_CHUNK external filter constants ────────────────────────────────

def test_ext_filter_constants_configured():
    """Verify filter cho external refs đặt đúng (theo feasibility test 04/2026)."""
    from kg_builder.builder import KGBuilder
    assert KGBuilder._EXT_ALLOWED_REF_TYPES == {"clause"}
    assert KGBuilder._EXT_CONFIDENCE_THRESHOLD == 0.7
