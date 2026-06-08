"""
Unit tests for Phase C score blending in fusion.py.

Tests don't need Neo4j — they monkey-patch the cross-encoder via
`get_reranker_model()` so we can fully control the rerank logits and verify:
  - Blending math (α·sigmoid + β·upstream + γ·canonical)
  - Floor bypass for canonical chunks (logit < 0 doesn't drop them)
  - Reserved slot guarantees top canonical inclusion in top_k
  - rerank_per_gap sorts by final_score
"""
import math
import sys
from pathlib import Path

import pytest

# Add rag-engine to sys.path so `from retrieval...` resolves.
RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
sys.path.insert(0, str(RAG_ENGINE_DIR))

import retrieval.fusion as fusion  # noqa: E402


class _FakeReranker:
    """Replaces the real cross-encoder for tests. Returns predetermined scores
    by checking whether each map key appears as a substring of the content
    fed to the cross-encoder. Substring lookup so tests remain stable even
    when fusion._augment_for_rerank prepends section_title to content."""

    def __init__(self, score_map: dict[str, float]):
        self._map = score_map

    def predict(self, pairs):
        import numpy as np
        out = []
        for _, content in pairs:
            score = 0.0
            for key, value in self._map.items():
                if key in content:
                    score = value
                    break
            out.append(score)
        return np.array(out)


@pytest.fixture
def patch_reranker(monkeypatch):
    """Returns a setter that swaps in a fake cross-encoder for one test."""

    def _set(score_map: dict[str, float]):
        monkeypatch.setattr(fusion, "get_reranker_model", lambda: _FakeReranker(score_map))

    return _set


# ---------- _sigmoid --------------------------------------------------------


class TestSigmoid:
    def test_zero(self):
        assert abs(fusion._sigmoid(0.0) - 0.5) < 1e-9

    def test_clip_high(self):
        assert fusion._sigmoid(50.0) == 1.0

    def test_clip_low(self):
        assert fusion._sigmoid(-50.0) == 0.0

    def test_typical_logit(self):
        # σ(3.140) ≈ 0.958 — used as Q2 ts_23_434 example in the plan
        assert abs(fusion._sigmoid(3.140) - 0.958) < 0.01


# ---------- _rerank_with_dedup ---------------------------------------------


class TestRerankWithDedup:
    def test_blends_score_correctly(self, patch_reranker):
        # logit=2.0 → sigmoid≈0.881 ; upstream=0.9 ; canonical=0
        # final = 0.6·0.881 + 0.3·0.9 + 0 = 0.799
        patch_reranker({"chunk_a": 2.0})
        chunks = [{
            "chunk_id": "c1", "content": "chunk_a", "spec_id": "ts_x",
            "section": "S", "score": 0.9,
        }]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        assert len(out) == 1
        assert out[0]["rerank_score"] == 2.0
        expected = 0.6 * fusion._sigmoid(2.0) + 0.3 * 0.9 + 0.0
        assert abs(out[0]["final_score"] - expected) < 1e-6

    def test_canonical_bonus_fires_at_threshold(self, patch_reranker):
        # upstream=1.0 (Pattern A canonical) → +γ·1.0 bonus
        patch_reranker({"canon": 1.0, "vec": 1.0})
        chunks = [
            {"chunk_id": "v1", "content": "vec",   "score": 0.95},  # canonical THRESHOLD strict 1.0 — not canonical
            {"chunk_id": "c1", "content": "canon", "score": 1.0},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=2)
        # Both have same logit but canonical gets +γ·1.0 → ranks first.
        assert out[0]["chunk_id"] == "c1"
        assert out[1]["chunk_id"] == "v1"

    def test_floor_drops_negative_non_canonical(self, patch_reranker):
        # logit -5.0, upstream 0.5 → dropped by floor (default 0.0)
        patch_reranker({"a": -5.0, "b": 1.0})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.5},
            {"chunk_id": "b", "content": "b", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        ids = [c["chunk_id"] for c in out]
        assert ids == ["b"]  # 'a' filtered by floor

    def test_canonical_survives_mild_negative_logit(self, patch_reranker):
        # Canonical with logit in [-2, 0) — cross-encoder uncertain — kept.
        patch_reranker({"canon": -1.5, "noise": 1.0})
        chunks = [
            {"chunk_id": "canon", "content": "canon", "score": 1.0},
            {"chunk_id": "noise", "content": "noise", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        ids = [c["chunk_id"] for c in out]
        assert "canon" in ids

    def test_canonical_dropped_when_strongly_negative(self, patch_reranker):
        # Pattern B over-match scenario: canonical chunk (upstream=1.0) BUT
        # cross-encoder logit < -2 → cross-encoder confident it's irrelevant,
        # drop. Prevents short-identifier regex matches (SA, N6, UE) from
        # polluting top_k just because section_title contains the identifier.
        patch_reranker({"canon": -4.88, "noise": 1.0})
        chunks = [
            {"chunk_id": "canon", "content": "canon", "score": 1.0},
            {"chunk_id": "noise", "content": "noise", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        ids = [c["chunk_id"] for c in out]
        assert "canon" not in ids
        assert "noise" in ids

    def test_no_reserved_slot_canonical_can_be_displaced(self, patch_reranker):
        # When 6 vector chunks have high logits and canonical has slightly
        # negative logit, blended sort puts canonical at #6 or below — no
        # forced inclusion (Phương án 2 removed reserved slot).
        score_map = {f"v{i}": 6.0 for i in range(6)}
        score_map["canon"] = -1.5  # mild negative — survives floor bypass
        patch_reranker(score_map)
        chunks = [{"chunk_id": f"v{i}", "content": f"v{i}", "score": 0.93} for i in range(6)]
        chunks.append({"chunk_id": "canon", "content": "canon", "score": 1.0})
        out = fusion._rerank_with_dedup("q", chunks, top_k=6)
        # Canonical's final = 0.6·sig(-1.5) + 0.3 + 0.15 = 0.6·0.182 + 0.45 = 0.559
        # v_i final = 0.6·sig(6) + 0.3·0.93 + 0 = 0.598 + 0.279 = 0.877
        # All 6 v_i beat canonical → canonical excluded.
        ids = {c["chunk_id"] for c in out}
        assert "canon" not in ids
        assert len(out) == 6

    def test_canonical_promoted_when_blend_wins(self, patch_reranker):
        # When canonical ranks #1 by blend math (positive logit + bonus),
        # it lands at top by sort, no special slot needed.
        patch_reranker({"canon": 5.0, "v1": 0.0, "v2": 0.0})
        chunks = [
            {"chunk_id": "v1",    "content": "v1",    "score": 0.5},
            {"chunk_id": "canon", "content": "canon", "score": 1.0},
            {"chunk_id": "v2",    "content": "v2",    "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=3)
        assert out[0]["chunk_id"] == "canon"

    def test_no_canonical_uses_pure_blend_sort(self, patch_reranker):
        # No upstream==1.0 — sort purely by final_score, no reserved slot logic.
        patch_reranker({"a": 3.0, "b": 1.0, "c": 0.5})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.9},
            {"chunk_id": "b", "content": "b", "score": 0.9},
            {"chunk_id": "c", "content": "c", "score": 0.9},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=3)
        ids = [c["chunk_id"] for c in out]
        assert ids == ["a", "b", "c"]  # logit-ordered

    def test_empty_input(self, patch_reranker):
        patch_reranker({})
        assert fusion._rerank_with_dedup("q", [], top_k=5) == []

    def test_dedup_preserves_first(self, patch_reranker):
        # Same normalised chunk_id ("ts_23.288_4.1" vs "ts_23_288_4.1") collapses.
        patch_reranker({"first": 2.0, "dup": 5.0})
        chunks = [
            {"chunk_id": "ts_23.288_4.1", "content": "first", "score": 0.9},
            {"chunk_id": "ts_23_288_4.1", "content": "dup",   "score": 0.9},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "ts_23.288_4.1"


# ---------- RERANK_MIN_KEEP soft floor bypass -------------------------------


class TestMinKeepBypass:
    """Soft floor bypass: when the strict floor drops everything (or almost
    everything), RERANK_MIN_KEEP > 0 supplements from the best-rejected
    chunks so the answer LLM is never handed empty context."""

    def test_default_zero_keeps_strict_floor(self, monkeypatch, patch_reranker):
        # MIN_KEEP=0 (default) — strict floor still drops everything if all
        # logits are below it. Same as legacy behaviour.
        monkeypatch.setattr(fusion, "_RERANK_MIN_KEEP", 0)
        patch_reranker({"a": -3.0, "b": -3.0})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.5},
            {"chunk_id": "b", "content": "b", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        assert out == []

    def test_min_keep_supplements_from_dropped(self, monkeypatch, patch_reranker):
        # MIN_KEEP=2, strict floor would drop both chunks; bypass keeps the
        # two best-by-final_score even though logits are below floor.
        monkeypatch.setattr(fusion, "_RERANK_MIN_KEEP", 2)
        patch_reranker({"a": -2.5, "b": -3.5})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.7},  # higher upstream
            {"chunk_id": "b", "content": "b", "score": 0.4},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        ids = [c["chunk_id"] for c in out]
        assert len(out) == 2
        # 'a' wins on blended score because its upstream is higher and its
        # logit is less negative (sigmoid(-2.5) > sigmoid(-3.5)).
        assert ids == ["a", "b"]

    def test_min_keep_does_not_displace_passed(self, monkeypatch, patch_reranker):
        # MIN_KEEP=3, but only 1 chunk passes floor — bypass tops up with
        # the next-best rejected chunks while keeping the passed one.
        monkeypatch.setattr(fusion, "_RERANK_MIN_KEEP", 3)
        patch_reranker({"good": 4.0, "ok": -1.5, "bad": -4.0})
        chunks = [
            {"chunk_id": "good", "content": "good", "score": 0.5},
            {"chunk_id": "ok",   "content": "ok",   "score": 0.5},
            {"chunk_id": "bad",  "content": "bad",  "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        ids = [c["chunk_id"] for c in out]
        assert len(out) == 3
        # 'good' on top by final_score; 'ok' and 'bad' supplemented.
        assert ids[0] == "good"
        assert set(ids) == {"good", "ok", "bad"}

    def test_min_keep_capped_by_top_k(self, monkeypatch, patch_reranker):
        # top_k=2 < MIN_KEEP=4 — final slice still trims to top_k.
        monkeypatch.setattr(fusion, "_RERANK_MIN_KEEP", 4)
        patch_reranker({"a": -2.0, "b": -2.5, "c": -3.0, "d": -3.5})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.5},
            {"chunk_id": "b", "content": "b", "score": 0.5},
            {"chunk_id": "c", "content": "c", "score": 0.5},
            {"chunk_id": "d", "content": "d", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=2)
        assert len(out) == 2

    def test_min_keep_with_empty_dropped(self, monkeypatch, patch_reranker):
        # All chunks pass the floor — no bypass needed; behaviour identical
        # to MIN_KEEP=0.
        monkeypatch.setattr(fusion, "_RERANK_MIN_KEEP", 5)
        patch_reranker({"a": 2.0, "b": 1.0})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.5},
            {"chunk_id": "b", "content": "b", "score": 0.5},
        ]
        out = fusion._rerank_with_dedup("q", chunks, top_k=5)
        assert len(out) == 2
        assert [c["chunk_id"] for c in out] == ["a", "b"]


# ---------- rerank_per_gap sort key ----------------------------------------


class TestRerankPerGapSortKey:
    def test_sorts_by_final_score(self, patch_reranker):
        patch_reranker({"a": 1.0, "b": 2.0, "c": 3.0})
        chunks = [
            {"chunk_id": "a", "content": "a", "score": 0.9},
            {"chunk_id": "b", "content": "b", "score": 0.9},
            {"chunk_id": "c", "content": "c", "score": 0.9},
        ]
        # Two gaps → exercises the per-gap path.
        out = fusion.rerank_per_gap(
            question="overall", gaps=["gap1", "gap2"], chunks=chunks, total_top_k=3,
        )
        assert len(out) == 3
        # final_score must be monotonically decreasing
        finals = [c.get("final_score") or 0.0 for c in out]
        assert finals == sorted(finals, reverse=True)
