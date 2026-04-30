"""
Result fusion: Reciprocal Rank Fusion (RRF) + cross-encoder reranking.

Both stages dedup by normalised chunk_id so the KG-ingestion artefact where
`ts_23.288_4.1` and `ts_23_288_4.1` are stored as two separate nodes for the
same logical chunk doesn't burn slots in the final top_k.

`rerank` filters out chunks whose cross-encoder score falls below
`min_score` (default 0.0) — a negative score from the cross-encoder is the
model saying "this chunk is irrelevant to the query". Including such chunks
in the answer prompt encourages the LLM to fabricate from off-topic content.
When all candidates are negatively scored, the rerank returns an empty list,
which lets the answer prompt's grounding rule fire ("Context does not cover
…") instead of forcing a hallucinated answer.
"""
import os

from models import get_reranker_model


# Configurable floor on cross-encoder scores. 0.0 is the natural "model thinks
# query and chunk are unrelated" boundary for ms-marco-MiniLM style rerankers.
# Override via env if you want to keep marginally-negative chunks (e.g. for
# eval where you want to see what the model produces from weak evidence).
_DEFAULT_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))


# Normalise a chunk_id so logical duplicates (KG produced `ts_23.288_4.1` and
# `ts_23_288_4.1` for the same section) collapse to one entry in dedup checks.
def _norm_id(cid):
    return (cid or "").replace(".", "_") if cid else ""


def rrf_fusion(
    lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Formula: RRF(d) = Σ wᵢ / (k + rankᵢ(d))

    Dedups by normalised chunk_id so ts_23.288 / ts_23_288 variants collapse
    to one entry — without this, both formats accumulate separate RRF scores
    and both could land in the final top_k.
    """
    if weights is None:
        weights = [1.0] * len(lists)

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for ranked_list, weight in zip(lists, weights):
        for rank, chunk in enumerate(ranked_list):
            cid = _norm_id(chunk.get("chunk_id"))
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            if cid not in chunks:
                chunks[cid] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    result = []
    for cid in sorted_ids[:top_k]:
        c = dict(chunks[cid])
        c["rrf_score"] = scores[cid]
        result.append(c)
    return result


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 6,
    min_score: float | None = None,
) -> list[dict]:
    """
    Cross-encoder reranking with dedup + negative-score filter. Used by both
    fixed mode and react_agent fallback (when there's only 0-1 gap to fan out
    across).
    """
    return _rerank_with_dedup(query, chunks, top_k, min_score=min_score)


def rerank_per_gap(
    question: str,
    gaps: list[str],
    chunks: list[dict],
    total_top_k: int = 6,
    min_score: float | None = None,
) -> list[dict]:
    """
    Per-sub-question reranking for compound queries.

    The cross-encoder is run separately for EACH gap (sub-question) so chunks
    that focus on one specific gap aren't drowned out by chunks that match the
    full compound query superficially. We allocate slots round-robin across
    gaps, then backfill any remaining with overall reranking against the
    original question.

    Dedups by normalised chunk_id so the ts_23.288 / ts_23_288 KG duplicates
    don't burn two slots. Filters chunks below `min_score` per pass — a chunk
    can only fill a gap's slot if it scores positively against that gap.
    """
    if not chunks:
        return []
    if not gaps or len(gaps) <= 1:
        # Fall back to overall rerank when there's nothing to fan out across.
        return _rerank_with_dedup(question, chunks, total_top_k, min_score=min_score)

    # Per-gap rerank: each chunk gets a separate score against each gap, plus
    # an overall score against the original question (used for backfill).
    per_gap_ranked: list[list[dict]] = []
    for gap in gaps:
        scored = _rerank_with_dedup(gap, chunks, top_k=len(chunks), min_score=min_score)
        per_gap_ranked.append(scored)
    overall_ranked = _rerank_with_dedup(
        question, chunks, top_k=len(chunks), min_score=min_score
    )

    # Round-robin pick: take the next-best chunk from each gap in turn until
    # we fill total_top_k slots, deduping by normalised chunk_id.
    selected: list[dict] = []
    selected_keys: set[str] = set()
    cursors = [0] * len(gaps)
    while len(selected) < total_top_k:
        progressed = False
        for gi in range(len(gaps)):
            if len(selected) >= total_top_k:
                break
            ranked = per_gap_ranked[gi]
            # Advance cursor past any chunk already selected for another gap.
            while cursors[gi] < len(ranked):
                candidate = ranked[cursors[gi]]
                cursors[gi] += 1
                key = _norm_id(candidate.get("chunk_id"))
                if not key or key in selected_keys:
                    continue
                # Annotate which gap "won" this chunk for trail visibility.
                candidate["picked_for_gap"] = gaps[gi]
                selected.append(candidate)
                selected_keys.add(key)
                progressed = True
                break
        if not progressed:
            break

    # Backfill from overall ranking if we still have slots (e.g. some gap had
    # no positively-scored chunks left to contribute).
    if len(selected) < total_top_k:
        for c in overall_ranked:
            if len(selected) >= total_top_k:
                break
            key = _norm_id(c.get("chunk_id"))
            if not key or key in selected_keys:
                continue
            c["picked_for_gap"] = "(backfill)"
            selected.append(c)
            selected_keys.add(key)

    # Final ordering: by rerank_score. Note this score is whatever the LAST
    # rerank pass left on the chunk (the overall-question pass), since we
    # mutate in place — that's intentional, gives a consistent ranking for
    # display.
    selected.sort(key=lambda c: c.get("rerank_score") or 0.0, reverse=True)
    return selected


# Run a single rerank pass: dedup-by-normalised-chunk_id, score with the
# cross-encoder, drop chunks with score < min_score, return top_k.
#
# Returns NEW dict objects (shallow copies of input + rerank_score). Avoids
# mutating input chunks so multiple per-gap passes in `rerank_per_gap` don't
# cascade-overwrite each other's scores on the same dicts.
def _rerank_with_dedup(
    query: str,
    chunks: list[dict],
    top_k: int,
    min_score: float | None = None,
) -> list[dict]:
    if not chunks:
        return []
    floor = _DEFAULT_MIN_SCORE if min_score is None else min_score

    # Dedup input: when two chunks share a normalised chunk_id, keep the first.
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        k = _norm_id(c.get("chunk_id"))
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    model = get_reranker_model()
    pairs = [(query, c.get("content") or "") for c in deduped]
    scores = model.predict(pairs).tolist()

    # Build new dicts with rerank_score; drop those below floor. Cross-encoder
    # negative scores mean "irrelevant" — keeping them encourages fabrication.
    rescored: list[dict] = []
    for c, s in zip(deduped, scores):
        score = float(s)
        if score < floor:
            continue
        rescored.append({**c, "rerank_score": score})

    rescored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return rescored[:top_k]
