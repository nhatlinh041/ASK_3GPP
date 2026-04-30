"""
Retrieval evaluation metrics: Precision@k, Recall@k, MRR, nDCG.
All functions accept ranked list of retrieved IDs and a set of relevant IDs.
"""
import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document; 0 if none found."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr(results: list[tuple[list[str], set[str]]]) -> float:
    """Mean Reciprocal Rank over a set of queries."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized DCG at k — DCG / ideal DCG."""
    ideal = sorted([1 if r in relevant else 0 for r in retrieved[:k]], reverse=True)
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg


def evaluate_batch(
    queries: list[dict],
    k: int = 10,
) -> dict[str, float]:
    """
    Run all metrics over a batch of queries.
    Each query dict: { 'retrieved': [chunk_id, ...], 'relevant': [chunk_id, ...] }
    Returns averaged metrics.
    """
    p_scores, r_scores, mrr_pairs, ndcg_scores = [], [], [], []

    for q in queries:
        retrieved = q["retrieved"]
        relevant = set(q["relevant"])
        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        mrr_pairs.append((retrieved, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))

    return {
        f"precision@{k}": sum(p_scores) / len(p_scores),
        f"recall@{k}": sum(r_scores) / len(r_scores),
        "mrr": mrr(mrr_pairs),
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
    }
