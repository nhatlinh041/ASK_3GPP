"""CLI to triage benchmark wrong-answer cases.

Two modes:

* **Single run** — read ``<run_dir>/results.json`` + ``<run_dir>/debug.log``,
  classify every wrong answer with rule codes, and write
  ``<run_dir>/failure_analysis.md``.

* **Diff** — same on two runs, then compare which questions flipped (regress /
  recover), how the Cypher pattern distribution shifted, and which questions
  improved their graph hit but stayed wrong. Output goes to
  ``<run_b>/diff_vs_<run_a_basename>.md``.

This module is intentionally dependency-light (stdlib only) so it can run on
any checkout without spinning up Neo4j or the rag-engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script (`python tests/benchmark/analyze_failures.py …`)
# without the package being on PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.benchmark._debug_log_parser import (  # noqa: E402
        QuestionTrace,
        parse_debug_log,
    )
    from tests.benchmark._failure_classifier import (  # noqa: E402
        RULE_DESCRIPTIONS,
        classify_all,
    )
    from tests.benchmark._kg_lookup import from_env as kg_from_env  # noqa: E402
else:
    from ._debug_log_parser import QuestionTrace, parse_debug_log
    from ._failure_classifier import RULE_DESCRIPTIONS, classify_all
    from ._kg_lookup import from_env as kg_from_env


@dataclass
class RunData:
    name: str
    path: Path
    metadata: dict
    overall: dict
    results: list[dict]
    traces: list[QuestionTrace]
    labels: dict[int, list[str]]

    def by_idx(self) -> dict[int, QuestionTrace]:
        return {t.idx: t for t in self.traces}


# Single source of truth for ordering rule codes in tables. OUT_OF_DOMAIN
# (truly outside 3GPP) and GOLD_PARAPHRASED (in-domain but gold worded
# differently) lead the list because they're the first triage axis: "is the
# question even answerable from 3GPP, and if so is the gold text reachable?"
# Everything below is conditional on the answer being yes.
RULE_ORDER = [
    "OUT_OF_DOMAIN",
    "GOLD_PARAPHRASED",
    "REFUSAL",
    "EXTRACT_FUZZY",
    "GRAPH_ERROR",
    "GRAPH_PATTERN_C",
    "GRAPH_ZERO_ROW",
    "EMPTY_CONTEXT_GUESS",
    "WEAK_RERANK",
    "RETRIEVAL_NO_GOLD_TEXT",
    "RETRIEVAL_OK_LLM_WRONG",
    "UNKNOWN",
]


def load_run(run_dir: Path, kg_lookup=None, kg_topic_check=None) -> RunData:
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    results_path = run_dir / "results.json"
    debug_path = run_dir / "debug.log"
    if not results_path.exists():
        raise SystemExit(f"Missing results.json in {run_dir}")
    if not debug_path.exists():
        raise SystemExit(f"Missing debug.log in {run_dir}")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    overall = payload.get("overall") or {}
    results = payload.get("detailed_results") or []
    traces = list(parse_debug_log(debug_path))
    labels = classify_all(
        traces,
        results,
        kg_lookup=kg_lookup,
        kg_topic_check=kg_topic_check,
    )

    if len(traces) != len(results):
        # Best-effort warning. The classifier already pads with empty dicts.
        print(
            f"  [warn] {run_dir.name}: {len(traces)} traces vs {len(results)} results — alignment assumed by 1-based idx.",
            file=sys.stderr,
        )

    return RunData(
        name=run_dir.name,
        path=run_dir,
        metadata=metadata,
        overall=overall,
        results=results,
        traces=traces,
        labels=labels,
    )


def _question_fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()[:12]


def _rule_count(labels: dict[int, list[str]]) -> Counter:
    counts: Counter = Counter()
    for rules in labels.values():
        for r in rules:
            counts[r] += 1
    return counts


def _category_breakdown(run: RunData) -> dict[str, dict]:
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "rules": Counter()}
    )
    for trace in run.traces:
        cat = trace.category
        by_cat[cat]["total"] += 1
        if trace.correct:
            by_cat[cat]["correct"] += 1
        for rule in run.labels.get(trace.idx, []):
            by_cat[cat]["rules"][rule] += 1
    return by_cat


# ---------- single-run report ------------------------------------------------


def _section_summary(run: RunData) -> list[str]:
    md, ov = run.metadata, run.overall
    total = ov.get("total_questions", len(run.traces))
    correct = ov.get("correct_answers", 0)
    lines = [
        f"# Failure analysis — {run.name}",
        "",
        f"- **Mode**: {md.get('mode', '?')}    **Model**: {md.get('model', '?')}    **Think**: {md.get('think', '?')}",
        f"- **Total**: {total}    **Correct**: {correct}    **Accuracy**: {ov.get('accuracy', '?')}%",
        f"- **Errors**: {ov.get('total_errors', 0)}    **Extraction failures**: {ov.get('extraction_failures', '?')}",
        f"- **Avg latency**: {ov.get('avg_latency_ms', '?')} ms",
        f"- **Graph search succeeded**: {ov.get('graph_search_success', '?')}/{total}    **Graph errors**: {ov.get('graph_search_errors', 0)}",
    ]

    # Effective accuracy = đúng / (tổng - câu OUT_OF_DOMAIN strict). The
    # strict OOD bucket only fires when both gold and topic are missing from
    # the KG; GOLD_PARAPHRASED is reported separately because those are still
    # in-domain (fixable by better retrieval/rerank).
    ood = sum(1 for rules in run.labels.values() if "OUT_OF_DOMAIN" in rules)
    paraphrased = sum(1 for rules in run.labels.values() if "GOLD_PARAPHRASED" in rules)
    if (ood or paraphrased) and isinstance(total, int) and isinstance(correct, int) and total > ood:
        eff = 100 * correct / (total - ood)
        lines.append(
            f"- **OUT_OF_DOMAIN**: {ood}    **GOLD_PARAPHRASED** (in-domain, gold paraphrased): {paraphrased}"
        )
        lines.append(
            f"- **Effective accuracy** (chỉ loại OOD): "
            f"{correct}/{total - ood} = **{eff:.1f}%**"
        )

    lines.append("")
    return lines


def _section_root_causes(run: RunData) -> list[str]:
    wrong_idxs = [i for i, r in run.labels.items() if r]
    counts = _rule_count(run.labels)
    total_wrong = len(wrong_idxs) or 1

    lines = [
        "## Root-cause histogram",
        "",
        "Mỗi câu sai có thể được gán nhiều rule (multi-label). `count` = số câu sai khớp rule, `% wrong` chia trên tổng câu sai.",
        "",
        "| Rule | Count | % wrong | Mô tả |",
        "|---|---:|---:|---|",
    ]
    for rule in RULE_ORDER:
        c = counts.get(rule, 0)
        if c == 0:
            continue
        lines.append(
            f"| `{rule}` | {c} | {100 * c / total_wrong:.1f}% | {RULE_DESCRIPTIONS[rule]} |"
        )
    lines.append("")
    return lines


def _section_category_table(run: RunData) -> list[str]:
    by_cat = _category_breakdown(run)
    rule_present = {
        rule for v in by_cat.values() for rule in v["rules"] if v["rules"][rule] > 0
    }
    cols = [r for r in RULE_ORDER if r in rule_present]
    headers = ["Category", "Q", "Correct", "Acc"] + cols
    lines = [
        "## Category × root-cause",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for cat, info in sorted(by_cat.items(), key=lambda kv: -kv[1]["total"]):
        acc = 100 * info["correct"] / max(1, info["total"])
        row = [cat, str(info["total"]), str(info["correct"]), f"{acc:.1f}%"]
        for rule in cols:
            row.append(str(info["rules"].get(rule, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _section_examples(run: RunData, max_per_rule: int = 5) -> list[str]:
    by_idx = run.by_idx()
    rule_to_idxs: dict[str, list[int]] = defaultdict(list)
    for idx, rules in run.labels.items():
        for r in rules:
            rule_to_idxs[r].append(idx)

    lines = ["## Ví dụ theo từng rule", ""]
    for rule in RULE_ORDER:
        idxs = rule_to_idxs.get(rule) or []
        if not idxs:
            continue
        lines.append(f"### `{rule}` — {RULE_DESCRIPTIONS[rule]}")
        lines.append("")
        for idx in idxs[:max_per_rule]:
            t = by_idx.get(idx)
            r = run.results[idx - 1] if 0 <= idx - 1 < len(run.results) else {}
            if t is None or not r:
                continue
            top_score = max(t.rerank_top_scores) if t.rerank_top_scores else 0.0
            q = (r.get("question") or "").splitlines()[0][:90]
            gold_text = (r.get("expected_answer") or "")[:100]
            pred_choice = "None" if t.pred is None else chr(ord("A") + t.pred)
            gold_choice = chr(ord("A") + t.gold) if t.gold >= 0 else "?"
            resolved = ", ".join(t.resolved_terms[:4]) if t.resolved_terms else "—"
            lines.append(
                f"- **#{idx}** · {t.category} · pred={pred_choice} gold={gold_choice} · "
                f"intent={t.intent} · pattern={t.cypher_pattern} · graph={t.graph_count} · "
                f"src={len(t.sources)} · top={top_score:.2f}"
            )
            lines.append(f"  - Q: {q}")
            lines.append(f"  - gold: {gold_text!r}")
            lines.append(f"  - terms: {resolved}")
        lines.append("")
    return lines


def _section_pattern_samples(run: RunData) -> list[str]:
    """Inline a Cypher sample for each pattern class — useful to spot
    spurious A1 matches like ``MATCH (t:Term {abbreviation: 'What'})``."""
    by_pattern: dict[str, list[QuestionTrace]] = defaultdict(list)
    for t in run.traces:
        if t.cypher and t.cypher_pattern:
            by_pattern[t.cypher_pattern].append(t)

    lines = ["## Cypher samples theo pattern", ""]
    for pat in ["A1", "A2", "B", "C", "unknown"]:
        items = by_pattern.get(pat) or []
        if not items:
            continue
        lines.append(f"### Pattern {pat} ({len(items)} câu)")
        lines.append("")
        for t in items[:2]:
            lines.append(f"- #{t.idx} ({t.category}) intent={t.intent} graph_count={t.graph_count} resolved={t.resolved_terms or '—'}")
            cypher = (t.cypher or "").strip()
            if len(cypher) > 700:
                cypher = cypher[:700] + " …"
            lines.append("```cypher")
            lines.append(cypher)
            lines.append("```")
        lines.append("")
    return lines


def render_single_report(run: RunData) -> str:
    parts: list[str] = []
    parts += _section_summary(run)
    parts += _section_root_causes(run)
    parts += _section_category_table(run)
    parts += _section_examples(run)
    parts += _section_pattern_samples(run)
    return "\n".join(parts)


# ---------- diff report ------------------------------------------------------


def _align_runs(a: RunData, b: RunData) -> list[tuple[int, int]]:
    """Return [(idx_a, idx_b)] pairs for questions present in both runs.

    Matches by question-text fingerprint. If question lists are identical
    length and order, this collapses to a 1:1 zip.
    """
    fp_a = {_question_fingerprint(r.get("question", "")): r for r in a.results}
    pairs: list[tuple[int, int]] = []
    for j, rb in enumerate(b.results, start=1):
        fp = _question_fingerprint(rb.get("question", ""))
        ra = fp_a.get(fp)
        if ra is None:
            continue
        # Locate idx_a from index in a.results
        try:
            idx_a = a.results.index(ra) + 1
        except ValueError:
            continue
        pairs.append((idx_a, j))
    return pairs


def _section_diff_summary(a: RunData, b: RunData, pairs: list[tuple[int, int]]) -> list[str]:
    sa, sb = a.overall or {}, b.overall or {}
    lines = [
        f"# Diff — {b.name}  vs  {a.name}",
        "",
        f"- Common questions matched by fingerprint: **{len(pairs)}**",
        "",
        "| Metric | A: " + a.name + " | B: " + b.name + " | Δ |",
        "|---|---:|---:|---:|",
    ]

    def _row(label: str, key: str, fmt: str = "{}") -> str:
        va, vb = sa.get(key), sb.get(key)
        try:
            delta = (vb or 0) - (va or 0)
            return f"| {label} | {fmt.format(va)} | {fmt.format(vb)} | {fmt.format(delta)} |"
        except Exception:
            return f"| {label} | {va} | {vb} | — |"

    lines.append(_row("Total", "total_questions"))
    lines.append(_row("Correct", "correct_answers"))
    lines.append(_row("Accuracy %", "accuracy", "{:.1f}"))
    lines.append(_row("Avg latency ms", "avg_latency_ms", "{:.0f}"))
    lines.append(_row("Graph hit", "graph_search_success"))
    lines.append(_row("Graph errors", "graph_search_errors"))
    lines.append(_row("Extraction fails", "extraction_failures"))
    lines.append("")
    return lines


def _section_flips(a: RunData, b: RunData, pairs: list[tuple[int, int]]) -> list[str]:
    by_a, by_b = a.by_idx(), b.by_idx()
    regressions: list[tuple[int, int]] = []  # (idx_a, idx_b) correct→wrong
    gains: list[tuple[int, int]] = []  # wrong→correct
    persistent_wrong: list[tuple[int, int]] = []  # wrong in both

    for ia, ib in pairs:
        ta, tb = by_a.get(ia), by_b.get(ib)
        if ta is None or tb is None:
            continue
        if ta.correct and not tb.correct:
            regressions.append((ia, ib))
        elif not ta.correct and tb.correct:
            gains.append((ia, ib))
        elif not ta.correct and not tb.correct:
            persistent_wrong.append((ia, ib))

    lines = [
        "## Flips",
        "",
        f"- Regressions (A correct → B wrong): **{len(regressions)}**",
        f"- Gains (A wrong → B correct): **{len(gains)}**",
        f"- Persistent wrong (both wrong): **{len(persistent_wrong)}**",
        "",
    ]

    def _flip_table(title: str, items: list[tuple[int, int]], src: str, max_rows: int = 15) -> None:
        if not items:
            return
        lines.append(f"### {title} (top {min(len(items), max_rows)})")
        lines.append("")
        lines.append(
            "| #A → #B | Category | Q (head) | Pat A → Pat B | Graph A → B | Src A → B | Top A → B | Rules (B) |"
        )
        lines.append("|---|---|---|---|---:|---:|---|---|")
        for ia, ib in items[:max_rows]:
            ta, tb = by_a[ia], by_b[ib]
            res = (a.results[ia - 1] if src == "a" else b.results[ib - 1])
            q = (res.get("question") or "").splitlines()[0][:60]
            top_a = max(ta.rerank_top_scores) if ta.rerank_top_scores else 0.0
            top_b = max(tb.rerank_top_scores) if tb.rerank_top_scores else 0.0
            rules = ",".join(b.labels.get(ib, [])) or "—"
            lines.append(
                f"| {ia} → {ib} | {tb.category} | {q} | {ta.cypher_pattern} → {tb.cypher_pattern} | "
                f"{ta.graph_count} → {tb.graph_count} | {len(ta.sources)} → {len(tb.sources)} | "
                f"{top_a:.2f} → {top_b:.2f} | {rules} |"
            )
        lines.append("")

    _flip_table("Regressions", regressions, src="b")
    _flip_table("Gains", gains, src="b")

    # Sub-finding: persistent wrong but B improved retrieval. Helps confirm
    # whether better graph hits actually translate into better answers.
    improved_no_help: list[tuple[int, int]] = [
        (ia, ib)
        for ia, ib in persistent_wrong
        if (by_b[ib].graph_count > by_a[ia].graph_count) or (len(by_b[ib].sources) > len(by_a[ia].sources))
    ]
    if improved_no_help:
        lines.append(f"### Persistent wrong, but B retrieved more chunks ({len(improved_no_help)})")
        lines.append("")
        lines.append("| #A → #B | Category | Graph A→B | Src A→B | Top A→B |")
        lines.append("|---|---|---:|---:|---|")
        for ia, ib in improved_no_help[:15]:
            ta, tb = by_a[ia], by_b[ib]
            top_a = max(ta.rerank_top_scores) if ta.rerank_top_scores else 0.0
            top_b = max(tb.rerank_top_scores) if tb.rerank_top_scores else 0.0
            lines.append(
                f"| {ia} → {ib} | {tb.category} | {ta.graph_count} → {tb.graph_count} | "
                f"{len(ta.sources)} → {len(tb.sources)} | {top_a:.2f} → {top_b:.2f} |"
            )
        lines.append("")
    return lines


def _section_stage_diff(a: RunData, b: RunData, pairs: list[tuple[int, int]]) -> list[str]:
    by_a, by_b = a.by_idx(), b.by_idx()
    pat_a, pat_b = Counter(), Counter()
    intent_a, intent_b = Counter(), Counter()
    top_a_vals: list[float] = []
    top_b_vals: list[float] = []
    for ia, ib in pairs:
        ta, tb = by_a.get(ia), by_b.get(ib)
        if not (ta and tb):
            continue
        pat_a[ta.cypher_pattern] += 1
        pat_b[tb.cypher_pattern] += 1
        intent_a[ta.intent] += 1
        intent_b[tb.intent] += 1
        if ta.rerank_top_scores:
            top_a_vals.append(max(ta.rerank_top_scores))
        if tb.rerank_top_scores:
            top_b_vals.append(max(tb.rerank_top_scores))

    lines = ["## Stage-level distribution", ""]
    lines.append("### Cypher pattern")
    lines.append("")
    lines.append("| Pattern | A | B | Δ |")
    lines.append("|---|---:|---:|---:|")
    keys = sorted(set(pat_a) | set(pat_b))
    for k in keys:
        lines.append(f"| {k} | {pat_a[k]} | {pat_b[k]} | {pat_b[k] - pat_a[k]:+d} |")
    lines.append("")

    lines.append("### Intent")
    lines.append("")
    lines.append("| Intent | A | B | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in sorted(set(intent_a) | set(intent_b), key=lambda x: -(intent_a[x] + intent_b[x])):
        lines.append(f"| {k} | {intent_a[k]} | {intent_b[k]} | {intent_b[k] - intent_a[k]:+d} |")
    lines.append("")

    if top_a_vals and top_b_vals:
        lines.append("### Top reranked score (per question with sources)")
        lines.append("")
        lines.append(
            f"- A: count={len(top_a_vals)}  mean={statistics.mean(top_a_vals):.3f}  median={statistics.median(top_a_vals):.3f}"
        )
        lines.append(
            f"- B: count={len(top_b_vals)}  mean={statistics.mean(top_b_vals):.3f}  median={statistics.median(top_b_vals):.3f}"
        )
        lines.append("")
    return lines


def _section_rule_diff(a: RunData, b: RunData) -> list[str]:
    counts_a = _rule_count(a.labels)
    counts_b = _rule_count(b.labels)
    lines = ["## Root-cause distribution (toàn bộ câu sai trong từng run)", ""]
    lines.append("| Rule | A wrong | B wrong | Δ |")
    lines.append("|---|---:|---:|---:|")
    for rule in RULE_ORDER:
        ca, cb = counts_a.get(rule, 0), counts_b.get(rule, 0)
        if ca == 0 and cb == 0:
            continue
        lines.append(f"| `{rule}` | {ca} | {cb} | {cb - ca:+d} |")
    lines.append("")
    return lines


def render_diff_report(a: RunData, b: RunData) -> str:
    pairs = _align_runs(a, b)
    parts: list[str] = []
    parts += _section_diff_summary(a, b, pairs)
    parts += _section_rule_diff(a, b)
    parts += _section_flips(a, b, pairs)
    parts += _section_stage_diff(a, b, pairs)
    return "\n".join(parts)


# ---------- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Triage benchmark wrong-answers (single-run or diff)."
    )
    p.add_argument("run", type=Path, help="Run directory (must contain results.json + debug.log).")
    p.add_argument(
        "--diff",
        type=Path,
        default=None,
        metavar="OTHER_RUN",
        help="Compare RUN against OTHER_RUN. RUN is treated as run B (the newer one).",
    )
    p.add_argument(
        "--kg-lookup",
        action="store_true",
        help="Probe the live Neo4j KG for each gold answer to detect OUT_OF_DOMAIN questions (requires Neo4j running and .env credentials).",
    )
    p.add_argument(
        "-o", "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: <run>/failure_analysis.md or <run>/diff_vs_<other>.md).",
    )
    args = p.parse_args(argv)

    kg = None
    if args.kg_lookup:
        kg = kg_from_env()
        if kg is None:
            print("  [warn] --kg-lookup requested but KG unreachable; OUT_OF_DOMAIN/GOLD_PARAPHRASED will be skipped", file=sys.stderr)
    kg_gold = kg.has_content if kg is not None else None
    kg_topic = kg.has_topic_in_kg if kg is not None else None

    try:
        run_b = load_run(args.run, kg_lookup=kg_gold, kg_topic_check=kg_topic)
        print(f"Loaded {run_b.name}: {len(run_b.traces)} traces, {sum(1 for r in run_b.labels.values() if r)} wrong")

        if args.diff is None:
            out = args.out or (run_b.path / "failure_analysis.md")
            out.write_text(render_single_report(run_b), encoding="utf-8")
            print(f"Wrote {out}")
            return 0

        run_a = load_run(args.diff, kg_lookup=kg_gold, kg_topic_check=kg_topic)
        print(f"Loaded {run_a.name}: {len(run_a.traces)} traces, {sum(1 for r in run_a.labels.values() if r)} wrong")
        out = args.out or (run_b.path / f"diff_vs_{run_a.name}.md")
        out.write_text(render_diff_report(run_a, run_b), encoding="utf-8")
        print(f"Wrote {out}")
        return 0
    finally:
        if kg is not None:
            kg.close()


if __name__ == "__main__":
    raise SystemExit(main())
