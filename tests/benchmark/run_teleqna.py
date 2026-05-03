"""
TeleQnA Benchmark Runner for the 3GPP Demo backend.

Mirrors the parent project's `tests/benchmark/run_tele_qna_benchmark.py` flow,
but instead of importing rag_system_v3 it talks to the demo's HTTP SSE endpoint
(POST /api/query). Outputs `benchmark_results_<name>.json`, `.log`, and
`_analysis.md` next to this script.

Usage:
    python tests/benchmark/run_teleqna.py <output_name>                  # default file, fixed mode
    python tests/benchmark/run_teleqna.py v1 --mode react_agent          # ReAct adaptive
    python tests/benchmark/run_teleqna.py v1 --category Definition       # single category
    python tests/benchmark/run_teleqna.py v1 --limit 100                 # 100 questions, balanced
    python tests/benchmark/run_teleqna.py v1 --include-research          # keep "Research publications"

Default behaviour:
- Loads `tele_qna/tele_qna_test.json` (raw TeleQnA JSONL, 10000 records, 5 subjects).
- When the input is JSONL it's auto-grouped by `subject`; if `--limit N` is given,
  N is split evenly across subjects (deterministic via --seed) so every subject
  gets a fair share.
- Grouped JSON files (one with explicit `category` buckets) are loaded as-is.
- Excludes `subject == "Research publications"` (academic papers, not 3GPP specs).
- Uses fixed pipeline + qwen3:14b + think:False for speed.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root so the script is portable
# regardless of where the repo lives on disk.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# parents[0] = tests/, parents[1] = project root (parent of tests/)
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BENCHMARK_FILE = PROJECT_ROOT / "tele_qna" / "tele_qna_test.json"
DEFAULT_API = os.getenv("DEMO_API_URL", "http://localhost:8000/api/query")

# Subjects we treat as "academic / non-3GPP" and skip by default. The KG only
# covers 3GPP standards; questions about IEEE/research papers can't be answered
# with our chunks and contaminate accuracy numbers.
RESEARCH_SUBJECTS = {"Research publications"}


# ---------------------------------------------------------------------------
# Logging — tee stdout to file (matches parent script's pattern)
# ---------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    category: str
    subject: str
    question: str
    expected_answer_index: int          # 0-based — matches dataset and choices[index]
    expected_answer_text: str
    model_response: str                 # full LLM reply (NOT truncated)
    extracted_choice: Optional[int]     # 0-based; None if extraction failed
    correct: bool
    execution_time_ms: float
    sources_count: int
    error: Optional[str] = None
    # Number of chunks returned by the graph branch (`retrieval_graph.count`).
    # 0 = empty result (Pattern C sentinel or LLM-generated Cypher matched
    # nothing). Used by analysis.md to count "graph search succeeded".
    graph_count: int = 0
    graph_error: Optional[str] = None
    # Every SSE event from rag-engine for this question (intent / retrieval_*
    # / rerank / answer / sources). Kept verbatim so debug.log can reconstruct
    # the full pipeline trail. Excluded from results.json to keep that file
    # small — see generate_report().
    pipeline_events: list[dict] = field(default_factory=list)


@dataclass
class CategoryStats:
    category: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_execution_time_ms: float
    errors: int
    # `graph_search_success` = retrieval_graph returned ≥1 chunk and no error.
    # Useful signal of whether the graph branch is contributing per category;
    # questions where the LLM-generated Cypher matches nothing fall to vector.
    graph_search_success: int = 0
    graph_search_errors: int = 0


# Markdown decoration the LLM may sprinkle around an Answer: prefix
# (e.g. "**Answer:** B", "*Answer*: B", "### Answer: B"). Strip these
# before regex-matching so the patterns below don't have to know about them.
_MD_DECORATION = re.compile(r"[*_`#>]+")

# Phrases the model emits when it explicitly declines to answer. When any of
# these appear inside an "Answer: ..." prefix we treat that as a refusal —
# the fuzzy fallback would otherwise pick a random-looking choice from prose
# context and report a false positive. Matched case-insensitively.
_REFUSAL_PATTERNS = (
    re.compile(r"\bcontext\s+does\s+not\s+(?:cover|specify|include|provide|address|mention|contain)\b", re.IGNORECASE),
    re.compile(r"\bnone\s+of\s+the\s+(?:options|choices|answers)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+(?:be\s+)?(?:determined|determine)\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:context|information)\b", re.IGNORECASE),
)


def _is_refusal(text: str) -> bool:
    """True when the response is an explicit "I cannot answer" reply.

    Triggers only when the refusal phrase appears in the FIRST line (i.e. as
    the LLM's headline answer), not when it shows up later as part of a
    citation or justification.
    """
    if not text:
        return False
    head = text.split("\n", 1)[0]
    # Also accept "Answer: <refusal phrase>" as a refusal regardless of
    # whether the phrase reaches into line 2.
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# MCQ extraction — ported and adapted from parent's regex priority chain.
# Returns 0-based choice index (matches dataset convention) or None.
# ---------------------------------------------------------------------------
def extract_choice(response: str, choices: list[str]) -> Optional[int]:
    if not response:
        return None
    text = response.strip()
    # Strip markdown decoration (*, _, `, #, >) so patterns matching at line
    # start work for "**Answer:** B", "### Answer: B", etc.
    cleaned = _MD_DECORATION.sub("", text).strip()
    n = len(choices)

    # 1. "Answer: <number>" — handles (1)/(2)/... and 1/2/...
    m = re.match(r"^\s*(?:the\s+)?answer[:\s]+\(?\s*(\d+)\s*\)?", cleaned, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1  # convert 1-based reply → 0-based
        if 0 <= idx < n:
            return idx

    # 2. "Answer: A/B/C/..."
    m = re.match(r"^\s*(?:the\s+)?answer[:\s]+([A-Za-z])\b", cleaned, re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < n:
            return idx

    # If the model explicitly refused at the top of the reply, stop here.
    # The fuzzy fallback (steps 6-7) would otherwise hallucinate a choice
    # from incidental keyword overlap with the question text.
    if _is_refusal(cleaned):
        return None

    # 3. "answer is N" / "answer: N" anywhere
    m = re.search(r"\banswer\s*(?:is)?\s*[:=]?\s*\(?\s*(\d+)\s*\)?", cleaned, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx

    # 4. "answer is X" letter form anywhere
    m = re.search(r"\banswer\s*(?:is)?\s*[:=]?\s*([A-Za-z])\b", cleaned, re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < n:
            return idx

    # 5. "X. <text>" at start of a line — model may default to letter without "Answer:"
    for i in range(n):
        letter = chr(ord("A") + i)
        if re.search(rf"(^|\n)\s*{letter}\s*[\.\):]", cleaned, re.IGNORECASE):
            return i

    # 6. Verbatim choice text — first occurrence wins. Word-boundary anchored
    # so short choices like "Class A" don't false-positive inside compound
    # words ("Class A" matches "class addresses" without \b).
    text_lower = cleaned.lower()
    first_pos = len(cleaned)
    first_idx: Optional[int] = None
    for i, choice in enumerate(choices):
        choice_clean = choice.strip().lower()
        if len(choice_clean) < 6:
            continue   # skip very short choices (e.g. numbers) — high false-positive
        try:
            m = re.search(rf"\b{re.escape(choice_clean)}\b", text_lower)
        except re.error:
            # Fallback to substring if escape produces an invalid pattern (rare)
            pos = text_lower.find(choice_clean)
            m = None
            if 0 <= pos < first_pos:
                first_pos = pos
                first_idx = i
            continue
        if m and m.start() < first_pos:
            first_pos = m.start()
            first_idx = i
    if first_idx is not None:
        return first_idx

    # 7. Fuzzy match against choices (last resort)
    best_idx: Optional[int] = None
    best_score = 0.55
    for i, choice in enumerate(choices):
        choice_lower = choice.lower()
        ratio = SequenceMatcher(None, choice_lower, text_lower).ratio()
        choice_words = set(choice_lower.split())
        text_words = set(text_lower.split())
        overlap = len(choice_words & text_words) / max(1, len(choice_words))
        score = max(ratio, overlap)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Demo backend client — POSTs to /api/query, drains the SSE stream, returns
# the accumulated answer text + a couple of metadata bits.
# ---------------------------------------------------------------------------
def query_demo(
    api_url: str, prompt: str, mode: str, model: str, think: bool,
    timeout: float = 600.0,
) -> tuple[str, int, int, list[dict]]:
    payload = {"question": prompt, "mode": mode, "model": model, "think": think}
    answer_parts: list[str] = []
    sources_count = 0
    # Verbatim list of every SSE event (preserves order). Returned to caller
    # so the full pipeline trail can be persisted for debugging.
    events: list[dict] = []
    t0 = time.monotonic()
    with requests.post(api_url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                ev = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            events.append(ev)
            stage = ev.get("stage")
            data = ev.get("data")
            if stage == "answer" and isinstance(data, str):
                answer_parts.append(data)
            elif stage == "sources" and isinstance(data, list):
                sources_count = len(data)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return "".join(answer_parts), elapsed_ms, sources_count, events


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
class TeleQnABenchmark:
    def __init__(self, *, output_name: str, mode: str, model: str, think: bool,
                 benchmark_file: Path, api_url: str, include_research: bool,
                 limit: Optional[int] = None, seed: int = 42):
        self.output_name = output_name
        self.mode = mode
        self.model = model
        self.think = think
        self.benchmark_file = benchmark_file
        self.api_url = api_url
        self.include_research = include_research
        # `limit` is also enforced again by run_category(); we read it here so
        # JSONL inputs can be balance-sampled at load time instead of being
        # truncated sequentially through one subject.
        self.limit = limit
        self.seed = seed

        # Each run lives in its own folder under results/ (gitignored) so
        # artefacts don't clutter the script dir or the repo. Re-running with
        # the same name overwrites.
        self.run_dir = SCRIPT_DIR / "results" / output_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.run_dir / "results.json"
        self.log_path = self.run_dir / "run.log"
        self.analysis_path = self.run_dir / "analysis.md"
        # Source question records (choices, answer, subject, explanation) used
        # in this run. Re-runnable: feed it back via `--benchmark-file` to
        # repeat the exact same set of questions.
        self.questions_path = self.run_dir / "questions.json"
        # Per-question prompt + raw model reply + gold/pred metadata, appended
        # as we go so a crash mid-run still leaves a partial trail. Truncated
        # at construction time so reruns don't accumulate stale entries.
        self.debug_path = self.run_dir / "debug.log"
        self.debug_path.write_text("", encoding="utf-8")

        self.results: list[BenchmarkResult] = []
        # Source records actually exercised this run, grouped by category in
        # the same shape as the input benchmark file so it's drop-in
        # re-runnable. Filled as we iterate categories.
        self._questions_by_category: dict[str, list[dict]] = {}
        self._category_descriptions: dict[str, str] = {}

    # ------- dataset loading ----------------------------------------------
    # Try JSON first; fall back to JSONL on "Extra data" (the raw TeleQnA file
    # ships with .json extension but is actually one record per line).
    def _read_records(self) -> tuple[object, str]:
        raw = self.benchmark_file.read_text(encoding="utf-8")
        try:
            return json.loads(raw), "json"
        except json.JSONDecodeError:
            recs = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
            return recs, "jsonl"

    # Balanced sampling: floor(limit / N_subjects) per subject + leftover slots
    # to the first subjects (alphabetical) so the chosen total equals `limit`.
    def _balance_sample(
        self, by_subject: dict[str, list[dict]], limit: int,
    ) -> dict[str, list[dict]]:
        rng = random.Random(self.seed)
        subjects = sorted(by_subject.keys())
        n = len(subjects)
        if n == 0:
            return {}
        base = limit // n
        extra = limit - base * n
        out: dict[str, list[dict]] = {}
        for i, subj in enumerate(subjects):
            target = min(base + (1 if i < extra else 0), len(by_subject[subj]))
            out[subj] = rng.sample(by_subject[subj], target) if target > 0 else []
        return out

    def load_benchmark_data(self) -> list[dict]:
        """Return list of category dicts: [{category, description, questions: [...]}].

        Accepts two input shapes:
        - Grouped JSON (top-level list of {category, description, questions})
        - Raw JSONL of TeleQnA records — auto-bucketed by `subject`, and
          balance-sampled to `self.limit` if given.
        """
        data, fmt = self._read_records()

        if fmt == "jsonl":
            # Bucket flat records by subject, then balance-sample.
            by_subj: dict[str, list[dict]] = defaultdict(list)
            for r in data:
                by_subj[r.get("subject", "Unknown")].append(r)
            chosen = (
                self._balance_sample(dict(by_subj), self.limit)
                if self.limit is not None else dict(by_subj)
            )
            data = [
                {
                    "category": subj,
                    "description": f"TeleQnA subject: {subj}",
                    "questions": qs,
                }
                for subj, qs in sorted(chosen.items()) if qs
            ]

        if not isinstance(data, list):
            raise ValueError(f"Expected top-level list, got {type(data).__name__}")
        # Filter Research publications subject from each category's questions
        if not self.include_research:
            for cat in data:
                qs = cat.get("questions") or []
                cat["questions"] = [q for q in qs if q.get("subject") not in RESEARCH_SUBJECTS]
        return data

    # Append one full question/response block to debug.log. Idx is 1-based and
    # matches the running index printed in the console. Streams every SSE
    # pipeline stage (intent → retrieval_vector → retrieval_graph → rerank →
    # answer → sources) as JSON; the noisy 'answer' token chunks are coalesced
    # into a single block at their first occurrence.
    def _append_debug(self, formatted_prompt: str, r: BenchmarkResult) -> None:
        idx = len(self.results) + 1
        mark = "OK" if r.correct else ("ERR" if r.error else "WRONG")
        with self.debug_path.open("a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"[{idx:3d}] {mark}  Category: {r.category}  |  Subject: {r.subject}\n")
            f.write("=" * 80 + "\n\n")
            f.write("PROMPT:\n")
            f.write(formatted_prompt + "\n\n")

            # Pipeline stages dump. Walk events in order; any stage that
            # streams (emits >1 event — e.g. answer, graph_cypher_token) is
            # coalesced into a single block at its first occurrence so the
            # trail stays readable. All other stages are dumped verbatim.
            f.write("PIPELINE STAGES:\n")
            stage_counts = Counter(ev.get("stage") for ev in r.pipeline_events)
            streaming = {s for s, c in stage_counts.items() if c > 1}
            emitted_streaming: set[str] = set()
            for ev in r.pipeline_events:
                stage = ev.get("stage")
                data = ev.get("data")
                if stage in streaming:
                    if stage in emitted_streaming:
                        continue
                    parts = []
                    for e in r.pipeline_events:
                        if e.get("stage") != stage:
                            continue
                        d = e.get("data")
                        parts.append(d if isinstance(d, str)
                                     else json.dumps(d, ensure_ascii=False))
                    f.write(f"-- stage: {stage} (coalesced × {stage_counts[stage]}) --\n")
                    f.write("".join(parts) + "\n\n")
                    emitted_streaming.add(stage)
                    continue
                f.write(f"-- stage: {stage} --\n")
                try:
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
                except (TypeError, ValueError):
                    f.write(repr(data))
                f.write("\n\n")
            if not r.pipeline_events:
                f.write("<no events captured>\n\n")

            f.write(
                f"GOLD: {r.expected_answer_index} = {r.expected_answer_text!r}\n"
                f"PRED: {r.extracted_choice}  |  CORRECT: {r.correct}  |  "
                f"TIME: {r.execution_time_ms:.0f}ms  |  SOURCES: {r.sources_count}\n"
            )
            if r.error:
                f.write(f"ERROR: {r.error}\n")
            f.write("\n")

    # ------- single question runner ---------------------------------------
    def run_question(self, qd: dict, category: str) -> BenchmarkResult:
        question = qd["question"]
        choices = list(qd.get("choices") or [])
        gold_idx = qd["answer"]   # 0-based per dataset
        subject = qd.get("subject", "")

        # Format MCQ — letter prefix matches the parent script's prompt and the
        # extractor's letter-pattern regex. Built up front so it's available
        # for the debug log even on early-exit paths.
        formatted = f"{question}\n\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(ord('A') + i)}. {choice}\n"
        formatted += (
            "\nPick the SINGLE best choice. Begin your reply with `Answer: <letter>` "
            "(e.g. `Answer: B`), then a short justification with spec citations."
        )

        if not (0 <= gold_idx < len(choices)):
            r = BenchmarkResult(
                category=category, subject=subject, question=question,
                expected_answer_index=gold_idx, expected_answer_text="",
                model_response="", extracted_choice=None, correct=False,
                execution_time_ms=0.0, sources_count=0, error="bad gold index",
            )
            self._append_debug(formatted, r)
            return r
        gold_text = choices[gold_idx]

        try:
            response, elapsed_ms, n_sources, events = query_demo(
                self.api_url, formatted, self.mode, self.model, self.think,
            )
        except Exception as e:
            r = BenchmarkResult(
                category=category, subject=subject, question=question,
                expected_answer_index=gold_idx, expected_answer_text=gold_text,
                model_response="", extracted_choice=None, correct=False,
                execution_time_ms=0.0, sources_count=0, error=str(e),
            )
            self._append_debug(formatted, r)
            return r

        predicted = extract_choice(response, choices)
        correct = (predicted == gold_idx) if predicted is not None else False
        # Pull graph branch outcome from the `retrieval_graph` event. Missing
        # event (e.g. error before graph stage) → defaults to count=0/no error.
        graph_count = 0
        graph_error: Optional[str] = None
        for ev in events:
            if ev.get("stage") == "retrieval_graph":
                d = ev.get("data") or {}
                graph_count = int(d.get("count") or 0)
                graph_error = d.get("error")
                break
        r = BenchmarkResult(
            category=category, subject=subject, question=question,
            expected_answer_index=gold_idx, expected_answer_text=gold_text,
            model_response=response, extracted_choice=predicted, correct=correct,
            execution_time_ms=float(elapsed_ms), sources_count=n_sources,
            graph_count=graph_count, graph_error=graph_error,
            pipeline_events=events,
        )
        self._append_debug(formatted, r)
        return r

    # Capture the planned question set up front so questions.json exists from
    # second zero — survives Ctrl+C / crashes during the run loop. Also
    # respects the same --limit truncation the run loop applies.
    def snapshot_planned_questions(
        self, categories: list[dict], max_questions: Optional[int],
    ) -> None:
        running = 0
        for cat in categories:
            name = cat["category"]
            qs = list(cat.get("questions") or [])
            if max_questions is not None:
                remaining = max_questions - running
                if remaining <= 0:
                    break
                qs = qs[:remaining]
            if not qs:
                continue
            self._category_descriptions[name] = cat.get("description", "")
            self._questions_by_category.setdefault(name, []).extend(qs)
            running += len(qs)

    # ------- per-category run ---------------------------------------------
    def run_category(self, cat: dict, max_questions: Optional[int]) -> None:
        name = cat["category"]
        questions = cat.get("questions") or []
        if max_questions is not None:
            remaining = max_questions - len(self.results)
            if remaining <= 0:
                return
            questions = questions[:remaining]
        if not questions:
            return

        print(f"\n{'='*60}")
        print(f"Category: {name}  ({len(questions)} questions)")
        print(f"{'='*60}")
        for i, qd in enumerate(questions, start=1):
            r = self.run_question(qd, name)
            self.results.append(r)
            mark = "✓" if r.correct else ("⚠" if r.error else "✗")
            running_correct = sum(1 for x in self.results if x.correct)
            running_acc = running_correct / len(self.results)
            pred = r.extracted_choice if r.extracted_choice is not None else "?"
            print(
                f"  [{len(self.results):3d}] {mark} {name[:18]:18s} "
                f"pred={pred} gold={r.expected_answer_index} "
                f"({r.execution_time_ms/1000:.1f}s, {r.sources_count} src)  "
                f"acc={running_acc:.3f}"
            )
            self.save_incremental_json()

    # ------- aggregation + reports ----------------------------------------
    def calculate_category_stats(self, category: str) -> CategoryStats:
        cat_results = [r for r in self.results if r.category == category]
        total = len(cat_results)
        correct = sum(1 for r in cat_results if r.correct)
        errors = sum(1 for r in cat_results if r.error is not None)
        acc = (correct / total * 100) if total else 0.0
        avg_time = sum(r.execution_time_ms for r in cat_results) / total if total else 0.0
        graph_ok = sum(
            1 for r in cat_results if r.graph_count > 0 and not r.graph_error
        )
        graph_err = sum(1 for r in cat_results if r.graph_error)
        return CategoryStats(
            category, total, correct, acc, avg_time, errors, graph_ok, graph_err,
        )

    def generate_report(self) -> dict:
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        accuracy = (correct / total * 100) if total else 0.0
        cats = sorted({r.category for r in self.results})
        category_stats = {c: self.calculate_category_stats(c) for c in cats}

        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "model": self.model,
                "think": self.think,
                "benchmark_file": str(self.benchmark_file),
                "include_research_publications": self.include_research,
                "api_url": self.api_url,
            },
            "overall": {
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": round(accuracy, 2),
                "total_errors": sum(1 for r in self.results if r.error is not None),
                "extraction_failures": sum(
                    1 for r in self.results if r.extracted_choice is None and r.error is None
                ),
                "avg_latency_ms": round(
                    sum(r.execution_time_ms for r in self.results) / total, 1
                ) if total else 0.0,
                # Graph-branch success = retrieval_graph returned ≥1 chunk
                # and reported no error. A low number means LLM-generated
                # Cypher is matching nothing (or sentinel Pattern C is firing).
                "graph_search_success": sum(
                    1 for r in self.results
                    if r.graph_count > 0 and not r.graph_error
                ),
                "graph_search_errors": sum(
                    1 for r in self.results if r.graph_error
                ),
            },
            "categories": {
                name: {
                    "total_questions": s.total_questions,
                    "correct_answers": s.correct_answers,
                    "accuracy": round(s.accuracy, 2),
                    "avg_execution_time_ms": round(s.avg_execution_time_ms, 1),
                    "errors": s.errors,
                    "graph_search_success": s.graph_search_success,
                    "graph_search_errors": s.graph_search_errors,
                }
                for name, s in category_stats.items()
            },
            "detailed_results": [
                {
                    "category": r.category,
                    "subject": r.subject,
                    "question": r.question,
                    "expected_answer_index": r.expected_answer_index,
                    "expected_answer": r.expected_answer_text,
                    "model_response": r.model_response,   # full, untruncated
                    "extracted_choice": r.extracted_choice,
                    "correct": r.correct,
                    "execution_time_ms": round(r.execution_time_ms, 1),
                    "sources_count": r.sources_count,
                    "graph_count": r.graph_count,
                    "graph_error": r.graph_error,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def save_incremental_json(self) -> None:
        try:
            report = self.generate_report()
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [warn] incremental save failed: {e}", file=sys.stderr)

    # Save the verbatim source question records used in this run, in the same
    # top-level shape as the input benchmark file so it can be fed straight
    # back via `--benchmark-file <this>` to re-run the exact same set later.
    def save_questions_file(self) -> None:
        out: list[dict] = []
        for name, questions in self._questions_by_category.items():
            out.append({
                "category": name,
                "description": self._category_descriptions.get(name, ""),
                "questions": questions,
            })
        try:
            with self.questions_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [warn] questions-file save failed: {e}", file=sys.stderr)

    # ------- pretty summary + analysis md ---------------------------------
    def print_summary(self, report: dict) -> None:
        print("\n" + "=" * 80)
        print(f"TeleQnA Benchmark — {self.output_name}")
        print("=" * 80)
        meta = report["metadata"]
        ov = report["overall"]
        print(f"Mode: {meta['mode']}    Model: {meta['model']}    Think: {meta['think']}")
        print(f"Total: {ov['total_questions']}    Correct: {ov['correct_answers']}    Accuracy: {ov['accuracy']}%")
        print(f"Errors: {ov['total_errors']}    Extraction failures: {ov['extraction_failures']}    Avg latency: {ov['avg_latency_ms']:.0f}ms")
        print(
            f"Graph search succeeded: {ov['graph_search_success']}/{ov['total_questions']}    "
            f"Graph errors: {ov['graph_search_errors']}"
        )
        print("=" * 80)
        print(f"\n{'Category':<26} {'Q':>4} {'Correct':>8} {'Acc':>8} {'Avg ms':>9} {'Err':>4} {'GraphOK':>8}")
        print("-" * 80)
        for name in sorted(report["categories"]):
            s = report["categories"][name]
            print(
                f"{name:<26} {s['total_questions']:>4} {s['correct_answers']:>8} "
                f"{s['accuracy']:>7.2f}% {s['avg_execution_time_ms']:>9.0f} {s['errors']:>4} "
                f"{s['graph_search_success']:>8}"
            )
        print("=" * 80 + "\n")

    def write_analysis_md(self, report: dict) -> None:
        ov = report["overall"]
        meta = report["metadata"]
        cats = report["categories"]
        sorted_cats = sorted(cats.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
        best = sorted_cats[0] if sorted_cats else None
        worst = sorted_cats[-1] if sorted_cats else None

        lines: list[str] = []
        lines.append(f"# TeleQnA Benchmark Analysis — {self.output_name}\n")
        lines.append("## Overview")
        lines.append(f"- **Timestamp**: {meta['timestamp']}")
        lines.append(f"- **Mode**: {meta['mode']}")
        lines.append(f"- **Model**: {meta['model']}    **Think**: {meta['think']}")
        lines.append(f"- **Total questions**: {ov['total_questions']}")
        lines.append(f"- **Overall accuracy**: **{ov['accuracy']}%**")
        lines.append(f"- **Errors**: {ov['total_errors']}    **Extraction failures**: {ov['extraction_failures']}")
        lines.append(f"- **Avg latency**: {ov['avg_latency_ms']:.0f} ms")
        lines.append(f"- **Excludes Research publications**: {not meta['include_research_publications']}")
        # Graph branch contribution: how many questions had retrieval_graph
        # come back with ≥1 chunk and no error. Low ratio → vector branch
        # carrying most of the load (or LLM-Cypher pattern needs work).
        gs_ok = ov.get("graph_search_success", 0)
        gs_err = ov.get("graph_search_errors", 0)
        gs_pct = (gs_ok / ov["total_questions"] * 100) if ov["total_questions"] else 0.0
        lines.append(
            f"- **Graph search succeeded**: {gs_ok}/{ov['total_questions']} "
            f"({gs_pct:.1f}%)    **Graph errors**: {gs_err}"
        )
        lines.append("")
        lines.append("## Accuracy by category\n")
        for name, s in sorted_cats:
            bar = "█" * int(s["accuracy"] / 5)
            lines.append(f"- **{name}**: {s['accuracy']:.1f}%  {bar}")
        lines.append("")
        lines.append("## Graph search success by category\n")
        # Sort by absolute success count so the strongest categories are visible
        # first. Ties broken alphabetically for stability.
        sorted_graph = sorted(
            cats.items(),
            key=lambda kv: (-kv[1].get("graph_search_success", 0), kv[0]),
        )
        for name, s in sorted_graph:
            ok = s.get("graph_search_success", 0)
            errs = s.get("graph_search_errors", 0)
            tot = s["total_questions"]
            pct = (ok / tot * 100) if tot else 0.0
            err_note = f"  (errors: {errs})" if errs else ""
            lines.append(f"- **{name}**: {ok}/{tot} ({pct:.1f}%){err_note}")
        lines.append("")
        if best and worst and best != worst:
            lines.append(f"### Best: {best[0]} ({best[1]['accuracy']}%)")
            lines.append(f"### Worst: {worst[0]} ({worst[1]['accuracy']}%)")
            lines.append("")
        # Wrong-answer sample (up to 10) for quick inspection
        wrong = [r for r in self.results if not r.correct and r.error is None]
        if wrong:
            lines.append("## Wrong-answer sample (first 10)\n")
            for r in wrong[:10]:
                lines.append(f"- **{r.category}** · gold={r.expected_answer_index} pred={r.extracted_choice}")
                lines.append(f"  - Q: {r.question}")
                lines.append(f"  - expected: \"{r.expected_answer_text}\"")
                lines.append(f"  - got: {r.model_response[:200]!r}")
                lines.append("")
        with self.analysis_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="TeleQnA benchmark runner for the demo backend.",
    )
    p.add_argument("output_name", help="Suffix for output files (no extension)")
    p.add_argument("--mode", choices=["fixed", "react_agent"], default="fixed")
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--think", action="store_true",
                   help="Enable LLM thinking phase (slower)")
    p.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE,
                   help="Path to benchmark JSON (defaults to the curated comprehensive file under tests/benchmark/)")
    p.add_argument("--category", default=None,
                   help="Run only this category (case-sensitive)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total question count (when input is JSONL, splits "
                        "evenly across subjects)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for balanced JSONL sampling (reproducible)")
    p.add_argument("--include-research", action="store_true",
                   help="Keep questions with subject 'Research publications'")
    p.add_argument("--api", default=DEFAULT_API,
                   help=f"Demo API URL (default {DEFAULT_API})")
    args = p.parse_args()

    if not args.benchmark_file.exists():
        print(f"Benchmark file not found: {args.benchmark_file}", file=sys.stderr)
        sys.exit(1)

    bench = TeleQnABenchmark(
        output_name=args.output_name,
        mode=args.mode,
        model=args.model,
        think=args.think,
        benchmark_file=args.benchmark_file,
        api_url=args.api,
        include_research=args.include_research,
        limit=args.limit,
        seed=args.seed,
    )

    # Tee everything to a log file alongside the json/md outputs.
    tee = TeeLogger(bench.log_path)
    sys.stdout = tee
    try:
        print(f"Loading benchmark from {args.benchmark_file}")
        categories = bench.load_benchmark_data()
        if args.category:
            categories = [c for c in categories if c.get("category") == args.category]
            if not categories:
                print(f"No category named {args.category!r}; aborting", file=sys.stderr)
                sys.exit(1)

        total_to_run = sum(len(c.get("questions") or []) for c in categories)
        capped = min(total_to_run, args.limit) if args.limit else total_to_run
        print(f"Will run {capped} question(s) across {len(categories)} category(ies)")
        print(f"Mode={args.mode}  Model={args.model}  Think={args.think}")
        print(f"Output dir: {bench.run_dir}")

        # Snapshot the planned question set BEFORE running anything so it
        # survives a Ctrl+C / crash mid-run. Re-saved at the end too in case
        # --limit truncated the actual run.
        bench.snapshot_planned_questions(categories, args.limit)
        bench.save_questions_file()

        for cat in categories:
            bench.run_category(cat, max_questions=args.limit)

        report = bench.generate_report()
        bench.save_incremental_json()
        bench.save_questions_file()
        bench.write_analysis_md(report)
        bench.print_summary(report)
        print(f"Wrote: {bench.json_path}")
        print(f"Wrote: {bench.questions_path}")
        print(f"Wrote: {bench.log_path}")
        print(f"Wrote: {bench.analysis_path}")
        print(f"Wrote: {bench.debug_path}")
    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
