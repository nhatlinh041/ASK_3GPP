"""
TeleQnA Resumable Runner — chạy hết toàn bộ TeleQnA qua nhiều lần (resumable).

Khác với `run_teleqna.py` (chạy 1 lần, ghi đè results.json):
- Output dồn về `tests/benchmark/result_all_tele_qna/<output_name>/`.
- Mỗi câu hỏi có 1 `qid` ổn định = chỉ số dòng trong file JSONL gốc.
- Kết quả ghi append-only vào `results.jsonl` (mỗi dòng 1 record).
- Mỗi lần chạy: đọc `results.jsonl` đã có → bỏ các qid đã xong → chạy tiếp
  tối đa `--batch-size` câu còn lại → append → regenerate `results.json` +
  `analysis.md` tổng hợp.
- Crash giữa chừng vẫn an toàn: phần đã làm còn nguyên trong results.jsonl.
- python tests/benchmark/run_teleqna_all.py teleqna_full \
        --mode fixed --model qwen3:14b --batch-size 500 --include-research


Usage:
    # Chạy lần đầu 200 câu — chỉ rõ mode / model / think
    # (mặc định: --mode fixed --model qwen3:14b, --think tắt; bỏ Research publications)
    python tests/benchmark/run_teleqna_all.py teleqna_full \
        --mode fixed --model qwen3:14b --batch-size 200

    # Lần 2 chạy tiếp 200 câu nữa cùng tên run (tự tiếp tục từ câu chưa chạy)
    # Lưu ý: dùng CÙNG --mode / --model / --think như lần đầu để kết quả nhất quán
    python tests/benchmark/run_teleqna_all.py teleqna_full \
        --mode fixed --model qwen3:14b --batch-size 200

    # Bật think (chậm hơn nhưng accuracy thường cao hơn)
    python tests/benchmark/run_teleqna_all.py teleqna_full_think \
        --mode fixed --model qwen3:14b --think --batch-size 200

    # Xem tiến độ không chạy
    python tests/benchmark/run_teleqna_all.py teleqna_full --status

    # Xoá hết và làm lại từ đầu
    python tests/benchmark/run_teleqna_all.py teleqna_full --reset

    # Chạy hết phần còn lại trong 1 lần (không giới hạn)
    python tests/benchmark/run_teleqna_all.py teleqna_full --batch-size 0

    # Đổi model + bật think + 100 câu (đặt output_name khác để không trộn
    # với run trước; --think chậm hơn nhưng thường accuracy cao hơn)
    python tests/benchmark/run_teleqna_all.py teleqna_qwen_think \
        --model qwen3:14b --think --batch-size 100

    # ReAct adaptive mode + deepseek
    python tests/benchmark/run_teleqna_all.py teleqna_react_dsk \
        --mode react_agent --model deepseek-r1:14b --batch-size 50

    # Chạy lại các qid trước đó bị error (network blip, OOM, …)
    python tests/benchmark/run_teleqna_all.py teleqna_full \
        --retry-errors --batch-size 100

    # Trỏ sang backend khác (vd: chạy benchmark trên server staging)
    python tests/benchmark/run_teleqna_all.py teleqna_staging \
        --api http://10.0.0.5:8000/api/query --batch-size 200

    # Giữ luôn câu thuộc Research publications (mặc định loại bỏ)
    python tests/benchmark/run_teleqna_all.py teleqna_with_research \
        --include-research --batch-size 200
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import lại helpers/dataclasses từ script gốc để không trùng lặp logic
# (extract_choice, query_demo, BenchmarkResult, CategoryStats, TeeLogger,
#  RESEARCH_SUBJECTS, DEFAULT_BENCHMARK_FILE, DEFAULT_API).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_teleqna import (  # noqa: E402
    BenchmarkResult,
    CategoryStats,
    DEFAULT_API,
    DEFAULT_BENCHMARK_FILE,
    RESEARCH_SUBJECTS,
    TeeLogger,
    extract_choice,
    query_demo,
)

PROJECT_ROOT = SCRIPT_DIR.parents[1]
# Thư mục chứa các run resumable — tách biệt với `results/` (dùng cho
# run_teleqna.py) để dễ tìm và không trộn lẫn output 2 cách chạy.
RUNS_ROOT = SCRIPT_DIR / "result_all_tele_qna"


# ---------------------------------------------------------------------------
# Tải toàn bộ dataset theo thứ tự ổn định và gán qid = chỉ số dòng JSONL gốc.
# qid là khoá định danh duy nhất giữa các lần chạy → kiểm tra "đã xong chưa".
# ---------------------------------------------------------------------------
def load_questions_with_qid(
    benchmark_file: Path, include_research: bool,
) -> list[dict]:
    """Đọc file benchmark, gán qid cho từng câu, lọc Research publications nếu cần.

    Trả về list[{qid, question, choices, answer, subject, explanation}] theo
    đúng thứ tự xuất hiện trong file.
    """
    raw = benchmark_file.read_text(encoding="utf-8")
    # Thử parse như JSON list trước (nếu là grouped JSON dạng cũ),
    # rồi fallback sang JSONL (file gốc TeleQnA).
    records: list[dict] = []
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "questions" in data[0]:
            # grouped JSON: flatten theo thứ tự category → question
            for cat in data:
                for q in cat.get("questions") or []:
                    records.append(q)
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("unexpected top-level shape")
    except (json.JSONDecodeError, ValueError):
        for line in raw.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Gán qid = chỉ số dòng trong file gốc (TRƯỚC khi lọc) để giá trị
    # ổn định kể cả khi đổi --include-research giữa các lần chạy.
    out: list[dict] = []
    for idx, r in enumerate(records):
        subj = r.get("subject", "Unknown")
        if not include_research and subj in RESEARCH_SUBJECTS:
            continue
        # Bản sao có qid để không sửa input
        rec = dict(r)
        rec["qid"] = idx
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Quét results.jsonl để biết qid nào đã xong (kể cả error case — cũng coi là
# "đã chạy" để không lặp lại). Người dùng muốn rerun riêng các qid lỗi có thể
# xoá thủ công dòng tương ứng hoặc dùng --retry-errors.
# ---------------------------------------------------------------------------
def read_done_qids(results_jsonl: Path, retry_errors: bool = False) -> tuple[set[int], list[dict]]:
    """Đọc results.jsonl, trả về (set qid đã xong, list raw records).

    retry_errors=True → các record có `error` không tính là đã xong (sẽ chạy lại).
    """
    done: set[int] = set()
    records: list[dict] = []
    if not results_jsonl.exists():
        return done, records
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            qid = rec.get("qid")
            if qid is None:
                continue
            if retry_errors and rec.get("error"):
                continue
            done.add(int(qid))
    return done, records


# ---------------------------------------------------------------------------
# Ghi 1 result vào results.jsonl (append). Bỏ pipeline_events khỏi file này
# để giữ kích thước nhỏ — full trail đã có trong debug.log.
# ---------------------------------------------------------------------------
def append_result(results_jsonl: Path, qid: int, r: BenchmarkResult) -> None:
    rec = {
        "qid": qid,
        "category": r.category,
        "subject": r.subject,
        "question": r.question,
        "expected_answer_index": r.expected_answer_index,
        "expected_answer": r.expected_answer_text,
        "model_response": r.model_response,
        "extracted_choice": r.extracted_choice,
        "correct": r.correct,
        "execution_time_ms": round(r.execution_time_ms, 1),
        "sources_count": r.sources_count,
        "graph_count": r.graph_count,
        "graph_error": r.graph_error,
        "error": r.error,
    }
    with results_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Append 1 block debug vào debug.log (chỉ chứa các câu CHẠY trong lần này +
# các lần trước — append-only, không truncate). Format giống run_teleqna.py.
# ---------------------------------------------------------------------------
def append_debug(
    debug_path: Path, run_idx: int, qid: int, formatted_prompt: str,
    r: BenchmarkResult,
) -> None:
    mark = "OK" if r.correct else ("ERR" if r.error else "WRONG")
    with debug_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"[run#{run_idx:3d} qid={qid}] {mark}  "
            f"Category: {r.category}  |  Subject: {r.subject}\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write("PROMPT:\n")
        f.write(formatted_prompt + "\n\n")
        # Dump các sự kiện pipeline; coalesce token-stream để dễ đọc
        f.write("PIPELINE STAGES:\n")
        from collections import Counter
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


# ---------------------------------------------------------------------------
# Chạy 1 câu hỏi (gọi backend, extract choice, return BenchmarkResult).
# Tách thành hàm độc lập để dùng được mà không cần class.
# ---------------------------------------------------------------------------
def run_one(qd: dict, api_url: str, mode: str, model: str, think: bool) -> tuple[BenchmarkResult, str]:
    question = qd["question"]
    choices = list(qd.get("choices") or [])
    gold_idx = qd["answer"]
    subject = qd.get("subject", "")
    # Dùng subject làm "category" để analysis.md group theo subject như run gốc
    category = subject or "Unknown"

    # Format MCQ — letter prefix khớp với regex extractor
    formatted = f"{question}\n\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(ord('A') + i)}. {choice}\n"
    formatted += (
        "\nPick the SINGLE best choice. Begin your reply with `Answer: <letter>` "
        "(e.g. `Answer: B`), then a short justification with spec citations."
    )

    if not (0 <= gold_idx < len(choices)):
        return BenchmarkResult(
            category=category, subject=subject, question=question,
            expected_answer_index=gold_idx, expected_answer_text="",
            model_response="", extracted_choice=None, correct=False,
            execution_time_ms=0.0, sources_count=0, error="bad gold index",
        ), formatted

    gold_text = choices[gold_idx]
    try:
        response, elapsed_ms, n_sources, events = query_demo(
            api_url, formatted, mode, model, think,
        )
    except Exception as e:
        return BenchmarkResult(
            category=category, subject=subject, question=question,
            expected_answer_index=gold_idx, expected_answer_text=gold_text,
            model_response="", extracted_choice=None, correct=False,
            execution_time_ms=0.0, sources_count=0, error=str(e),
        ), formatted

    predicted = extract_choice(response, choices)
    correct = (predicted == gold_idx) if predicted is not None else False
    # Trích graph branch outcome từ sự kiện retrieval_graph (nếu có)
    graph_count = 0
    graph_error: Optional[str] = None
    for ev in events:
        if ev.get("stage") == "retrieval_graph":
            d = ev.get("data") or {}
            graph_count = int(d.get("count") or 0)
            graph_error = d.get("error")
            break
    return BenchmarkResult(
        category=category, subject=subject, question=question,
        expected_answer_index=gold_idx, expected_answer_text=gold_text,
        model_response=response, extracted_choice=predicted, correct=correct,
        execution_time_ms=float(elapsed_ms), sources_count=n_sources,
        graph_count=graph_count, graph_error=graph_error,
        pipeline_events=events,
    ), formatted


# ---------------------------------------------------------------------------
# Aggregate report — đọc lại toàn bộ results.jsonl, build dict cùng schema
# với run_teleqna.py để các tool phân tích cũ (analyze_failures.py, v.v.)
# tái sử dụng được.
# ---------------------------------------------------------------------------
def build_report(
    records: list[dict], *, mode: str, model: str, think: bool,
    benchmark_file: Path, include_research: bool, api_url: str,
    total_planned: int,
) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    errors = sum(1 for r in records if r.get("error"))
    extraction_failures = sum(
        1 for r in records
        if r.get("extracted_choice") is None and not r.get("error")
    )
    accuracy = (correct / total * 100) if total else 0.0
    avg_latency = (
        sum(r.get("execution_time_ms", 0.0) for r in records) / total
    ) if total else 0.0
    graph_ok = sum(
        1 for r in records
        if (r.get("graph_count") or 0) > 0 and not r.get("graph_error")
    )
    graph_err = sum(1 for r in records if r.get("graph_error"))

    # Group theo category (subject) để tính per-category stats
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r.get("category", "Unknown"), []).append(r)

    categories = {}
    for name in sorted(by_cat):
        rs = by_cat[name]
        n = len(rs)
        c = sum(1 for r in rs if r.get("correct"))
        e = sum(1 for r in rs if r.get("error"))
        g_ok = sum(
            1 for r in rs
            if (r.get("graph_count") or 0) > 0 and not r.get("graph_error")
        )
        g_err = sum(1 for r in rs if r.get("graph_error"))
        categories[name] = {
            "total_questions": n,
            "correct_answers": c,
            "accuracy": round((c / n * 100) if n else 0.0, 2),
            "avg_execution_time_ms": round(
                sum(r.get("execution_time_ms", 0.0) for r in rs) / n if n else 0.0, 1
            ),
            "errors": e,
            "graph_search_success": g_ok,
            "graph_search_errors": g_err,
        }

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model": model,
            "think": think,
            "benchmark_file": str(benchmark_file),
            "include_research_publications": include_research,
            "api_url": api_url,
            "resumable_run": True,
            "total_planned": total_planned,
            "completed": total,
            "remaining": max(0, total_planned - total),
        },
        "overall": {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": round(accuracy, 2),
            "total_errors": errors,
            "extraction_failures": extraction_failures,
            "avg_latency_ms": round(avg_latency, 1),
            "graph_search_success": graph_ok,
            "graph_search_errors": graph_err,
        },
        "categories": categories,
        "detailed_results": records,
    }


def write_analysis_md(report: dict, out_path: Path, run_name: str) -> None:
    ov = report["overall"]
    meta = report["metadata"]
    cats = report["categories"]
    sorted_cats = sorted(cats.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    best = sorted_cats[0] if sorted_cats else None
    worst = sorted_cats[-1] if sorted_cats else None

    lines: list[str] = []
    lines.append(f"# TeleQnA Resumable Benchmark — {run_name}\n")
    lines.append("## Overview")
    lines.append(f"- **Timestamp**: {meta['timestamp']}")
    lines.append(f"- **Mode**: {meta['mode']}")
    lines.append(f"- **Model**: {meta['model']}    **Think**: {meta['think']}")
    lines.append(
        f"- **Progress**: {meta['completed']}/{meta['total_planned']} "
        f"(còn {meta['remaining']})"
    )
    lines.append(f"- **Overall accuracy**: **{ov['accuracy']}%**")
    lines.append(
        f"- **Errors**: {ov['total_errors']}    "
        f"**Extraction failures**: {ov['extraction_failures']}"
    )
    lines.append(f"- **Avg latency**: {ov['avg_latency_ms']:.0f} ms")
    lines.append(
        f"- **Excludes Research publications**: "
        f"{not meta['include_research_publications']}"
    )
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
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Status / reset helpers
# ---------------------------------------------------------------------------
def print_status(run_dir: Path, total_planned: int) -> None:
    results_jsonl = run_dir / "results.jsonl"
    done, records = read_done_qids(results_jsonl)
    n = len(done)
    correct = sum(1 for r in records if r.get("correct"))
    errors = sum(1 for r in records if r.get("error"))
    acc = (correct / n * 100) if n else 0.0
    print(f"Run dir: {run_dir}")
    print(f"Tổng số câu (sau khi lọc): {total_planned}")
    print(f"Đã chạy: {n} ({n / total_planned * 100:.1f}%)")
    print(f"Còn lại: {total_planned - n}")
    print(f"Đúng: {correct} ({acc:.2f}%)    Lỗi: {errors}")


def reset_run(run_dir: Path) -> None:
    if not run_dir.exists():
        print(f"Run dir chưa tồn tại: {run_dir}")
        return
    # Chỉ xoá file artefact, KHÔNG xoá thư mục — tránh hiệu ứng phụ ngoài ý muốn
    for name in ("results.jsonl", "results.json", "analysis.md", "debug.log", "run.log"):
        p = run_dir / name
        if p.exists():
            p.unlink()
            print(f"  removed {p.name}")
    print(f"Đã reset {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Resumable TeleQnA runner — chạy hết toàn bộ qua nhiều batch.",
    )
    p.add_argument("output_name",
                   help="Tên thư mục run (results/<name>); chạy lại cùng tên = continue")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Số câu chạy tối đa trong lần này (0 = chạy hết phần còn lại). Mặc định 100")
    p.add_argument("--mode", choices=["fixed", "react_agent"], default="fixed")
    p.add_argument("--model", default="qwen3:14b")
    p.add_argument("--think", action="store_true",
                   help="Bật pha think của LLM (chậm hơn)")
    p.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE,
                   help="Đường dẫn file benchmark (mặc định tele_qna_test.json)")
    p.add_argument("--include-research", action="store_true",
                   help="Không loại bỏ subject 'Research publications'")
    p.add_argument("--api", default=DEFAULT_API,
                   help=f"URL backend (mặc định {DEFAULT_API})")
    p.add_argument("--retry-errors", action="store_true",
                   help="Chạy lại các qid trước đó bị error")
    p.add_argument("--status", action="store_true",
                   help="Chỉ in tiến độ rồi thoát, không chạy gì")
    p.add_argument("--reset", action="store_true",
                   help="Xoá kết quả cũ rồi thoát (cần xác nhận output_name)")
    args = p.parse_args()

    if not args.benchmark_file.exists():
        print(f"Benchmark file not found: {args.benchmark_file}", file=sys.stderr)
        sys.exit(1)

    run_dir = RUNS_ROOT / args.output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = run_dir / "results.jsonl"
    results_json = run_dir / "results.json"
    analysis_md = run_dir / "analysis.md"
    debug_log = run_dir / "debug.log"
    run_log = run_dir / "run.log"

    # Reset nhánh ngắn — xoá xong là thoát, không chạy gì
    if args.reset:
        reset_run(run_dir)
        return

    # Tải toàn bộ câu hỏi (đã gán qid + đã lọc Research nếu cần)
    all_questions = load_questions_with_qid(
        args.benchmark_file, include_research=args.include_research,
    )
    total_planned = len(all_questions)

    if args.status:
        print_status(run_dir, total_planned)
        return

    # Xác định các qid còn cần chạy
    done_qids, _existing_records = read_done_qids(
        results_jsonl, retry_errors=args.retry_errors,
    )
    remaining = [q for q in all_questions if q["qid"] not in done_qids]
    if not remaining:
        print(f"Đã chạy hết {total_planned} câu. Không còn gì để làm.")
        # Vẫn regenerate aggregate report phòng khi file bị xoá
        _, records = read_done_qids(results_jsonl)
        report = build_report(
            records, mode=args.mode, model=args.model, think=args.think,
            benchmark_file=args.benchmark_file,
            include_research=args.include_research, api_url=args.api,
            total_planned=total_planned,
        )
        with results_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        write_analysis_md(report, analysis_md, args.output_name)
        return

    # Cắt batch — batch_size=0 nghĩa là chạy hết phần còn lại trong 1 lượt
    batch = remaining if args.batch_size == 0 else remaining[: args.batch_size]

    # Tee stdout sang run.log (append để giữ log các lần chạy trước)
    log_fh = open(run_log, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, term, fh):
            self.terminal = term
            self.fh = fh
        def write(self, m):
            self.terminal.write(m)
            self.fh.write(m)
            self.fh.flush()
        def flush(self):
            self.terminal.flush()
            self.fh.flush()

    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_fh)

    try:
        print("\n" + "=" * 80)
        print(f"TeleQnA Resumable Runner — {args.output_name}")
        print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Mode: {args.mode}  Model: {args.model}  Think: {args.think}")
        print(f"Benchmark: {args.benchmark_file}")
        print(
            f"Tổng kế hoạch: {total_planned}    Đã xong: {len(done_qids)}    "
            f"Còn lại: {len(remaining)}    Batch lần này: {len(batch)}"
        )
        print("=" * 80)

        # Chạy từng câu — append vào results.jsonl ngay sau mỗi câu để
        # Ctrl+C giữa chừng cũng không mất dữ liệu.
        running_correct = 0
        running_total = 0
        t_start = time.monotonic()
        for i, qd in enumerate(batch, start=1):
            qid = qd["qid"]
            r, formatted = run_one(qd, args.api, args.mode, args.model, args.think)
            append_result(results_jsonl, qid, r)
            append_debug(debug_log, i, qid, formatted, r)
            running_total += 1
            if r.correct:
                running_correct += 1
            mark = "✓" if r.correct else ("⚠" if r.error else "✗")
            pred = r.extracted_choice if r.extracted_choice is not None else "?"
            acc = running_correct / running_total
            global_done = len(done_qids) + running_total
            print(
                f"  [{i:4d}/{len(batch)}] qid={qid:5d} {mark} "
                f"{r.category[:18]:18s} pred={pred} gold={r.expected_answer_index} "
                f"({r.execution_time_ms/1000:.1f}s, {r.sources_count} src)  "
                f"batch_acc={acc:.3f}  total={global_done}/{total_planned}"
            )

        elapsed = time.monotonic() - t_start
        print("\n" + "-" * 80)
        print(
            f"Batch xong: {running_total} câu trong {elapsed:.1f}s "
            f"(avg {elapsed / max(1, running_total):.1f}s/câu)"
        )
        print(
            f"Batch accuracy: {running_correct}/{running_total} "
            f"({running_correct / max(1, running_total) * 100:.2f}%)"
        )

        # Regenerate aggregate report từ TOÀN BỘ results.jsonl
        _, all_records = read_done_qids(results_jsonl)
        report = build_report(
            all_records, mode=args.mode, model=args.model, think=args.think,
            benchmark_file=args.benchmark_file,
            include_research=args.include_research, api_url=args.api,
            total_planned=total_planned,
        )
        with results_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        write_analysis_md(report, analysis_md, args.output_name)

        ov = report["overall"]
        meta = report["metadata"]
        print(
            f"\nTổng tích luỹ: {ov['total_questions']}/{meta['total_planned']}    "
            f"Đúng: {ov['correct_answers']} ({ov['accuracy']}%)    "
            f"Lỗi: {ov['total_errors']}"
        )
        print(f"Còn lại: {meta['remaining']} câu — chạy tiếp với cùng output_name để continue.")
        print(f"\nWrote: {results_jsonl}")
        print(f"Wrote: {results_json}")
        print(f"Wrote: {analysis_md}")
        print(f"Wrote: {debug_log}")
        print(f"Wrote: {run_log}")
    finally:
        sys.stdout = original_stdout
        log_fh.close()


if __name__ == "__main__":
    main()
