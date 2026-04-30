"""
TeleQnA benchmark runner — evaluates end-to-end answer accuracy.

Usage:
    python benchmark_teleqna.py --dataset ../../tele_qna/TeleQnA.json \
        --model qwen3:14b --top-k 100 --mode fixed
"""
import argparse
import json
import time
from pathlib import Path

import requests

RAG_ENGINE_URL = "http://localhost:8000"


def load_teleqna(path: str) -> list[dict]:
    """Load TeleQnA dataset — supports both JSONL and JSON dict formats."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        # Format: {"question 1": {question, options, answer}, ...}
        return list(data.values())
    return data


def run_query(question: str, model: str, mode: str) -> dict:
    """Call RAG engine and collect all SSE events into a result dict."""
    response = requests.post(
        f"{RAG_ENGINE_URL}/api/query",
        json={"question": question, "mode": mode, "model": model},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    answer_tokens = []
    sources = []
    stages = []

    for line in response.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        event = json.loads(line[5:].strip())
        stage = event.get("stage")
        stages.append(stage)
        if stage == "answer":
            answer_tokens.append(event["data"])
        elif stage == "sources":
            sources = event["data"]

    return {
        "answer": "".join(answer_tokens),
        "sources": sources,
        "stages": stages,
    }


def check_answer(result: dict, expected_answer: str, options: dict | None) -> bool:
    """
    Simple correctness check: does the answer contain the expected answer key/text?
    For MCQ: check if the answer letter (A/B/C/D) or option text appears.
    """
    answer_text = result["answer"].lower()
    if not expected_answer:
        return False

    # Direct text match
    if expected_answer.lower() in answer_text:
        return True

    # MCQ option match
    if options:
        option_text = options.get(expected_answer, "").lower()
        if option_text and option_text[:30] in answer_text:
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../../tele_qna/TeleQnA.json")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--mode", default="fixed", choices=["fixed", "react_agent"])
    parser.add_argument("--top-k", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    questions = load_teleqna(args.dataset)[: args.top_k]
    print(f"Running benchmark: {len(questions)} questions, model={args.model}, mode={args.mode}")

    results = []
    correct = 0
    start = time.time()

    for i, q in enumerate(questions, 1):
        question_text = q.get("question", "")
        expected = q.get("answer", "")
        options = q.get("options")

        try:
            result = run_query(question_text, args.model, args.mode)
            is_correct = check_answer(result, expected, options)
            if is_correct:
                correct += 1

            results.append({
                "question": question_text,
                "expected": expected,
                "answer": result["answer"][:500],
                "correct": is_correct,
                "sources": result["sources"],
            })

            accuracy = correct / i * 100
            print(f"[{i}/{len(questions)}] acc={accuracy:.1f}% {'✓' if is_correct else '✗'}")

        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            results.append({"question": question_text, "error": str(e), "correct": False})

    elapsed = time.time() - start
    final_accuracy = correct / len(questions) * 100

    summary = {
        "accuracy": final_accuracy,
        "correct": correct,
        "total": len(questions),
        "model": args.model,
        "mode": args.mode,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }

    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nAccuracy: {final_accuracy:.2f}% ({correct}/{len(questions)}) in {elapsed:.0f}s")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
