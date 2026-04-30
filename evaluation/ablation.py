"""
Ablation study — toggle retrieval components to measure their contribution.

Usage:
    python ablation.py --dataset ../../tele_qna/TeleQnA.json \
        --model qwen3:14b --top-k 50

Configs tested:
    full        — vector + graph + multihop + rerank (baseline)
    no_graph    — vector + multihop + rerank (no graph search)
    no_multihop — vector + graph + rerank (no multi-hop)
    no_rerank   — vector + graph + multihop, RRF only (no cross-encoder)
    vector_only — vector search only, no graph/multihop/rerank
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    "full": {"graph": True, "multihop": True, "rerank": True},
    "no_graph": {"graph": False, "multihop": True, "rerank": True},
    "no_multihop": {"graph": True, "multihop": False, "rerank": True},
    "no_rerank": {"graph": True, "multihop": True, "rerank": False},
    "vector_only": {"graph": False, "multihop": False, "rerank": False},
}


def run_benchmark(dataset: str, model: str, top_k: int, config_name: str, output: str) -> float:
    """Run benchmark_teleqna.py for one config and return accuracy."""
    cmd = [
        sys.executable,
        "benchmark_teleqna.py",
        "--dataset", dataset,
        "--model", model,
        "--top-k", str(top_k),
        "--output", output,
    ]
    subprocess.run(cmd, check=True)

    results = json.loads(Path(output).read_text())
    return results["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../../tele_qna/TeleQnA.json")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = parser.parse_args()

    print(f"Ablation study: {args.top_k} questions, model={args.model}")
    print(f"Configs: {args.configs}\n")

    summary = {}
    for config_name in args.configs:
        output = f"ablation_{config_name}.json"
        print(f"Running config: {config_name}")
        try:
            accuracy = run_benchmark(args.dataset, args.model, args.top_k, config_name, output)
            summary[config_name] = accuracy
            print(f"  {config_name}: {accuracy:.2f}%\n")
        except Exception as e:
            print(f"  {config_name}: ERROR — {e}\n")
            summary[config_name] = None

    print("=" * 40)
    print("Ablation Results:")
    for config, acc in sorted(summary.items(), key=lambda x: x[1] or 0, reverse=True):
        bar = "█" * int((acc or 0) / 2)
        print(f"  {config:<15} {acc:>6.2f}%  {bar}")

    Path("ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary saved to ablation_summary.json")


if __name__ == "__main__":
    main()
