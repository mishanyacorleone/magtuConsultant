"""
Скрипт 02: RAGChecker с разбивкой по категориям из данных (без хардкода).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

RESULTS_DIR = Path(__file__).parent.parent / "results"
VLLM_BASE_URL = os.environ.get("EVAL_VLLM_URL", "http://localhost:9002/v1")
VLLM_MODEL = os.environ.get("EVAL_VLLM_MODEL",
    "/root/.cache/huggingface/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", data) if isinstance(data, dict) else data


def to_ragchecker_format(records: list[dict]) -> dict:
    return {
        "results": [
            {
                "query_id": r["query_id"],
                "query": r["question"],
                "gt_answer": r.get("ground_truth", ""),
                "response": r["actual_answer"],
                "retrieved_context": r.get("retrieved_context", []),
            }
            for r in records
            if r.get("actual_answer") and not r.get("error")
        ]
    }


def run_ragchecker_group(records: list[dict], model: str, vllm_url: str) -> dict:
    try:
        from ragchecker import RAGResults, RAGChecker
    except ImportError:
        print("Ошибка: pip install ragchecker")
        sys.exit(1)

    try:
        from ragchecker.metrics import all_metrics, faithfulness, hallucination, self_knowledge
    except ImportError:
        try:
            from ragchecker.metrics.retriever_metrics import faithfulness
            from ragchecker.metrics.generator_metrics import hallucination, self_knowledge
            from ragchecker.metrics import all_metrics
        except ImportError as e:
            print(f"Ошибка импорта метрик: {e}")
            sys.exit(1)

    data = to_ragchecker_format(records)
    if not data["results"]:
        return {"count": 0, "metrics": {}}

    rag_results = RAGResults.from_dict(data)
    has_gt = any(r.get("gt_answer", "").strip() for r in data["results"])
    litellm_model = model if model.startswith("openai/") else f"openai/{model}"

    checker = RAGChecker(
        extractor_name=litellm_model,
        checker_name=litellm_model,
        extractor_api_base=vllm_url,
        checker_api_base=vllm_url,
        batch_size_extractor=2,
        batch_size_checker=2,
    )

    metrics_arg = "all_metrics" if has_gt else [faithfulness, hallucination, self_knowledge]
    checker.evaluate(rag_results, metrics=metrics_arg)

    try:
        aggregate = rag_results.get_aggregate_metrics()
    except AttributeError:
        sums: dict = defaultdict(float)
        counts: dict = defaultdict(int)
        for r in rag_results.results:
            for k, v in (r.metrics if isinstance(r.metrics, dict) else {}).items():
                if v is not None:
                    sums[k] += float(v)
                    counts[k] += 1
        aggregate = {k: round(sums[k] / counts[k], 4) for k in sums if counts[k] > 0}

    return {"count": len(data["results"]), "has_gt": has_gt,
            "metrics": {k: round(v, 4) for k, v in aggregate.items()}}


def print_group(label: str, result: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}  ({result.get('count', 0)} вопросов)")
    print(f"{'=' * 55}")
    for k, v in result.get("metrics", {}).items():
        print(f"  {k:<35} {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "raw_answers.json")
    parser.add_argument("--model", default=VLLM_MODEL)
    parser.add_argument("--vllm-url", default=VLLM_BASE_URL)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Ошибка: {args.input} не найден. Сначала запустите 01_collect_answers.py")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    valid = [r for r in records if r.get("actual_answer") and not r.get("error")]
    print(f"Записей: {len(valid)} из {len(records)}")

    # Группировка по категориям из данных
    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r.get("category", "unknown")].append(r)

    print(f"Категории: {sorted(by_cat.keys())}\n")

    cat_results = {}
    for cat in sorted(by_cat.keys()):
        print(f"\n--- RAGChecker: {cat} ({len(by_cat[cat])} вопросов) ---")
        cat_results[cat] = run_ragchecker_group(by_cat[cat], args.model, args.vllm_url)

    print(f"\n--- RAGChecker: общие ({len(valid)} вопросов) ---")
    overall = run_ragchecker_group(valid, args.model, args.vllm_url)

    print("\n\n" + "█" * 55)
    print("  РЕЗУЛЬТАТЫ RAGCHECKER")
    print("█" * 55)
    print_group("ОБЩИЕ МЕТРИКИ", overall)
    for cat in sorted(cat_results.keys()):
        print_group(f"Категория: {cat}", cat_results[cat])

    report = {"overall": overall, "by_category": cat_results}
    report_path = RESULTS_DIR / "ragchecker_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    lines = ["=== RAGChecker Summary ===\n", f"ОБЩИЕ ({overall.get('count', 0)} вопросов):"]
    for k, v in overall.get("metrics", {}).items():
        lines.append(f"  {k:<35} {v:.4f}")
    for cat in sorted(cat_results.keys()):
        r = cat_results[cat]
        lines.append(f"\n{cat} ({r.get('count', 0)} вопросов):")
        for k, v in r.get("metrics", {}).items():
            lines.append(f"  {k:<35} {v:.4f}")

    summary_path = RESULTS_DIR / "ragchecker_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Отчёт: {report_path}\n✓ Сводка: {summary_path}")


if __name__ == "__main__":
    main()