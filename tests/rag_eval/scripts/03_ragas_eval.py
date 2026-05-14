"""
Скрипт 03: RAGAS с разбивкой по категориям.

Вход:  tests/rag_eval/results/raw_answers.json
Выход: tests/rag_eval/results/ragas_report.json
       tests/rag_eval/results/ragas_summary.txt

Использование:
    python tests/rag_eval/scripts/03_ragas_eval.py
    python tests/rag_eval/scripts/03_ragas_eval.py --vllm-url http://localhost:9002/v1
"""

import argparse
import importlib
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
ST_EMBEDDINGS_MODEL = "/mnt/mishutqa/PycharmProjects/abitBot/app/models/deepvk/USER-bge-m3"

MAX_CHUNKS = 3
MAX_CHUNK_CHARS = 500


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", data) if isinstance(data, dict) else data


def build_dataset(records: list[dict]):
    from datasets import Dataset
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for r in records:
        contexts = [c["text"][:MAX_CHUNK_CHARS]
                    for c in r.get("retrieved_context", [])[:MAX_CHUNKS] if c.get("text")]
        if not contexts:
            contexts = ["(контекст не получен)"]
        rows["question"].append(r["question"])
        rows["answer"].append(r["actual_answer"])
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(r.get("ground_truth", ""))
    return Dataset.from_dict(rows)


def get_llm_wrapper(base_url: str, model: str):
    from langchain_openai import ChatOpenAI
    LangchainLLMWrapper = None
    for mod_name in ("ragas.llms", "ragas"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "LangchainLLMWrapper"):
                LangchainLLMWrapper = mod.LangchainLLMWrapper
                break
        except Exception:
            pass
    llm = ChatOpenAI(base_url=base_url, api_key="EMPTY", model=model,
                     temperature=0, max_tokens=512)
    return LangchainLLMWrapper(llm)


def get_emb_wrapper(model_path: str):
    LangchainEmbeddingsWrapper = None
    for mod_name in ("ragas.embeddings", "ragas"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "LangchainEmbeddingsWrapper"):
                LangchainEmbeddingsWrapper = mod.LangchainEmbeddingsWrapper
                break
        except Exception:
            pass
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    print(f"  Загружаем embeddings: {model_path}")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_path))


def _try_metric(name: str):
    try:
        return getattr(importlib.import_module("ragas.metrics"), name, None)
    except Exception:
        return None


def run_ragas_group(records: list[dict], llm_wrapper, emb_wrapper) -> dict:
    """Запускает RAGAS для группы записей."""
    from ragas import evaluate

    dataset = build_dataset(records)
    has_gt = any(r.get("ground_truth", "").strip() for r in records)

    _ALWAYS = ["faithfulness", "answer_relevancy", "context_relevancy"]
    _WITH_GT = ["context_precision", "context_recall", "answer_correctness"]
    candidate_names = _ALWAYS + (_WITH_GT if has_gt else [])

    metrics, metric_names = [], []
    for name in candidate_names:
        m = _try_metric(name)
        if m is not None:
            metrics.append(m)
            metric_names.append(name)

    for m in metrics:
        if hasattr(m, "llm"):
            m.llm = llm_wrapper
        if hasattr(m, "embeddings"):
            m.embeddings = emb_wrapper

    try:
        result = evaluate(dataset, metrics=metrics, llm=llm_wrapper, embeddings=emb_wrapper)
    except TypeError:
        result = evaluate(dataset, metrics=metrics)

    aggregate = {k: round(float(result[k]), 4) for k in metric_names if k in result}
    return {"count": len(records), "has_gt": has_gt, "metrics": aggregate}


def print_group_metrics(label: str, result: dict) -> None:
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {label}  ({result.get('count', 0)} вопросов)")
    print(sep)
    if "error" in result:
        print(f"  ⚠ {result['error']}")
        return
    for k, v in result.get("metrics", {}).items():
        print(f"  {k:<25} {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RAGAS оценка")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "raw_answers.json")
    parser.add_argument("--model", default=VLLM_MODEL)
    parser.add_argument("--vllm-url", default=VLLM_BASE_URL)
    parser.add_argument("--embeddings-model", default=ST_EMBEDDINGS_MODEL)
    parser.add_argument("--no-skip-compromat", action="store_true", default=False)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Ошибка: {args.input} не найден. Сначала запустите 01_collect_answers.py")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    valid = [r for r in records if r.get("actual_answer") and not r.get("error")]
    print(f"Загружено записей: {len(valid)} из {len(records)}")

    skip_compromat = not args.no_skip_compromat

    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r.get("category", "Unknown")].append(r)

    categories_to_eval = {
        cat: recs for cat, recs in by_cat.items()
        if not (skip_compromat and cat.lower() == "compomat")
    }

    all_for_overall = []
    for recs in categories_to_eval.values():
        all_for_overall.extend(recs)

    print("\nЗагружаем LLM и embeddings...")
    llm_wrapper = get_llm_wrapper(args.vllm_url, args.model)
    emb_wrapper = get_emb_wrapper(args.embeddings_model)

    # По категориям
    cat_results = {}
    for cat, recs in sorted(categories_to_eval.items()):
        print(f"\n--- RAGAS: категория {cat} ({len(recs)} вопросов) ---")
        cat_results[cat] = run_ragas_group(recs, llm_wrapper, emb_wrapper)

    # Общие
    print(f"\n--- RAGAS: общие метрики ({len(all_for_overall)} вопросов) ---")
    overall = run_ragas_group(all_for_overall, llm_wrapper, emb_wrapper)

    # Консоль
    print("\n\n" + "█" * 55)
    print("  РЕЗУЛЬТАТЫ RAGAS")
    print("█" * 55)
    print_group_metrics("ОБЩИЕ МЕТРИКИ", overall)
    for cat, result in sorted(cat_results.items()):
        print_group_metrics(f"Категория: {cat}", result)

    # Сохранение
    report = {"overall": overall, "by_category": cat_results}
    report_path = RESULTS_DIR / "ragas_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["=== RAGAS Evaluation Summary ===\n", "--- ОБЩИЕ МЕТРИКИ ---",
             f"  Вопросов: {overall.get('count', 0)}"]
    for k, v in overall.get("metrics", {}).items():
        lines.append(f"  {k:<25} {v:.4f}")
    for cat, result in sorted(cat_results.items()):
        lines.append(f"\n--- Категория: {cat} ({result.get('count', 0)} вопросов) ---")
        for k, v in result.get("metrics", {}).items():
            lines.append(f"  {k:<25} {v:.4f}")

    summary_path = RESULTS_DIR / "ragas_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ Отчёт:  {report_path}")
    print(f"✓ Сводка: {summary_path}")


if __name__ == "__main__":
    main()