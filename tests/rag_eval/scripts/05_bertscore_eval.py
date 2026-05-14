"""
Скрипт 05: BERTScore через локальную модель USER-bge-m3.

Вход:  tests/rag_eval/results/raw_answers.json
Выход: tests/rag_eval/results/bertscore_report.json
       tests/rag_eval/results/bertscore_summary.txt

Использование:
    python tests/rag_eval/scripts/05_bertscore_eval.py
    python tests/rag_eval/scripts/05_bertscore_eval.py --no-skip-compromat
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOCAL_MODEL = Path("/mnt/mishutqa/PycharmProjects/abitBot/app/models/deepvk/USER-bge-m3")
DEFAULT_MODEL = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "deepvk/USER-bge-m3"


def _auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", data) if isinstance(data, dict) else data


def encode_texts(texts: list[str], model_path: str, device: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Ошибка: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(model_path, device=device)
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)


def compute_scores_for_group(records: list[dict], model_path: str,
                              device: str, batch_size: int) -> dict:
    """Считает BERTScore для группы записей."""
    responses = [r["actual_answer"] for r in records]
    has_gt = any(r.get("ground_truth", "").strip() for r in records)

    # Ответ vs Контекст
    ctx_refs = []
    for r in records:
        chunks = [c["text"] for c in r.get("retrieved_context", []) if c.get("text")]
        ctx_refs.append(" ".join(chunks) if chunks else "(контекст не получен)")

    pred_vecs = encode_texts(responses, model_path, device, batch_size)
    ctx_vecs = encode_texts(ctx_refs, model_path, device, batch_size)
    ctx_scores = [_cosine(p.tolist(), c.tolist()) for p, c in zip(pred_vecs, ctx_vecs)]

    result = {
        "count": len(records),
        "vs_context": {
            "precision": _avg(ctx_scores),
            "recall": _avg(ctx_scores),
            "f1": _avg(ctx_scores),
        }
    }

    # Ответ vs Эталон
    if has_gt:
        gt_pairs = [(i, r) for i, r in enumerate(records) if r.get("ground_truth", "").strip()]
        gt_cands = [r["actual_answer"] for _, r in gt_pairs]
        gt_refs = [r["ground_truth"] for _, r in gt_pairs]

        gt_pred_vecs = encode_texts(gt_cands, model_path, device, batch_size)
        gt_ref_vecs = encode_texts(gt_refs, model_path, device, batch_size)
        gt_scores = [_cosine(p.tolist(), r.tolist())
                     for p, r in zip(gt_pred_vecs, gt_ref_vecs)]

        result["vs_gt"] = {
            "precision": _avg(gt_scores),
            "recall": _avg(gt_scores),
            "f1": _avg(gt_scores),
        }

    return result


def print_group_metrics(label: str, metrics: dict) -> None:
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {label}  ({metrics['count']} вопросов)")
    print(sep)
    vc = metrics["vs_context"]
    print(f"  Ответ vs Контекст:")
    print(f"    Precision: {vc['precision']:.4f}  Recall: {vc['recall']:.4f}  F1: {vc['f1']:.4f}")
    if "vs_gt" in metrics:
        vg = metrics["vs_gt"]
        print(f"  Ответ vs Эталон:")
        print(f"    Precision: {vg['precision']:.4f}  Recall: {vg['recall']:.4f}  F1: {vg['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="BERTScore оценка")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "raw_answers.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=_auto_device())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-skip-compromat", action="store_true", default=False)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Ошибка: {args.input} не найден. Сначала запустите 01_collect_answers.py")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    valid = [r for r in records if r.get("actual_answer") and not r.get("error")]
    print(f"Загружено записей: {len(valid)} из {len(records)}")
    print(f"Модель: {args.model}  |  Устройство: {args.device}\n")

    # Группируем по категориям
    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r.get("category", "Unknown")].append(r)

    skip_compromat = not args.no_skip_compromat
    categories_to_eval = {
        cat: recs for cat, recs in by_cat.items()
        if not (skip_compromat and cat.lower() == "compomat")
    }

    # Все записи для общей метрики (без Compomat если skip)
    all_for_overall = []
    for recs in categories_to_eval.values():
        all_for_overall.extend(recs)

    report = {
        "model": args.model,
        "device": args.device,
        "overall": {},
        "by_category": {},
        "per_sample": [],
    }

    # Метрики по каждой категории
    cat_results = {}
    for cat, recs in sorted(categories_to_eval.items()):
        print(f"\n--- Считаем BERTScore для категории: {cat} ({len(recs)} вопросов) ---")
        cat_results[cat] = compute_scores_for_group(recs, args.model, args.device, args.batch_size)

    # Общие метрики
    print(f"\n--- Считаем общий BERTScore ({len(all_for_overall)} вопросов) ---")
    overall = compute_scores_for_group(all_for_overall, args.model, args.device, args.batch_size)

    # Вывод в консоль
    print("\n\n" + "█" * 55)
    print("  РЕЗУЛЬТАТЫ BERTSCORE")
    print("█" * 55)

    print_group_metrics("ОБЩИЕ МЕТРИКИ", overall)

    for cat, metrics in sorted(cat_results.items()):
        print_group_metrics(f"Категория: {cat}", metrics)

    # Сохранение
    report["overall"] = overall
    report["by_category"] = cat_results

    report_path = RESULTS_DIR / "bertscore_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Текстовая сводка
    lines = ["=== BERTScore Evaluation Summary ===\n",
             f"Модель: {args.model}", f"Устройство: {args.device}", ""]

    lines.append("--- ОБЩИЕ МЕТРИКИ ---")
    lines.append(f"  Вопросов: {overall['count']}")
    for mode, scores in {k: v for k, v in overall.items() if isinstance(v, dict)}.items():
        lines.append(f"  {mode}:")
        for k, v in scores.items():
            lines.append(f"    {k:<12} {v:.4f}")

    for cat, metrics in sorted(cat_results.items()):
        lines.append(f"\n--- Категория: {cat} ({metrics['count']} вопросов) ---")
        for mode, scores in {k: v for k, v in metrics.items() if isinstance(v, dict)}.items():
            lines.append(f"  {mode}:")
            for k, v in scores.items():
                lines.append(f"    {k:<12} {v:.4f}")

    summary_path = RESULTS_DIR / "bertscore_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ Отчёт:  {report_path}")
    print(f"✓ Сводка: {summary_path}")


if __name__ == "__main__":
    main()