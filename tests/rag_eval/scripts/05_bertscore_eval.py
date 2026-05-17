"""
Скрипт 05: BERTScore с разбивкой по категориям из Excel (без хардкода).

Категории берутся из raw_answers.json как есть (lower).
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


def encode(texts: list[str], model_path: str, device: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Ошибка: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(model_path, device=device)
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)


def compute_group(records: list[dict], model_path: str, device: str, batch_size: int) -> dict:
    responses = [r["actual_answer"] for r in records]
    has_gt = any(r.get("ground_truth", "").strip() for r in records)

    # Vs context
    ctx_refs = []
    for r in records:
        chunks = [c["text"] for c in r.get("retrieved_context", []) if c.get("text")]
        ctx_refs.append(" ".join(chunks) if chunks else "(контекст не получен)")

    pred_vecs = encode(responses, model_path, device, batch_size)
    ctx_vecs = encode(ctx_refs, model_path, device, batch_size)
    ctx_scores = [_cosine(p.tolist(), c.tolist()) for p, c in zip(pred_vecs, ctx_vecs)]

    result: dict = {
        "count": len(records),
        "vs_context": {"precision": _avg(ctx_scores), "recall": _avg(ctx_scores), "f1": _avg(ctx_scores)},
    }

    # Vs ground_truth
    if has_gt:
        gt_pairs = [(i, r) for i, r in enumerate(records) if r.get("ground_truth", "").strip()]
        gt_cands = [r["actual_answer"] for _, r in gt_pairs]
        gt_refs = [r["ground_truth"] for _, r in gt_pairs]
        gt_pred = encode(gt_cands, model_path, device, batch_size)
        gt_ref = encode(gt_refs, model_path, device, batch_size)
        gt_scores = [_cosine(p.tolist(), r.tolist()) for p, r in zip(gt_pred, gt_ref)]
        result["vs_gt"] = {"precision": _avg(gt_scores), "recall": _avg(gt_scores), "f1": _avg(gt_scores)}

    return result


def print_group(label: str, metrics: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {label}  ({metrics['count']} вопросов)")
    print(f"{'=' * 55}")
    vc = metrics["vs_context"]
    print(f"  Ответ vs Контекст:  P={vc['precision']:.4f}  R={vc['recall']:.4f}  F1={vc['f1']:.4f}")
    if "vs_gt" in metrics:
        vg = metrics["vs_gt"]
        print(f"  Ответ vs Эталон:    P={vg['precision']:.4f}  R={vg['recall']:.4f}  F1={vg['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "raw_answers.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=_auto_device())
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Ошибка: {args.input} не найден. Сначала запустите 01_collect_answers.py")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    valid = [r for r in records if r.get("actual_answer") and not r.get("error")]
    print(f"Записей: {len(valid)} из {len(records)}")
    print(f"Модель: {args.model}  |  Устройство: {args.device}")

    # Группировка по категориям из данных (без хардкода)
    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r.get("category", "unknown")].append(r)

    print(f"Категории: {sorted(by_cat.keys())}\n")

    # Считаем по каждой категории
    cat_results = {}
    for cat in sorted(by_cat.keys()):
        print(f"\n--- BERTScore: {cat} ({len(by_cat[cat])} вопросов) ---")
        cat_results[cat] = compute_group(by_cat[cat], args.model, args.device, args.batch_size)

    # Общие по всем
    print(f"\n--- BERTScore: общие ({len(valid)} вопросов) ---")
    overall = compute_group(valid, args.model, args.device, args.batch_size)

    # Консольный вывод
    print("\n\n" + "█" * 55)
    print("  РЕЗУЛЬТАТЫ BERTSCORE")
    print("█" * 55)
    print_group("ОБЩИЕ МЕТРИКИ", overall)
    for cat in sorted(cat_results.keys()):
        print_group(f"Категория: {cat}", cat_results[cat])

    # Сохранение
    report = {"model": args.model, "device": args.device,
              "overall": overall, "by_category": cat_results}
    report_path = RESULTS_DIR / "bertscore_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"=== BERTScore Summary ===\nМодель: {args.model}\n"]
    lines.append(f"ОБЩИЕ ({overall['count']} вопросов):")
    for mode, s in {k: v for k, v in overall.items() if isinstance(v, dict)}.items():
        lines.append(f"  {mode}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f}")
    for cat in sorted(cat_results.keys()):
        m = cat_results[cat]
        lines.append(f"\n{cat} ({m['count']} вопросов):")
        for mode, s in {k: v for k, v in m.items() if isinstance(v, dict)}.items():
            lines.append(f"  {mode}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f}")

    summary_path = RESULTS_DIR / "bertscore_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Отчёт: {report_path}\n✓ Сводка: {summary_path}")


if __name__ == "__main__":
    main()