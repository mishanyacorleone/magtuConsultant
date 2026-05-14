"""Семантическое сходство через локальную модель USER-bge-m3.

Загружает модель напрямую через sentence-transformers — никаких сетевых запросов.
"""

import logging
import math
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Путь к локальной модели
_LOCAL_MODEL = Path("/mnt/mishutqa/PycharmProjects/abitBot/app/models/deepvk/USER-bge-m3")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


async def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Считает косинусное сходство через локальную модель USER-bge-m3.

    Использует sentence-transformers напрямую — без сетевых запросов.
    """
    if not predictions or not references:
        return {"error": "empty predictions or references"}

    try:
        from sentence_transformers import SentenceTransformer

        model_path = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "deepvk/USER-bge-m3"
        logger.info("Загружаем BERTScore модель: %s", model_path)

        model = SentenceTransformer(model_path, device="cpu")
        pred_vecs = model.encode(predictions, batch_size=8, show_progress_bar=False).tolist()
        ref_vecs = model.encode(references, batch_size=8, show_progress_bar=False).tolist()

        scores = [_cosine(p, r) for p, r in zip(pred_vecs, ref_vecs)]

        return {
            "f1_mean": round(sum(scores) / len(scores), 4),
            "f1_min": round(min(scores), 4),
            "f1_max": round(max(scores), 4),
            "f1_scores": [round(s, 4) for s in scores],
            "samples_count": len(scores),
            "model": model_path,
        }

    except ImportError:
        return {"error": "sentence-transformers не установлен: pip install sentence-transformers"}
    except Exception as exc:
        logger.warning("BERTScore failed: %s", exc)
        return {"error": str(exc)}