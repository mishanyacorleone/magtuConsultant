"""RAGChecker метрика через локальный vLLM."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# LiteLLM требует OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


async def compute_ragchecker(results: list[dict], llm_url: str = "http://localhost:9002/v1") -> dict:
    """RAGChecker: claim-level покрытие эталонного ответа.

    Использует локальный vLLM через LiteLLM (openai/<model> префикс).
    """
    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import overall_metrics
    except ImportError:
        return {"error": "ragchecker не установлен: pip install ragchecker"}

    pairs = [r for r in results if r.get("actual_answer") and r.get("ground_truth")]
    if not pairs:
        return {"error": "нет пар с actual_answer и ground_truth"}

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from app.core.config import get_settings
        s = get_settings()
        llm_base_url = llm_url  # используем переданный URL (внешний localhost)
        llm_model = s.vllm_llm_model
    except Exception:
        llm_base_url = llm_url
        llm_model = "default"

    # LiteLLM требует префикс openai/ для OpenAI-совместимых эндпоинтов
    litellm_model = llm_model if llm_model.startswith("openai/") else f"openai/{llm_model}"

    try:
        # query_id обязателен для RAGChecker
        rag_data = {
            "results": [
                {
                    "query_id": f"q_{i}",
                    "query": r["question"],
                    "gt_answer": r["ground_truth"],
                    "response": r["actual_answer"],
                    "retrieved_context": [
                        {"text": ctx} for ctx in r.get("contexts", []) if ctx
                    ],
                }
                for i, r in enumerate(pairs)
            ]
        }

        rag_results = RAGResults.from_dict(rag_data)

        checker = RAGChecker(
            extractor_name=litellm_model,
            checker_name=litellm_model,
            extractor_api_base=llm_base_url,
            checker_api_base=llm_base_url,
            batch_size_extractor=2,
            batch_size_checker=2,
        )

        checker.evaluate(rag_results, metrics=overall_metrics)

        try:
            aggregate = rag_results.get_aggregate_metrics()
        except AttributeError:
            from collections import defaultdict
            sums: dict = defaultdict(float)
            counts: dict = defaultdict(int)
            for r in rag_results.results:
                metrics_dict = r.metrics if isinstance(r.metrics, dict) else {}
                for k, v in metrics_dict.items():
                    if v is not None:
                        sums[k] += float(v)
                        counts[k] += 1
            aggregate = {k: sums[k] / counts[k] for k in sums if counts[k] > 0}

        return {
            "claim_recall": round(aggregate.get("claim_recall", 0.0), 4),
            "claim_precision": round(aggregate.get("claim_precision", 0.0), 4),
            "hallucination_rate": round(aggregate.get("hallucination_rate", 0.0), 4),
            "samples_count": len(pairs),
            "raw": {k: round(v, 4) for k, v in aggregate.items()},
        }

    except Exception as exc:
        logger.warning("RAGChecker failed: %s", exc)
        return {"error": str(exc)}