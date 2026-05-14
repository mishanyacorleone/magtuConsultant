"""RAGAS метрики через локальный vLLM и локальные embeddings."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_EMB_MODEL = Path("/mnt/mishutqa/PycharmProjects/abitBot/app/models/deepvk/USER-bge-m3")

# RAGAS/LiteLLM требуют OPENAI_API_KEY даже для локального vLLM
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


async def compute_ragas(results: list[dict], llm_url: str = "http://localhost:9002/v1") -> dict:
    """RAGAS метрики: faithfulness, answer_relevancy, context_recall.

    Использует локальный vLLM как LLM-судью и локальную модель для embeddings.
    """
    try:
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        return {"error": f"зависимость не установлена: {exc}"}

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

    try:
        # LLM — локальный vLLM
        llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key="EMPTY",
            model=llm_model,
            temperature=0,
            max_tokens=512,
        )
        llm_wrapper = LangchainLLMWrapper(llm)

        # Embeddings — локальная модель через sentence-transformers
        emb_model_path = str(_LOCAL_EMB_MODEL) if _LOCAL_EMB_MODEL.exists() else "deepvk/USER-bge-m3"
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name=emb_model_path)
            emb_wrapper = LangchainEmbeddingsWrapper(emb)
        except ImportError:
            logger.warning("langchain-community не установлен, embeddings будут через vLLM")
            from langchain_openai import OpenAIEmbeddings
            # infinity-emb endpoint без /v1
            emb_base = llm_base_url.rstrip("/v1").rstrip("/")
            emb = OpenAIEmbeddings(base_url=f"{emb_base}", api_key="EMPTY", model=llm_model)
            emb_wrapper = LangchainEmbeddingsWrapper(emb)

        # Строим датасет
        MAX_CHUNKS = 3
        MAX_CHUNK_CHARS = 500
        rows = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for r in pairs:
            contexts = [
                ctx[:MAX_CHUNK_CHARS] for ctx in r.get("contexts", [])[:MAX_CHUNKS]
                if ctx
            ]
            if not contexts:
                contexts = ["(контекст не получен)"]
            rows["question"].append(r["question"])
            rows["answer"].append(r["actual_answer"])
            rows["contexts"].append(contexts)
            rows["ground_truth"].append(r["ground_truth"])

        dataset = Dataset.from_dict(rows)
        has_gt = any(r["ground_truth"].strip() for r in pairs)

        # Метрики
        from ragas.metrics import faithfulness, answer_relevancy
        metrics = [faithfulness, answer_relevancy]

        if has_gt:
            try:
                from ragas.metrics import context_recall
                metrics.append(context_recall)
            except ImportError:
                pass

        for m in metrics:
            if hasattr(m, "llm"):
                m.llm = llm_wrapper
            if hasattr(m, "embeddings"):
                m.embeddings = emb_wrapper

        try:
            result = evaluate(dataset, metrics=metrics, llm=llm_wrapper, embeddings=emb_wrapper)
        except TypeError:
            result = evaluate(dataset, metrics=metrics)

        output = {}
        for k in ["faithfulness", "answer_relevancy", "context_recall"]:
            if k in result:
                output[k] = round(float(result[k]), 4)

        output["samples_count"] = len(pairs)
        return output

    except Exception as exc:
        logger.warning("RAGAS failed: %s", exc)
        return {"error": str(exc)}