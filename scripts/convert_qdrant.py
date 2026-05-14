"""Конвертер исходных JSON файлов в формат documents.json для ingest_qdrant.py.

Читает 4 исходных файла из data/qdrant/source/ и собирает единый
data/qdrant/documents.json в формате:
    [{"id": int, "text": str, "source": str, "group": str}, ...]

Запуск:
    python scripts/convert_qdrant.py

После запуска проверь data/qdrant/documents.json вручную,
затем запусти: python scripts/ingest_qdrant.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Настройки источников — заполни URL перед запуском
# =============================================================================

SOURCES = {
    "qdrant_pravila_priema.json": {
        "source": "TODO_URL_PRAVILA_PRIEMA",          # например: https://abit.magtu.ru/priem/pravila-priema
        "group": "Правила приёма",
    },
    "qdrant_ind_achievements_bak_spec.json": {
        "source": "TODO_URL_IND_ACHIEVEMENTS_BAK",    # например: https://abit.magtu.ru/priem/ind-dostizheniya-bak
        "group": "Индивидуальные достижения (бакалавриат/специалитет)",
    },
    "qdrant_ind_achievements_mag.json": {
        "source": "TODO_URL_IND_ACHIEVEMENTS_MAG",    # например: https://abit.magtu.ru/priem/ind-dostizheniya-mag
        "group": "Индивидуальные достижения (магистратура)",
    },
    "qdrant_osobie_prava.json": {
        "source": "TODO_URL_OSOBIE_PRAVA",            # URL возьми из поля meta[0] в исходном файле
        "group": "Особые права и преимущества",
    },
}

SOURCE_DIR = Path("data/qdrant/source")
OUTPUT_FILE = Path("data/qdrant/documents.json")


# =============================================================================
# Обработчики — по одному на каждую структуру файла
# =============================================================================

def process_pravila_priema(data: dict, source: str, group: str) -> list[dict]:
    """Формат: {"Глава I": ["параграф 1", "параграф 2", ...]}"""
    results = []
    for chapter, paragraphs in data.items():
        for paragraph in paragraphs:
            text = f"Документ: Правила приёма. {chapter}. {paragraph}"
            results.append({"text": text.strip(), "source": source, "group": group})
    return results


def process_ind_achievements(data: dict, source: str, group: str) -> list[dict]:
    """Формат: {"Раздел": [{"Название индивидуального достижения": ..., "Документы": ..., "Балл": ...}]}"""
    results = []
    for section_name, items in data.items():
        for item in items:
            parts = []

            if "Название индивидуального достижения" in item:
                parts.append(
                    f"Индивидуальное достижение: {item['Название индивидуального достижения']}"
                )
            if "Документы" in item:
                parts.append(f"Подтверждающие документы: {item['Документы']}")
            if "Балл" in item:
                parts.append(f"Дополнительные баллы: {item['Балл']}")

            text = ". ".join(parts)
            results.append({"text": text.strip(), "source": source, "group": group})

    return results


def process_osobie_prava(data: list, source: str, group: str) -> list[dict]:
    """Формат: [{"text": ..., "meta": ["url", "title", "level"]}]
    URL берётся из meta[0] если есть, иначе из source-заглушки.
    """
    results = []
    for item in data:
        text = item.get("text", "").strip()
        if not text:
            continue

        # Используем URL из meta если он есть и не пустой
        meta = item.get("meta", [])
        item_source = meta[0] if meta and meta[0] else source

        results.append({"text": text, "source": item_source, "group": group})

    return results


# =============================================================================
# Роутер — выбирает обработчик по имени файла
# =============================================================================

PROCESSORS = {
    "qdrant_pravila_priema.json": process_pravila_priema,
    "qdrant_ind_achievements_bak_spec.json": process_ind_achievements,
    "qdrant_ind_achievements_mag.json": process_ind_achievements,
    "qdrant_osobie_prava.json": process_osobie_prava,
}


# =============================================================================
# Main
# =============================================================================

def convert() -> None:
    if not SOURCE_DIR.exists():
        print(f"Error: source directory not found: {SOURCE_DIR}")
        print("Создай директорию data/qdrant/source/ и положи туда исходные JSON файлы.")
        sys.exit(1)

    all_documents: list[dict] = []

    for filename, meta in SOURCES.items():
        file_path = SOURCE_DIR / filename

        if not file_path.exists():
            print(f"  ⚠ Файл не найден, пропускаю: {filename}")
            continue

        processor = PROCESSORS.get(filename)
        if not processor:
            print(f"  ⚠ Нет обработчика для: {filename}")
            continue

        print(f"  Обрабатываю {filename}...", end=" ", flush=True)

        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)

        chunks = processor(data, source=meta["source"], group=meta["group"])
        all_documents.extend(chunks)

        print(f"{len(chunks)} чанков ✓")

    if not all_documents:
        print("Нет документов для сохранения.")
        sys.exit(1)

    # Проставляем последовательные id
    for idx, doc in enumerate(all_documents):
        doc["id"] = idx + 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Готово. Всего документов: {len(all_documents)}")
    print(f"  Сохранено: {OUTPUT_FILE}")
    print()
    print("Следующий шаг:")
    print("  1. Проверь data/qdrant/documents.json")
    print("  2. Замени TODO_URL_* в scripts/convert_qdrant.py на реальные ссылки")
    print("  3. Запусти: python scripts/ingest_qdrant.py")


if __name__ == "__main__":
    convert()