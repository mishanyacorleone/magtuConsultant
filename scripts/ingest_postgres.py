"""Скрипт загрузки CSV таблиц в PostgreSQL.

Особенности формата:
    - Разделитель: табуляция
    - Массивы TEXT[]: значения через точку с запятой
    - Числовые колонки: кастуем из строки в int
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

ARRAY_COLUMNS: dict[str, list[str]] = {
    "vi_soo_vo": ["required_vi", "optional_vi_ege", "optional_vi_vuz"],
    "vi_spo":    ["required_vi"],
    "vi_mag":    ["vi_form"],
}

INTEGER_COLUMNS: dict[str, list[str]] = {
    "marks_last_years": ["mark", "year"],
    "min_max_marks":    ["min_marks", "max_marks"],
    "prices":           ["price"],
    "spec_info":        ["Plan_Budg", "Plan_Comm", "Years", "Months"],
    "vi_mag":           ["min_marks", "max_marks"],
}


def parse_array_value(value: str) -> list[str] | None:
    """Конвертирует 'a;b;c' в Python list — asyncpg передаст как TEXT[]."""
    if not value or not value.strip():
        return None
    items = [item.strip() for item in value.split(";") if item.strip()]
    return items if items else None


def parse_int_value(value: str) -> int | None:
    if not value or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def process_row(row: dict, table_name: str) -> dict:
    array_cols = ARRAY_COLUMNS.get(table_name, [])
    int_cols = INTEGER_COLUMNS.get(table_name, [])
    result = {}
    for key, value in row.items():
        if key in array_cols:
            result[key] = parse_array_value(value)
        elif key in int_cols:
            result[key] = parse_int_value(value)
        elif value == "" or value is None:
            result[key] = None
        else:
            result[key] = value
    return result


async def load_table(conn: asyncpg.Connection, csv_path: Path) -> int:
    """Загружает CSV напрямую через asyncpg (минуя SQLAlchemy для массивов)."""
    table_name = csv_path.stem

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_rows = list(reader)

    if not raw_rows:
        logger.warning("ingest_postgres_empty", table=table_name)
        return 0

    rows = [process_row(row, table_name) for row in raw_rows]
    columns = list(rows[0].keys())

    # Строим INSERT с позиционными параметрами $1, $2, ...
    col_str = ", ".join(f'"{col}"' for col in columns)
    placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
    insert_sql = f'INSERT INTO "{table_name}" ({col_str}) VALUES ({placeholders})'  # noqa: S608

    try:
        await conn.execute(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE')  # noqa: S608

        # Конвертируем список словарей в список кортежей
        records = [tuple(row[col] for col in columns) for row in rows]
        await conn.executemany(insert_sql, records)

        logger.info("ingest_postgres_table_done", table=table_name, rows=len(rows))
        return len(rows)

    except Exception as exc:
        logger.error("ingest_postgres_table_error", table=table_name, error=str(exc))
        raise


async def ingest(data_dir: Path, only_table: str | None) -> None:
    settings = get_settings()

    # Парсим DSN для asyncpg (убираем +asyncpg из драйвера)
    dsn = settings.postgres_dsn.replace("postgresql+asyncpg://", "postgresql://")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in {data_dir}")
        sys.exit(1)

    if only_table:
        csv_files = [f for f in csv_files if f.stem == only_table]
        if not csv_files:
            print(f"Error: {only_table}.csv not found in {data_dir}")
            sys.exit(1)

    logger.info("ingest_postgres_start", tables=[f.stem for f in csv_files])

    conn = await asyncpg.connect(dsn)
    total_rows = 0

    try:
        for csv_path in csv_files:
            print(f"  Loading {csv_path.stem}...", end=" ", flush=True)
            try:
                count = await load_table(conn, csv_path)
                total_rows += count
                print(f"{count} rows ✓")
            except Exception as exc:
                print(f"ERROR: {exc}")
                await conn.close()
                sys.exit(1)
    finally:
        await conn.close()

    print(f"\n✓ Загружено {total_rows} строк в {len(csv_files)} таблиц")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV files into PostgreSQL")
    parser.add_argument("--data", type=Path, default=Path("data/postgres"))
    parser.add_argument("--table", type=str, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: directory not found: {args.data}")
        sys.exit(1)

    asyncio.run(ingest(args.data, args.table))


if __name__ == "__main__":
    main()