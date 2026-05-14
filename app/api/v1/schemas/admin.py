from typing import Any
from pydantic import BaseModel


# Qdrant schemas
class AddDocumentRequest(BaseModel):
    text: str
    source: str = ""
    group: str = ""

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Приём документов начинается 20 июня 2026 года",
                "source": "https://abit.magtu.ru/priem/pravila",
                "group": "Правила приёма"
            }
        }
    }


class AddDocumentBatchRequest(BaseModel):
    documents: list[dict]

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": [
                    {"text": "Документ 1", "source": "https://...", "group": "Правила приёма"},
                    {"text": "Документ 2", "source": "https://...", "group": "FAQ"}
                ]
            }
        }
    }


class DeleteByTextRequest(BaseModel):
    substring: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class LoadJsonPathRequest(BaseModel):
    json_path: str


# PostgreSQL schemas
class AddRowRequest(BaseModel):
    row_data: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "row_data": {
                    "code": "09.03.03",
                    "profile_spec_name": "прикладная информатика",
                    "mark": 193,
                    "year": 2025
                }
            }
        }
    }


class SqlQueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "example": {"query": "SELECT * FROM marks_last_years LIMIT 5"}
        }
    }


# Generic response
class AdminResponse(BaseModel):
    status: str
    message: str
    data: Any = None