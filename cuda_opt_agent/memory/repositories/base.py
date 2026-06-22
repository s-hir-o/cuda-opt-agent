"""Base repository with generic CRUD and JSON serialization helpers."""

from __future__ import annotations

from typing import Any, Generic, Sequence, Type, TypeVar

from pydantic import BaseModel as PydanticModel
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from cuda_opt_agent.core.exceptions import StorageError
from cuda_opt_agent.memory.database import Base

OrmT = TypeVar("OrmT", bound=Base)
PydT = TypeVar("PydT", bound=PydanticModel)


# ── JSON serialization helpers ─────────────────────────────────

def to_json_dict(model: PydanticModel | None) -> dict | None:
    """Serialize a Pydantic model to a JSON-compatible dict. None-in, None-out."""
    if model is None:
        return None
    return model.model_dump(mode="json")


def from_json_dict(data: dict | None, cls: Type[PydT]) -> PydT | None:
    """Deserialize a JSON dict to a Pydantic model. None-in, None-out."""
    if data is None:
        return None
    return cls.model_validate(data)


def to_json_list(models: Sequence[PydanticModel]) -> list[dict]:
    """Serialize a list of Pydantic models."""
    return [m.model_dump(mode="json") for m in models]


def from_json_list(data: list[dict] | None, cls: Type[PydT]) -> list[PydT]:
    """Deserialize a list of JSON dicts to Pydantic models."""
    if not data:
        return []
    return [cls.model_validate(item) for item in data]


# ── Base Repository ────────────────────────────────────────────

class BaseRepository(Generic[OrmT]):
    """Generic CRUD operations for a single ORM model.

    The session is injected — never created internally.
    Callers manage the session lifecycle via database.get_session().
    """

    def __init__(self, model: Type[OrmT], session: Session) -> None:
        self._model = model
        self._session = session

    def add(self, obj: OrmT) -> OrmT:
        """Insert and flush. Returns the object with populated PK."""
        try:
            self._session.add(obj)
            self._session.flush()
            return obj
        except Exception as exc:
            raise StorageError(f"Failed to insert {self._model.__name__}: {exc}") from exc

    def get_by_id(self, id: Any) -> OrmT | None:
        """Retrieve by primary key."""
        return self._session.get(self._model, id)

    def update(self, obj: OrmT) -> OrmT:
        """Merge (insert-or-update) and flush."""
        try:
            merged = self._session.merge(obj)
            self._session.flush()
            return merged
        except Exception as exc:
            raise StorageError(f"Failed to update {self._model.__name__}: {exc}") from exc

    def delete(self, obj: OrmT) -> None:
        """Delete and flush."""
        try:
            self._session.delete(obj)
            self._session.flush()
        except Exception as exc:
            raise StorageError(f"Failed to delete {self._model.__name__}: {exc}") from exc

    def list_all(self, offset: int = 0, limit: int = 100) -> Sequence[OrmT]:
        """Paginated list of all rows."""
        stmt = select(self._model).offset(offset).limit(limit)
        return self._session.execute(stmt).scalars().all()

    def count(self) -> int:
        """Total row count."""
        stmt = select(func.count()).select_from(self._model)
        return self._session.execute(stmt).scalar_one()