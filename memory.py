"""
memory.py
─────────────────────────────────────────────────────────────────────────
Long-Term Memory API used by the agents / graph / main.py.

Public functions:
    save_project_memory(...)     -> upsert a full project record
    load_project_memory(...)     -> fetch one project by project_id
    search_similar_projects(...) -> full-text search over past requirements
    update_project_memory(...)   -> partial update of an existing record
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from database import get_db_session
from models import ProjectMemory

_ALLOWED_UPDATE_FIELDS = {
    "thread_id",
    "requirement",
    "user_stories",
    "architecture",
    "module_plans",
    "generated_code",
    "review_scores",
    "completed_modules",
    "human_feedback",
    "delivery_package_metadata",
    "execution_history",
}


def save_project_memory(
    project_id: str,
    thread_id: str,
    requirement: str,
    user_stories: Optional[List[str]] = None,
    architecture: Optional[str] = None,
    module_plans: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, str]] = None,
    review_scores: Optional[Dict[str, Any]] = None,
    completed_modules: Optional[List[str]] = None,
    human_feedback: str = "",
    delivery_package_metadata: Optional[Dict[str, Any]] = None,
    execution_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Insert a new project memory row, or UPDATE it in place if project_id
    already exists (upsert). Safe to call multiple times for the same
    project_id (e.g. once from delivery_node, once from main.py).
    """
    with get_db_session() as session:
        record = (
            session.query(ProjectMemory)
            .filter(ProjectMemory.project_id == project_id)
            .one_or_none()
        )

        if record is None:
            record = ProjectMemory(project_id=project_id)
            session.add(record)

        record.thread_id = thread_id
        record.requirement = requirement
        record.user_stories = user_stories or []
        record.architecture = architecture
        record.module_plans = module_plans or {}
        record.generated_code = generated_code or {}
        record.review_scores = review_scores or {}
        record.completed_modules = completed_modules or []
        record.human_feedback = human_feedback or ""
        record.delivery_package_metadata = delivery_package_metadata or {}
        record.execution_history = execution_history or []
        record.updated_at = datetime.now(timezone.utc)

        session.flush()
        return record.to_dict()


def load_project_memory(project_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one project's stored memory. Returns None if not found."""
    if not project_id:
        return None

    with get_db_session() as session:
        record = (
            session.query(ProjectMemory)
            .filter(ProjectMemory.project_id == project_id)
            .one_or_none()
        )
        return record.to_dict() if record else None


def search_similar_projects(requirement: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Full-text search over previously stored requirements using PostgreSQL's
    built-in tsvector/tsquery ranking (no extra extensions required).
    Returns the most relevant past projects, most relevant first.
    """
    if not requirement or not requirement.strip():
        return []

    sql = text(
        """
        SELECT
            project_id,
            thread_id,
            requirement,
            architecture,
            module_plans,
            generated_code,
            completed_modules,
            updated_at,
            ts_rank(
                to_tsvector('english', requirement),
                plainto_tsquery('english', :query)
            ) AS rank
        FROM project_memory
        WHERE to_tsvector('english', requirement) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit
        """
    )

    with get_db_session() as session:
        rows = session.execute(sql, {"query": requirement, "limit": limit}).mappings().all()

        results: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            if d.get("updated_at"):
                d["updated_at"] = d["updated_at"].isoformat()
            d.pop("rank", None)
            results.append(d)
        return results


def update_project_memory(project_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """
    Partially update an existing project memory record.
    Only keys in _ALLOWED_UPDATE_FIELDS are applied. Returns None if the
    project_id does not exist yet (use save_project_memory to create it).
    """
    if not project_id:
        return None

    with get_db_session() as session:
        record = (
            session.query(ProjectMemory)
            .filter(ProjectMemory.project_id == project_id)
            .one_or_none()
        )
        if record is None:
            return None

        for key, value in fields.items():
            if key in _ALLOWED_UPDATE_FIELDS:
                setattr(record, key, value)

        record.updated_at = datetime.now(timezone.utc)
        session.flush()
        return record.to_dict()
