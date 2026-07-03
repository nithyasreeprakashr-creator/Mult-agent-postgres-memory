"""
models.py
─────────────────────────────────────────────────────────────────────────
SQLAlchemy ORM model(s) for Long-Term Memory storage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProjectMemory(Base):
    """
    One row per project. Re-runs of the same project_id UPDATE this row
    (upsert semantics handled in memory.py), so the table always holds the
    latest known state of every project the factory has ever built.
    """

    __tablename__ = "project_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity
    project_id = Column(String(64), unique=True, nullable=False, index=True)
    thread_id = Column(String(64), nullable=True, index=True)

    # Planning artifacts
    requirement = Column(Text, nullable=False)
    user_stories = Column(JSONB, nullable=False, default=list)
    architecture = Column(Text, nullable=True)
    module_plans = Column(JSONB, nullable=False, default=dict)

    # Build artifacts
    generated_code = Column(JSONB, nullable=False, default=dict)
    review_scores = Column(JSONB, nullable=False, default=dict)
    completed_modules = Column(JSONB, nullable=False, default=list)

    # Human-in-the-loop
    human_feedback = Column(Text, nullable=False, default="")

    # Delivery
    delivery_package_metadata = Column(JSONB, nullable=False, default=dict)

    # Audit trail
    execution_history = Column(JSONB, nullable=False, default=list)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "thread_id": self.thread_id,
            "requirement": self.requirement,
            "user_stories": self.user_stories or [],
            "architecture": self.architecture,
            "module_plans": self.module_plans or {},
            "generated_code": self.generated_code or {},
            "review_scores": self.review_scores or {},
            "completed_modules": self.completed_modules or [],
            "human_feedback": self.human_feedback or "",
            "delivery_package_metadata": self.delivery_package_metadata or {},
            "execution_history": self.execution_history or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ProjectMemory project_id={self.project_id!r} updated_at={self.updated_at!r}>"
