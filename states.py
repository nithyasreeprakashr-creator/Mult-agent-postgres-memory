from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# Pydantic models (structured LLM output)

class ModuleList(BaseModel):
    stories: List[str] = Field(description="User stories derived from requirements")
    modules: List[str] = Field(description="Backend module names, e.g. auth, users, inventory")


class ArchitectureDoc(BaseModel):
    tech_stack: str = Field(description="Primary tech stack description")
    db_schema: str = Field(description="Database schema definitions")
    api_endpoints: str = Field(description="Key API endpoint designs")
    folder_structure: str = Field(description="Project folder structure as ASCII tree")
    architecture_diagram: str = Field(description="ASCII diagram of module communication")


class ModuleFile(BaseModel):
    path: str
    purpose: str
    exports: List[str]


class ModulePlan(BaseModel):
    module_name: str
    files: List[ModuleFile]
    dependencies: List[str]
    api_routes: List[str]


class CodeReview(BaseModel):
    score: int = Field(description="Code quality score 1–10")
    issues: List[str] = Field(description="Issues found in the code")
    logic_correctness: str = Field(description="Brief logic analysis")
    security_check: str = Field(description="Brief security analysis")


# Reducer helpers

def _merge_dicts(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    """Merge two dicts; b wins on key collision (latest write)."""
    return {**a, **b}


def _append_list(a: List[str], b: List[str]) -> List[str]:
    """Append without duplicates."""
    seen = set(a)
    return a + [x for x in b if x not in seen]


# Top-level graph state  (shared across all nodes)

class SoftwareState(TypedDict):
    # Input
    requirement: str

    # Long-Term Memory (PostgreSQL)
    project_id: str
    thread_id: str
    memory_context: Optional[Dict]

    # Planning artifacts
    stories: List[str]
    architecture: Optional[str]
    modules: List[str]

    # Parallel worker tracking
    completed_modules: Annotated[List[str], _append_list]

    # Code artifacts (merged from all workers)
    generated_code: Annotated[Dict[str, str], operator.or_]
    tests: Annotated[Dict[str, str], operator.or_]

    # Long-Term Memory artifacts merged back from parallel workers
    module_plans: Annotated[Dict[str, str], operator.or_]
    review_scores: Annotated[Dict[str, int], operator.or_]

    human_approved: bool
    human_feedback: str
    # Final output 
    delivery_package: Optional[str]


class WorkerState(TypedDict):
    # Inherited from the dispatcher Send payload
    module_name: str
    requirement: str
    architecture: str
    stories: List[str]
    quality_guide: str

    # Long-Term Memory (PostgreSQL) — inherited from SoftwareState
    project_id: str
    thread_id: str
    memory_context: Optional[Dict]

    # Worker-local working data
    module_plan: Optional[str]
    generated_code: Dict[str, str]   
    tests: Dict[str, str]               

    # Review loop (fully local — no race conditions)
    review_score: Optional[int]
    review_issues: List[str]
    fix_attempts: int
    max_fix_attempts: int

      # -------------------------
    # Human In The Loop
    # -------------------------

    human_approved: bool
    human_feedback: str

    # Worker completion signal — merged back into SoftwareState
    completed_modules: List[str]
    module_plans: Annotated[Dict[str, str], operator.or_]
    review_scores: Annotated[Dict[str, int], operator.or_]
