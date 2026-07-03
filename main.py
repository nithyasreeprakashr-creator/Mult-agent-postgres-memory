from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph import app
from state import SoftwareState
from langgraph.types import Command

from database import check_connection, init_db
from memory import load_project_memory, save_project_memory, search_similar_projects


# Initial state

REQUIREMENT = (
    "Build a Hospital Management System with Authentication and User Management. "
    "Include role-based access control (admin, doctor, receptionist), "
    "patient registration, appointment scheduling, medical records management, "
    "prescription tracking, and billing management."
)

# Long-Term Memory (PostgreSQL) — identity + previous-context loading
#
# Set PROJECT_ID in .env to resume/continue a specific past project;
# otherwise a fresh project_id is generated for this run.
PROJECT_ID = os.getenv("PROJECT_ID") or str(uuid.uuid4())
THREAD_ID = os.getenv("THREAD_ID") or f"thread-{PROJECT_ID}"

_pg_available = check_connection()
memory_context = None

if _pg_available:
    init_db()
    existing_project = load_project_memory(PROJECT_ID)
    similar_projects = search_similar_projects(REQUIREMENT, limit=3)
    memory_context = {"current": existing_project, "similar": similar_projects}

    if existing_project:
        print(f"🧠  Resuming project_id={PROJECT_ID} from PostgreSQL long-term memory.")
    if similar_projects:
        print(f"🧠  Found {len(similar_projects)} similar past project(s) in memory.")
else:
    print("⚠  PostgreSQL unavailable — continuing WITHOUT long-term memory context.")


initial_state: SoftwareState = {
    "requirement": REQUIREMENT,

    "project_id": PROJECT_ID,
    "thread_id": THREAD_ID,
    "memory_context": memory_context,

    "stories": [],
    "architecture": None,
    "modules": [],

    "completed_modules": [],

    "generated_code": {},
    "tests": {},

    "module_plans": {},
    "review_scores": {},

    "human_approved": False,
    "human_feedback": "",

    "delivery_package": None,
}

# Run

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   🏭  PARALLEL AI SOFTWARE FACTORY  (LangGraph)     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": 300,
    }
    print(f"🧵  project_id = {PROJECT_ID}")
    print(f"🧵  thread_id  = {THREAD_ID}")

    t0 = time.time()

    try:
        # Start the workflow
        result = app.invoke(initial_state, config=config)

        # Human-in-the-Loop
        while "__interrupt__" in result:

            interrupt_data = result["__interrupt__"][0].value

            print("\n" + "=" * 60)
            print(" HUMAN REVIEW REQUIRED ")
            print("=" * 60)

            print(f"Module : {interrupt_data['module']}")
            print(f"Score  : {interrupt_data['review_score']}/10")

            print("\nReviewer Issues:")
            for issue in interrupt_data["review_issues"]:
                print(f"- {issue}")

            print("\nGenerated Code:\n")
            print(interrupt_data["generated_code"])

            choice = input("\nApprove? (yes/no): ").strip().lower()

            feedback = ""

            if choice == "no":
                feedback = input("Enter feedback: ")

            result = app.invoke(
                Command(
                    resume={
                        "approved": choice == "yes",
                        "feedback": feedback,
                    }
                ),
                config=config,
            )
    except Exception as exc:
        print(f"\n❌  Fatal error: {exc}")
        raise

    elapsed = time.time() - t0

    # Summary
    print()
    print("═" * 60)
    print("🎉  WORKFLOW COMPLETE")
    print("═" * 60)
    print(f"\n⏱  Total time        : {elapsed:.1f}s")
    print(f"📦  Modules completed : {result.get('completed_modules', [])}")
    print(f"📄  Code files        : {len(result.get('generated_code', {}))}")
    print(f"🧪  Test files        : {len(result.get('tests', {}))}")

    print("\n── Architecture (first 600 chars) ──────────────────────")
    arch = result.get("architecture") or ""
    print((arch[:600] + "…") if len(arch) > 600 else arch)

    print("\n── Generated Modules ────────────────────────────────────")
    for module, code in result.get("generated_code", {}).items():
        snippet = code[:200].replace("\n", " ")
        print(f"\n  [{module}]  ({len(code):,} chars)")
        print(f"  {snippet}…")

    print("\n── Delivery Package ─────────────────────────────────────")
    pkg = result.get("delivery_package") or ""
    print(f"  Size  : {len(pkg):,} chars")
    print(f"  Saved : outputs/delivery_package.md")
    print(f"  Saved : outputs/architecture.md")
    print()

    # Save project memory to PostgreSQL (safety-net upsert — delivery_node
    # already saved this inside the graph; this confirms it persisted even
    # if the graph exited via an unexpected path).
    if _pg_available:
        try:
            save_project_memory(
                project_id=PROJECT_ID,
                thread_id=THREAD_ID,
                requirement=REQUIREMENT,
                user_stories=result.get("stories", []),
                architecture=result.get("architecture"),
                module_plans=result.get("module_plans", {}),
                generated_code=result.get("generated_code", {}),
                review_scores=result.get("review_scores", {}),
                completed_modules=result.get("completed_modules", []),
                human_feedback=result.get("human_feedback", ""),
                delivery_package_metadata={
                    "path": "outputs/delivery_package.md",
                    "size_chars": len(pkg),
                    "modules_delivered": result.get("completed_modules", []),
                },
                execution_history=[
                    {
                        "event": "workflow_complete",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(elapsed, 2),
                    }
                ],
            )
            print(f"💾  Long-term memory confirmed in PostgreSQL (project_id={PROJECT_ID})\n")
        except Exception as exc:
            print(f"⚠  Could not save project memory to PostgreSQL: {exc}\n")
