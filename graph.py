from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import Send
from langgraph.graph import END, StateGraph
from psycopg_pool import ConnectionPool

from database import CHECKPOINT_DATABASE_URL

from agents import (
    _build_worker_payload,
    architect_node,
    delivery_node,
    planner_node,
    qa_node,
    worker_coder,
    worker_complete,
    worker_fixer,
    worker_module_planner,
    worker_reviewer,
    human_review_node,
)
from state import SoftwareState, WorkerState


# WORKER SUBGRAPH

def _route_after_review(state: WorkerState) -> Literal["worker_fixer", "worker_complete"]:

    score = state.get("review_score", 0) or 0
    attempts = state.get("fix_attempts", 0)
    max_att = state.get("max_fix_attempts", 3)
    threshold = 8

    if score >= threshold:
        print(f"      ✓ Score {score}/10 PASS")
        return "worker_complete"

    if attempts >= max_att:
        print(f"      ⚠ Max retry reached")
        return "worker_complete"

    print(f"      ✗ Score {score}/10 RETRY")
    return "worker_fixer"


worker_builder = StateGraph(WorkerState)

worker_builder.add_node("worker_module_planner", worker_module_planner)
worker_builder.add_node("worker_coder", worker_coder)
worker_builder.add_node("worker_reviewer", worker_reviewer)
worker_builder.add_node("worker_fixer", worker_fixer)
worker_builder.add_node("worker_complete", worker_complete)

worker_builder.set_entry_point("worker_module_planner")
worker_builder.add_edge("worker_module_planner", "worker_coder")
worker_builder.add_edge("worker_coder","worker_reviewer")

worker_builder.add_conditional_edges(
    "worker_reviewer",
    _route_after_review,
    {
        "worker_fixer": "worker_fixer",
        "worker_complete": "worker_complete",
    },
)
worker_builder.add_edge("worker_fixer",    "worker_reviewer")   # loop back
worker_builder.add_edge("worker_complete", END)

worker_graph = worker_builder.compile()

def worker_entry_node(state):
    result = worker_graph.invoke(state)

    return {
        "completed_modules": result.get(
            "completed_modules",
            []
        ),
        "generated_code": result.get(
            "generated_code",
            {}
        ),
        "tests": result.get(
            "tests",
            {}
        ),
        "module_plans": result.get(
            "module_plans",
            {}
        ),
        "review_scores": result.get(
            "review_scores",
            {}
        ),
    }


# MAIN GRAPH — dispatcher (fan-out node)

def dispatcher_node(state: SoftwareState) -> list[Send]:
    """
    Convert each module name into a Send targeting 'worker_entry'.
    LangGraph will execute all Sends in parallel.
    """
    modules = state.get("modules", [])
    print("─" * 60)
    print(f"🚀  DISPATCHER: Spawning {len(modules)} parallel workers …")
    for m in modules:
        print(f"   → Worker: {m}")

    return [
        Send("worker_entry", _build_worker_payload(state, module))
        for module in modules
    ]


# MAIN GRAPH construction

main_builder = StateGraph(SoftwareState)

#  Register nodes 
main_builder.add_node("planner",      planner_node)
main_builder.add_node("architect",    architect_node)

main_builder.add_node("dispatcher",   lambda s: {}) # pass-through

main_builder.add_node(
    "worker_entry",
    worker_entry_node
)   # compiled subgraph

main_builder.add_node("human_review",human_review_node)
main_builder.add_node("qa",           qa_node)
main_builder.add_node("delivery",     delivery_node)

# Linear backbone 
main_builder.set_entry_point("planner")
main_builder.add_edge("planner",   "architect")
main_builder.add_edge("architect", "dispatcher")
# Fan-out: dispatcher → parallel workers
main_builder.add_conditional_edges(
    "dispatcher",
    dispatcher_node,
    ["worker_entry"],
)


# Route after Human Approval
def route_after_human(state: SoftwareState) -> Literal["qa", "dispatcher"]:

    if state["human_approved"]:
        print("\n✅ Human Approved. Proceeding to QA...\n")
        return "qa"

    print("\n🔄 Human requested regeneration...\n")
    return "dispatcher"


# Fan-in: after ALL workers complete → qa 
main_builder.add_edge("worker_entry","human_review")

main_builder.add_conditional_edges(
    "human_review",
    route_after_human,
    {
        "qa": "qa",
        "dispatcher": "dispatcher",
    },
)

main_builder.add_edge(
    "qa",
    "delivery"
)

main_builder.add_edge(
    "delivery",
    END
)

#  Compile with PostgreSQL-backed checkpointing (Long-Term Memory)
#
#  PostgresSaver persists LangGraph's short-term/thread checkpoints
#  (the conversational/interrupt state) to PostgreSQL instead of RAM,
#  so a thread survives process restarts. This is separate from — and
#  complementary to — the ProjectMemory table in models.py, which stores
#  the durable, cross-project "long-term memory" the agents read from.
_checkpoint_pool = ConnectionPool(
    conninfo=CHECKPOINT_DATABASE_URL,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)

checkpointer = PostgresSaver(_checkpoint_pool)
checkpointer.setup()  # idempotent — creates checkpoint tables on first run

app = main_builder.compile(checkpointer=checkpointer)
