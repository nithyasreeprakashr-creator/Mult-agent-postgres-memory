from __future__ import annotations
from langgraph.types import interrupt

import json
import os
from datetime import datetime, timezone
from typing import Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from memory import save_project_memory
from state import (
    ArchitectureDoc,
    CodeReview,
    ModuleList,
    ModulePlan,
    SoftwareState,
    WorkerState,
)

load_dotenv()

# LLM setup

BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "minimax-m2.1:cloud")

print("BASE_URL =", os.getenv("OLLAMA_BASE_URL"))
print("MODEL =", os.getenv("OLLAMA_MODEL"))
print("API KEY EXISTS =", bool(os.getenv("OLLAMA_API_KEY")))

llm = ChatOllama(base_url=BASE_URL, model=MODEL_NAME, temperature=0)

planner_llm       = llm.with_structured_output(ModuleList,    method="json_mode")
architect_llm     = llm.with_structured_output(ArchitectureDoc, method="json_mode")
module_planner_llm = llm.with_structured_output(ModulePlan,   method="json_mode")
reviewer_llm      = llm.with_structured_output(CodeReview,    method="json_mode")

def invoke_model(model, prompt, **kwargs):
    messages = prompt.format_messages(**kwargs)
    return model.invoke(messages)
# Quality guide 

QUALITY_GUIDE = """
## Quality Requirements

### Python & FastAPI
- Use `datetime.now(timezone.utc)` NOT `datetime.utcnow()`
- ALL database sessions must come via `Depends(get_db)` — never create `Session()` directly
- Use `async def` for all endpoints with async SQLAlchemy sessions
- Add type hints on ALL function parameters and return values

### Security
- NEVER hardcode secrets — use `os.getenv()` via a `config` module
- Hash passwords with bcrypt via `passlib`
- Use JWT with explicit expiration for authentication tokens
- Validate ALL user input via Pydantic schemas

### Architecture
- Separate files: `schemas.py`, `models.py`, `crud.py`, `routes.py`
- CRUD layer must NOT raise `HTTPException`
- Use dependency injection (`Depends`) for all services and DB access

### Database
- Use proper transactions: commit in the API layer
- Use `selectinload` / `joinedload` to avoid N+1 queries
- Add `order_by` for all paginated queries
- Set sensible defaults AND maximum bounds for `limit` / `offset` pagination

### Code Quality
- No unused imports
- No bare `except:` — catch specific exception types
- No mutable default arguments
- Use `enum.Enum` for fixed value sets
""".strip()


# Utility helpers

def _dict_to_str(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return "\n".join(f"- {_dict_to_str(v)}" for v in val)
    if isinstance(val, dict):
        return "\n".join(f"{k}: {_dict_to_str(v)}" for k, v in val.items())
    return str(val)

def _format_memory_snippet(
    memory_context: Optional[dict],
    module: Optional[str] = None,
    max_chars: int = 1200,
) -> str:
    """
    Turn the long-term memory (loaded from PostgreSQL in main.py) into a
    short, prompt-friendly text block. Used by every agent so it can build
    on prior runs instead of starting from a blank slate.

    memory_context shape (built in main.py):
        {
            "current": {...ProjectMemory dict...} | None,   # same project_id, if resumed
            "similar": [ {...ProjectMemory dict...}, ... ]  # full-text search matches
        }
    """
    if not memory_context:
        return "No previous project memory available."

    parts: list[str] = []

    current = memory_context.get("current")
    if current:
        parts.append(f"Resumed project requirement: {current.get('requirement', '')}")
        if current.get("architecture"):
            parts.append(
                f"Previously stored architecture:\n{current['architecture'][:max_chars]}"
            )
        if module:
            prev_code = (current.get("generated_code") or {}).get(module)
            if prev_code:
                parts.append(
                    f"Previously generated code for '{module}':\n{prev_code[:max_chars]}"
                )
            prev_plan = (current.get("module_plans") or {}).get(module)
            if prev_plan:
                parts.append(
                    f"Previous module plan for '{module}':\n{prev_plan[:max_chars]}"
                )

    similar = memory_context.get("similar") or []
    for proj in similar[:2]:
        if current and proj.get("project_id") == current.get("project_id"):
            continue  # avoid duplicating the resumed project
        parts.append(
            f"Similar past project (id={proj.get('project_id')}): "
            f"{(proj.get('requirement') or '')[:200]}"
        )
        if proj.get("architecture"):
            parts.append(f"  Architecture excerpt: {proj['architecture'][:max_chars]}")
        if module:
            prev_code = (proj.get("generated_code") or {}).get(module)
            if prev_code:
                parts.append(
                    f"  Code excerpt for '{module}' from that project:\n{prev_code[:max_chars]}"
                )

    return "\n\n".join(parts) if parts else "No previous project memory available."


def _safe_json(text: str) -> dict:
    """Strip markdown fences and parse JSON; return {} on failure."""
    clean = text.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        return {}


# MAIN GRAPH — NODE 1  PLANNER

def planner_node(state: SoftwareState) -> dict:
    print("─" * 60)
    print("📝  PLANNER: Analysing requirements …")

    memory_snippet = _format_memory_snippet(state.get("memory_context"))

    prompt = ChatPromptTemplate.from_template(
        "You are a software planner. Analyse the requirement and "
        "produce user stories plus backend module names.\n"
        "Module names: simple lowercase strings like 'auth', 'users', 'inventory'.\n\n"
        "Requirement: {requirement}\n\n"
        "Previous Project Memory (reuse useful patterns, avoid repeating mistakes):\n"
        "{memory}\n\n"
        "Return ONLY a flat JSON object:\n"
        '{{"stories": ["..."], "modules": ["..."]}}'
    )

    try:
       result: ModuleList = invoke_model(
            planner_llm,
            prompt,
            requirement=state["requirement"],
            memory=memory_snippet,
        )
    except (OutputParserException, Exception) as exc:
        print(f"   ⚠  Planner structured output failed ({exc}); falling back …")
        messages = prompt.format_messages(
            requirement=state["requirement"],
            memory=memory_snippet,
        )
        raw = llm.invoke(messages)
        raw_dict = _safe_json(raw.content)
        stories = raw_dict.get("stories") or [f"Implement {state['requirement']}"]
        modules = raw_dict.get("modules") or ["app"]
        result = ModuleList(stories=stories, modules=modules)

    # Normalise modules to plain strings
    clean_modules = []
    for m in result.modules:
        if isinstance(m, dict):
            clean_modules.append(m.get("name", str(m)))
        else:
            clean_modules.append(str(m))

    print(f"   Stories: {len(result.stories)} | Modules: {clean_modules}")
    return {
        "stories":           result.stories,
        "modules":           clean_modules,
        "completed_modules": [],
        "generated_code":    {},
        "tests":             {},
        "module_plans":      {},
        "review_scores":     {},
    }


# MAIN GRAPH — NODE 2  ARCHITECT

def architect_node(state: SoftwareState) -> dict:
    print("─" * 60)
    print("🏗️   ARCHITECT: Designing system …")

    memory_snippet = _format_memory_snippet(state.get("memory_context"))

    prompt = ChatPromptTemplate.from_template(
        "You are a software architect. Design a FastAPI backend system.\n"
        "User stories:\n{stories}\n\n"
        "Modules to design:\n{modules}\n\n"
        "Previous Project Memory (reuse conventions/tech stack where sensible "
        "for consistency across projects):\n{memory}\n\n"
        "Respond ONLY with a flat JSON object containing exactly these five string fields:\n"
        "tech_stack, db_schema, api_endpoints, folder_structure, architecture_diagram.\n"
        "Each value must be a plain string (no nested objects or arrays).\n"
        "folder_structure: multi-line ASCII tree.\n"
        "architecture_diagram: ASCII diagram showing modules, layers, auth flow.\n\n"
        '{{"tech_stack":"...","db_schema":"...","api_endpoints":"...",'
        '"folder_structure":"...","architecture_diagram":"..."}}'
    )

    try:
        doc: ArchitectureDoc = invoke_model(
            architect_llm,
            prompt,
            stories="\n".join(f"- {s}" for s in state["stories"]),
            modules="\n".join(f"- {m}" for m in state["modules"]),
            memory=memory_snippet,
        )
    except (OutputParserException, Exception) as exc:
        print(f"   ⚠  Architect structured output failed ({exc}); falling back …")
        messages = prompt.format_messages(
            stories="\n".join(f"- {s}" for s in state["stories"]),
            modules="\n".join(f"- {m}" for m in state["modules"]),
            memory=memory_snippet,
        )

        raw = llm.invoke(messages)
        d = _safe_json(raw.content)
        doc = ArchitectureDoc(
            tech_stack=_dict_to_str(d.get("tech_stack", "FastAPI, SQLAlchemy, PostgreSQL")),
            db_schema=_dict_to_str(d.get("db_schema", "See modules")),
            api_endpoints=_dict_to_str(d.get("api_endpoints", "See modules")),
            folder_structure=_dict_to_str(d.get("folder_structure", "app/")),
            architecture_diagram=_dict_to_str(d.get("architecture_diagram", "N/A")),
        )

    arch_text = (
        f"## Tech Stack\n{doc.tech_stack}\n\n"
        f"## Database Schema\n{doc.db_schema}\n\n"
        f"## API Endpoints\n{doc.api_endpoints}\n\n"
        f"## Folder Structure\n{doc.folder_structure}\n\n"
        f"## Architecture Diagram\n{doc.architecture_diagram}"
    )

    print("   Architecture designed ✓")
    return {"architecture": arch_text}


# MAIN GRAPH — NODE 3  DISPATCHER  (fans-out parallel workers)

def _build_worker_payload(state: SoftwareState, module_name: str) -> WorkerState:
    """Construct the initial WorkerState for one parallel worker."""
    return WorkerState(
        module_name=module_name,
        requirement=state["requirement"],
        architecture=state.get("architecture", ""),
        stories=state.get("stories", []),
        quality_guide=QUALITY_GUIDE,
        project_id=state.get("project_id", ""),
        thread_id=state.get("thread_id", ""),
        memory_context=state.get("memory_context"),
        module_plan=None,
        generated_code={},
        tests={},
        review_score=None,
        review_issues=[],
        fix_attempts=0,
        max_fix_attempts=3,
        human_approved=False,
        human_feedback=state.get(
            "human_feedback",
            ""
        ),
        completed_modules=[],
        module_plans={},
        review_scores={},
    )


# WORKER SUBGRAPH — NODE A  MODULE PLANNER

def worker_module_planner(state: WorkerState) -> dict:
    module = state["module_name"]
    print(f"   📐 [{module}] Module Planner …")

    memory_snippet = _format_memory_snippet(state.get("memory_context"), module=module)

    prompt = ChatPromptTemplate.from_template(
        "You are a senior engineer planning the '{module}' backend module.\n\n"
        "Overall architecture:\n{architecture}\n\n"
        "Previous Project Memory for this module (reuse prior file layout/"
        "dependencies where they still make sense):\n{memory}\n\n"
        "Return ONLY a flat JSON object with these keys:\n"
        "module_name (string), files (array of {{path, purpose, exports}}), "
        "dependencies (array of strings), api_routes (array of strings).\n\n"
        '{{"module_name":"{module}","files":[{{"path":"...","purpose":"...","exports":["..."]}}],'
        '"dependencies":["..."],"api_routes":["..."]}}'
    )

    try:
        plan: ModulePlan = invoke_model(
            module_planner_llm,
            prompt,
            module=module,
            architecture=state["architecture"],
            memory=memory_snippet,
        )
        plan_text = (
            f"Module: {plan.module_name}\n"
            f"Files: {[f.path for f in plan.files]}\n"
            f"Dependencies: {plan.dependencies}\n"
            f"Routes: {plan.api_routes}"
        )
    except (OutputParserException, Exception) as exc:
        print(f"      ⚠  [{module}] Module planner failed ({exc}); using default plan.")
        plan_text = f"Module: {module}\nFiles: [{module}/routes.py, {module}/models.py, {module}/schemas.py, {module}/crud.py]\nDependencies: [fastapi, sqlalchemy]\nRoutes: [GET /{module}, POST /{module}]"

    return {"module_plan": plan_text}


# WORKER SUBGRAPH — NODE B  MODULE CODER

def worker_coder(state: WorkerState) -> dict:
    module = state["module_name"]
    print(f"   💻 [{module}] Coder generating code …")

    memory_snippet = _format_memory_snippet(state.get("memory_context"), module=module)

    prompt = ChatPromptTemplate.from_template(
    "You are a senior Python backend engineer.\n"
    "Generate production-ready FastAPI code for the '{module}' module.\n\n"

    "Module plan:\n{plan}\n\n"

    "Overall architecture:\n{architecture}\n\n"

    "Previous Project Memory (reuse prior implementation patterns for this "
    "module where they remain valid; otherwise improve on them):\n{memory}\n\n"

    "Human Feedback (if available):\n{feedback}\n\n"

    "Quality requirements (follow EVERY rule):\n{quality}\n\n"

    "If human feedback is provided, you MUST improve the code based on that feedback.\n\n"

    "Return ONLY Python source code inside a single markdown code block."
)

    response = invoke_model(
        llm,
        prompt,
        module=module,
        plan=state.get("module_plan", ""),
        architecture=state.get("architecture", ""),
        memory=memory_snippet,
        feedback=state.get("human_feedback", ""),
        quality=state.get("quality_guide", ""),
    )

    return {"generated_code": {module: response.content}}


# WORKER SUBGRAPH — NODE C  REVIEWER

def worker_reviewer(state: WorkerState) -> dict:
    module = state["module_name"]
    code   = state["generated_code"].get(module, "")
    print(f"   🔍 [{module}] Reviewer (attempt {state.get('fix_attempts', 0) + 1}) …")

    if not code:
        print(f"      ⚠  No code to review for [{module}].")
        return {"review_score": 1, "review_issues": [f"No code generated for '{module}'"]}

    memory_snippet = _format_memory_snippet(state.get("memory_context"), module=module)

    prompt = ChatPromptTemplate.from_template(
        "You are a strict code reviewer. Review this '{module}' module code.\n\n"
        "Code:\n{code}\n\n"
        "Overall architecture (validate the code stays consistent with it):\n"
        "{architecture}\n\n"
        "Previous Project Memory (flag inconsistencies with prior implementations "
        "of this module, if any):\n{memory}\n\n"
        "Return ONLY a flat JSON object:\n"
        '{{"score": 0, "issues": ["..."], "logic_correctness": "...", "security_check": "..."}}'
    )

    try:
        review: CodeReview = invoke_model(
            reviewer_llm,
            prompt,
            module=module,
            code=code,
            architecture=state.get("architecture", ""),
            memory=memory_snippet,
        )
        score = review.score

        # Handle percentage style scores
        if score > 10:
            score = round(score / 10)

        # Final safety
        score = max(1, min(score, 10))

        review.score = score
    except (OutputParserException, Exception) as exc:
        print(f"      ⚠  [{module}] Reviewer failed ({exc}); defaulting score=5.")
        review = CodeReview(score=5, issues=[str(exc)], logic_correctness="unknown", security_check="unknown")

    review.score = max(
        1,
        min(review.score, 10)
    )
    print(f"      Score: {review.score}/10 | Issues: {len(review.issues)}")
    for iss in review.issues[:3]:
        print(f"        • {iss}")

    return {"review_score": review.score, "review_issues": review.issues}

def human_review_node(state: SoftwareState):

    print("\n" + "=" * 60)
    print("👤 HUMAN IN THE LOOP")
    print("=" * 60)

    print("\n📦 Completed Modules:")
    for module in state.get("completed_modules", []):
        print(f"✔ {module}")

    print("\n📄 Generated Modules:")
    for module in state.get("generated_code", {}).keys():
        print(f"✔ {module}")

    choice = input("\nApprove Project? (yes/no): ").strip().lower()

    feedback = ""

    if choice == "no":
        feedback = input("Enter feedback: ")

    return {
        "human_approved": choice == "yes",
        "human_feedback": feedback,
    }
# WORKER SUBGRAPH — NODE D  FIXER

def worker_fixer(state: WorkerState) -> dict:
    module   = state["module_name"]
    attempts = state.get("fix_attempts", 0) + 1
    code     = state["generated_code"].get(module, "")
    issues   = state.get("review_issues", [])
    print(f"   🛠️  [{module}] Fixer (attempt {attempts}) …")

    memory_snippet = _format_memory_snippet(state.get("memory_context"), module=module)

    prompt = ChatPromptTemplate.from_template(
        "Fix the following code based on the reviewer's issues.\n"
        "Module: {module}\n\n"
        "Current code:\n{code}\n\n"
        "Issues:\n{issues}\n\n"
        "Previous Project Memory (preserve consistency with this prior "
        "implementation's conventions unless it conflicts with fixing the issues):\n"
        "{memory}\n\n"
        "Human Feedback:\n{feedback}\n\n"
        "Quality requirements:\n{quality}\n\n"
        "If human feedback is provided, you MUST improve the code based on that feedback.\n\n"
        "Return ONLY the corrected Python code in a markdown code block."
    )

    response = invoke_model(
        llm,
        prompt,
        module=module,
        code=code,
        issues="\n".join(f"- {i}" for i in issues),
        memory=memory_snippet,
        feedback=state.get("human_feedback", ""),
        quality=state.get("quality_guide", ""),
    )

    return {
        "generated_code": {module: response.content},
        "fix_attempts":   attempts,
    }


# WORKER SUBGRAPH — NODE E  COMPLETE MODULE

def worker_complete(state: WorkerState) -> dict:
    module = state["module_name"]
    score  = state.get("review_score", 0) or 0
    print(f"   ✅  [{module}] Complete (final score={score})")

    # Return values that will be merged back into SoftwareState via reducers
    return {
        "completed_modules": [module],
        "generated_code":    state.get("generated_code", {}),
        "tests":             state.get("tests", {}),
        "module_plans":      {module: state.get("module_plan", "") or ""},
        "review_scores":     {module: score},
    }


def qa_node(state: SoftwareState) -> dict:
    print("─" * 60)
    print("🧪  QA AGENT: Generating tests for all modules …")

    all_code = state.get("generated_code", {})
    tests: dict[str, str] = {}

    for module, code in all_code.items():
        print(f"   Writing tests for [{module}] …")

        prompt = ChatPromptTemplate.from_template(
            "You are a QA engineer. Write comprehensive pytest unit tests "
            "AND integration test suggestions for the '{module}' module.\n\n"
            "Also append an ## API Validation Checklist section.\n\n"
            "Code:\n{code}\n\n"
            "Return ONLY Python test code in a markdown code block."
        )

        response = invoke_model(
            llm,
            prompt,
            module=module,
            code=code,
        )
        tests[module] = response.content

    print(f"   Tests generated for {len(tests)} module(s).")
    return {"tests": tests}

def human_review_node(state: SoftwareState):

    print("\n" + "=" * 60)
    print("👤 HUMAN IN THE LOOP")
    print("=" * 60)

    print("\n📦 Completed Modules:")
    for module in state.get("completed_modules", []):
        print(f"✔ {module}")

    print("\n📄 Generated Modules:")
    for module in state.get("generated_code", {}).keys():
        print(f"✔ {module}")

    choice = input("\nApprove Project? (yes/no): ").strip().lower()

    feedback = ""

    if choice == "no":
        feedback = input("Enter feedback: ")

        # Clear previous outputs before regeneration
        return {
            "human_approved": False,
            "human_feedback": feedback,

            "completed_modules": [],
            "generated_code": {},
            "tests": {},
        }

    return {
        "human_approved": True,
        "human_feedback": "",
    }

# MAIN GRAPH — NODE 5  DELIVERY

def delivery_node(state: SoftwareState) -> dict:
    print("─" * 60)
    print("📦  DELIVERY: Compiling final package …")

    lines: list[str] = []

    # ── architecture.md ──────────────────────────
    arch_md = f"# Architecture\n\n{state.get('architecture', 'N/A')}\n"
    output_dir = os.getenv("OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "architecture.md"), "w", encoding="utf-8") as f:
        f.write(arch_md)
    print("   Saved outputs/architecture.md")

    # ── delivery_package.md ──────────────────────
    lines += [
        "# ══════════════════════════════════════════",
        "# SOFTWARE DELIVERY PACKAGE",
        "# ══════════════════════════════════════════\n",
        "## Requirement\n",
        state["requirement"],
        "\n## User Stories\n",
    ]
    for s in state.get("stories", []):
        lines.append(f"- {s}")

    lines += ["\n## Completed Modules\n"]
    for m in state.get("completed_modules", []):
        lines.append(f"- {m}")

    lines += [f"\n## Architecture\n", state.get("architecture", "N/A")]

    lines += ["\n\n## ═══ SOURCE CODE ═══\n"]
    for module, code in state.get("generated_code", {}).items():
        lines += [f"\n### Module: `{module}`\n", code]

    lines += ["\n\n## ═══ TEST CODE ═══\n"]
    for module, test in state.get("tests", {}).items():
        lines += [f"\n### Tests: `{module}`\n", test]

    package = "\n".join(lines)
    pkg_path = os.path.join(output_dir, "delivery_package.md")
    with open(pkg_path, "w", encoding="utf-8") as f:
        f.write(package)

    print(f"   Saved {pkg_path}  ({len(package):,} chars)")

    # ── Save complete project into PostgreSQL Long-Term Memory ──
    delivery_metadata = {
        "path": pkg_path,
        "architecture_path": os.path.join(output_dir, "architecture.md"),
        "size_chars": len(package),
        "modules_delivered": state.get("completed_modules", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    project_id = state.get("project_id", "")
    if project_id:
        try:
            save_project_memory(
                project_id=project_id,
                thread_id=state.get("thread_id", ""),
                requirement=state["requirement"],
                user_stories=state.get("stories", []),
                architecture=state.get("architecture"),
                module_plans=state.get("module_plans", {}),
                generated_code=state.get("generated_code", {}),
                review_scores=state.get("review_scores", {}),
                completed_modules=state.get("completed_modules", []),
                human_feedback=state.get("human_feedback", ""),
                delivery_package_metadata=delivery_metadata,
                execution_history=[
                    {
                        "event": "delivery_complete",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "modules_delivered": state.get("completed_modules", []),
                    }
                ],
            )
            print(f"   💾  Project memory saved to PostgreSQL (project_id={project_id})")
        except Exception as exc:
            print(f"   ⚠  Failed to save project memory to PostgreSQL: {exc}")
    else:
        print("   ⚠  No project_id in state — skipping PostgreSQL save.")

    return {"delivery_package": package}
