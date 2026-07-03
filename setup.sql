-- ═══════════════════════════════════════════════════════════════════════
-- software_factory_memory — PostgreSQL setup script
-- Run once, as a superuser (e.g. `psql -U postgres -f setup.sql`)
-- ═══════════════════════════════════════════════════════════════════════

-- 1. Create the database (run this part connected to the default "postgres" db)
-- ───────────────────────────────────────────────────────────────────────
-- NOTE: CREATE DATABASE cannot run inside a transaction block / DO block,
-- so keep this as a standalone statement. If the DB already exists, skip it.
CREATE DATABASE software_factory_memory;

-- Now connect to the new database before running the rest:
--   \c software_factory_memory

-- 2. project_memory table (Long-Term Memory)
--    NOTE: This table is also auto-created by SQLAlchemy's Base.metadata.create_all()
--    via database.init_db() the first time the app runs — this script is provided
--    for teams that prefer to provision schema via DBA-managed migrations instead.
-- ───────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS project_memory (
    id                          SERIAL PRIMARY KEY,

    project_id                  VARCHAR(64) NOT NULL UNIQUE,
    thread_id                   VARCHAR(64),

    requirement                 TEXT NOT NULL,
    user_stories                JSONB NOT NULL DEFAULT '[]',
    architecture                TEXT,
    module_plans                JSONB NOT NULL DEFAULT '{}',

    generated_code              JSONB NOT NULL DEFAULT '{}',
    review_scores               JSONB NOT NULL DEFAULT '{}',
    completed_modules           JSONB NOT NULL DEFAULT '[]',

    human_feedback              TEXT NOT NULL DEFAULT '',

    delivery_package_metadata   JSONB NOT NULL DEFAULT '{}',
    execution_history           JSONB NOT NULL DEFAULT '[]',

    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Indexes
-- ───────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS ix_project_memory_project_id ON project_memory (project_id);
CREATE INDEX IF NOT EXISTS ix_project_memory_thread_id  ON project_memory (thread_id);

-- Full-text search index used by memory.search_similar_projects()
CREATE INDEX IF NOT EXISTS ix_project_memory_requirement_fts
    ON project_memory
    USING GIN (to_tsvector('english', requirement));

-- 4. updated_at auto-touch trigger (keeps psql-side inserts/updates honest too)
-- ───────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_project_memory_touch_updated_at ON project_memory;
CREATE TRIGGER trg_project_memory_touch_updated_at
    BEFORE UPDATE ON project_memory
    FOR EACH ROW
    EXECUTE FUNCTION touch_updated_at();

-- 5. LangGraph checkpoint tables (short-term / thread state)
--    These are created automatically by `checkpointer.setup()` in graph.py
--    the first time the app runs (checkpoints, checkpoint_blobs,
--    checkpoint_writes, checkpoint_migrations). No manual action needed —
--    listed here purely for operational awareness.
