-- Search quality: compact FTS ranking table, duplicate marking, canonical
-- legal areas.
--
-- None of these touch existing documents rows in bulk: every UPDATE of
-- documents is non-HOT and re-inserts into three HNSW indexes, which on this
-- instance runs at ~30 rows/min. New structures therefore live in side tables
-- (or a nullable column set only on the few affected rows).
--
-- Backfills: scripts/backfill_fts_core.py, scripts/mark_duplicates.py and
-- scripts/canonicalize_legal_domains.py (all also run by scripts/run_refresh.py).

-- 1. Compact FTS for ranking ----------------------------------------------
-- documents.fts includes the full decision text (weight D), so ranking every
-- match means reading large TOASTed tsvectors (~20s for a 15k-match query).
-- The API therefore ranked only the first 1,500 matches in index order, an
-- effectively arbitrary subset (3/40 overlap with the true top 40).
-- document_fts_core holds the A/B/C-weighted curated fields only, small enough
-- to rank all matches.
CREATE TABLE IF NOT EXISTS public.document_fts_core (
    doc_id text PRIMARY KEY REFERENCES public.documents(doc_id) ON DELETE CASCADE,
    fts_core tsvector NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_fts_core_fts
ON public.document_fts_core USING gin (fts_core);

CREATE OR REPLACE FUNCTION public.document_fts_core_vector(
    summary text, metadata jsonb
) RETURNS tsvector
LANGUAGE sql STABLE AS $$
    SELECT
        setweight(to_tsvector('portuguese', unaccent(
            coalesce(replace(metadata->>'keywords_search_query', ',', ' '), '') || ' ' ||
            coalesce(array_to_string(ARRAY(SELECT jsonb_array_elements_text(
                coalesce(metadata->'legal_descriptors', '[]'::jsonb))), ' '), '')
        )), 'A') ||
        setweight(to_tsvector('portuguese', unaccent(
            coalesce(summary, '') || ' ' || coalesce(metadata->>'legal_question', '')
        )), 'B') ||
        setweight(to_tsvector('portuguese', unaccent(
            coalesce(metadata->>'ratio_decidendi', '')
        )), 'C')
$$;

CREATE OR REPLACE FUNCTION public.document_fts_core_sync() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO public.document_fts_core (doc_id, fts_core)
    VALUES (NEW.doc_id, public.document_fts_core_vector(NEW.summary, NEW.metadata))
    ON CONFLICT (doc_id) DO UPDATE SET fts_core = EXCLUDED.fts_core;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS documents_fts_core_sync ON public.documents;
CREATE TRIGGER documents_fts_core_sync
AFTER INSERT OR UPDATE OF summary, metadata ON public.documents
FOR EACH ROW EXECUTE FUNCTION public.document_fts_core_sync();

-- 2. Duplicate decisions ----------------------------------------------------
-- DGSI sometimes publishes the same decision twice under different ids.
-- Non-canonical copies point at the kept copy and are excluded from search.
-- Adding a nullable column without default is a metadata-only change.
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS duplicate_of text;

-- Tiny partial index so "doc_id NOT IN (duplicates)" anti-joins never scan
-- the documents heap.
CREATE INDEX IF NOT EXISTS idx_documents_duplicates
ON public.documents (doc_id) WHERE duplicate_of IS NOT NULL;

-- Without statistics the planner assumes "duplicate_of IS NULL" is highly
-- selective and turns filtered vector searches into exact sorts over the
-- whole filtered set (court=STJ: 0.09s -> 28.5s). mark_duplicates.py
-- re-analyzes after every run.
ANALYZE public.documents (duplicate_of);

-- 3. Canonical legal areas --------------------------------------------------
-- One row per distinct raw documents.legal_domain value, mapped onto
-- api/legal_areas.CANONICAL_AREAS. Filters join through this table instead
-- of rewriting 395k documents rows.
CREATE TABLE IF NOT EXISTS public.legal_domain_map (
    raw text PRIMARY KEY,
    areas text[] NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_legal_domain_map_areas
ON public.legal_domain_map USING gin (areas);
