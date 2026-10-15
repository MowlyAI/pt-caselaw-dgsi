-- Permanent disability degree per decision, for numeric search filters.
--
-- Searches such as "indemnização para invalidez superior a 15 pontos" matched
-- the number as a word: every top result had exactly 15 points. The degree is
-- already extracted as free text (metadata.injuries[].disability_degree);
-- scripts/backfill_disability_points.py parses it (api/disability.py) into
-- this side table, which the disability_points_* filters of /search read.
--
-- A side table rather than a documents column: every UPDATE of documents is
-- non-HOT and re-inserts into three HNSW indexes (see 20261010_...).
-- Additive only; the API reads the table only when a request sends a
-- disability_points_* filter.

CREATE TABLE IF NOT EXISTS public.document_disability_points (
    doc_id text PRIMARY KEY REFERENCES public.documents(doc_id) ON DELETE CASCADE,
    points real NOT NULL CHECK (points > 0 AND points <= 100)
);

CREATE INDEX IF NOT EXISTS idx_document_disability_points_points
ON public.document_disability_points (points);
