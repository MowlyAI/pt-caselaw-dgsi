-- Judge rapporteur (relator) per decision, for the `judge` search filter and
-- the GET /judges autocomplete.
--
-- The relator was only inside metadata->>'judge_name', with no index, and is
-- written with different case and accents across courts ("JOSE CORREIA" /
-- "José Correia"). scripts/backfill_judges.py reads the page's "Relator:"
-- field (api/judges.py) into this side table with a normalised judge_key, and
-- refreshes judge_directory, the per-court summary the autocomplete reads.
--
-- A side table rather than a documents column: every UPDATE of documents is
-- non-HOT and re-inserts into three HNSW indexes (see 20261010_...).
-- Additive only; the API reads these objects only when a request sends a
-- judge filter or calls GET /judges.

CREATE TABLE IF NOT EXISTS public.document_judges (
    doc_id text PRIMARY KEY REFERENCES public.documents(doc_id) ON DELETE CASCADE,
    judge_name text NOT NULL,
    judge_key text NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_judges_judge_key
ON public.document_judges (judge_key);

-- One row per judge and court, without duplicate decisions. A few thousand
-- rows, so the autocomplete scans it whole.
CREATE MATERIALIZED VIEW IF NOT EXISTS public.judge_directory AS
SELECT j.judge_key,
       d.court_short,
       mode() WITHIN GROUP (ORDER BY j.judge_name) AS judge_name,
       count(*)::int AS doc_count,
       min(d.decision_date) AS first_date,
       max(d.decision_date) AS last_date
FROM public.document_judges j
JOIN public.documents d ON d.doc_id = j.doc_id
WHERE d.duplicate_of IS NULL
GROUP BY j.judge_key, d.court_short;

-- Required by REFRESH MATERIALIZED VIEW CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_judge_directory_key_court
ON public.judge_directory (judge_key, court_short);
