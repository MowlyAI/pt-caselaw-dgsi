-- B-tree expression indexes on the jsonb filter columns of documents.
--
-- Problem addressed:
--   metadata->>'decision_type' and metadata->>'extraction_confidence' are
--   first-class search filters (Filters.decision_type / extraction_confidence
--   in api/main.py), but as jsonb expressions they had no index and no
--   planner statistics. PostgreSQL therefore fell back to the default 0.5%
--   selectivity estimate for metadata->>'decision_type' = 'Acórdão' when
--   the true selectivity is ~98%. Filtered vector queries then misplanned
--   into "B-tree pre-filter + exact distance sort" over tens of thousands of
--   rows — e.g. court=STJ + decision_type='Acórdão' exact-sorted 70,275 rows
--   with TOASTed vector reads in ~37s, blowing the 30s statement timeout
--   (production 504s on POST /search).
--
--   These indexes give the planner real MCV statistics for the expressions,
--   so filtered vector queries choose correctly between the HNSW iterative
--   scan (large matching sets) and a B-tree pre-filter (small sets). They can
--   also serve as the access path for selective values (e.g. 'Sentenca').
--
-- Measured on production (371,737 rows, pgvector 0.8.0):
--   court=STJ + decision_type='Acórdão', LIMIT 80:  37,327 ms -> 16 ms (warm)
--
-- Applied to production with CREATE INDEX CONCURRENTLY on 2026-09-28; the
-- plain statements below are for fresh environments (brief write lock).

CREATE INDEX IF NOT EXISTS idx_documents_decision_type
ON public.documents ((metadata->>'decision_type'));

CREATE INDEX IF NOT EXISTS idx_documents_extraction_confidence
ON public.documents ((metadata->>'extraction_confidence'));

-- Populate planner statistics for the new expressions.
ANALYZE public.documents;
