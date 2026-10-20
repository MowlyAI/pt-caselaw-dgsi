-- Lookup of cited decisions by process number (POST /documents/resolve-citations).
--
-- The expression must stay identical to api/citations.PROCESS_KEY_SQL or the
-- planner will not use the index. Additive only. On production run it as
-- CREATE INDEX CONCURRENTLY outside a transaction to avoid locking documents.

CREATE INDEX IF NOT EXISTS idx_documents_process_number_key
ON public.documents ((upper(regexp_replace(process_number, '\s+', '', 'g'))));

-- Expression statistics are gathered only by a whole-table ANALYZE.
ANALYZE public.documents;
