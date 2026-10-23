-- PGR pareceres share the documents table with court decisions but are left
-- out of searches that do not name them (api/main.py NON_COURT_SOURCES).
-- Tiny partial index so the "doc_id NOT IN (non-court)" anti-join on the
-- FTS fast path never scans the documents heap. Keep the list in sync.
CREATE INDEX IF NOT EXISTS idx_documents_non_court
ON public.documents (doc_id) WHERE court_short IN ('PGR');
