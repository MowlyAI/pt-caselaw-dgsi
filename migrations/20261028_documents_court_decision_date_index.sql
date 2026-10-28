-- Fallback lookup of cited decisions by court and date
-- (POST /documents/resolve-citations, when the cited number is written
-- differently from the corpus). Additive only. On production run it as
-- CREATE INDEX CONCURRENTLY outside a transaction to avoid locking documents.

CREATE INDEX IF NOT EXISTS idx_documents_court_decision_date
ON public.documents (court_short, decision_date);
