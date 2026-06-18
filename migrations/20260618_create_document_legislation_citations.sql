-- Normalized lookup table for fast legislation citation searches.
--
-- Problem addressed:
--   JSONB containment on documents.metadata works for selective laws, but common
--   codes such as Código Civil can match >100k documents and force expensive
--   heap/TOAST reads before sorting. This table stores one narrow row per
--   cited law/article/document so search can resolve doc_ids first and hydrate
--   only the requested page of documents.
--
-- Backfill data with scripts/create_legislation_citations_table.py. The backfill
-- is intentionally separated from schema creation so it can run in bounded
-- decision_date batches on production-sized databases.

CREATE TABLE IF NOT EXISTS public.document_legislation_citations (
    doc_id text NOT NULL REFERENCES public.documents(doc_id) ON DELETE CASCADE,
    law text NOT NULL,
    article text NOT NULL DEFAULT '',
    decision_date date,
    court_short text,
    is_auj boolean,
    legal_domain text,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, law, article)
);

CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_law_date
ON public.document_legislation_citations (law, decision_date DESC, doc_id);

CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_law_article_prefix
ON public.document_legislation_citations (
    law,
    article text_pattern_ops,
    decision_date DESC,
    doc_id
)
WHERE article <> '';

CREATE INDEX IF NOT EXISTS idx_doc_leg_citations_doc_id
ON public.document_legislation_citations (doc_id);

ANALYZE public.document_legislation_citations;