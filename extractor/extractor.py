"""
LLM extractor: turns the raw full text of a DGSI decision into a rich
structured JSON following `schema.ExtractedInfo`.

Reads `doc["full_text"]` and returns the model's JSON as a dict that can be
attached to the document under `llm_extracted`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from typing import Optional

import httpx
from rich.console import Console

from extractor.schema import ExtractedInfo

console = Console()


def _strip_accents(s: str) -> str:
    """Transliterate accented Latin letters to plain ASCII.
    é→e, á→a, ç→c, ñ→n, ü→u, etc. Non-combinable marks dropped."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_KW_ALLOWED = re.compile(r"[^a-z0-9 ]")
_KW_WS = re.compile(r"\s+")


def _normalize_keywords(raw: Optional[str]) -> Optional[str]:
    """Deterministic normalisation of `keywords_search_query`.

    Guarantees the stored value is comma-separated lowercase ASCII phrases of
    2–6 tokens each, no accents, no punctuation, no duplicates.
    """
    if not raw:
        return None
    # Split on commas; fall back to newline if model used newlines.
    parts = [p for p in raw.split(",")]
    if len(parts) == 1 and "\n" in raw:
        parts = raw.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        s = _strip_accents(p).lower().replace("-", " ")
        s = _KW_ALLOWED.sub(" ", s)
        s = _KW_WS.sub(" ", s).strip()
        if not s:
            continue
        toks = s.split()
        # Drop single-word (not a phrase) and overlong (not a useful retrieval unit).
        if len(toks) < 2 or len(toks) > 6:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return ", ".join(out) if out else None


# ---------------------------------------------------------------------------
# Legislation post-processing.
#
# The LLM is instructed to emit one entry per distinct (article, n.º, alínea)
# tuple and to always write the law's full canonical Portuguese name. In
# practice ~7% of generated articles use compound forms like "527.º, n.ºs 1 e 2"
# or "11.º, 12.º, 13.º, 19.º". We fix this deterministically here so every
# downstream dedup / join on (article, law) is exact across the corpus.
# ---------------------------------------------------------------------------

_LAW_ABBREV: dict[str, str] = {
    "cc": "Código Civil",
    "c.c.": "Código Civil",
    "cod. civil": "Código Civil",
    "cpc": "Código de Processo Civil",
    "ncpc": "Código de Processo Civil",
    "novo cpc": "Código de Processo Civil",
    "cp": "Código Penal",
    "cpp": "Código de Processo Penal",
    "ct": "Código do Trabalho",
    "cpt": "Código de Processo do Trabalho",
    "csc": "Código das Sociedades Comerciais",
    "crp": "Constituição da República Portuguesa",
    "c.r.p.": "Constituição da República Portuguesa",
    "cpa": "Código do Procedimento Administrativo",
    "cpta": "Código de Processo nos Tribunais Administrativos",
    "ccp": "Código dos Contratos Públicos",
    "rcp": "Regulamento das Custas Processuais",
}

_CTX_PRIORITY = {"supporting": 3, "distinguishing": 2, "criticizing": 1, "referencing": 0}


def _canonicalize_law(raw: str) -> str:
    """Map common abbreviations to full Portuguese names and normalise
    statute/decree-law numbering to the canonical `Lei n.º X/YYYY` form."""
    if not raw:
        return raw
    s = raw.strip()
    # Drop trailing promulgation dates e.g. ", de 4 de julho".
    s = re.sub(r",\s*de\s+\d+\s+de\s+\w+(?:\s+de\s+\d+)?\s*$", "", s, flags=re.IGNORECASE)
    low = s.lower().strip()
    if low in _LAW_ABBREV:
        return _LAW_ABBREV[low]
    stripped = low.strip(". ")
    if stripped in _LAW_ABBREV:
        return _LAW_ABBREV[stripped]
    m = re.match(r"(?:decreto[\s-]?lei|dl)\s*(?:n[°º\.\s]*)?\s*(\d+(?:-[A-Za-z])?/\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Decreto-Lei n.º {m.group(1)}"
    m = re.match(r"lei\s*(?:n[°º\.\s]*)?\s*(\d+(?:-[A-Za-z])?/\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Lei n.º {m.group(1)}"
    return s


def _clean_article_text(s: str) -> str:
    """Harmonise ordinal markers, `alínea` spelling and whitespace."""
    if not s:
        return s
    s = s.strip()
    s = s.replace("°", "º")
    # bare "º" after an article number → ".º"
    s = re.sub(r"(\d+(?:-[A-Za-z])?)\s*º(?!\.)", r"\1.º", s)
    # "67-A" (no ordinal marker) → "67.º-A"
    s = re.sub(r"\b(\d+)(?=-[A-Za-z]\b)(?!\.º)", r"\1.º", s)
    # "al." → "alínea "
    s = re.sub(r"\bal\.\s*", "alínea ", s, flags=re.IGNORECASE)
    # ASCII "alinea" → "alínea"
    s = re.sub(r"\balinea(s?)\b", lambda m: "alínea" + m.group(1), s, flags=re.IGNORECASE)
    # "N.º" / "Nº" / "N°" / "n.ºs" → lowercase (preserve trailing "s")
    s = re.sub(r"\b[Nn]\.?\s*[º°](s?)\b", lambda m: "n.º" + (m.group(1) or ""), s)
    # strip an accidental "artigo" / "art." prefix
    s = re.sub(r"^(?:artigos?|arts?\.?)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" .,;:")


def _split_number_list(nums_str: str) -> list[str]:
    """Parse `'1 e 2'`, `'1, 2 e 3'` or `'2 a 4'` → `['1','2','3','4']`."""
    s = nums_str.strip().rstrip(" .,;:")
    m = re.match(r"^(\d+)\s+a\s+(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 < b - a <= 20:
            return [str(i) for i in range(a, b + 1)]
        return []
    parts = re.split(r"\s*(?:,|\be\b)\s*", s)
    return [p for p in (p.strip() for p in parts) if re.fullmatch(r"\d+", p)]


def _split_letter_list(letters_str: str) -> list[str]:
    """Extract every `a)`-style alínea letter from a fragment."""
    return [m.group(1) for m in re.finditer(r"\b([a-z])\)", letters_str)]


def _expand_article_ranges(raw_article: str) -> list[str]:
    """Expand a possibly compound article locator into canonical singletons.

    Handles the patterns the LLM still emits despite the prompt:
        '527.º, n.ºs 1 e 2'           → ['527.º, n.º 1', '527.º, n.º 2']
        '394.º, n.º 2, alíneas b) e f)' → one entry per alínea
        '615.º, n.º 1, b)'            → '615.º, n.º 1, alínea b)'  (implicit alínea)
        '11.º, 12.º, 13.º, 19.º'      → one entry per article
        '180.º, n.ºs 2 a 4'           → n.º 2, n.º 3, n.º 4
    Unparseable residues fall back to the cleaned single-string form.
    """
    if not raw_article:
        return []
    s = _clean_article_text(raw_article)
    if not s:
        return []

    # Pure multi-article list ("11.º, 12.º e 13.º"): split and return.
    if re.fullmatch(r"(\d+\.º(?:-[A-Za-z])?)(\s*(?:,|\be\b)\s*\d+\.º(?:-[A-Za-z])?)+", s):
        return [m.group(0) for m in re.finditer(r"\d+\.º(?:-[A-Za-z])?", s)]

    # Article range ("234.º a 240.º").
    m_rng = re.fullmatch(r"(\d+)\.º\s+a\s+(\d+)\.º", s)
    if m_rng:
        a, b = int(m_rng.group(1)), int(m_rng.group(2))
        if 0 < b - a <= 20:
            return [f"{i}.º" for i in range(a, b + 1)]
        return [s]

    art_m = re.match(r"^(\d+\.º(?:-[A-Za-z])?)", s)
    if not art_m:
        return [s]
    article = art_m.group(1)
    rest = s[art_m.end():].lstrip(",. ")

    # Trailing alínea block (explicit keyword).
    alinea_letters: list[str] = []
    m_al = re.search(r"al[íi]neas?\s+(.+)$", rest, re.IGNORECASE)
    if m_al:
        alinea_letters = _split_letter_list(m_al.group(1))
        rest = rest[:m_al.start()].rstrip(",. ")
    else:
        # Implicit alínea: ", b)" at the end without the word "alínea".
        m_impl = re.search(r",\s*([a-z]\)(?:\s*(?:,|\be\b)\s*[a-z]\))*)\s*$", rest, re.IGNORECASE)
        if m_impl:
            letters = _split_letter_list(m_impl.group(1))
            if letters:
                alinea_letters = letters
                rest = rest[:m_impl.start()].rstrip(",. ")

    # Optional n.º block.
    numrefs: list[str] = []
    m_n = re.search(r"n\.ºs?\s*(.+?)$", rest, re.IGNORECASE)
    if m_n:
        numrefs = _split_number_list(m_n.group(1))
        rest = rest[:m_n.start()].rstrip(",. ")

    if rest.strip(",. "):
        return [s]  # residue → keep cleaned string verbatim

    nums: list[Optional[str]] = list(numrefs) if numrefs else [None]
    lets: list[Optional[str]] = list(alinea_letters) if alinea_letters else [None]
    out: list[str] = []
    seen: set[str] = set()
    for n in nums:
        for l in lets:
            p = article
            if n is not None:
                p += f", n.º {n}"
            if l is not None:
                p += f", alínea {l})"
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out or [s]


def _normalize_legislation(items: list[dict]) -> list[dict]:
    """Expand compound articles, canonicalise law names, dedup on (article, law)
    keeping the highest-priority citation_context."""
    if not items:
        return []
    expanded: list[dict] = []
    for it in items:
        article = (it.get("article") or "").strip()
        law = _canonicalize_law((it.get("law") or "").strip())
        ctx = (it.get("citation_context") or "referencing").strip().lower()
        if ctx not in _CTX_PRIORITY:
            ctx = "referencing"
        arts = _expand_article_ranges(article) if article else [""]
        for a in arts:
            expanded.append({"article": a, "law": law, "citation_context": ctx})
    by_key: dict[tuple[str, str], dict] = {}
    for it in expanded:
        key = (it["article"], it["law"])
        cur = by_key.get(key)
        if cur is None or _CTX_PRIORITY[it["citation_context"]] > _CTX_PRIORITY[cur["citation_context"]]:
            by_key[key] = it
    return list(by_key.values())


_AUJ_RE = re.compile(
    # Accent-stripped, lowercased match. Covers the canonical variants plus
    # the occasional DGSI raw-data typo "uniforizacao" (missing 'm').
    r"(unifor[a-z]{0,6}zacao"           # uniformização / uniforização (typo)
    r"|fixacao\s+de\s+jurispr"          # fixação de jurisprudência
    r"|fixar\s+jurispr"                 # fixar jurisprudência
    r"|uniformizador)"                  # acórdão uniformizador
)


def _is_jurisprudence_unification(
    procedural_type: Optional[str],
    case_type: Optional[str],
) -> bool:
    """Detect Acórdãos de Uniformização/Fixação de Jurisprudência (AUJ).

    AUJs are high-authority STJ/STA decisions that fix binding jurisprudence.
    Validated on a 500-doc STJ sample: 26/500 (5.2 %) captured with zero false
    positives; both `procedural_type` and `case_type` are checked because
    either may carry the signal depending on how the LLM routed the label.
    """
    for s in (procedural_type, case_type):
        if s and _AUJ_RE.search(_strip_accents(s).lower()):
            return True
    return False


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-flash")
MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "80000"))


# The system message is built ONCE at import time so its bytes are
# byte-identical across every request — maximising prompt-cache hits on
# providers that support it (OpenAI, Anthropic, DeepSeek, and others).
# Put ALL static content (rules + full JSON schema) here; the user message
# contains only per-document variable data.
_SCHEMA_JSON = json.dumps(ExtractedInfo.model_json_schema(), ensure_ascii=False)

SYSTEM_PROMPT = f"""# ROLE
You are a specialised legal-information extraction engine for Portuguese jurisprudence
published on DGSI (www.dgsi.pt). Every input is the full raw text of one judicial
decision. Every output is a single JSON object that strictly matches the schema below.

# OUTPUT CONTRACT (hard rules — any violation invalidates the answer)
1. Return VALID JSON ONLY. No prose, no markdown fences, no commentary, no trailing text.
2. The JSON MUST parse against the schema at the end of this message.
3. Missing scalar fields → null. Missing array fields → []. Never omit a field.
4. All dates MUST be ISO 8601 (YYYY-MM-DD). If only year/month is known, use YYYY-MM-01;
   year-only → YYYY-01-01.
5. All free-text fields MUST be in the original language of the decision
   (Portuguese for DGSI — do NOT translate to English).
6. `citation_context` MUST be exactly one of: supporting | distinguishing | criticizing | referencing.
7. `extraction_confidence` MUST be exactly one of: high | medium | low.
8. Currency amounts: `amount` is a JSON number (no thousand separators, "." as decimal);
   default `currency` = "EUR".
9. Do NOT invent facts, parties, laws, precedents or authors. If a field is not in the
   source, leave it null / empty array.

# FIELD GUIDANCE

## IDENTITY
- process_number: exact case number as shown (e.g. "4779/24.6T8VNG.P1.S1").
- court_name: full court name in Portuguese.
- judge_name: relator/rapporteur, in the form shown (typically ALL CAPS).
- decision_date: the date this decision was handed down by the court of the current instance.
- decision_type: Acórdão | Sentença | Despacho | Deliberação | …
- case_type: specific procedural variety ("Revista excepcional", "Recurso Contraordenacional",
  "Processo Especial de Revitalização", …).

## SUBSTANTIVE CONTENT
- summary: ≤300 words, third-person, in Portuguese. Focus on WHAT the court decided and WHY.
- legal_question: the single precise question the court had to answer (one sentence).
  Prefer the wording the decision itself uses.
- decision_outcome: what was actually ordered (procedência / improcedência / anulação / …).
- ratio_decidendi: the binding legal reasoning, tightly worded (3–6 sentences). Always cite
  the legal basis (article + code) when it appears.

## LEGAL METADATA
- legal_descriptors: verbatim from the "Descritores" block (ALL CAPS as in source, one
  string per item). If there is no such block, leave [].
- legal_domain: high-level area (Direito Civil, Penal, Administrativo, do Trabalho, …).
  Compose with "/" when multiple (e.g. "Direito Comercial / Direito de Família").
- procedural_type: type of appeal / action (REVISTA, APELAÇÃO, AGRAVO, RECURSO PENAL, …).
- instance_level: formal name of the instance (e.g. "Supremo Tribunal de Justiça (3ª instância)",
  "Tribunal da Relação (2ª instância)", "1ª instância").

## CITATIONS (be exhaustive, but only include what the text actually cites)

### legislation_cited[] — {{article, law, citation_context}}
CRITICAL: every reference to the same article of the same code must produce a byte-identical
(article, law) pair across the whole corpus. Abbreviations, slashes, missing dots and date
suffixes break deduplication downstream. Follow the canonical format exactly.

`law` — always the FULL canonical Portuguese name, WITH accents, NEVER abbreviated, NEVER
prefixed with "Artigo ...", NEVER suffixed with a promulgation date. Canonical names to use:
    "Código Civil"                                       (not CC, C.C., Cód. Civil)
    "Código de Processo Civil"                           (not CPC, NCPC, Novo CPC)
    "Código Penal"                                       (not CP)
    "Código de Processo Penal"                           (not CPP)
    "Código do Trabalho"                                 (not CT)
    "Código de Processo do Trabalho"                     (not CPT)
    "Código das Sociedades Comerciais"                   (not CSC)
    "Constituição da República Portuguesa"               (not CRP, C.R.P.)
    "Código do Procedimento Administrativo"              (not CPA)
    "Código de Processo nos Tribunais Administrativos"   (not CPTA)
    "Código dos Contratos Públicos"                      (not CCP)
    "Regulamento das Custas Processuais"                 (not RCP)
    "Código do IVA" / "Código do IRC" / "Código do IRS"  (short forms ARE canonical)
For specific statutes: "Lei n.º <NUM>/<ANO>" (e.g. "Lei n.º 23/2007" — drop ", de 4 de julho").
For decree-laws:       "Decreto-Lei n.º <NUM>/<ANO>" (e.g. "Decreto-Lei n.º 15/93").
For EU regulations:    "Regulamento (UE) n.º <NUM>/<ANO>" / "Diretiva <NUM>/<ANO>/UE".
If a code you meet is not in the list above, expand its abbreviation to the full official
Portuguese name (e.g. "Código da Estrada", "Código do Registo Predial") with accents.

`article` — pure article locator, lowercase "n.º" and "alínea", ALWAYS ".º" (dot + masculine
ordinal), NEVER "º" alone, NEVER "°", NEVER "/" as separator, NEVER "al." for alínea.
    "394.º"                          — single article
    "394.º, n.º 1"                   — article + number
    "394.º, n.º 1, alínea a)"        — article + number + alínea (lowercase letter, ")")
    "394.º, n.ºs 1 e 2"              — FORBIDDEN: split into two entries instead
    "394.º e 395.º"                  — FORBIDDEN: split into two entries instead
    "5.º-A"                          — sub-article with letter (uppercase, hyphen)
    "672.º, n.º 1, alíneas a) e b)"  — FORBIDDEN: split into two entries (one per alínea)
Rules:
    • Do NOT include the word "artigo" in `article`.
    • Do NOT include the law name in `article`.
    • If the source cites a range or several paragraphs/alíneas, emit ONE entry per
      distinct (article, n.º, alínea) tuple so deduplication works.

`citation_context` — one of: supporting | distinguishing | criticizing | referencing.
    • supporting   — the ratio decidendi relies on this article.
    • distinguishing — used to differentiate the case from another.
    • criticizing  — argued against or questioned.
    • referencing  — neutral mention, tangential cite, backdrop.

Deduplicate: one entry per distinct (article, law) pair. If the same article is cited
multiple times, keep the most relevant citation_context (supporting > distinguishing >
criticizing > referencing).

### jurisprudence_cited[] — {{process_number, court_name, court_abbreviation, decision_date, citation_context}}
Every prior decision cited. Keep `process_number` verbatim, `decision_date` as ISO,
`court_abbreviation` as STJ | TRL | TRP | TRE | TRC | TRG | TC | TCAS | TCAN | STA | … .

### doctrine_cited[] — {{author, title, citation, text_cited, citation_context, impact_on_decision}}
Every scholar / treatise cited. Only include authorities actually named or quoted.

DO NOT fabricate any citation in any of the three arrays above.

## PARTIES & FACTS
- parties[]: {{role, name, type}}. role as in the decision ("Recorrente", "Autor",
  "Arguido", "Assistente", …). type ∈ {{"individual","company","state","other"}}.
- amounts_involved[]: every explicit monetary figure — {{amount, currency, description}}.
  Use concrete descriptions ("indemnização por danos morais"), not just "valor".
- timeline_events[]: dated procedural or factual events — {{date, event, location}}.

## CASE-SPECIFIC FLAGS (domain-aware booleans)
- is_jurisprudence_unification: true ONLY when THIS decision is itself an Acórdão
  de Uniformização de Jurisprudência (AUJ) or Acórdão de Fixação de Jurisprudência
  — i.e. issued under the STJ/STA mechanisms that resolve contradictory prior
  caselaw and fix binding jurisprudence (CPP arts. 437.º–448.º; CPC art. 688.º;
  CPTA art. 152.º). Positive signals (ANY of these is sufficient):
    • `procedural_type` / `case_type` reads "Uniformização de Jurisprudência",
      "Fixação de Jurisprudência", "Acórdão Uniformizador", or the DGSI typo
      "Uniforização de Jurisprudência".
    • the dispositive part explicitly fixes/uniformiza binding jurisprudence
      ("fixa-se a seguinte jurisprudência: …" / "uniformiza-se a jurisprudência
      nos seguintes termos: …").
    • the decision is published as an "Acórdão n.º X/ANO" in the Diário da
      República série I (the canonical AUJ publication channel).
  FALSE for ordinary revistas, apelações, recursos, habeas corpus, reclamações
  and every other procedural variety — even when they CITE an AUJ. Merely
  quoting or following an AUJ does NOT make the current decision one.
- documentary_evidence / expert_testimony / medical_evidence / witness_testimony:
    • true  → the decision explicitly relies on or analyses this evidence type;
    • false → the decision explicitly rejects it or notes its absence;
    • null  → the decision does not discuss it at all (common for procedural decisions).
- liability_found: true/false for civil/criminal liability decisions; null when the case
  is not about liability (e.g. procedural appeals, conflicts of competence).
- liability_reasoning: one short sentence explaining the liability conclusion.
- insurance_companies[]: names of insurers mentioned (strings).
- injuries[]: {{name, description, severity, disability_degree}}.
    severity ∈ {{"minor","moderate","severe","permanent"}}.

## SEARCH HELPERS (critical — drive retrieval downstream)
- semantic_search_query: ONE natural-language sentence in Portuguese (≤30 words)
  describing "what kind of case is this?" Designed for vector similarity search on legal
  questions. Focus on facts + legal issue, NOT on keywords. Full sentence, ends with period.
- keywords_search_query: 8–15 multi-word Portuguese legal phrases, COMMA-separated, each
  phrase being a retrieval unit a lawyer or lay user would type in a search box.
  Output format: "phrase one, phrase two, phrase three, ..." (single comma + single space).
  NORMALISATION RULES — apply character-by-character to every phrase before emitting it:
    1. Lowercase only (a–z, 0–9, space).
    2. Strip ALL diacritics/accents:
         á â ã à ä → a    é ê ë è → e    í î ï ì → i
         ó ô õ ò ö → o    ú û ü ù → u    ç → c    ñ → n
    3. Replace every hyphen "-" with a single space.
    4. Remove every other non-alphanumeric character (.,;:()[]{{}}"'!?/\<>|&*+#@$%^=`~
       INCLUDING the masculine ordinal ".º" and "º"). Collapse runs of spaces to one space.
    5. Each phrase MUST contain 2–6 tokens. NO single-word phrases.
    6. Use the FULL Portuguese name of any code/law, never an abbreviation (write
       "codigo de processo civil", never "cpc"; "constituicao da republica portuguesa",
       never "crp").
    7. Include article references as "artigo <N> <codigo>"
       (e.g. "artigo 643 codigo de processo civil") so an article search matches the doc.
    8. Include BOTH a formal legal phrasing and a colloquial phrasing where they differ
       (e.g. "responsabilidade civil extracontratual" and "pedido de indemnizacao";
       "resolucao com justa causa" and "rescisao do contrato de trabalho").
    9. Deduplicate phrases; do not repeat the same phrase twice.
  Good example: "recurso de revista, reclamacao de indeferimento, intempestividade do
    recurso, justo impedimento, artigo 643 codigo de processo civil, admissibilidade do
    recurso, despacho do relator, dupla conforme, poderes do supremo tribunal de justica"
  BAD examples (will be rejected):
    "recurso revista cpc 643"     ← abbreviations, no phrases, space-separated
    "Recurso de Revista, …"       ← uppercase
    "artigo 643.º CPC, …"         ← special chars ".º", abbreviation
    "responsabilidade, civil, …"  ← single-word phrases

## CONFIDENCE
- high   → decision is full-text, metadata and body present, most fields populated from
           explicit content.
- medium → some sections missing or noisy, but the main outcome / ratio is extractable.
- low    → text truncated, corrupted, metadata-only, or clearly not a decision.

# OUTPUT JSON SCHEMA (Pydantic model — authoritative)
{_SCHEMA_JSON}

# FINAL INSTRUCTION
After reading the document in the user message, return ONLY the JSON object described above."""


def build_extraction_prompt(doc: dict) -> str:
    """User-message content — only variable per-document data, placed AFTER the
    cached system prefix. Keep the header minimal so the cacheable share of the
    total token bill stays large."""
    full_text = (doc.get("full_text") or "")[:MAX_INPUT_CHARS]
    court = doc.get("court") or doc.get("court_short") or ""
    return (
        f"Court: {court}\n"
        f"URL: {doc.get('url', '')}\n"
        f"\n"
        f"=== FULL DOCUMENT TEXT ===\n"
        f"{full_text}\n"
        f"=== END ===\n"
        f"\n"
        f"Return the JSON object now."
    )


async def extract_document(
    client: httpx.AsyncClient,
    doc: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    usage_sink: Optional[list] = None,
) -> Optional[dict]:
    """Extract structured info from a document using OpenRouter LLM.

    If `usage_sink` is provided, the OpenRouter `usage` dict (containing
    prompt_tokens / completion_tokens / cached_tokens / cost if exposed)
    is appended to it per successful request.
    """
    async with semaphore:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_extraction_prompt(doc)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "usage": {"include": True},
        }
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/pt-caselaw-dgsi",
                        "X-Title": "pt-caselaw-dgsi",
                    },
                    timeout=240,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if usage_sink is not None and isinstance(data.get("usage"), dict):
                        usage_sink.append(data["usage"])
                    content = data["choices"][0]["message"]["content"]
                    # Some models return JSON wrapped in ```json fences — strip them.
                    content = content.strip()
                    if content.startswith("```"):
                        content = content.strip("`")
                        if content.lower().startswith("json"):
                            content = content[4:].lstrip()
                    parsed = json.loads(content)
                    result = ExtractedInfo(**parsed).model_dump()
                    result["keywords_search_query"] = _normalize_keywords(
                        result.get("keywords_search_query")
                    )
                    result["legislation_cited"] = _normalize_legislation(
                        result.get("legislation_cited") or []
                    )
                    # AUJ is primarily LLM-generated (full document context).
                    # The deterministic detector acts as a safety-net: if the
                    # LLM missed the flag but `procedural_type` / `case_type`
                    # carry an unambiguous AUJ signal, we still capture it.
                    result["is_jurisprudence_unification"] = (
                        bool(result.get("is_jurisprudence_unification"))
                        or _is_jurisprudence_unification(
                            result.get("procedural_type"),
                            result.get("case_type"),
                        )
                    )
                    return result
                if resp.status_code in (429, 500, 502, 503, 504, 529):
                    wait = (2 ** attempt) + (attempt * 0.5)
                    await asyncio.sleep(wait)
                    continue
                console.print(f"[red]HTTP {resp.status_code} for {doc.get('url','')}: {resp.text[:200]}[/red]")
                return None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if attempt == max_retries - 1:
                    console.print(f"[yellow]Parse error {type(e).__name__}: {str(e)[:200]}[/yellow]")
                    return None
                await asyncio.sleep(2 ** attempt)
            except (httpx.TimeoutException, httpx.RequestError):
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None
