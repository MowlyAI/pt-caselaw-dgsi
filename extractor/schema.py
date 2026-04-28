"""
Pydantic schema for LLM-extracted structured data from DGSI decisions.
Mirrors the user-specified JSON schema one-to-one.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CitationContext = Literal["supporting", "distinguishing", "criticizing", "referencing"]


class LegislationCited(BaseModel):
    article: Optional[str] = None
    law: Optional[str] = None
    citation_context: Optional[str] = None


class JurisprudenceCited(BaseModel):
    process_number: Optional[str] = None
    court_name: Optional[str] = None
    court_abbreviation: Optional[str] = None
    decision_date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    citation_context: Optional[str] = None


class DoctrineCited(BaseModel):
    author: Optional[str] = None
    title: Optional[str] = None
    citation: Optional[str] = None
    text_cited: Optional[str] = None
    citation_context: Optional[str] = None
    impact_on_decision: Optional[str] = None


class Party(BaseModel):
    role: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = Field(default=None, description="individual/company")


class AmountInvolved(BaseModel):
    amount: Optional[float] = None
    currency: Optional[str] = "EUR"
    description: Optional[str] = None


class TimelineEvent(BaseModel):
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    event: Optional[str] = None
    location: Optional[str] = None


class Injury(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = Field(default=None, description="minor/moderate/severe/permanent")
    disability_degree: Optional[str] = None


class ExtractedInfo(BaseModel):
    """Complete structured representation of a DGSI decision."""
    process_number: Optional[str] = None
    court_name: Optional[str] = None
    judge_name: Optional[str] = None
    decision_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    decision_type: Optional[str] = Field(default=None, description="Acórdão/Sentença/Despacho/…")
    case_type: Optional[str] = Field(default=None, description="high-level category")
    summary: Optional[str] = Field(default=None, description="≤300 words")
    legal_question: Optional[str] = None
    decision_outcome: Optional[str] = None
    ratio_decidendi: Optional[str] = None
    legal_descriptors: list[str] = Field(default_factory=list)
    legislation_cited: list[LegislationCited] = Field(default_factory=list)
    jurisprudence_cited: list[JurisprudenceCited] = Field(default_factory=list)
    doctrine_cited: list[DoctrineCited] = Field(default_factory=list)
    parties: list[Party] = Field(default_factory=list)
    voting: Optional[str] = None
    procedural_type: Optional[str] = None
    legal_domain: Optional[str] = None
    instance_level: Optional[str] = None
    is_jurisprudence_unification: bool = Field(
        default=False,
        description=(
            "True ONLY when this decision is itself an Acórdão de Uniformização/"
            "Fixação de Jurisprudência (AUJ) under CPP arts. 437.º–448.º, "
            "CPC art. 688.º or CPTA art. 152.º. False for ordinary decisions, "
            "even when they cite an AUJ."
        ),
    )
    amounts_involved: list[AmountInvolved] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    liability_found: Optional[bool] = None
    liability_reasoning: Optional[str] = None
    documentary_evidence: Optional[bool] = None
    expert_testimony: Optional[bool] = None
    medical_evidence: Optional[bool] = None
    witness_testimony: Optional[bool] = None
    insurance_companies: list[str] = Field(default_factory=list)
    injuries: list[Injury] = Field(default_factory=list)
    semantic_search_query: Optional[str] = None
    keywords_search_query: Optional[str] = None
    extraction_confidence: Optional[str] = Field(default=None, description="high/medium/low")
