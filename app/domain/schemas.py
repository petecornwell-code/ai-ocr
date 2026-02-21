from datetime import datetime

from pydantic import BaseModel


# ── Extraction schema (input) ─────────────────────────────────────────


class ExtractionFieldSpec(BaseModel):
    """Describes a single field to extract from the document."""

    type: str = "string"
    description: str = ""


class ExtractionSchema(BaseModel):
    """JSON schema that tells the crew which fields to pull from the image."""

    fields: dict[str, ExtractionFieldSpec]


# ── Vote result structures (output) ───────────────────────────────────


class FieldVotes(BaseModel):
    agent_a: str | None = None
    agent_b: str | None = None
    agent_c: str | None = None


class FieldResult(BaseModel):
    value: str | None = None
    status: str  # "consensus" | "intervention_required"
    votes: FieldVotes


class VoteSummary(BaseModel):
    total_fields: int
    consensus_count: int
    intervention_count: int


class VoteResult(BaseModel):
    fields: dict[str, FieldResult]
    summary: VoteSummary


# ── Job responses ─────────────────────────────────────────────────────


class OCRJobResponse(BaseModel):
    id: int
    filename: str
    status: str
    extraction_schema: dict | None = None
    extracted_text: str | None = None
    crew_analysis: dict | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OCRJobListResponse(BaseModel):
    jobs: list[OCRJobResponse]
    total: int


class OCRJobStatusResponse(BaseModel):
    id: int
    status: str
