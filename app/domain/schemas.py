from datetime import datetime

from pydantic import BaseModel


class OCRJobResponse(BaseModel):
    id: int
    filename: str
    status: str
    extracted_text: str | None = None
    crew_analysis: str | None = None
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
