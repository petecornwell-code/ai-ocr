import asyncio
import json

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.domain.enums import JobStatus
from app.domain.schemas import (
    ExtractionSchema,
    OCRJobListResponse,
    OCRJobResponse,
    OCRJobStatusResponse,
)
from app.service.ocr_service import (
    create_job,
    delete_job,
    get_job,
    get_job_count,
    get_jobs,
    process_ocr_job,
    save_upload_file,
)

router = APIRouter(prefix="/ocr", tags=["OCR"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".pdf"}


def _job_to_response(job) -> dict:
    """Convert an OCRJob ORM object to a response-friendly dict."""
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "extraction_schema": (
            json.loads(job.extraction_schema) if job.extraction_schema else None
        ),
        "extracted_text": job.extracted_text,
        "crew_analysis": (
            json.loads(job.crew_analysis) if job.crew_analysis else None
        ),
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.post("/upload", response_model=OCRJobResponse, status_code=201)
async def upload_file(
    file: UploadFile,
    db: Session = Depends(get_db),
    extraction_schema: str | None = Form(default=None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB",
        )

    schema_dict = None
    if extraction_schema:
        try:
            schema_dict = json.loads(extraction_schema)
            ExtractionSchema(**schema_dict)
        except (json.JSONDecodeError, Exception) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid extraction_schema: {e}",
            )

    file_path = save_upload_file(content, file.filename)
    job = create_job(db, file.filename, file_path, extraction_schema=schema_dict)
    return _job_to_response(job)


@router.post("/jobs/{job_id}/process", response_model=OCRJobResponse)
async def process_job(job_id: int, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Job is already {job.status.value}")

    # Run the blocking OCR + crew work in a separate thread so the event
    # loop stays free for other requests.  process_ocr_job manages its own
    # DB session internally, so the request-scoped session is not shared.
    result_dict = await asyncio.to_thread(process_ocr_job, job_id)

    # Deserialize the JSON string fields for the response.
    result_dict["extraction_schema"] = (
        json.loads(result_dict["extraction_schema"])
        if result_dict["extraction_schema"]
        else None
    )
    result_dict["crew_analysis"] = (
        json.loads(result_dict["crew_analysis"])
        if result_dict["crew_analysis"]
        else None
    )
    return result_dict


@router.get("/jobs", response_model=OCRJobListResponse)
def list_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    jobs = get_jobs(db, skip=skip, limit=limit)
    total = get_job_count(db)
    return OCRJobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=OCRJobResponse)
def get_job_detail(job_id: int, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/jobs/{job_id}/status", response_model=OCRJobStatusResponse)
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return OCRJobStatusResponse(id=job.id, status=job.status.value)


@router.delete("/jobs/{job_id}", status_code=204)
def remove_job(job_id: int, db: Session = Depends(get_db)):
    if not delete_job(db, job_id):
        raise HTTPException(status_code=404, detail="Job not found")
