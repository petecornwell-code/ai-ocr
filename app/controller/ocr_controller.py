from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.domain.enums import JobStatus
from app.domain.schemas import OCRJobListResponse, OCRJobResponse, OCRJobStatusResponse
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


@router.post("/upload", response_model=OCRJobResponse, status_code=201)
async def upload_file(file: UploadFile, db: Session = Depends(get_db)):
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

    file_path = save_upload_file(content, file.filename)
    job = create_job(db, file.filename, file_path)
    return job


@router.post("/jobs/{job_id}/process", response_model=OCRJobResponse)
def process_job(job_id: int, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Job is already {job.status.value}")

    result = process_ocr_job(db, job_id)
    return result


@router.get("/jobs", response_model=OCRJobListResponse)
def list_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    jobs = get_jobs(db, skip=skip, limit=limit)
    total = get_job_count(db)
    return OCRJobListResponse(jobs=jobs, total=total)


@router.get("/jobs/{job_id}", response_model=OCRJobResponse)
def get_job_detail(job_id: int, db: Session = Depends(get_db)):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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
