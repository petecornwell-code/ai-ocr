import json
import os
import uuid

import pytesseract
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.enums import JobStatus
from app.domain.models import OCRJob
from app.service.crew_service import run_ocr_analysis


def save_upload_file(file_content: bytes, filename: str) -> str:
    os.makedirs(settings.upload_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(settings.upload_dir, unique_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def perform_ocr(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text.strip()


def create_job(
    db: Session,
    filename: str,
    file_path: str,
    extraction_schema: dict | None = None,
) -> OCRJob:
    job = OCRJob(
        filename=filename,
        file_path=file_path,
        status=JobStatus.PENDING,
        extraction_schema=json.dumps(extraction_schema) if extraction_schema else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def process_ocr_job(job_id: int) -> dict:
    """Run OCR and crew analysis in a thread-safe way with its own DB session.

    This function is designed to be called via ``asyncio.to_thread`` so it
    does not block the event loop.  It opens and closes its own database
    session rather than sharing one across threads.

    Returns a response-ready dict (not an ORM object) so the caller does
    not need to touch the session.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(OCRJob).filter(OCRJob.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.PROCESSING
        db.commit()

        try:
            extracted_text = perform_ocr(job.file_path)
            job.extracted_text = extracted_text

            extraction_schema = (
                json.loads(job.extraction_schema) if job.extraction_schema else None
            )

            if extraction_schema:
                crew_result = run_ocr_analysis(extracted_text, extraction_schema)
                job.crew_analysis = json.dumps(crew_result)
            else:
                job.crew_analysis = None

            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)

        db.commit()
        db.refresh(job)

        # Snapshot all attributes while the session is still open so the
        # caller can build a response without touching the ORM object.
        return {
            "id": job.id,
            "filename": job.filename,
            "status": job.status.value if hasattr(job.status, "value") else job.status,
            "extraction_schema": job.extraction_schema,
            "extracted_text": job.extracted_text,
            "crew_analysis": job.crew_analysis,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
    finally:
        db.close()


def get_job(db: Session, job_id: int) -> OCRJob | None:
    return db.query(OCRJob).filter(OCRJob.id == job_id).first()


def get_jobs(db: Session, skip: int = 0, limit: int = 100) -> list[OCRJob]:
    return db.query(OCRJob).offset(skip).limit(limit).all()


def get_job_count(db: Session) -> int:
    return db.query(OCRJob).count()


def delete_job(db: Session, job_id: int) -> bool:
    job = db.query(OCRJob).filter(OCRJob.id == job_id).first()
    if not job:
        return False
    if os.path.exists(job.file_path):
        os.remove(job.file_path)
    db.delete(job)
    db.commit()
    return True
