import os

import pytesseract
from PIL import Image
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.enums import JobStatus
from app.domain.models import OCRJob
from app.service.crew_service import run_ocr_analysis


def save_upload_file(file_content: bytes, filename: str) -> str:
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def perform_ocr(file_path: str) -> str:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text.strip()


def create_job(db: Session, filename: str, file_path: str) -> OCRJob:
    job = OCRJob(
        filename=filename,
        file_path=file_path,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def process_ocr_job(db: Session, job_id: int) -> OCRJob:
    job = db.query(OCRJob).filter(OCRJob.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job.status = JobStatus.PROCESSING
    db.commit()

    try:
        extracted_text = perform_ocr(job.file_path)
        job.extracted_text = extracted_text

        crew_result = run_ocr_analysis(extracted_text)
        job.crew_analysis = crew_result

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)

    db.commit()
    db.refresh(job)
    return job


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
