from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.controller import health_controller, ocr_controller
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="OCR microservice powered by CrewAI for intelligent document analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_controller.router)
app.include_router(ocr_controller.router)
