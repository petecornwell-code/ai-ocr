import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

import app.database as _db_module
from app.database import Base, get_db
from app.main import app

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_db):
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(setup_db):
    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    # Also patch SessionLocal so that process_ocr_job (which opens its
    # own session in a background thread) uses the same test database.
    with patch.object(_db_module, "SessionLocal", TestingSessionLocal):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture
def upload_dir():
    d = tempfile.mkdtemp()
    os.environ["OCR_UPLOAD_DIR"] = d
    yield d
    shutil.rmtree(d, ignore_errors=True)
    os.environ.pop("OCR_UPLOAD_DIR", None)


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal valid PNG file for testing uploads."""
    from PIL import Image

    img = Image.new("RGB", (100, 30), color="white")
    path = tmp_path / "sample.png"
    img.save(str(path))
    return path
