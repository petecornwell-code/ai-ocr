import io
import json
from unittest.mock import patch

from PIL import Image


def _make_png_bytes():
    """Generate a minimal PNG image as bytes."""
    img = Image.new("RGB", (100, 30), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


SAMPLE_SCHEMA = {
    "fields": {
        "invoice_number": {"type": "string", "description": "Invoice number"},
        "total_amount": {"type": "string", "description": "Total amount due"},
    }
}

SAMPLE_CREW_RESULT = {
    "fields": {
        "invoice_number": {
            "value": "INV-001",
            "status": "consensus",
            "votes": {"agent_a": "INV-001", "agent_b": "INV-001", "agent_c": "INV-0O1"},
        },
        "total_amount": {
            "value": "$500.00",
            "status": "consensus",
            "votes": {"agent_a": "$500.00", "agent_b": "$500.00", "agent_c": "$500.00"},
        },
    },
    "summary": {
        "total_fields": 2,
        "consensus_count": 2,
        "intervention_count": 0,
    },
}


class TestUpload:
    def test_upload_valid_image(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test.png"
        assert data["status"] == "pending"
        assert data["id"] is not None
        assert data["extraction_schema"] is None

    def test_upload_with_extraction_schema(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"extraction_schema": json.dumps(SAMPLE_SCHEMA)},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["extraction_schema"] == SAMPLE_SCHEMA

    def test_upload_with_invalid_schema(self, client):
        png_bytes = _make_png_bytes()
        response = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"extraction_schema": "not valid json"},
        )
        assert response.status_code == 400
        assert "Invalid extraction_schema" in response.json()["detail"]

    def test_upload_invalid_extension(self, client):
        response = client.post(
            "/ocr/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]

    def test_upload_no_filename(self, client):
        response = client.post(
            "/ocr/upload",
            files={"file": ("", b"content", "image/png")},
        )
        assert response.status_code in (400, 422)

    def test_upload_allowed_extensions(self, client):
        png_bytes = _make_png_bytes()
        for ext in ["png", "jpg", "jpeg", "tiff", "bmp", "gif", "pdf"]:
            response = client.post(
                "/ocr/upload",
                files={"file": (f"test.{ext}", png_bytes, "image/png")},
            )
            assert response.status_code == 201, f"Failed for extension: {ext}"


class TestListJobs:
    def test_list_jobs_empty(self, client):
        response = client.get("/ocr/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_jobs_with_data(self, client):
        png_bytes = _make_png_bytes()
        client.post(
            "/ocr/upload",
            files={"file": ("a.png", png_bytes, "image/png")},
        )
        client.post(
            "/ocr/upload",
            files={"file": ("b.png", png_bytes, "image/png")},
        )
        response = client.get("/ocr/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["jobs"]) == 2

    def test_list_jobs_pagination(self, client):
        png_bytes = _make_png_bytes()
        for i in range(5):
            client.post(
                "/ocr/upload",
                files={"file": (f"img{i}.png", png_bytes, "image/png")},
            )
        response = client.get("/ocr/jobs?skip=2&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5


class TestGetJob:
    def test_get_job_detail(self, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        response = client.get(f"/ocr/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["filename"] == "test.png"

    def test_get_job_not_found(self, client):
        response = client.get("/ocr/jobs/9999")
        assert response.status_code == 404

    def test_get_job_status(self, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        response = client.get(f"/ocr/jobs/{job_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "pending"

    def test_get_status_not_found(self, client):
        response = client.get("/ocr/jobs/9999/status")
        assert response.status_code == 404


class TestProcessJob:
    @patch("app.service.ocr_service.perform_ocr", return_value="Extracted OCR text")
    @patch("app.service.ocr_service.run_ocr_analysis", return_value=SAMPLE_CREW_RESULT)
    def test_process_job_with_schema(self, mock_crew, mock_ocr, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"extraction_schema": json.dumps(SAMPLE_SCHEMA)},
        )
        job_id = upload_resp.json()["id"]

        response = client.post(f"/ocr/jobs/{job_id}/process")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["extracted_text"] == "Extracted OCR text"
        assert data["crew_analysis"]["summary"]["consensus_count"] == 2
        assert data["crew_analysis"]["fields"]["invoice_number"]["status"] == "consensus"

    @patch("app.service.ocr_service.perform_ocr", return_value="Extracted OCR text")
    def test_process_job_without_schema(self, mock_ocr, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        response = client.post(f"/ocr/jobs/{job_id}/process")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["extracted_text"] == "Extracted OCR text"
        assert data["crew_analysis"] is None

    @patch("app.service.ocr_service.perform_ocr", side_effect=Exception("OCR failed"))
    def test_process_job_failure(self, mock_ocr, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        response = client.post(f"/ocr/jobs/{job_id}/process")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "OCR failed" in data["error_message"]

    def test_process_job_not_found(self, client):
        response = client.post("/ocr/jobs/9999/process")
        assert response.status_code == 404

    @patch("app.service.ocr_service.perform_ocr", return_value="text")
    def test_process_job_already_processed(self, mock_ocr, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        client.post(f"/ocr/jobs/{job_id}/process")
        response = client.post(f"/ocr/jobs/{job_id}/process")
        assert response.status_code == 400
        assert "already" in response.json()["detail"]


class TestDeleteJob:
    def test_delete_job(self, client):
        png_bytes = _make_png_bytes()
        upload_resp = client.post(
            "/ocr/upload",
            files={"file": ("test.png", png_bytes, "image/png")},
        )
        job_id = upload_resp.json()["id"]

        response = client.delete(f"/ocr/jobs/{job_id}")
        assert response.status_code == 204

        response = client.get(f"/ocr/jobs/{job_id}")
        assert response.status_code == 404

    def test_delete_job_not_found(self, client):
        response = client.delete("/ocr/jobs/9999")
        assert response.status_code == 404
