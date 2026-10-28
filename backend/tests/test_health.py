from fastapi.testclient import TestClient

from ghostline.api.app import create_app


def test_health_reports_ok():
    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
