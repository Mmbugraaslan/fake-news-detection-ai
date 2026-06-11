from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint() -> None:
    response = client.post(
        "/predict",
        json={"text": "Deneme haber metni", "model_name": "classical"},
    )

    assert response.status_code in (200, 400)