"""Maintained runtime health contract for the technical demo."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_is_liveness_not_dependency_readiness() -> None:
    app = create_app()

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_reflects_chat_graph_initialization() -> None:
    app = create_app()
    client = TestClient(app)

    assert client.get("/ready").status_code == 503

    app.state.chat_ready = True
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["chat"] == "initialized"


def test_metrics_endpoint_is_available_for_the_monitoring_profile() -> None:
    response = TestClient(create_app()).get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
