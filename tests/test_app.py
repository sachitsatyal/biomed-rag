from fastapi.testclient import TestClient
from serving_fakes import fake_components

from biomed_rag.serving.app import create_app


def client() -> TestClient:
    return TestClient(create_app(load=fake_components))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "model": "fake-model"}


def test_ask_returns_cited_answer():
    with client() as c:
        response = c.post("/ask", json={"question": "Does lecanemab work?", "mode": "rag"})
    assert response.status_code == 200
    assert response.json()["claims"][0]["pmids"] == ["111"]


def test_invalid_request_is_rejected_with_422():
    with client() as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422


def test_pipeline_failure_returns_503():
    with client() as c:
        response = c.post("/ask", json={"question": "please fail now", "mode": "rag"})
    assert response.status_code == 503


def test_home_page_is_served():
    with client() as c:
        response = c.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "biomed-rag" in response.text
