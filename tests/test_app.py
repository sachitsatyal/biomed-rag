from fastapi.testclient import TestClient
from serving_fakes import fake_components

from biomed_rag.serving.app import create_app
from biomed_rag.serving.rate_limit import RateLimiter


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


def test_stats_count_answers_and_errors():
    with client() as c:
        c.post("/ask", json={"question": "Does lecanemab work?", "mode": "rag"})
        c.post("/ask", json={"question": "please fail now", "mode": "rag"})
        stats = c.get("/stats").json()
    assert stats["requests_total"] == 2
    assert stats["errors"] == 1
    assert stats["recent_errors"] == ["RuntimeError"]
    assert stats["by_mode"]["rag"] == 2


def test_rate_limit_returns_429_with_retry_after():
    limiter = RateLimiter(per_minute=1, per_day=10, global_per_day=10)
    body = {"question": "Does lecanemab work?", "mode": "rag"}
    with TestClient(create_app(load=fake_components, limiter=limiter)) as c:
        assert c.post("/ask", json=body).status_code == 200
        response = c.post("/ask", json=body)
        stats = c.get("/stats").json()
    assert response.status_code == 429
    assert "per minute" in response.json()["detail"]
    assert int(response.headers["retry-after"]) > 0
    assert stats["rate_limited"] == 1
    assert stats["requests_total"] == 1  # the blocked request never reached the pipeline
