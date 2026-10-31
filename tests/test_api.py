import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from cc_copilot.api import app, get_model_choice
from cc_copilot.types.model_choice import ModelChoice

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def client():
    app.dependency_overrides[get_model_choice] = lambda: ModelChoice(model=TestModel())
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_rejects_wrong_key(client):
    assert client.get("/api/v1/health", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_mask_endpoint(client):
    response = client.post("/api/v1/pii/mask", json={"text": "جوالي 0551234567"}, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["masked_text"] == "جوالي [MOBILE]"
    assert body["findings"][0]["type"] == "MOBILE"


def test_chat_review_endpoint(client):
    response = client.post("/api/v1/review/chat", json={"transcript": "Agent: هلا والله"}, headers=AUTH)
    assert response.status_code == 200
    assert "masked_transcript" in response.json()


def test_call_endpoint_rejects_non_audio(client):
    files = {"file": ("note.txt", b"just text", "text/plain")}
    assert client.post("/api/v1/review/call", files=files, headers=AUTH).status_code == 415


def test_llm_endpoint_returns_503_when_gemini_is_not_configured(monkeypatch):
    from cc_copilot.env import get_default_model_choice

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_GOOGLE_MODEL", raising=False)
    get_default_model_choice.cache_clear()
    response = TestClient(app).post("/api/v1/review/chat", json={"transcript": "Agent: hi"}, headers=AUTH)
    assert response.status_code == 503
