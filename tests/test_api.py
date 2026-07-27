from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_chat_returns_tutor_answer():
    response = client.post(
        "/chat",
        json={"question": "What is one lakh?", "subject": "mathematics", "grade": 7},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_used"] == "tutor_agent"
    assert body["citations"]

