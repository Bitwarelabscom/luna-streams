"""Integration tests for the API endpoints."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from luna_streams.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


class TestEventIngestion:
    def test_ingest_single_event(self, client):
        resp = client.post("/api/events", json={
            "events": [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "memory_entry",
                "source": "conversation",
                "content": {
                    "entities": ["Luna"],
                    "topic_tags": ["test"],
                    "sentiment": 0.5,
                    "importance": 0.7,
                    "summary": "Test event",
                },
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
        assert data["queued"] is True

    def test_ingest_batch(self, client):
        events = []
        for i in range(5):
            events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "memory_entry",
                "source": "conversation",
                "content": {"summary": f"Event {i}"},
            })
        resp = client.post("/api/events", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 5

    def test_invalid_event_type(self, client):
        resp = client.post("/api/events", json={
            "events": [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "invalid_type",
                "source": "conversation",
                "content": {},
            }],
        })
        assert resp.status_code == 422


class TestStreamQueries:
    def test_get_streams(self, client):
        resp = client.get("/api/streams")
        assert resp.status_code == 200
        streams = resp.json()["streams"]
        assert "user_model" in streams
        assert "knowledge_graph" in streams
        assert "conversation_dynamics" in streams

    def test_get_stream_state(self, client):
        resp = client.get("/api/streams/user_model/state")
        assert resp.status_code == 200

    def test_get_unknown_stream(self, client):
        resp = client.get("/api/streams/nonexistent/state")
        assert resp.status_code == 404


class TestContextEndpoint:
    def test_get_context(self, client):
        resp = client.get("/api/context")
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "changed" in data
        assert "token_count" in data

    def test_context_with_user_id(self, client):
        resp = client.get("/api/context?user_id=test-user")
        assert resp.status_code == 200
