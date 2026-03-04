"""Tests for the structured event schema."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from luna_streams.api.schemas import (
    ConversationMeta,
    EventBatchRequest,
    EventContent,
    EventSource,
    EventType,
    Relation,
    StructuredEvent,
)


def make_event(**overrides) -> dict:
    """Helper to create a valid event dict."""
    base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "memory_entry",
        "source": "conversation",
        "content": {
            "entities": ["Luna", "Mamba"],
            "relations": [],
            "topic_tags": ["architecture"],
            "sentiment": 0.7,
            "importance": 0.85,
            "summary": "Test event",
        },
    }
    base.update(overrides)
    return base


class TestStructuredEvent:
    def test_valid_memory_entry(self):
        event = StructuredEvent(**make_event())
        assert event.event_type == EventType.MEMORY_ENTRY
        assert event.source == EventSource.CONVERSATION
        assert len(event.content.entities) == 2

    def test_valid_entity_update(self):
        event = StructuredEvent(**make_event(event_type="entity_update"))
        assert event.event_type == EventType.ENTITY_UPDATE

    def test_valid_edge_update(self):
        data = make_event(event_type="edge_update")
        data["content"]["relations"] = [
            {"from": "Luna", "to": "Mamba", "type": "integrates", "weight": 0.8}
        ]
        event = StructuredEvent(**data)
        assert len(event.content.relations) == 1
        assert event.content.relations[0].type == "integrates"

    def test_valid_conversation_meta(self):
        data = make_event(event_type="conversation_meta")
        data["conversation_meta"] = {
            "message_length": 142,
            "response_time_ms": 3200,
            "session_duration_min": 45,
            "active_persona": "Sol",
            "active_model": "grok-4.1",
            "turn_number": 12,
        }
        event = StructuredEvent(**data)
        assert event.conversation_meta.active_persona == "Sol"
        assert event.conversation_meta.turn_number == 12

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            StructuredEvent(**make_event(event_type="invalid"))

    def test_sentiment_bounds(self):
        with pytest.raises(ValidationError):
            data = make_event()
            data["content"]["sentiment"] = 2.0
            StructuredEvent(**data)

    def test_importance_bounds(self):
        with pytest.raises(ValidationError):
            data = make_event()
            data["content"]["importance"] = -0.5
            StructuredEvent(**data)

    def test_minimal_event(self):
        """Event with only required fields."""
        event = StructuredEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.MEMORY_ENTRY,
            source=EventSource.CONVERSATION,
            content=EventContent(),
        )
        assert event.content.entities == []
        assert event.content.sentiment == 0.0

    def test_batch_request(self):
        events = [make_event() for _ in range(3)]
        batch = EventBatchRequest(events=[StructuredEvent(**e) for e in events])
        assert len(batch.events) == 3
