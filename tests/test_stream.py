"""Tests for stream model loading and compact event encoding."""

from datetime import datetime, timezone

import numpy as np
import pytest

from luna_streams.api.schemas import (
    ConversationMeta,
    EventContent,
    EventSource,
    EventType,
    StructuredEvent,
)
from luna_streams.streams.user_model import UserModelStream


def make_event(**overrides) -> StructuredEvent:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        event_type=EventType.MEMORY_ENTRY,
        source=EventSource.CONVERSATION,
        content=EventContent(
            entities=["Luna", "Henke"],
            topic_tags=["architecture"],
            sentiment=0.6,
            importance=0.8,
            summary="Working on Mamba integration",
        ),
    )
    defaults.update(overrides)
    return StructuredEvent(**defaults)


class TestCompactEncoding:
    def test_memory_entry_format(self):
        stream = UserModelStream()
        event = make_event()
        text = stream.event_to_tokens(event)
        assert text.startswith("mem_e conv")
        assert "Luna,Henke" in text
        assert "architecture" in text

    def test_entity_update_format(self):
        stream = UserModelStream()
        event = make_event(event_type=EventType.ENTITY_UPDATE)
        text = stream.event_to_tokens(event)
        assert text.startswith("ent_u conv")

    def test_token_count_compact(self):
        """Compact encoding should produce ~16 tokens on average."""
        stream = UserModelStream()
        event = make_event()
        text = stream.event_to_tokens(event)
        # Rough estimate: whitespace-split words ~ tokens
        word_count = len(text.split())
        assert word_count <= 20, f"Too many words ({word_count}): {text}"

    def test_entities_truncated_to_3(self):
        stream = UserModelStream()
        event = make_event(
            content=EventContent(
                entities=["A", "B", "C", "D", "E"],
                sentiment=0.5,
                importance=0.5,
            ),
        )
        text = stream.event_to_tokens(event)
        # Should only include first 3 entities
        assert "D" not in text
        assert "E" not in text

    def test_summary_truncated(self):
        stream = UserModelStream()
        long_summary = "a" * 100
        event = make_event(
            content=EventContent(summary=long_summary, sentiment=0.0, importance=0.5),
        )
        text = stream.event_to_tokens(event)
        # Summary should be truncated to ~30 chars
        assert len(text) < 60

    def test_with_conversation_meta(self):
        stream = UserModelStream()
        event = make_event(
            conversation_meta=ConversationMeta(
                active_model="grok-4.1",
                response_time_ms=3200,
                turn_number=12,
            ),
        )
        text = stream.event_to_tokens(event)
        assert "grok-4.1" in text
        assert "3200ms" in text


class TestStreamStubMode:
    def test_accepts_memory_entry(self):
        stream = UserModelStream()
        event = make_event()
        assert stream.accepts_event(event) is True

    def test_rejects_conversation_meta(self):
        stream = UserModelStream()
        event = make_event(event_type=EventType.CONVERSATION_META)
        assert stream.accepts_event(event) is False

    def test_process_event_stub(self):
        stream = UserModelStream()
        event = make_event()
        result = stream.process_event(event)
        assert "drift_signal" in result
        assert "state_norm" in result
        assert stream.events_processed == 1

    def test_multiple_events_update_ema(self):
        stream = UserModelStream()
        for i in range(10):
            event = make_event(
                content=EventContent(
                    sentiment=float(i) / 10,
                    importance=0.5,
                    summary=f"Event {i}",
                ),
            )
            stream.process_event(event)
        assert stream.events_processed == 10
        assert stream.ema.step_count == 10


class TestGGUFLoading:
    def test_load_missing_model_falls_to_stub(self):
        stream = UserModelStream()
        stream.load_model("/nonexistent/path/model.gguf")
        assert stream.model is None

    def test_get_status_stub(self):
        stream = UserModelStream()
        status = stream.get_status()
        assert status["status"] == "stub"
        assert status["events_processed"] == 0

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "/opt/luna-streams/models/mamba-370m/mamba-370m-q8_0.gguf"
        ).exists(),
        reason="GGUF model not downloaded",
    )
    def test_load_real_model(self):
        stream = UserModelStream()
        stream.load_model(
            "/opt/luna-streams/models/mamba-370m/mamba-370m-q8_0.gguf"
        )
        assert stream.model is not None
        status = stream.get_status()
        assert status["status"] == "running"

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "/opt/luna-streams/models/mamba-370m/mamba-370m-q8_0.gguf"
        ).exists(),
        reason="GGUF model not downloaded",
    )
    def test_real_inference(self):
        stream = UserModelStream()
        stream.load_model(
            "/opt/luna-streams/models/mamba-370m/mamba-370m-q8_0.gguf"
        )
        event = make_event()
        result = stream.process_event(event)
        assert "drift_signal" in result
        assert "emotional_valence" in result
        assert "focus_intensity" in result
        assert stream.events_processed == 1
