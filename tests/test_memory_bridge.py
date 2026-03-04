"""Tests for the memory bridge conversions."""

from luna_streams.integration.memory_bridge import MemoryBridge


class TestMemoryBridge:
    def test_from_chat_interaction(self):
        payload = {
            "type": "message",
            "content": "I'm working on Mamba integration for Luna",
            "metadata": {"mode": "companion", "model": "grok-4.1"},
            "enrichment": {
                "emotionalValence": 0.6,
                "attentionScore": 0.8,
                "interMessageMs": 3200,
            },
        }
        event = MemoryBridge.from_chat_interaction(payload)
        assert event.event_type.value == "memory_entry"
        assert event.content.sentiment == 0.6
        assert event.content.importance == 0.8
        assert event.conversation_meta.response_time_ms == 3200
        assert event.conversation_meta.active_model == "grok-4.1"

    def test_from_graph_entities(self):
        payload = {
            "entities": [
                {"label": "Mamba", "type": "TOOL", "confidence": 0.9},
                {"label": "Luna", "type": "SOFTWARE", "confidence": 0.95},
            ],
            "cooccurrences": 2,
        }
        event = MemoryBridge.from_graph_entities(payload)
        assert event.event_type.value == "entity_update"
        assert "Mamba" in event.content.entities
        assert "Luna" in event.content.entities

    def test_from_edge_classification(self):
        payload = {
            "edges": [
                {"source": "Luna", "target": "Mamba", "edge_type": "integrates", "weight": 0.8},
                {"source": "Henke", "target": "Luna", "edge_type": "working_on", "weight": 0.9},
            ],
        }
        event = MemoryBridge.from_edge_classification(payload)
        assert event.event_type.value == "edge_update"
        assert len(event.content.relations) == 2
        assert "Henke" in event.content.entities

    def test_from_session_meta(self):
        payload = {
            "messageLength": 142,
            "responseTimeMs": 3200,
            "sessionDurationMin": 45,
            "activePersona": "Sol",
            "activeModel": "grok-4.1",
            "turnNumber": 12,
        }
        event = MemoryBridge.from_session_meta(payload)
        assert event.event_type.value == "conversation_meta"
        assert event.conversation_meta.active_persona == "Sol"
        assert event.conversation_meta.turn_number == 12

    def test_chat_interaction_empty_enrichment(self):
        """Should handle missing enrichment gracefully."""
        payload = {"type": "message", "content": "hello"}
        event = MemoryBridge.from_chat_interaction(payload)
        assert event.content.sentiment == 0.0
        assert event.content.importance == 0.5
