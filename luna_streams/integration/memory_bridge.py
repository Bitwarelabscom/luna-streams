"""Memory Bridge - converts Luna Chat formats to structured events.

This is the translation layer between Luna Chat's internal data
formats and the Mamba Streams event schema. The event schema is
an immutable API contract - this bridge adapts to it.
"""

import logging
from datetime import datetime, timezone

from ..api.schemas import (
    ConversationMeta,
    EventContent,
    EventSource,
    EventType,
    Relation,
    StructuredEvent,
)

logger = logging.getLogger("luna_streams.bridge")


class MemoryBridge:
    """Converts Luna Chat event payloads into StructuredEvent format.

    Luna Chat sends events via HTTP POST to /api/events. This class
    provides helper methods for the TypeScript client to construct
    properly formatted events.

    The actual conversion happens in luna-chat's TypeScript client
    (src/integration/luna-streams.client.ts). This Python-side bridge
    validates and normalizes incoming events.
    """

    @staticmethod
    def from_chat_interaction(payload: dict) -> StructuredEvent:
        """Convert a recordChatInteraction payload.

        Expected input (from luna-chat):
        {
            "type": "message" | "response",
            "content": "raw text",
            "metadata": {"mode": "companion", "model": "grok-4.1"},
            "enrichment": {
                "emotionalValence": 0.3,
                "attentionScore": 0.7,
                "interMessageMs": 5200
            }
        }
        """
        enrichment = payload.get("enrichment", {})
        metadata = payload.get("metadata", {})

        return StructuredEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.MEMORY_ENTRY,
            source=EventSource.CONVERSATION,
            content=EventContent(
                sentiment=enrichment.get("emotionalValence", 0.0),
                importance=enrichment.get("attentionScore", 0.5),
                summary=payload.get("content", "")[:200],
            ),
            conversation_meta=ConversationMeta(
                message_length=len(payload.get("content", "")),
                response_time_ms=enrichment.get("interMessageMs", 0),
                active_model=metadata.get("model"),
                active_persona=metadata.get("mode"),
            ),
        )

    @staticmethod
    def from_graph_entities(payload: dict) -> StructuredEvent:
        """Convert a graph entity extraction result.

        Expected input:
        {
            "entities": [{"label": "Mamba", "type": "TOOL", "confidence": 0.9}],
            "cooccurrences": 3
        }
        """
        entities = [e["label"] for e in payload.get("entities", [])]

        return StructuredEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.ENTITY_UPDATE,
            source=EventSource.CONVERSATION,
            content=EventContent(entities=entities),
        )

    @staticmethod
    def from_edge_classification(payload: dict) -> StructuredEvent:
        """Convert edge classification results.

        Expected input:
        {
            "edges": [
                {"source": "Luna", "target": "Mamba", "edge_type": "integrates", "weight": 0.8}
            ]
        }
        """
        relations = [
            Relation(
                **{
                    "from": e["source"],
                    "to": e["target"],
                    "type": e.get("edge_type", "co_occurrence"),
                    "weight": e.get("weight", 0.5),
                }
            )
            for e in payload.get("edges", [])
        ]

        entities = list({e["source"] for e in payload.get("edges", [])}
                       | {e["target"] for e in payload.get("edges", [])})

        return StructuredEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.EDGE_UPDATE,
            source=EventSource.CONVERSATION,
            content=EventContent(
                entities=entities,
                relations=relations,
            ),
        )

    @staticmethod
    def from_session_meta(payload: dict) -> StructuredEvent:
        """Convert session metadata (for Stream 3).

        Expected input:
        {
            "messageLength": 142,
            "responseTimeMs": 3200,
            "sessionDurationMin": 45,
            "activePersona": "Sol",
            "activeModel": "grok-4.1",
            "turnNumber": 12
        }
        """
        return StructuredEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=EventType.CONVERSATION_META,
            source=EventSource.CONVERSATION,
            content=EventContent(importance=0.3),
            conversation_meta=ConversationMeta(
                message_length=payload.get("messageLength", 0),
                response_time_ms=payload.get("responseTimeMs", 0),
                session_duration_min=payload.get("sessionDurationMin", 0),
                active_persona=payload.get("activePersona"),
                active_model=payload.get("activeModel"),
                turn_number=payload.get("turnNumber", 0),
            ),
        )
