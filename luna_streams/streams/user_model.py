"""Stream 1: User Model.

Continuously tracks the user - behavioral patterns, preferences,
emotional state, current focus, drift over time.

Built first. All other streams depend on this proving value.

Uses compact event encoding (~16 tokens avg) to stay under 150ms
per forward pass on CPU via llama.cpp GGUF.
"""

import logging
from pathlib import Path

from ..api.schemas import StructuredEvent
from ..config import settings
from ..heads.mlp_heads import create_user_model_heads
from .base_stream import BaseStream

logger = logging.getLogger("luna_streams.user_model")


class UserModelStream(BaseStream):
    """Stream 1 - User behavioral model."""

    def __init__(self):
        super().__init__(name="user_model", hidden_dim=1024)

        # Load trained MLP heads
        heads_path = Path(settings.model_dir) / settings.mlp_heads_path
        self.head_manager = create_user_model_heads(str(heads_path))
        if self.head_manager.is_loaded:
            logger.info(f"Loaded trained MLP heads from {heads_path}")
        else:
            logger.warning("No trained MLP heads - using heuristic fallback")

    def accepts_event(self, event: StructuredEvent) -> bool:
        """User model processes memory entries and entity updates."""
        return event.event_type.value in ("memory_entry", "entity_update")

    def event_to_tokens(self, event: StructuredEvent) -> str:
        """Convert event to compact text for tokenization.

        Target: ~16 tokens avg to stay under 150ms latency.
        Format: type_code source entities topics sent imp summary_snippet
        """
        # Compact type codes
        type_code = {
            "memory_entry": "mem_e",
            "entity_update": "ent_u",
            "edge_update": "edge_u",
            "conversation_meta": "conv_m",
        }.get(event.event_type.value, "unk")

        # Compact source codes
        src = {
            "conversation": "conv",
            "agent_dialogue": "agent",
            "neuralsleep": "sleep",
            "system": "sys",
        }.get(event.source.value, "?")

        parts = [type_code, src]
        c = event.content

        if c.entities:
            parts.append(",".join(c.entities[:3]))
        if c.topic_tags:
            parts.append(",".join(c.topic_tags[:2]))
        if c.sentiment != 0.0:
            parts.append(f"{c.sentiment:.1f}")
        if c.importance != 0.5:
            parts.append(f"{c.importance:.1f}")
        if c.summary:
            # Take first ~30 chars of summary
            parts.append(c.summary[:30].strip())

        meta = event.conversation_meta
        if meta:
            if meta.active_model:
                parts.append(meta.active_model.split("/")[-1][:8])
            if meta.response_time_ms:
                parts.append(f"{meta.response_time_ms}ms")

        return " ".join(parts)
