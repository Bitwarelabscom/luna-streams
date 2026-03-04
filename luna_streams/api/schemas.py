"""Pydantic models for the Luna Streams API.

The event schema is an immutable API contract - the memory bridge
adapts to it, not the other way around.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event schema (immutable contract from architecture spec)
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    MEMORY_ENTRY = "memory_entry"
    ENTITY_UPDATE = "entity_update"
    EDGE_UPDATE = "edge_update"
    CONVERSATION_META = "conversation_meta"


class EventSource(str, Enum):
    CONVERSATION = "conversation"
    AGENT_DIALOGUE = "agent_dialogue"
    NEURALSLEEP = "neuralsleep"
    SYSTEM = "system"


class Relation(BaseModel):
    from_entity: str = Field(alias="from")
    to_entity: str = Field(alias="to")
    type: str
    weight: float = 0.5

    model_config = {"populate_by_name": True}


class EventContent(BaseModel):
    entities: list[str] = []
    relations: list[Relation] = []
    topic_tags: list[str] = []
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""


class ConversationMeta(BaseModel):
    message_length: int = 0
    response_time_ms: int = 0
    session_duration_min: float = 0.0
    active_persona: Optional[str] = None
    active_model: Optional[str] = None
    turn_number: int = 0


class StructuredEvent(BaseModel):
    timestamp: datetime
    event_type: EventType
    source: EventSource
    content: EventContent
    conversation_meta: Optional[ConversationMeta] = None


# ---------------------------------------------------------------------------
# API request/response models
# ---------------------------------------------------------------------------

class EventBatchRequest(BaseModel):
    events: list[StructuredEvent]


class EventBatchResponse(BaseModel):
    accepted: int
    queued: bool


class StreamStatus(BaseModel):
    status: str  # "running", "idle", "stopped", "not_loaded"
    events_processed: int = 0
    last_event_at: Optional[datetime] = None
    state_norm: float = 0.0
    drift_signal: float = 0.0
    emotional_valence: float = 0.0


class StreamsResponse(BaseModel):
    streams: dict[str, StreamStatus]


class ContextResponse(BaseModel):
    context: str
    token_count: int
    changed: bool
    last_updated: Optional[datetime] = None


class StreamDetailResponse(BaseModel):
    status: str
    events_processed: int = 0
    last_event_at: Optional[datetime] = None
    state_norm: float = 0.0
    drift_signal: float = 0.0
    head_outputs: dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    streams_active: int
