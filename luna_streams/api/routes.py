"""API routes for Luna Streams.

All endpoints for event ingestion, stream state queries,
context retrieval, and health checks.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    ContextResponse,
    EventBatchRequest,
    EventBatchResponse,
    HealthResponse,
    StreamDetailResponse,
    StreamsResponse,
    StreamStatus,
    StructuredEvent,
)

logger = logging.getLogger("luna_streams")

router = APIRouter()

# Shared state - populated by main.py at startup
_start_time: float = time.time()
_event_queue: asyncio.Queue[StructuredEvent] = asyncio.Queue()
_stream_states: dict[str, dict] = {}

# Last context cache for delta injection
_last_context: dict[str, tuple[str, datetime]] = {}


def init_routes(event_queue: asyncio.Queue, stream_states: dict) -> None:
    """Initialize shared state from main application."""
    global _event_queue, _stream_states, _start_time
    _event_queue = event_queue
    _stream_states = stream_states
    _start_time = time.time()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health():
    active = sum(1 for s in _stream_states.values() if s.get("status") == "running")
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime_seconds=round(time.time() - _start_time, 1),
        streams_active=active,
    )


# ---------------------------------------------------------------------------
# Event ingestion
# ---------------------------------------------------------------------------

@router.post("/api/events", response_model=EventBatchResponse)
async def ingest_events(batch: EventBatchRequest):
    """Accept structured events and queue them for stream processing."""
    accepted = 0
    for event in batch.events:
        try:
            _event_queue.put_nowait(event)
            accepted += 1
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
            break

    return EventBatchResponse(accepted=accepted, queued=True)


# ---------------------------------------------------------------------------
# Stream state queries
# ---------------------------------------------------------------------------

@router.get("/api/streams", response_model=StreamsResponse)
async def get_streams():
    """Return summary of all streams' current state."""
    streams = {}
    for name, state in _stream_states.items():
        streams[name] = StreamStatus(
            status=state.get("status", "not_loaded"),
            events_processed=state.get("events_processed", 0),
            last_event_at=state.get("last_event_at"),
            state_norm=state.get("state_norm", 0.0),
            drift_signal=state.get("drift_signal", 0.0),
            emotional_valence=state.get("emotional_valence", 0.0),
        )
    return StreamsResponse(streams=streams)


@router.get("/api/streams/{stream_name}/state", response_model=StreamDetailResponse)
async def get_stream_state(stream_name: str):
    """Return detailed state for a specific stream."""
    if stream_name not in _stream_states:
        raise HTTPException(404, f"Stream '{stream_name}' not found")

    state = _stream_states[stream_name]
    return StreamDetailResponse(
        status=state.get("status", "not_loaded"),
        events_processed=state.get("events_processed", 0),
        last_event_at=state.get("last_event_at"),
        state_norm=state.get("state_norm", 0.0),
        drift_signal=state.get("drift_signal", 0.0),
        head_outputs=state.get("head_outputs", {}),
    )


@router.get("/api/streams/{stream_name}/summary")
async def get_stream_summary(stream_name: str):
    """Return the text summary from a stream's decoder head."""
    if stream_name not in _stream_states:
        raise HTTPException(404, f"Stream '{stream_name}' not found")

    state = _stream_states[stream_name]
    heads = state.get("head_outputs", {})
    summary = heads.get("context_summary", "Stream not yet producing summaries.")

    return {"stream": stream_name, "summary": summary}


# ---------------------------------------------------------------------------
# Context retrieval (critical path for Luna Chat)
# ---------------------------------------------------------------------------

@router.get("/api/context", response_model=ContextResponse)
async def get_context(user_id: Optional[str] = Query(None)):
    """Return combined context string for system prompt injection.

    Uses delta tracking: returns changed=false if context hasn't
    shifted significantly since last query.
    """
    # Build context from all active streams
    sections = []

    user_state = _stream_states.get("user_model", {})
    if user_state.get("status") == "running":
        heads = user_state.get("head_outputs", {})
        summary = heads.get("context_summary", "")
        if summary:
            sections.append(f"[User State] {summary}")

        drift = user_state.get("drift_signal", 0.0)
        if drift > 0.3:
            sections.append(f"[Drift] {drift:.2f} (high - unusual pattern)")
        elif drift > 0.1:
            sections.append(f"[Drift] {drift:.2f} (moderate shift)")

    knowledge_state = _stream_states.get("knowledge_graph", {})
    if knowledge_state.get("status") == "running":
        heads = knowledge_state.get("head_outputs", {})
        summary = heads.get("knowledge_summary", "")
        if summary:
            sections.append(f"[Knowledge State] {summary}")

    dynamics_state = _stream_states.get("conversation_dynamics", {})
    if dynamics_state.get("status") == "running":
        heads = dynamics_state.get("head_outputs", {})
        summary = heads.get("dynamics_summary", "")
        if summary:
            sections.append(f"[Dynamics] {summary}")

    context = "\n".join(sections) if sections else ""
    token_count = len(context.split()) if context else 0  # rough estimate

    # Delta check
    cache_key = user_id or "default"
    last = _last_context.get(cache_key)
    changed = True
    if last and last[0] == context:
        changed = False

    now = datetime.now(timezone.utc)
    if changed and context:
        _last_context[cache_key] = (context, now)

    return ContextResponse(
        context=context,
        token_count=token_count,
        changed=changed,
        last_updated=_last_context.get(cache_key, (None, None))[1],
    )


# ---------------------------------------------------------------------------
# Debug / tools
# ---------------------------------------------------------------------------

@router.get("/api/benchmark")
async def get_benchmark():
    """Return last benchmark results if available."""
    import json
    from pathlib import Path

    path = Path("benchmark_results.json")
    if not path.exists():
        raise HTTPException(404, "No benchmark results found. Run: python benchmark.py")

    return json.loads(path.read_text())


@router.get("/api/streams/{stream_name}/snapshots")
async def get_snapshots(stream_name: str):
    """List available state snapshots for a stream."""
    from pathlib import Path
    from ..config import settings

    state_dir = Path(settings.state_dir) / stream_name.replace("_", "/")
    if not state_dir.exists():
        state_dir = Path(settings.state_dir) / stream_name
    if not state_dir.exists():
        return {"stream": stream_name, "snapshots": []}

    snapshots = sorted(state_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "stream": stream_name,
        "snapshots": [
            {
                "filename": s.name,
                "size_mb": round(s.stat().st_size / (1024**2), 2),
                "modified": datetime.fromtimestamp(s.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            for s in snapshots
        ],
    }


@router.post("/api/streams/{stream_name}/rollback")
async def rollback_stream(stream_name: str, snapshot: str = Query(...)):
    """Rollback a stream to a previous snapshot."""
    # TODO: implement after stream runner is built (Phase 4)
    raise HTTPException(501, "Rollback not yet implemented")
