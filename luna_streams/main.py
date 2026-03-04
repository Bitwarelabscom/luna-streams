"""Entry point for Luna Mamba Streams.

Starts the FastAPI server and stream runner processes.
"""

import asyncio
import logging
import signal
import sys

import uvicorn

from .app import create_app
from .api.schemas import StructuredEvent
from .config import settings

logger = logging.getLogger("luna_streams")


async def stream_event_processor(
    event_queue: asyncio.Queue[StructuredEvent],
    stream_states: dict,
) -> None:
    """Background task that consumes events from the queue and feeds them to streams.

    This is a stub - the actual stream runner will be implemented in Phase 4
    once the model is trained and deployed. For now, it just counts events
    and updates state for API visibility.
    """
    logger.info("Stream event processor started (stub mode)")
    stream_states["user_model"]["status"] = "idle"

    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=5.0)

            # Route event to appropriate stream(s)
            etype = event.event_type.value

            if etype in ("memory_entry", "entity_update"):
                state = stream_states["user_model"]
                state["events_processed"] = state.get("events_processed", 0) + 1
                state["last_event_at"] = event.timestamp
                state["status"] = "running"
                logger.debug(f"User model processed event: {etype}")

            if etype in ("entity_update", "edge_update"):
                state = stream_states["knowledge_graph"]
                if state["status"] != "not_loaded":
                    state["events_processed"] = state.get("events_processed", 0) + 1
                    state["last_event_at"] = event.timestamp

            if etype == "conversation_meta" or event.conversation_meta:
                state = stream_states["conversation_dynamics"]
                if state["status"] != "not_loaded":
                    state["events_processed"] = state.get("events_processed", 0) + 1
                    state["last_event_at"] = event.timestamp

            event_queue.task_done()

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info("Stream event processor shutting down")
            break
        except Exception as e:
            logger.error(f"Stream event processor error: {e}", exc_info=True)
            await asyncio.sleep(1)


async def main() -> None:
    """Start the API server and stream processors."""
    logger.info("Luna Mamba Streams starting...")
    logger.info(f"Config: host={settings.host} port={settings.port} log_level={settings.log_level}")

    # Shared state between API and stream processors
    event_queue: asyncio.Queue[StructuredEvent] = asyncio.Queue(maxsize=10000)
    stream_states = {
        "user_model": {"status": "idle", "events_processed": 0, "head_outputs": {}},
        "knowledge_graph": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
        "conversation_dynamics": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
    }

    # Create FastAPI app with shared state
    app = create_app(event_queue, stream_states)

    # Start background stream processor
    processor_task = asyncio.create_task(
        stream_event_processor(event_queue, stream_states)
    )

    # Start uvicorn
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Graceful shutdown
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        processor_task.cancel()
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await server.serve()
    finally:
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    logger.info("Luna Mamba Streams stopped")


def run() -> None:
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
