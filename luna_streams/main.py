"""Entry point for Luna Mamba Streams.

Starts the FastAPI server and stream runner processes.
"""

import asyncio
import logging
import signal
import time

import uvicorn

from .api.schemas import StructuredEvent
from .config import settings
from .streams.user_model import UserModelStream

logger = logging.getLogger("luna_streams")


async def stream_event_processor(
    event_queue: asyncio.Queue[StructuredEvent],
    stream_states: dict,
    user_stream: UserModelStream,
) -> None:
    """Background task that consumes events from the queue and feeds them to streams."""
    mode = "GGUF" if user_stream.model is not None else "stub"
    logger.info(f"Stream event processor started ({mode} mode)")
    stream_states["user_model"]["status"] = "idle"

    last_save = time.monotonic()

    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=5.0)

            etype = event.event_type.value

            # Route to user_model stream
            if etype in ("memory_entry", "entity_update"):
                head_outputs = user_stream.process_event(event)

                # Update shared state from stream
                state = stream_states["user_model"]
                state["events_processed"] = user_stream.events_processed
                state["last_event_at"] = event.timestamp
                state["status"] = "running"
                state["state_norm"] = user_stream.ema.state_norm
                state["drift_signal"] = user_stream.ema.drift_signal
                state["emotional_valence"] = head_outputs.get("emotional_valence", 0.0)
                state["head_outputs"] = head_outputs
                logger.debug(
                    f"User model processed {etype}: "
                    f"drift={user_stream.ema.drift_signal:.4f} "
                    f"norm={user_stream.ema.state_norm:.4f}"
                )

            # Other streams (future - just count for now)
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

            # Periodic auto-save
            now = time.monotonic()
            if now - last_save > settings.auto_save_interval_sec:
                user_stream.save_state()
                last_save = now
                logger.info(
                    f"Auto-saved user_model state: "
                    f"events={user_stream.events_processed}, "
                    f"drift={user_stream.ema.drift_signal:.4f}"
                )

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info("Stream event processor shutting down")
            # Save state on shutdown
            if user_stream.events_processed > 0:
                user_stream.save_state()
                logger.info("Saved user_model state on shutdown")
            break
        except Exception as e:
            logger.error(f"Stream event processor error: {e}", exc_info=True)
            await asyncio.sleep(1)


async def main() -> None:
    """Start the API server and stream processors."""
    logger.info("Luna Mamba Streams starting...")
    logger.info(f"Config: host={settings.host} port={settings.port} log_level={settings.log_level}")

    # Initialize user model stream
    user_stream = UserModelStream()

    # Try to restore previous state
    if user_stream.restore_state():
        logger.info(f"Restored user_model: events={user_stream.events_processed}")
    else:
        logger.info("No previous user_model state found, starting fresh")

    # Load GGUF model
    user_stream.load_model()
    mode = "GGUF" if user_stream.model is not None else "stub"
    logger.info(f"User model stream initialized in {mode} mode")

    # Shared state between API and stream processors
    event_queue: asyncio.Queue[StructuredEvent] = asyncio.Queue(maxsize=10000)
    stream_states = {
        "user_model": {
            "status": "idle",
            "events_processed": user_stream.events_processed,
            "head_outputs": user_stream.head_outputs,
            "state_norm": user_stream.ema.state_norm,
            "drift_signal": user_stream.ema.drift_signal,
        },
        "knowledge_graph": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
        "conversation_dynamics": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
    }

    # Initialize context injector
    from .integration.context_injector import ContextInjector

    injector = ContextInjector()

    # Create FastAPI app with shared state
    from .app import create_app

    app = create_app(event_queue, stream_states, context_injector=injector)

    # Start background stream processor
    processor_task = asyncio.create_task(
        stream_event_processor(event_queue, stream_states, user_stream)
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
        await injector.close()

    logger.info("Luna Mamba Streams stopped")


def run() -> None:
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
