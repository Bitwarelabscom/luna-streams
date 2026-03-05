"""FastAPI application factory for Luna Streams."""

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI

from .api.routes import init_routes, router
from .config import settings

if TYPE_CHECKING:
    from .integration.context_injector import ContextInjector


def create_app(
    event_queue: asyncio.Queue | None = None,
    stream_states: dict | None = None,
    context_injector: "ContextInjector | None" = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = FastAPI(
        title="Luna Mamba Streams",
        description="Continuous cognition layer - three parallel Mamba SSM streams",
        version="0.1.0",
    )

    # Initialize shared state
    if event_queue is None:
        event_queue = asyncio.Queue(maxsize=10000)
    if stream_states is None:
        stream_states = {
            "user_model": {"status": "idle", "events_processed": 0, "head_outputs": {}},
            "knowledge_graph": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
            "conversation_dynamics": {"status": "not_loaded", "events_processed": 0, "head_outputs": {}},
        }

    init_routes(event_queue, stream_states, context_injector=context_injector)
    app.include_router(router)

    return app
