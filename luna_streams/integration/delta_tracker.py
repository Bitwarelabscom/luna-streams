"""Delta tracker for context injection optimization.

Tracks state norm changes and only triggers summary regeneration
when the Mamba state shifts significantly. Prevents redundant
context injection and saves compute on the summary decoder.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("luna_streams.delta")


@dataclass
class DeltaRecord:
    state_norm: float = 0.0
    context_text: str = ""
    updated_at: float = 0.0


class DeltaTracker:
    """Tracks state changes across streams for delta injection."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._records: dict[str, DeltaRecord] = {}

    def has_changed(self, stream_name: str, current_norm: float) -> bool:
        """Check if a stream's state has changed significantly."""
        record = self._records.get(stream_name)
        if record is None:
            return True

        delta = abs(current_norm - record.state_norm)
        return delta > self.threshold

    def update(self, stream_name: str, state_norm: float, context_text: str) -> None:
        """Record a new state snapshot after summary regeneration."""
        self._records[stream_name] = DeltaRecord(
            state_norm=state_norm,
            context_text=context_text,
            updated_at=time.time(),
        )

    def get_cached_context(self, stream_name: str) -> str:
        """Get the last cached context text for a stream."""
        record = self._records.get(stream_name)
        return record.context_text if record else ""
