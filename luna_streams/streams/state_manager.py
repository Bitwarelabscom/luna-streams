"""State persistence for Mamba streams.

Uses safetensors for fast binary serialization of both
fast state AND slow state (EMA). Maintains rolling snapshots
with configurable retention.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("luna_streams.state")


class StateManager:
    """Manages state persistence for a single stream."""

    def __init__(self, stream_name: str, state_dir: str, retention: int = 3):
        self.stream_name = stream_name
        self.state_dir = Path(state_dir) / stream_name
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.retention = retention
        self.last_save_time = 0.0

    def save(self, state_dict: dict[str, np.ndarray]) -> Path:
        """Save state as safetensors. Returns path to saved file."""
        try:
            import safetensors.numpy as stn
        except ImportError:
            import safetensors.torch as stn

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"state_{timestamp}.safetensors"
        filepath = self.state_dir / filename

        # Convert all arrays to numpy if needed
        clean_dict = {}
        for key, val in state_dict.items():
            if isinstance(val, np.ndarray):
                clean_dict[key] = val
            else:
                clean_dict[key] = np.array(val)

        try:
            import safetensors.numpy
            safetensors.numpy.save_file(clean_dict, str(filepath))
        except Exception:
            # Fallback: save as npz
            filepath = filepath.with_suffix(".npz")
            np.savez(str(filepath), **clean_dict)

        self.last_save_time = time.time()
        logger.info(f"State saved: {filepath} ({filepath.stat().st_size / 1024:.1f}KB)")

        # Rotate old snapshots
        self._rotate_snapshots()

        return filepath

    def load_latest(self) -> dict[str, np.ndarray] | None:
        """Load the most recent state snapshot. Returns None if no state found."""
        # Try safetensors first, then npz
        snapshots = sorted(
            list(self.state_dir.glob("state_*.safetensors"))
            + list(self.state_dir.glob("state_*.npz")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not snapshots:
            logger.info(f"No state snapshots found for {self.stream_name}")
            return None

        latest = snapshots[0]
        logger.info(f"Loading state from {latest}")

        try:
            if latest.suffix == ".safetensors":
                try:
                    import safetensors.numpy
                    return safetensors.numpy.load_file(str(latest))
                except ImportError:
                    import safetensors.torch
                    data = safetensors.torch.load_file(str(latest))
                    return {k: v.numpy() for k, v in data.items()}
            else:
                data = np.load(str(latest))
                return dict(data)
        except Exception as e:
            logger.error(f"Failed to load state from {latest}: {e}")
            # Try the next snapshot
            if len(snapshots) > 1:
                logger.info("Trying previous snapshot...")
                return self.load_latest()
            return None

    def list_snapshots(self) -> list[dict]:
        """List available snapshots with metadata."""
        snapshots = sorted(
            list(self.state_dir.glob("state_*.*")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [
            {
                "filename": s.name,
                "size_mb": round(s.stat().st_size / (1024**2), 3),
                "modified": datetime.fromtimestamp(
                    s.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for s in snapshots
        ]

    def _rotate_snapshots(self) -> None:
        """Keep only the last N snapshots."""
        snapshots = sorted(
            list(self.state_dir.glob("state_*.*")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old in snapshots[self.retention:]:
            logger.debug(f"Removing old snapshot: {old}")
            old.unlink(missing_ok=True)

    def should_save(self, interval_sec: int = 300) -> bool:
        """Check if enough time has elapsed since last save."""
        return (time.time() - self.last_save_time) >= interval_sec
