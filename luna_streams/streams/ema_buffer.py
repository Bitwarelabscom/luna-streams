"""Dual fast/slow state EMA buffer.

Each stream maintains two hidden state vectors:
- Fast state: current SSM hidden state, updates every step
- Slow state: exponential moving average (decay ~0.999), long-term baseline

Drift signal = L2 distance between fast and slow states.
High drift = user is in an unusual mode. Low drift = business as usual.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("luna_streams.ema")


@dataclass
class EMAState:
    """Serializable EMA buffer state."""
    fast: np.ndarray
    slow: np.ndarray
    step_count: int = 0


class EMABuffer:
    """Dual fast/slow state with EMA smoothing.

    The slow state prevents catastrophic forgetting: it anchors
    long-term patterns even as the fast state chases recent input.
    """

    def __init__(self, dim: int, decay: float = 0.999):
        self.dim = dim
        self.decay = decay
        self.fast = np.zeros(dim, dtype=np.float32)
        self.slow = np.zeros(dim, dtype=np.float32)
        self.step_count = 0

    def update(self, new_state: np.ndarray) -> None:
        """Update both fast and slow states with a new hidden state vector."""
        if new_state.shape != (self.dim,):
            # Flatten or reshape if needed
            new_state = new_state.flatten()[:self.dim]
            if len(new_state) < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[:len(new_state)] = new_state
                new_state = padded

        self.fast = new_state.astype(np.float32)
        self.slow = self.decay * self.slow + (1 - self.decay) * self.fast
        self.step_count += 1

    @property
    def drift_signal(self) -> float:
        """Divergence between fast and slow state. Range [0, inf), typically 0-1."""
        slow_norm = np.linalg.norm(self.slow)
        if slow_norm < 1e-8:
            return 0.0
        return float(np.linalg.norm(self.fast - self.slow) / slow_norm)

    @property
    def state_norm(self) -> float:
        """L2 norm of the fast state."""
        return float(np.linalg.norm(self.fast))

    def get_state(self) -> EMAState:
        """Export serializable state."""
        return EMAState(
            fast=self.fast.copy(),
            slow=self.slow.copy(),
            step_count=self.step_count,
        )

    def load_state(self, state: EMAState) -> None:
        """Restore from serialized state."""
        self.fast = state.fast.copy()
        self.slow = state.slow.copy()
        self.step_count = state.step_count

    def to_dict(self) -> dict:
        """Convert to dict for safetensors serialization."""
        return {
            "ema_fast": self.fast,
            "ema_slow": self.slow,
            "ema_step_count": np.array([self.step_count], dtype=np.int64),
        }

    @classmethod
    def from_dict(cls, data: dict, decay: float = 0.999) -> "EMABuffer":
        """Restore from safetensors dict."""
        fast = data["ema_fast"]
        buf = cls(dim=len(fast), decay=decay)
        buf.fast = fast.astype(np.float32)
        buf.slow = data["ema_slow"].astype(np.float32)
        buf.step_count = int(data["ema_step_count"][0])
        return buf
