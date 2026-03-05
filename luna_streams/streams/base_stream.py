"""Abstract base class for Mamba streams.

Handles model loading via llama.cpp GGUF, forward passes, state management,
and event queue consumption. Concrete streams (user_model, knowledge_graph,
conversation_dynamics) extend this with stream-specific logic.

Inference path: mamba-370m-hf Q8_0 GGUF via llama-cpp-python
Benchmark result: 97ms mean latency, 10.3 events/sec, 509MB RAM
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ..api.schemas import StructuredEvent
from ..config import settings
from ..heads.mlp_heads import HeadManager
from .ema_buffer import EMABuffer
from .state_manager import StateManager

logger = logging.getLogger("luna_streams.stream")


class BaseStream(ABC):
    """Abstract base for a Mamba stream."""

    def __init__(self, name: str, hidden_dim: int = 1024):
        self.name = name
        self.hidden_dim = hidden_dim
        self.ema = EMABuffer(dim=hidden_dim, decay=settings.ema_decay)
        self.state_manager = StateManager(
            stream_name=name,
            state_dir=settings.state_dir,
            retention=settings.snapshot_retention,
        )
        self.events_processed = 0
        self.model = None
        self.head_outputs: dict = {}
        self.head_manager: Optional[HeadManager] = None

    @abstractmethod
    def accepts_event(self, event: StructuredEvent) -> bool:
        """Return True if this stream should process this event type."""
        ...

    @abstractmethod
    def event_to_tokens(self, event: StructuredEvent) -> str:
        """Convert a structured event to compact text for tokenization."""
        ...

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the Mamba GGUF model via llama.cpp."""
        if model_path is None:
            model_path = str(Path(settings.model_dir) / settings.gguf_model)

        if not Path(model_path).exists():
            logger.warning(f"No GGUF model at {model_path} - running in stub mode")
            return

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=model_path,
                n_ctx=settings.gguf_n_ctx,
                n_threads=settings.gguf_n_threads,
                n_threads_batch=settings.gguf_n_threads,
                verbose=False,
            )
            logger.info(
                f"Loaded GGUF model for {self.name}: {model_path} "
                f"(n_ctx={settings.gguf_n_ctx}, threads={settings.gguf_n_threads})"
            )
        except Exception as e:
            logger.error(f"Failed to load GGUF model for {self.name}: {e}")
            self.model = None

    def process_event(self, event: StructuredEvent) -> dict:
        """Process a single event through the stream.

        Returns dict of head outputs.
        """
        if not self.accepts_event(event):
            return {}

        text = self.event_to_tokens(event)

        if self.model is not None:
            hidden_state = self._forward(text)
            self.ema.update(hidden_state)
            self.head_outputs = self._run_heads(hidden_state)
        else:
            # Stub mode: update EMA with a deterministic pseudo-state
            pseudo_state = self._pseudo_hidden_state(event)
            self.ema.update(pseudo_state)
            self.head_outputs = {
                "drift_signal": self.ema.drift_signal,
                "state_norm": self.ema.state_norm,
            }

        self.events_processed += 1

        if self.state_manager.should_save(settings.auto_save_interval_sec):
            self.save_state()

        return self.head_outputs

    def _forward(self, text: str) -> np.ndarray:
        """Run forward pass through the GGUF model via llama.cpp.

        Tokenizes the compact event text, evaluates through all layers,
        then extracts a hidden state vector from the logits distribution.
        The logit vector serves as our "hidden state" - it encodes the
        model's compressed understanding of the input sequence.
        """
        tokens = self.model.tokenize(text.encode())

        # Truncate to n_ctx if needed
        max_tokens = settings.gguf_n_ctx - 1
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Run forward pass - processes all tokens through the SSM
        self.model.eval(tokens)

        # Extract logits as hidden state representation
        # llama.cpp exposes scores after eval
        # We use the output embedding (logits) as a proxy for the hidden state
        # since llama.cpp doesn't expose intermediate SSM states directly
        scores = self.model.scores
        if scores is not None and len(scores) > 0:
            # Last token's logits - take first hidden_dim components
            logits = np.array(scores[-1], dtype=np.float32)
            # Project to hidden_dim via hashing/folding if vocab > hidden_dim
            if len(logits) > self.hidden_dim:
                # Fold the logit vector down to hidden_dim
                state = np.zeros(self.hidden_dim, dtype=np.float32)
                for i in range(0, len(logits), self.hidden_dim):
                    chunk = logits[i : i + self.hidden_dim]
                    state[: len(chunk)] += chunk
                state /= (len(logits) / self.hidden_dim)
            else:
                state = np.zeros(self.hidden_dim, dtype=np.float32)
                state[: len(logits)] = logits
        else:
            state = np.zeros(self.hidden_dim, dtype=np.float32)

        # Reset the model's internal state for next event
        self.model.reset()

        return state

    def _run_heads(self, hidden_state: np.ndarray) -> dict:
        """Run output heads on the hidden state.

        Uses trained MLP heads if loaded, otherwise falls back to heuristics.
        """
        result = {
            "drift_signal": self.ema.drift_signal,
            "state_norm": self.ema.state_norm,
        }

        if self.head_manager and self.head_manager.is_loaded:
            head_out = self.head_manager.run_all(hidden_state)
            if "emotional_valence" in head_out:
                result["emotional_valence"] = float(head_out["emotional_valence"][0])
            if "focus_topics" in head_out:
                topics = head_out["focus_topics"]
                top_indices = np.argsort(topics)[-5:][::-1]
                result["focus_topics"] = [int(i) for i in top_indices]
                result["focus_intensity"] = float(np.max(topics))
            if "next_event" in head_out:
                result["next_event_type"] = int(np.argmax(head_out["next_event"]))
        else:
            # Heuristic fallback
            result["emotional_valence"] = float(np.tanh(hidden_state[0]))
            result["focus_intensity"] = float(np.clip(np.linalg.norm(hidden_state[:64]), 0, 1))

        return result

    def _pseudo_hidden_state(self, event: StructuredEvent) -> np.ndarray:
        """Generate a deterministic pseudo hidden state for stub mode testing."""
        rng = np.random.RandomState(hash(str(event.timestamp)) % (2**31))
        state = rng.randn(self.hidden_dim).astype(np.float32) * 0.1

        state[0] = event.content.sentiment
        state[1] = event.content.importance
        if event.content.entities:
            state[2] = len(event.content.entities) / 10.0
        if event.content.topic_tags:
            state[3] = len(event.content.topic_tags) / 10.0

        return state

    def save_state(self) -> None:
        """Persist current state (both fast and slow EMA)."""
        state_dict = self.ema.to_dict()
        state_dict["events_processed"] = np.array([self.events_processed], dtype=np.int64)
        self.state_manager.save(state_dict)

    def restore_state(self) -> bool:
        """Restore state from the latest snapshot. Returns True if successful."""
        data = self.state_manager.load_latest()
        if data is None:
            return False

        try:
            self.ema = EMABuffer.from_dict(data, decay=settings.ema_decay)
            if "events_processed" in data:
                self.events_processed = int(data["events_processed"][0])
            logger.info(
                f"Restored {self.name}: step={self.ema.step_count}, "
                f"events={self.events_processed}, drift={self.ema.drift_signal:.4f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to restore state for {self.name}: {e}")
            return False

    def get_status(self) -> dict:
        """Return current stream status for API."""
        return {
            "status": "running" if self.model is not None else "stub",
            "events_processed": self.events_processed,
            "state_norm": self.ema.state_norm,
            "drift_signal": self.ema.drift_signal,
            "head_outputs": self.head_outputs,
            "step_count": self.ema.step_count,
        }
