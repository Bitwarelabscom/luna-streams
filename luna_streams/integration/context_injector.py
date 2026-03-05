"""Context injector - produces text summaries for chat model prompt.

Reads head outputs from active streams, generates concise natural language
summaries via Qwen 3.5:9b, and formats them for system prompt injection.

Uses DeltaTracker to skip regeneration when state hasn't shifted.
Falls back to template-based summaries if Qwen is unreachable.
"""

import logging
from typing import Optional

import httpx

from ..config import settings
from .delta_tracker import DeltaTracker

logger = logging.getLogger("luna_streams.context_injector")

QWEN_PROMPT = """You are a concise observer of user behavioral state from neural stream data.
Given these real-time signals from a continuous Mamba SSM tracking a user's interactions:

- Emotional valence: {emotional_valence:.3f} (range -1 to 1, negative=distressed, positive=engaged)
- Focus intensity: {focus_intensity:.3f} (0=unfocused, 1=deeply focused)
- Drift signal: {drift_signal:.3f} (0=normal pattern, high=unusual behavior)
- State norm: {state_norm:.3f} (overall activation level)
- Events processed: {events_processed}

Write a single ~40 word observation about the user's current cognitive/emotional state.
Be specific about what these numbers suggest. No hedging, no "it seems". Direct observation only.
Do not mention the numbers themselves - translate them into human-readable insight."""


class ContextInjector:
    """Generates context strings for Luna Chat system prompt injection."""

    def __init__(self, threshold: float = 0.01):
        self.delta = DeltaTracker(threshold=threshold)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate_context(self, stream_states: dict) -> str:
        """Generate context string from current stream states.

        Returns formatted context for system prompt injection,
        or empty string if no streams are active.
        """
        user_state = stream_states.get("user_model", {})
        if user_state.get("status") not in ("running", "idle"):
            return ""

        heads = user_state.get("head_outputs", {})
        events_processed = user_state.get("events_processed", 0)
        if events_processed == 0:
            return ""

        state_norm = user_state.get("state_norm", heads.get("state_norm", 0.0))
        drift_signal = user_state.get("drift_signal", heads.get("drift_signal", 0.0))
        emotional_valence = heads.get("emotional_valence", 0.0)
        focus_intensity = heads.get("focus_intensity", 0.0)

        # Check if state has changed enough to regenerate
        if not self.delta.has_changed("user_model", state_norm):
            cached = self.delta.get_cached_context("user_model")
            if cached:
                return cached

        # Generate summary
        summary = await self._generate_qwen_summary(
            emotional_valence=emotional_valence,
            focus_intensity=focus_intensity,
            drift_signal=drift_signal,
            state_norm=state_norm,
            events_processed=events_processed,
        )

        # Format output
        sections = ["[Continuous Cognition - Mamba Streams]"]
        sections.append(f"[User State] {summary}")

        if drift_signal > 0.3:
            sections.append(f"[Drift] {drift_signal:.2f} (high - unusual pattern detected)")
        elif drift_signal > 0.1:
            sections.append(f"[Drift] {drift_signal:.2f} (moderate behavioral shift)")

        context = "\n".join(sections)

        # Cache the result
        self.delta.update("user_model", state_norm, context)

        return context

    async def _generate_qwen_summary(
        self,
        emotional_valence: float,
        focus_intensity: float,
        drift_signal: float,
        state_norm: float,
        events_processed: int,
    ) -> str:
        """Generate a natural language summary via Qwen."""
        prompt = QWEN_PROMPT.format(
            emotional_valence=emotional_valence,
            focus_intensity=focus_intensity,
            drift_signal=drift_signal,
            state_norm=state_norm,
            events_processed=events_processed,
        )

        try:
            client = self._get_client()
            resp = await client.post(
                f"{settings.qwen_ollama_url}/api/chat",
                json={
                    "model": settings.qwen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 80,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data.get("message", {}).get("content", "").strip()
            if summary:
                # Clean up - take first sentence or first ~60 words
                words = summary.split()
                if len(words) > 60:
                    summary = " ".join(words[:60])
                logger.debug(f"Qwen summary: {summary[:80]}...")
                return summary
        except httpx.TimeoutException:
            logger.warning("Qwen timeout - falling back to template summary")
        except Exception as e:
            logger.warning(f"Qwen error: {e} - falling back to template summary")

        # Template fallback
        return self._template_summary(
            emotional_valence, focus_intensity, drift_signal, state_norm, events_processed
        )

    def _template_summary(
        self,
        emotional_valence: float,
        focus_intensity: float,
        drift_signal: float,
        state_norm: float,
        events_processed: int,
    ) -> str:
        """Generate a template-based summary when Qwen is unreachable."""
        # Emotional state
        if emotional_valence > 0.3:
            mood = "positively engaged"
        elif emotional_valence > 0.0:
            mood = "mildly engaged"
        elif emotional_valence > -0.3:
            mood = "neutral"
        else:
            mood = "showing signs of frustration"

        # Focus
        if focus_intensity > 0.7:
            focus = "deeply focused"
        elif focus_intensity > 0.3:
            focus = "moderately attentive"
        else:
            focus = "in exploratory mode"

        # Drift
        if drift_signal > 0.3:
            drift_note = " Behavioral pattern diverges significantly from baseline."
        elif drift_signal > 0.1:
            drift_note = " Some deviation from typical interaction patterns."
        else:
            drift_note = ""

        return (
            f"User is {mood} and {focus} "
            f"({events_processed} interactions tracked).{drift_note}"
        )
