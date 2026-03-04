"""Context injector - produces text summaries for chat model prompt.

Phase 5: Initially uses Qwen (Option B) via Ollama for summary generation.
Phase 7: Replaced by trained autoregressive decoder (Option A).
"""

# TODO: Implement in Phase 5
# - Read output heads from active streams
# - For text summary: call Qwen via Ollama HTTP
# - Format: "[User State] {summary}\n[Drift] {drift_signal}"
# - Cache generated summary via DeltaTracker
