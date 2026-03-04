"""Autoregressive text summary decoder head.

MLPs cannot generate coherent natural language. This decoder
takes the projected hidden state as initial context and
autoregressively generates ~40 tokens.

Phase 5: Uses Qwen via Ollama (Option B)
Phase 7: Trained small Mamba/Transformer decoder (Option A, ~30M params)

The bootstrap: Qwen generates high-quality summaries that also serve
as training labels for the self-contained decoder.
"""

# TODO: Implement Option B (Qwen bridge) in Phase 5
# TODO: Implement Option A (trained decoder) in Phase 7
