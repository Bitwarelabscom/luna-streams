"""MLP output heads for structured outputs.

Standard small MLP networks (10-20M params each) for:
- emotional_valence: Float [-1.0, 1.0]
- focus_topics: List of (topic, weight)
- clustering_signal, centrality_shift, etc.

These work fine for structured outputs - floats, enums, scored lists.
"""

# TODO: Implement in Phase 3/4 after training
# Each head: hidden_state -> Linear(hidden_dim, 256) -> ReLU -> Linear(256, output_dim)
