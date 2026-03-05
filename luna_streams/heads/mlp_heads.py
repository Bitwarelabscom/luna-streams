"""MLP output heads - numpy-only inference.

Loads trained head weights (safetensors) and runs forward passes
without PyTorch. Designed for deployment on the Hetzner server
where we only have numpy + llama-cpp-python.

Each head: hidden_state -> Linear(1024, 256) -> ReLU -> Linear(256, out_dim) -> activation
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("luna_streams.heads")


class MLPHead:
    """Single MLP head with numpy forward pass."""

    def __init__(
        self,
        name: str,
        layer_dims: list[int],
        activation: str = "tanh",
    ):
        self.name = name
        self.layer_dims = layer_dims
        self.activation = activation
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.loaded = False

    def load_weights(self, weight_dict: dict[str, np.ndarray]) -> None:
        """Load weights from a flat dict (safetensors format).

        Keys use nn.Sequential indices: {name}.layer{seq_idx}.weight
        ReLU layers are skipped, so Linear layers are at indices 0, 2, 4...
        """
        self.weights = []
        self.biases = []

        # Find all weight keys for this head, sorted by layer index
        prefix = f"{self.name}.layer"
        layer_indices = sorted({
            int(k.split(".")[1].replace("layer", ""))
            for k in weight_dict
            if k.startswith(prefix) and k.endswith(".weight")
        })

        if not layer_indices:
            logger.warning(f"No weights found for head '{self.name}'")
            return

        for idx in layer_indices:
            w_key = f"{self.name}.layer{idx}.weight"
            b_key = f"{self.name}.layer{idx}.bias"

            w = weight_dict[w_key].astype(np.float32)
            b = weight_dict.get(b_key, np.zeros(w.shape[0], dtype=np.float32))
            b = b.astype(np.float32)

            self.weights.append(w)
            self.biases.append(b)

        self.loaded = True
        logger.info(f"Loaded head '{self.name}': {len(self.weights)} linear layers")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        if not self.loaded:
            return np.zeros(self.layer_dims[-1], dtype=np.float32)

        h = x.astype(np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Linear: h = h @ w.T + b
            h = h @ w.T + b

            # Activation (skip on last layer - applied separately)
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)  # ReLU

        # Output activation
        if self.activation == "tanh":
            h = np.tanh(h)
        elif self.activation == "softmax":
            h = h - np.max(h)  # stability
            exp_h = np.exp(h)
            h = exp_h / np.sum(exp_h)
        elif self.activation == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))

        return h


class HeadManager:
    """Manages all MLP output heads for a stream."""

    def __init__(self):
        self.heads: dict[str, MLPHead] = {}
        self._loaded = False

    def register_head(self, name: str, layer_dims: list[int], activation: str = "tanh") -> None:
        """Register a head configuration."""
        self.heads[name] = MLPHead(name=name, layer_dims=layer_dims, activation=activation)

    def load_all(self, weights_path: str) -> bool:
        """Load all head weights from a safetensors file."""
        path = Path(weights_path)
        if not path.exists():
            logger.warning(f"Head weights not found: {weights_path}")
            return False

        try:
            from safetensors.numpy import load_file

            weight_dict = load_file(str(path))

            for head in self.heads.values():
                head.load_weights(weight_dict)

            loaded_count = sum(1 for h in self.heads.values() if h.loaded)
            logger.info(f"Loaded {loaded_count}/{len(self.heads)} heads from {weights_path}")
            self._loaded = loaded_count > 0
            return self._loaded

        except Exception as e:
            logger.error(f"Failed to load head weights: {e}")
            return False

    def run_all(self, hidden_state: np.ndarray) -> dict[str, np.ndarray]:
        """Run all loaded heads on a hidden state vector."""
        results = {}
        for name, head in self.heads.items():
            if head.loaded:
                results[name] = head.forward(hidden_state)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_user_model_heads(weights_path: Optional[str] = None) -> HeadManager:
    """Create and optionally load the user model head configuration."""
    manager = HeadManager()
    manager.register_head("emotional_valence", [1024, 256, 1], activation="tanh")
    manager.register_head("focus_topics", [1024, 256, 50], activation="softmax")
    manager.register_head("next_event", [1024, 128, 4], activation="softmax")

    if weights_path:
        manager.load_all(weights_path)

    return manager
