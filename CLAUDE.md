# Claude Code Project Instructions

## Code Style
- Do not use em dash anywhere. Use regular hyphens (-) or double hyphens (--) instead.
- Python 3.11+, type hints on all public functions
- Pydantic for all data models (schemas.py is the immutable contract)

## Architecture
- Inference: GGUF via llama-cpp-python (NOT PyTorch/HuggingFace)
- Model: `mamba-370m-hf` Q8_0 (371M params, 97ms mean latency on CPU)
- State: safetensors persistence with 3-snapshot rotation
- EMA: dual fast/slow state with drift signal
- API: FastAPI on port 8100

## Key Constraints
- Compact event encoding (~16 tokens max) is critical for <150ms latency
- PyTorch/HuggingFace Mamba does NOT work on CPU (sequential Python fallback)
- torch.compile, int8 quantization both fail with Mamba architecture
- Model loading uses llama_cpp.Llama, NOT transformers.AutoModel

## Build Commands
```bash
source .venv/bin/activate
pytest tests/ -v               # Run all tests (44 tests)
python benchmark.py             # Priority Zero benchmark
python -m luna_streams          # Start server
```

## Docker
```bash
docker compose build
docker compose up -d
# Service joins luna-chat_luna-network (external)
```

## Source of Truth
Architecture spec: `/opt/luna-chat/clauding/MAMBA_STREAMS_ARCHITECTURE_v3_FINAL.md`

## Configuration
All env vars use `STREAMS_` prefix. See `luna_streams/config.py` for full list.

## Integration with Luna Chat
- Luna Chat emits events via POST /api/events (fire-and-forget)
- Luna Chat fetches context via GET /api/context (delta-tracked)
- Disabled by default: set `LUNA_STREAMS_ENABLED=true` in luna-chat docker-compose
