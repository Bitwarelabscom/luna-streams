# Claude Code Project Instructions

## Code Style
- Do not use em dash anywhere. Use regular hyphens (-) or double hyphens (--) instead.
- Python 3.11+, type hints on all public functions
- Pydantic for all data models (schemas.py is the immutable contract)

## Architecture
- Inference: GGUF via llama-cpp-python with CUDA GPU offload (NOT PyTorch/HuggingFace)
- Model: `mamba-2.8b-hf` Q8_0 (2.8B params, 48ms mean latency on Tesla P4)
- Previous: `mamba-370m-hf` Q8_0 (371M params, 97ms on CPU) -- kept for rollback
- State: safetensors persistence with 3-snapshot rotation
- EMA: dual fast/slow state with drift signal
- API: FastAPI on port 8100
- Context: Qwen 3.5:9b via Ollama generates text summaries from head outputs

## Hardware Layout (10.0.0.30)
- GPU 0 (RTX 3080, 10GB): Ollama on port 11434
- GPU 1 (Tesla P4, 8GB): Luna Streams (Mamba 2.8B Q8_0, ~3GB VRAM)
- Service managed via systemd: `sudo systemctl restart luna-streams`
- Service file: `/etc/systemd/system/luna-streams.service`
- Model dir: `/media/gpu/claude/luna-streams-models/`
- State dir: `/media/gpu/claude/luna-streams/state/`
- Venv: `/media/gpu/claude/llama-convert/venv/`

## Key Constraints
- Compact event encoding (~16 tokens max) is critical for fast inference
- PyTorch/HuggingFace Mamba does NOT work on CPU (sequential Python fallback)
- torch.compile, int8 quantization both fail with Mamba architecture
- Model loading uses llama_cpp.Llama, NOT transformers.AutoModel
- Llama constructor MUST include `logits_all=True` or scores are all zeros
- Logits index: `scores[len(tokens) - 1]`, NOT `scores[-1]`
- Tesla P4 is Pascal (compute 6.1) -- no tensor cores, Q8_0 is faster than F16

## Build Commands
```bash
source .venv/bin/activate
pytest tests/ -v               # Run all tests (44 tests)
python benchmark.py             # Priority Zero benchmark
python -m luna_streams          # Start server
```

## Deployment (production on 10.0.0.30)
```bash
# Sync code
rsync -av /opt/luna-streams/luna_streams/ 10.0.0.30:/media/gpu/claude/luna-streams/luna_streams/

# Restart service
ssh 10.0.0.30 'sudo systemctl restart luna-streams'

# Check logs
ssh 10.0.0.30 'journalctl -u luna-streams -f'

# Verify
curl http://10.0.0.30:8100/health
curl http://10.0.0.30:8100/api/streams/user_model/state
```

## Docker (alternative, not used in production)
```bash
docker compose build
docker compose up -d
# Dockerfile targets CUDA 12.4 / Pascal compute 6.1
```

## Source of Truth
Architecture spec: `/opt/luna-chat/clauding/MAMBA_STREAMS_ARCHITECTURE_v3_FINAL.md`

## Configuration
All env vars use `STREAMS_` prefix. See `luna_streams/config.py` for full list.
Key GPU settings: `STREAMS_GGUF_N_GPU_LAYERS`, `STREAMS_HIDDEN_DIM`, `CUDA_VISIBLE_DEVICES`.

## Integration with Luna Chat
- Luna Chat emits events via POST /api/events (fire-and-forget) from `streamMessage()` in chat.service.ts
- Luna Chat fetches context via GET /api/context (delta-tracked)
- Enabled via `LUNA_STREAMS_ENABLED=true` and `LUNA_STREAMS_URL=http://10.0.0.30:8100` in luna-chat docker-compose
- Client code: `/opt/luna-chat/src/integration/luna-streams.client.ts`
- Note: `processMessage()` also has emit calls but the frontend uses `streamMessage()`
