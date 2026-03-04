# Luna Mamba Streams

Continuous cognition layer for Luna - three parallel Mamba state-space models running 24/7 on CPU, processing memory events in real-time. Each stream maintains a persistent hidden state encoding compressed understanding. No context windows. No batch jobs. Always on.

## Architecture

```
Luna Chat                          Luna Streams (this service)
---------                          --------------------------
chat.service.ts  --POST /api/events-->  [Event Queue]
memory.service.ts                            |
                                    [User Model Stream] --> EMA Buffer --> State (safetensors)
                                    [Knowledge Graph Stream] (Phase 8)
                                    [Conversation Dynamics Stream] (Phase 8)
                                             |
luna.persona.ts  <--GET /api/context--  [Context Injector] (~120 tokens)
```

## Inference

- **Model:** `state-spaces/mamba-370m-hf` (371M params)
- **Format:** GGUF Q8_0 via llama-cpp-python (optimized C++ CPU kernels)
- **Hardware:** AMD Ryzen 5 3600 (6-core/12-thread, AVX2), 62GB RAM
- **Performance:** 97ms mean latency, 10.3 events/sec, 509MB RAM per stream
- **Target:** <150ms mean latency per step -- **PASSED**

### Why GGUF, not PyTorch?

The benchmark (Priority Zero) tested both paths:

| Path | Model | Mean Latency | Verdict |
|------|-------|-------------|---------|
| GGUF llama.cpp | 370M Q8_0 | **97ms** | PASS |
| PyTorch eager | 370M F32 | 595ms | FAIL |
| PyTorch eager | 790M F32 | ~2,670ms | FAIL |

PyTorch's HuggingFace Mamba implementation falls back to sequential Python loops on CPU (no CUDA kernels). `torch.compile` fails on Mamba's `MambaCache` (uses `mark_static_address`). `torch.ao.quantization.quantize_dynamic` (int8) breaks Mamba's custom conv1d/SSM layers. llama.cpp has native C++ Mamba support with proper AVX2 vectorization.

## Setup

```bash
# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Download GGUF model
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('devingulliver/mamba-gguf', 'mamba-370m/mamba-370m-q8_0.gguf', local_dir='models')
"

# Run tests
pytest tests/ -v

# Run benchmark
OMP_NUM_THREADS=8 python benchmark.py --skip-path-a --events 100

# Start server
python -m luna_streams
```

## Docker

```bash
# Build and run (joins luna-chat network)
docker compose build
docker compose up -d

# Health check
curl http://localhost:8100/health

# Copy GGUF model into the volume
docker cp models/mamba-370m/mamba-370m-q8_0.gguf luna-streams:/app/models/mamba-370m/
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST /api/events` | Ingest structured events (async queue) |
| `GET /api/context?user_id={id}` | Combined context string for prompt injection (delta-tracked) |
| `GET /api/streams` | All streams status summary |
| `GET /api/streams/{name}/state` | Detailed stream state + head outputs |
| `GET /health` | Health check |
| `GET /api/benchmark` | Last benchmark results |

### Event Schema

```json
{
  "events": [{
    "timestamp": "2026-03-04T17:00:00Z",
    "event_type": "memory_entry",
    "source": "conversation",
    "content": {
      "entities": ["Luna", "Henke"],
      "topic_tags": ["architecture"],
      "sentiment": 0.6,
      "importance": 0.8,
      "summary": "Working on Mamba integration"
    }
  }]
}
```

Event types: `memory_entry`, `entity_update`, `edge_update`, `conversation_meta`

## Compact Event Encoding

Events are tokenized into ~16 tokens for the Mamba model (critical for <150ms latency):

```
mem_e conv Luna,Henke architecture 0.6 0.8 Working on Mamba integration
```

vs full format (~38 tokens, too slow):
```
[memory_entry] entities: Luna, Henke | topics: architecture | sentiment: 0.60 | importance: 0.80 | summary: Working on Mamba integration
```

## EMA Dual-State Buffer

Each stream maintains two exponential moving averages of the hidden state:

- **Fast state:** Current SSM output (replaced each step)
- **Slow state:** EMA with decay 0.999 (tracks long-term trends)
- **Drift signal:** `L2(fast - slow) / norm(slow)` -- spikes on behavioral changes

Both states are persisted via safetensors with 3-snapshot rotation.

## Project Structure

```
luna_streams/
  config.py              # Pydantic Settings (STREAMS_ env prefix)
  app.py                 # FastAPI factory
  main.py                # Entry: API + background stream processor
  api/
    schemas.py           # Immutable event contract (StructuredEvent)
    routes.py            # All HTTP endpoints
  streams/
    base_stream.py       # GGUF model loading + forward pass
    user_model.py        # Stream 1 (compact event encoding)
    ema_buffer.py        # Dual fast/slow EMA with drift signal
    state_manager.py     # safetensors persistence
  integration/
    memory_bridge.py     # Luna Chat format -> StructuredEvent
    context_injector.py  # State -> text summary for prompt injection
    delta_tracker.py     # Change detection (avoids redundant context updates)
  heads/
    mlp_heads.py         # Float/list output heads (Phase 3)
    summary_decoder.py   # Text summary generation (Phase 5/7)
training/
  configs/               # LoRA fine-tuning configs per stream
  data_prep/             # SQL export + event sequence building
  train.py               # LoRA fine-tune (RTX 3080)
```

## Configuration

All settings via environment variables with `STREAMS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMS_PORT` | 8100 | HTTP port |
| `STREAMS_GGUF_MODEL` | `mamba-370m/mamba-370m-q8_0.gguf` | GGUF model path (relative to model_dir) |
| `STREAMS_GGUF_N_CTX` | 256 | Context window size |
| `STREAMS_GGUF_N_THREADS` | 8 | CPU threads for inference |
| `STREAMS_EMA_DECAY` | 0.999 | Slow state EMA decay |
| `STREAMS_DELTA_THRESHOLD` | 0.01 | Minimum state change for context update |
| `STREAMS_AUTO_SAVE_INTERVAL_SEC` | 300 | State persistence interval |
| `STREAMS_QWEN_OLLAMA_URL` | `http://10.0.0.30:11434` | Qwen for text summaries |
| `STREAMS_QWEN_MODEL` | `qwen3.5:9b` | Qwen model name |

## Phases

- [x] Phase 0: Priority Zero benchmark -- PASSED (97ms mean)
- [x] Phase 1: FastAPI app + event schema + memory bridge
- [x] Phase 4: Stream 1 deployment with GGUF inference
- [x] Phase 5: Luna Chat integration (client, emission, context injection)
- [ ] Phase 2: Training data preparation (export + Qwen labels)
- [ ] Phase 3: LoRA fine-tune Stream 1 on RTX 3080
- [ ] Phase 6: Validation gate (does Stream 1 provide value?)
- [ ] Phase 7: Autoregressive summary decoder
- [ ] Phase 8: Streams 2 (knowledge graph) + 3 (conversation dynamics)
- [ ] Phase 9: Cross-stream communication
- [ ] Phase 10: NeuralSleep integration

## Source of Truth

Architecture spec: `/opt/luna-chat/clauding/MAMBA_STREAMS_ARCHITECTURE_v3_FINAL.md`
