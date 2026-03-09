# Luna Mamba Streams

Continuous cognition layer for Luna - parallel Mamba state-space models running 24/7 on GPU, processing memory events in real-time. Each stream maintains a persistent hidden state encoding compressed understanding. No context windows. No batch jobs. Always on.

## Architecture

```
Luna Chat (Hetzner)                Luna Streams (10.0.0.30, Tesla P4)
-------------------                ------------------------------------
chat.service.ts  --POST /api/events-->  [Event Queue]
memory.service.ts                            |
                                    [User Model Stream] --> EMA Buffer --> State (safetensors)
                                    [Knowledge Graph Stream] (Phase 8)
                                    [Conversation Dynamics Stream] (Phase 8)
                                             |
luna.persona.ts  <--GET /api/context--  [Context Injector] (~120 tokens)
                                             |
                                    [Qwen 3.5:9b] (RTX 3080, text summaries)
```

## Inference

- **Model:** `state-spaces/mamba-2.8b-hf` (2.8B params), LoRA fine-tuned on user data
- **Format:** GGUF Q8_0 (2.83GB) via llama-cpp-python with CUDA GPU offload
- **Hardware:** Tesla P4 (8GB VRAM, Pascal compute 6.1), dedicated GPU 1
- **Performance:** 48ms mean latency, ~20 events/sec
- **Previous:** Mamba 370M on CPU (97ms mean) -- 7.5x parameter increase, 2x faster
- **MLP Heads:** 3 trained output heads (emotional_valence, focus_topics, next_event)
- **Context Generation:** Qwen 3.5:9b on RTX 3080 (GPU 0) via Ollama

### GPU Layout (10.0.0.30)

| GPU | Device | VRAM | Service |
|-----|--------|------|---------|
| 0 | RTX 3080 | 10GB | Ollama (Qwen 3.5:9b + others) |
| 1 | Tesla P4 | 8GB | Luna Streams (Mamba 2.8B Q8_0) |

### Why Q8_0 over F16?

Tesla P4 is Pascal (compute 6.1) with no tensor cores -- FP16 runs at FP32 speed. Q8_0 moves less data across the memory bus (~3GB vs ~5.6GB), so it's faster. Quality loss is negligible for logit-folding.

### Why GGUF, not PyTorch?

PyTorch's HuggingFace Mamba implementation falls back to sequential Python loops on CPU (no CUDA kernels). `torch.compile` fails on Mamba's `MambaCache`. llama.cpp has native C++ Mamba support with proper CUDA offload.

### Key Implementation Detail

llama-cpp-python requires `logits_all=True` in the Llama constructor to populate `model.scores` after `eval()`. Without this, scores are all zeros. The logits for the last token are at `scores[len(tokens) - 1]`, not `scores[-1]` (which indexes into the full n_ctx-sized buffer).

## Setup

### Native (recommended for GPU deployment)

```bash
# On GPU server (10.0.0.30)
python -m venv /path/to/venv
source /path/to/venv/bin/activate

# Install with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61" pip install llama-cpp-python
pip install -e ".[dev]"

# Run with GPU
CUDA_VISIBLE_DEVICES=1 \
STREAMS_GGUF_MODEL=mamba-2.8b/mamba-2.8b-user-q8_0.gguf \
STREAMS_MLP_HEADS_PATH=mamba-2.8b/mlp_heads.safetensors \
STREAMS_GGUF_N_GPU_LAYERS=-1 \
STREAMS_HIDDEN_DIM=2560 \
python -m luna_streams
```

### systemd (production)

The service is managed via systemd on 10.0.0.30:

```bash
sudo systemctl status luna-streams
sudo systemctl restart luna-streams
journalctl -u luna-streams -f  # live logs
```

Service file: `/etc/systemd/system/luna-streams.service`

### Docker (alternative)

```bash
# Build with CUDA (requires matching host CUDA version)
docker compose build
docker compose up -d

# Health check
curl http://localhost:8100/health
```

Models are bind-mounted from `./models/` (read-only). The Dockerfile targets CUDA 12.4 with Pascal compute 6.1.

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

### Head Outputs

```json
{
  "emotional_valence": 0.42,
  "focus_topics": [4, 2, 47, 6, 27],
  "focus_intensity": 0.55,
  "next_event_type": 3,
  "drift_signal": 0.15,
  "state_norm": 709.16
}
```

## Compact Event Encoding

Events are tokenized into ~16 tokens for the Mamba model:

```
USR|pos|0.6|attn:0.8|Luna,Henke|architecture|Working on Mamba integration
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
    base_stream.py       # GGUF model loading + forward pass (GPU/CPU)
    user_model.py        # Stream 1 (compact event encoding)
    ema_buffer.py        # Dual fast/slow EMA with drift signal
    state_manager.py     # safetensors persistence
  integration/
    memory_bridge.py     # Luna Chat format -> StructuredEvent
    context_injector.py  # State -> text summary via Qwen
    delta_tracker.py     # Change detection (avoids redundant context updates)
  heads/
    mlp_heads.py         # Numpy-only MLP inference (trained head weights)
    summary_decoder.py   # Text summary generation (Phase 7)
scripts/
  convert_mamba_2.8b.sh  # HF -> GGUF F16 -> Q8_0 conversion
training/
  TRAINING.md            # Full training guide
  configs/
    stream1_user.yaml    # 370M LoRA config
    stream1_user_2.8b.yaml  # 2.8B LoRA config (fp16, grad accum)
  data_prep/
    export_luna_data.py          # Export from PostgreSQL databases
    build_event_sequences.py     # Convert to compact token sequences
    generate_labels.py           # Qwen labeling oracle (370M)
    generate_labels_claude.py    # Multi-signal label generation (2.8B)
  train.py               # LoRA fine-tune (RTX 3080)
```

## Configuration

All settings via environment variables with `STREAMS_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMS_PORT` | 8100 | HTTP port |
| `STREAMS_GGUF_MODEL` | `mamba-370m/mamba-370m-user-q8_0.gguf` | GGUF model path (relative to model_dir) |
| `STREAMS_MLP_HEADS_PATH` | `mamba-370m/mlp_heads.safetensors` | Trained MLP head weights |
| `STREAMS_GGUF_N_CTX` | 256 | Context window size |
| `STREAMS_GGUF_N_THREADS` | 8 | CPU threads for inference |
| `STREAMS_GGUF_N_GPU_LAYERS` | 0 | GPU layers (-1=all on GPU, 0=CPU-only) |
| `STREAMS_HIDDEN_DIM` | 1024 | Model hidden dimension (1024 for 370M, 2560 for 2.8B) |
| `STREAMS_EMA_DECAY` | 0.999 | Slow state EMA decay |
| `STREAMS_DELTA_THRESHOLD` | 0.01 | Minimum state change for context update |
| `STREAMS_AUTO_SAVE_INTERVAL_SEC` | 300 | State persistence interval |
| `STREAMS_QWEN_OLLAMA_URL` | `http://10.0.0.30:11434` | Qwen for text summaries |
| `STREAMS_QWEN_MODEL` | `qwen3.5:9b` | Qwen model name |

## Phases

- [x] Phase 0: Priority Zero benchmark -- PASSED (97ms mean on CPU)
- [x] Phase 1: FastAPI app + event schema + memory bridge
- [x] Phase 2: Training data prep (DB export, event sequences, label generation)
- [x] Phase 3: LoRA fine-tune on RTX 3080 (370M: 7 epochs, val_loss 2.49)
- [x] Phase 4: Stream 1 deployment with GGUF inference + trained MLP heads
- [x] Phase 5: Luna Chat integration (client, emission, context injection)
- [x] Context Injector: Qwen-generated summaries from head outputs with delta tracking
- [x] GPU Upgrade: Mamba 2.8B on Tesla P4 (48ms, 2x faster, 7.5x params)
- [x] 2.8B Training: LoRA fine-tune (9 epochs, val_loss 1.985, next_acc 95.7%)
- [x] Production deployment: systemd service, Luna Chat connected
- [ ] Phase 6: Validation gate (does Stream 1 provide value?)
- [ ] Phase 7: Autoregressive summary decoder
- [ ] Phase 8: Streams 2 (knowledge graph) + 3 (conversation dynamics)
- [ ] Phase 9: Cross-stream communication
- [ ] Phase 10: NeuralSleep integration

## Source of Truth

Architecture spec: `/opt/luna-chat/clauding/MAMBA_STREAMS_ARCHITECTURE_v3_FINAL.md`
