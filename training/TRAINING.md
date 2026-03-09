# Luna Streams - Training Guide

LoRA fine-tune Mamba for user modeling. Produces a GGUF model + MLP head weights personalized to your Luna interaction history.

Supports both Mamba 370M (CPU deployment) and Mamba 2.8B (GPU deployment on Tesla P4).

## Requirements

- NVIDIA GPU with 10GB+ VRAM (tested on RTX 3080)
- CUDA 12.x toolkit installed
- Python 3.11+
- PostgreSQL databases running (luna-chat + memorycore)

## Training Time

| Model | GPU | CUDA Kernels | Time |
|-------|-----|-------------|------|
| 370M (fp32) | RTX 3080 | Yes | ~2 min (7 epochs) |
| 2.8B (fp16) | RTX 3080 | Yes | ~5 min (9 epochs) |
| Any | Any | No (CPU fallback) | ~40 min/epoch - unusable |

## Step-by-Step

### 1. Set up environment

```bash
cd /opt/luna-streams/training
python -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA - match your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Training dependencies
pip install transformers peft safetensors pyyaml

# GGUF conversion
pip install gguf
```

### 2. Install CUDA kernels (critical for speed)

Without these, Mamba falls back to a sequential CPU path that is 20x slower.

```bash
# Install wheel first (needed for --no-build-isolation)
pip install wheel

# causal-conv1d - must build from source against your torch version
git clone --depth 1 --branch v1.4.0 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d
cd /tmp/causal-conv1d
TORCH_CUDA_ARCH_LIST='8.6' pip install -e . --no-build-isolation

# mamba-ssm - must build from source against your torch version
git clone --depth 1 --branch v2.2.4 https://github.com/state-spaces/mamba.git /tmp/mamba-ssm
cd /tmp/mamba-ssm
TORCH_CUDA_ARCH_LIST='8.6' pip install -e . --no-build-isolation
```

**Important notes:**
- Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (8.6 for RTX 3080/3090, 8.9 for RTX 4090, 7.5 for RTX 2080)
- Use `--no-build-isolation` so it builds against your installed torch
- causal-conv1d v1.4.0 pairs with mamba-ssm v2.2.4 (v1.6.0 has an incompatible API)
- If mamba-ssm's `__init__.py` fails with `GreedySearchDecoderOnlyOutput` import error, patch it to wrap the model imports in try/except -- only the CUDA ops are needed

**Verify kernels work:**
```bash
python -c "
import torch
import selective_scan_cuda
import causal_conv1d_cuda
print('CUDA kernels OK')
"
```

### 3. Export your data

Edit `data_prep/export_luna_data.py` and update the container names and database names to match your setup, then run:

```bash
python -m training.data_prep.export_luna_data
```

This exports 7 tables from your luna-chat and memorycore PostgreSQL databases into `data_prep/raw/`. Verify the row counts look reasonable for your instance.

### 4. Build event sequences

```bash
python -m training.data_prep.build_event_sequences
```

Converts raw exports into compact-encoded event sequences grouped by session. Output: `data_prep/sequences.jsonl`.

### 5. Generate labels

Two options:

**Option A: Multi-signal heuristic labels (recommended for 2.8B)**

Uses keyword matching, sentiment analysis, and recency-weighted predictions. No LLM required.

```bash
python -m training.data_prep.generate_labels_claude
```

Produces 347 labeled sequences (193 sessions + 154 global chunks with 50% overlap).

**Option B: Qwen labeling oracle (original 370M approach)**

Requires Ollama running with Qwen 3.5:9b.

```bash
python -m training.data_prep.generate_labels
```

Output for both: `data_prep/labeled_sequences.jsonl`.

### 6. Stop GPU services

Free up VRAM before training:

```bash
systemctl stop ollama
# Stop any other GPU-using processes
nvidia-smi  # verify VRAM is free
```

### 7. Train

**For Mamba 2.8B (recommended):**
```bash
python train.py \
  --config configs/stream1_user_2.8b.yaml \
  --data data_prep/labeled_sequences.jsonl \
  --output-dir output \
  --epochs 10
```

Training details (2.8B):
- LoRA rank 16, alpha 32 on in_proj, x_proj, dt_proj
- fp16 with gradient accumulation 16 (effective batch 16)
- Batch size 1 (fits in 10GB VRAM with gradient checkpointing)
- Combined loss: causal LM + emotional valence MSE + focus topics BCE + 0.1 * next event CE
- Best result: val_loss 1.985, next_event accuracy 95.7%, valence MAE 0.171

**For Mamba 370M:**
```bash
python train.py \
  --config configs/stream1_user.yaml \
  --data data_prep/labeled_sequences.jsonl \
  --output-dir output \
  --epochs 10
```

Training details (370M):
- fp32 (the sequential fallback path produces NaN in fp16)
- Batch size 2, gradient accumulation 8

### 8. Convert to GGUF

**For 2.8B (two-step: F16 then Q8_0):**
```bash
# Use the conversion script
bash scripts/convert_mamba_2.8b.sh

# Or manually:
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  output/merged_model \
  --outfile output/mamba-2.8b-user-f16.gguf \
  --outtype f16

/path/to/llama.cpp/build/bin/llama-quantize \
  output/mamba-2.8b-user-f16.gguf \
  output/mamba-2.8b-user-q8_0.gguf \
  Q8_0
```

**For 370M (direct Q8_0):**
```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  output/merged_model \
  --outtype q8_0 \
  --outfile output/mamba-370m-user-q8_0.gguf
```

### 9. Deploy

Copy the GGUF and head weights to your model directory:

```bash
# For 2.8B GPU deployment
cp output/mamba-2.8b-user-q8_0.gguf /path/to/models/mamba-2.8b/
cp output/mlp_heads.safetensors /path/to/models/mamba-2.8b/
```

Update environment or systemd service:
```
STREAMS_GGUF_MODEL=mamba-2.8b/mamba-2.8b-user-q8_0.gguf
STREAMS_MLP_HEADS_PATH=mamba-2.8b/mlp_heads.safetensors
STREAMS_HIDDEN_DIM=2560
STREAMS_GGUF_N_GPU_LAYERS=-1
```

Restart and verify:
```bash
sudo systemctl restart luna-streams
curl http://localhost:8100/api/streams/user_model/state
```

You should see `head_outputs` with non-zero `state_norm`, `emotional_valence`, `focus_topics`, and `next_event_type` values.

### 10. Restart GPU services

```bash
systemctl start ollama
# Restart any other GPU services you stopped
```

## Config Reference

Two config files available:

| Config | Model | VRAM | dtype |
|--------|-------|------|-------|
| `configs/stream1_user.yaml` | mamba-370m-hf | ~4GB | fp32 |
| `configs/stream1_user_2.8b.yaml` | mamba-2.8b-hf | ~10GB | fp16 |

Key parameters:

| Parameter | 370M | 2.8B | Description |
|-----------|------|------|-------------|
| `model.base` | mamba-370m-hf | mamba-2.8b-hf | HuggingFace model ID |
| `model.hidden_dim` | 1024 | 2560 | Hidden state dimension |
| `lora.rank` | 16 | 16 | LoRA rank |
| `lora.alpha` | 32 | 32 | LoRA alpha (2x rank) |
| `training.batch_size` | 2 | 1 | Batch size (limited by VRAM) |
| `training.gradient_accumulation` | 8 | 16 | Effective batch = batch_size * grad_accum |
| `training.learning_rate` | 2e-4 | 1e-4 | AdamW learning rate |
| `training.fp16` | false | true | Mixed precision training |
| `training.max_seq_length` | 128 | 128 | Token sequence length |

## State Migration

When upgrading from 370M to 2.8B:
- Archive old state: `mv state/ state_370m_backup/`
- Fresh start required -- 1024-dim EMA buffers are incompatible with 2560-dim
- Old 370M models stay in `models/mamba-370m/` for rollback
- Rollback: set env vars back to 370M paths, `HIDDEN_DIM=1024`, `N_GPU_LAYERS=0`

## Security

The fine-tuned GGUF and head weights encode patterns from your personal data. They are not raw data, but memorization is possible with small datasets. Treat these files as sensitive -- do not publish or share them. The `.gitignore` excludes `*.gguf`, `*.safetensors`, and `training/data_prep/raw/`.

## Troubleshooting

**OOM during training**: Reduce `batch_size` to 1, reduce `max_seq_length` to 64, or enable gradient checkpointing (enabled by default for 2.8B).

**NaN losses (370M)**: Ensure you're using fp32. The Mamba sequential fallback path is numerically unstable in fp16. The 2.8B config uses fp16 safely because CUDA kernels handle the compute.

**PEFT error on out_proj**: Mamba's out_proj is blocked by PEFT. Only use `in_proj`, `x_proj`, `dt_proj` as LoRA targets.

**causal_conv1d_fwd argument mismatch**: Version mismatch between causal-conv1d and mamba-ssm. Use causal-conv1d v1.4.0 with mamba-ssm v2.2.4.

**ABI symbol errors (undefined symbol: _ZN3c10...)**: The CUDA kernels were compiled against a different PyTorch version. Uninstall both packages, clear pip cache and any cached .so files, rebuild both from source with `--no-build-isolation` and `--no-cache-dir`.

**Training extremely slow (40+ min/epoch)**: CUDA kernels not loaded. Mamba is falling back to sequential CPU path. Install mamba-ssm with CUDA support (see step 2).

**Root disk full during training**: HF model cache and pip build artifacts can consume 20GB+. Set `HF_HOME` to a larger disk, or symlink `~/.cache/huggingface` to an external drive.

**logits_all=True**: When using llama-cpp-python for inference, the Llama constructor MUST include `logits_all=True` or `model.scores` will be all zeros after `eval()`. This is a llama-cpp-python behavior, not a model issue.
