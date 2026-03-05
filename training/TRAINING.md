# Luna Streams - Training Guide

LoRA fine-tune Mamba 370M for user modeling. Produces a GGUF model + MLP head weights personalized to your Luna interaction history.

## Requirements

- NVIDIA GPU with 10GB+ VRAM (tested on RTX 3080)
- CUDA 12.x toolkit installed
- Python 3.11+
- PostgreSQL databases running (luna-chat + memorycore)
- Ollama with Qwen 3.5:9b (for label generation)

## Training Time

On an RTX 3080 with CUDA kernels: ~2 minutes total (7 epochs + merge + GGUF conversion).

Without CUDA kernels (CPU fallback): ~40 minutes per epoch - unusable. See "Installing CUDA Kernels" below.

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
- If mamba-ssm's `__init__.py` fails with `GreedySearchDecoderOnlyOutput` import error, patch it to wrap the model imports in try/except - only the CUDA ops are needed

**Verify kernels work:**
```bash
python -c "
import torch
import selective_scan_cuda
import causal_conv1d_cuda
print('CUDA kernels OK')
"
```

### 3. Download base model

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('state-spaces/mamba-370m-hf')
AutoTokenizer.from_pretrained('state-spaces/mamba-370m-hf')
print('Model cached')
"
```

### 4. Export your data

Edit `data_prep/export_luna_data.py` and update the container names and database names to match your setup, then run:

```bash
python -m training.data_prep.export_luna_data
```

This exports 7 tables from your luna-chat and memorycore PostgreSQL databases into `data_prep/raw/`. Verify the row counts look reasonable for your instance.

### 5. Build event sequences

```bash
python -m training.data_prep.build_event_sequences
```

Converts raw exports into compact-encoded event sequences grouped by session. Output: `data_prep/sequences.jsonl`.

### 6. Generate labels

Requires Ollama running with Qwen 3.5:9b. Edit the `QWEN_URL` in `data_prep/generate_labels.py` if your Ollama is at a different address.

```bash
python -m training.data_prep.generate_labels
```

Uses Qwen as a labeling oracle to produce:
- `emotional_valence` (float, -1 to 1)
- `focus_topics` (list of topic indices from a 50-class vocabulary)
- `next_event_type` (0=memory_entry, 1=entity_update, 2=edge_update, 3=conversation_meta)

Falls back to heuristic labels if Qwen is unreachable. Output: `data_prep/labeled_sequences.jsonl`.

### 7. Stop GPU services

Free up VRAM before training. Stop any services using the GPU (Ollama, inference servers, etc.):

```bash
systemctl stop ollama
# Stop any other GPU-using processes
nvidia-smi  # verify VRAM is free
```

### 8. Train

```bash
python train.py \
  --config configs/stream1_user.yaml \
  --data data_prep/labeled_sequences.jsonl \
  --output-dir output \
  --epochs 10
```

Training details:
- LoRA rank 16, alpha 32 on in_proj, x_proj, dt_proj (1.9% of params trainable)
- fp32 (the slow Mamba fallback produces NaN in fp16 - safe to use fp32 even with CUDA kernels)
- Batch size 2, gradient accumulation 8 (effective batch 16)
- Early stopping patience 3
- Combined loss: causal LM + emotional valence MSE + focus topics BCE + 0.1 * next event CE

Expected output:
```
Epoch 1/10 - train_loss: 3.87, val_loss: 2.98, valence_mae: 0.42, next_acc: 0.89
...
Epoch 7/10 - Early stopping
Best val_loss: 2.49
Merged model saved: output/merged_model
```

### 9. Convert to GGUF

```bash
# Clone llama.cpp for the convert script
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama-cpp

python /tmp/llama-cpp/convert_hf_to_gguf.py \
  output/merged_model \
  --outtype q8_0 \
  --outfile output/mamba-370m-user-q8_0.gguf
```

### 10. Deploy

Copy the GGUF and head weights to your Luna Streams deployment:

```bash
cp output/mamba-370m-user-q8_0.gguf /path/to/luna-streams/models/mamba-370m/
cp output/mlp_heads.safetensors /path/to/luna-streams/models/mamba-370m/
```

Update your environment or `docker-compose.yml`:
```
STREAMS_GGUF_MODEL=mamba-370m/mamba-370m-user-q8_0.gguf
STREAMS_MLP_HEADS_PATH=mamba-370m/mlp_heads.safetensors
```

Restart luna-streams. Verify with:
```bash
curl http://localhost:8100/api/streams/user_model/state
```

You should see `head_outputs` with real `emotional_valence`, `focus_topics`, and `next_event_type` values.

### 11. Restart GPU services

```bash
systemctl start ollama
# Restart any other GPU services you stopped
```

## Config Reference

See `configs/stream1_user.yaml` for all tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.base` | mamba-370m-hf | HuggingFace model ID |
| `lora.rank` | 16 | LoRA rank (higher = more capacity, more VRAM) |
| `lora.alpha` | 32 | LoRA alpha (typically 2x rank) |
| `training.batch_size` | 2 | Batch size (limited by VRAM) |
| `training.learning_rate` | 2e-4 | AdamW learning rate |
| `training.max_epochs` | 10 | Maximum training epochs |
| `training.max_seq_length` | 128 | Token sequence length |
| `training.early_stopping_patience` | 3 | Epochs without improvement before stopping |

## Security

The fine-tuned GGUF and head weights encode patterns from your personal data. They are not raw data, but memorization is possible with small datasets. Treat these files as sensitive - do not publish or share them. The `.gitignore` excludes `*.gguf`, `*.safetensors`, and `training/data_prep/raw/`.

## Troubleshooting

**OOM during training**: Reduce `batch_size` to 1, reduce `max_seq_length` to 64, or enable gradient checkpointing (already enabled in train.py).

**NaN losses**: Ensure you're using fp32 (dtype=torch.float32). The Mamba sequential fallback path is numerically unstable in fp16.

**PEFT error on out_proj**: Mamba's out_proj is blocked by PEFT. Only use `in_proj`, `x_proj`, `dt_proj` as LoRA targets.

**causal_conv1d_fwd argument mismatch**: Version mismatch between causal-conv1d and mamba-ssm. Use causal-conv1d v1.4.0 with mamba-ssm v2.2.4.

**ABI symbol errors (undefined symbol)**: The CUDA kernels were compiled against a different PyTorch version. Rebuild both from source with `--no-build-isolation`.

**Training extremely slow (40+ min/epoch)**: CUDA kernels not loaded. Mamba is falling back to sequential CPU path. Install mamba-ssm with CUDA support (see step 2).
