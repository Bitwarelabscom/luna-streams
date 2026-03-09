#!/usr/bin/env bash
# Convert Mamba 2.8B HuggingFace model to GGUF Q8_0
#
# Prerequisites:
#   - llama.cpp repo cloned (for convert_hf_to_gguf.py)
#   - Python with transformers, safetensors, numpy installed
#   - ~12GB disk space (HF model + GGUF output)
#
# Usage:
#   ./scripts/convert_mamba_2.8b.sh [LLAMA_CPP_DIR]

set -euo pipefail

LLAMA_CPP_DIR="${1:-$HOME/llama.cpp}"
MODEL_ID="state-spaces/mamba-2.8b-hf"
HF_DIR="./models/mamba-2.8b/hf"
GGUF_DIR="./models/mamba-2.8b"
GGUF_F16="${GGUF_DIR}/mamba-2.8b-f16.gguf"
GGUF_Q8="${GGUF_DIR}/mamba-2.8b-q8_0.gguf"

echo "=== Mamba 2.8B GGUF Conversion ==="

# Step 1: Download from HuggingFace
if [ ! -d "$HF_DIR" ]; then
    echo "Downloading ${MODEL_ID}..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', local_dir='${HF_DIR}')
"
    echo "Download complete."
else
    echo "HF model already exists at ${HF_DIR}, skipping download."
fi

# Step 2: Convert to GGUF F16
if [ ! -f "$GGUF_F16" ]; then
    echo "Converting to GGUF F16..."
    python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "$HF_DIR" \
        --outfile "$GGUF_F16" \
        --outtype f16
    echo "F16 conversion complete."
else
    echo "F16 GGUF already exists, skipping conversion."
fi

# Step 3: Quantize to Q8_0
if [ ! -f "$GGUF_Q8" ]; then
    echo "Quantizing to Q8_0..."
    "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
        "$GGUF_F16" \
        "$GGUF_Q8" \
        Q8_0
    echo "Q8_0 quantization complete."
else
    echo "Q8_0 GGUF already exists, skipping quantization."
fi

# Step 4: Verify
echo ""
echo "=== Output ==="
ls -lh "$GGUF_Q8"
echo ""
echo "GGUF model ready at: ${GGUF_Q8}"
echo "Set STREAMS_GGUF_MODEL=mamba-2.8b/mamba-2.8b-q8_0.gguf in your environment."

# Optional: clean up F16 intermediate
read -p "Delete intermediate F16 file (~5.6GB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$GGUF_F16"
    echo "Deleted ${GGUF_F16}"
fi
