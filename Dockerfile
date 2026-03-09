# Luna Mamba Streams - CUDA-enabled inference container
# GGUF via llama.cpp with GPU offload for Tesla P4 (compute 6.1)

# Stage 1: Build llama-cpp-python with CUDA support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake ninja-build \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY requirements.txt .

# Build llama-cpp-python with CUDA for Pascal (compute 6.1)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv libpython3.11 \
    curl ca-certificates libgomp1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY luna_streams/ ./luna_streams/
COPY benchmark.py .

RUN useradd -m -r streams \
    && mkdir -p /app/models /app/state \
    && chown -R streams:streams /app

USER streams
EXPOSE 8100

# GPU inference tuning - fewer CPU threads needed with GPU offload
ENV OMP_NUM_THREADS=2
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "luna_streams"]
