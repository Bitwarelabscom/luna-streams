# Luna Mamba Streams - CPU inference container
# GGUF via llama.cpp - optimized C++ kernels for Mamba on AMD Ryzen 5 3600

# Stage 1: Build llama-cpp-python (needs compiler for C++ backend)
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY luna_streams/ ./luna_streams/
COPY benchmark.py .

RUN useradd -m -r streams \
    && mkdir -p /app/models /app/state \
    && chown -R streams:streams /app

USER streams
EXPOSE 8100

# CPU tuning for llama.cpp
ENV OMP_NUM_THREADS=8
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "luna_streams"]
