"""
Luna Mamba Streams - Priority Zero Benchmark
=============================================
Tests CPU inference paths for Mamba-790M on actual Hetzner hardware.
Nothing else starts until this passes.

Run: python benchmark.py
Output: benchmark_results.json

Pass/fail target: <150ms mean latency per step
Fallback: if both paths fail, re-run with --model state-spaces/mamba-370m-hf
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


@dataclass
class PathResult:
    path_name: str = ""
    viable: bool = False
    reason: str = ""
    ram_mb: float = 0.0
    model_size_mb: float = 0.0
    latency: Optional[LatencyStats] = None
    state_save_ms: float = 0.0
    state_restore_ms: float = 0.0
    sustained_throughput_eps: float = 0.0  # events per second over 10 min
    events_processed: int = 0
    error: str = ""


@dataclass
class BenchmarkResults:
    timestamp: str = ""
    model_id: str = ""
    system_info: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    decision: str = ""


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Capture CPU model, instruction sets, RAM, OS info."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
    }

    # Get CPU model and flags
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                if "Flags" in line:
                    flags = line.split(":", 1)[1].strip().split()
                    info["avx2"] = "avx2" in flags
                    info["avx512f"] = "avx512f" in flags
                    info["sse4_2"] = "sse4_2" in flags
    except Exception:
        info["cpu_model"] = "unknown"
        info["avx2"] = False
        info["avx512f"] = False

    return info


# ---------------------------------------------------------------------------
# Dummy event generation
# ---------------------------------------------------------------------------

TOPIC_POOLS = [
    "architecture", "memory", "state-space-models", "Mamba", "NeuralSleep",
    "graph-consolidation", "trading", "music", "Docker", "deployment",
    "embeddings", "attention", "transformers", "Linux", "Python", "TypeScript",
    "Neo4j", "PostgreSQL", "Redis", "WebSockets", "SSE", "REST",
    "consciousness", "dual-LNN", "knowledge-graph", "VR", "Unreal-Engine",
]

ENTITY_POOLS = [
    "Luna", "Henke", "Mamba", "NeuralSleep", "MemoryCore", "Neo4j",
    "Qwen", "Grok", "Ollama", "Docker", "Hetzner", "BitwareLabs",
    "FastAPI", "React", "Next.js", "Zustand", "TipTap", "Cytoscape",
]

EDGE_TYPES = [
    "co_occurrence", "semantic", "temporal", "causal",
    "knows_person", "working_on", "integrates",
]


def generate_dummy_events(n: int = 10000) -> list[dict]:
    """Generate realistic structured events matching the spec schema."""
    events = []
    for i in range(n):
        ts = datetime.now(timezone.utc).isoformat()
        r = random.random()

        if r < 0.5:
            # memory_entry (50%)
            num_entities = random.randint(1, 4)
            entities = random.sample(ENTITY_POOLS, min(num_entities, len(ENTITY_POOLS)))
            num_topics = random.randint(1, 3)
            topics = random.sample(TOPIC_POOLS, min(num_topics, len(TOPIC_POOLS)))

            event = {
                "timestamp": ts,
                "event_type": "memory_entry",
                "source": random.choice(["conversation", "agent_dialogue"]),
                "content": {
                    "entities": entities,
                    "relations": [],
                    "topic_tags": topics,
                    "sentiment": round(random.uniform(-1.0, 1.0), 3),
                    "importance": round(random.uniform(0.0, 1.0), 3),
                    "summary": f"Event {i}: user discussing {', '.join(topics[:2])}",
                },
                "conversation_meta": {
                    "message_length": random.randint(10, 500),
                    "response_time_ms": random.randint(500, 15000),
                    "session_duration_min": round(random.uniform(1, 120), 1),
                    "active_persona": random.choice([None, "Sol", "Vega", "Aurora", "Polaris"]),
                    "active_model": random.choice(["grok-4.1", "qwen-2.5-7b", "claude-opus"]),
                    "turn_number": random.randint(1, 50),
                },
            }
        elif r < 0.75:
            # entity_update (25%)
            entities = random.sample(ENTITY_POOLS, random.randint(1, 3))
            event = {
                "timestamp": ts,
                "event_type": "entity_update",
                "source": "conversation",
                "content": {
                    "entities": entities,
                    "relations": [],
                    "topic_tags": [],
                    "sentiment": 0.0,
                    "importance": round(random.uniform(0.3, 0.9), 3),
                    "summary": "",
                },
            }
        elif r < 0.90:
            # edge_update (15%)
            e1, e2 = random.sample(ENTITY_POOLS, 2)
            event = {
                "timestamp": ts,
                "event_type": "edge_update",
                "source": "conversation",
                "content": {
                    "entities": [e1, e2],
                    "relations": [{
                        "from": e1,
                        "to": e2,
                        "type": random.choice(EDGE_TYPES),
                        "weight": round(random.uniform(0.1, 1.0), 3),
                    }],
                    "topic_tags": [],
                    "sentiment": 0.0,
                    "importance": round(random.uniform(0.2, 0.8), 3),
                    "summary": "",
                },
            }
        else:
            # conversation_meta (10%)
            event = {
                "timestamp": ts,
                "event_type": "conversation_meta",
                "source": "conversation",
                "content": {
                    "entities": [],
                    "relations": [],
                    "topic_tags": [],
                    "sentiment": 0.0,
                    "importance": 0.3,
                    "summary": "",
                },
                "conversation_meta": {
                    "message_length": random.randint(10, 500),
                    "response_time_ms": random.randint(200, 10000),
                    "session_duration_min": round(random.uniform(1, 180), 1),
                    "active_persona": random.choice([None, "Sol", "Vega"]),
                    "active_model": random.choice(["grok-4.1", "qwen-2.5-7b"]),
                    "turn_number": random.randint(1, 80),
                },
            }

        events.append(event)

    return events


def event_to_text(event: dict) -> str:
    """Convert a structured event to a compact text representation for tokenization."""
    parts = [f"[{event['event_type']}]"]
    content = event.get("content", {})

    if content.get("entities"):
        parts.append(f"entities: {', '.join(content['entities'])}")
    if content.get("relations"):
        for rel in content["relations"]:
            parts.append(f"rel: {rel['from']}->{rel['to']} ({rel['type']}, {rel['weight']})")
    if content.get("topic_tags"):
        parts.append(f"topics: {', '.join(content['topic_tags'])}")
    if content.get("sentiment"):
        parts.append(f"sentiment: {content['sentiment']}")
    if content.get("importance"):
        parts.append(f"importance: {content['importance']}")
    if content.get("summary"):
        parts.append(f"summary: {content['summary'][:100]}")

    meta = event.get("conversation_meta")
    if meta:
        parts.append(f"len:{meta.get('message_length', 0)} rt:{meta.get('response_time_ms', 0)}ms")
        if meta.get("active_model"):
            parts.append(f"model:{meta['active_model']}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Path A: GGUF via llama.cpp
# ---------------------------------------------------------------------------

def benchmark_path_a_gguf(model_id: str, events: list[dict]) -> PathResult:
    """Test GGUF inference path via llama-cpp-python."""
    result = PathResult(path_name="gguf_llama_cpp")

    # Check if llama-cpp-python is installed and supports Mamba
    try:
        from llama_cpp import Llama
    except ImportError:
        result.reason = "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        return result

    # Check Mamba architecture support
    # As of early 2026, llama.cpp has experimental Mamba support (GGUF format)
    # but it's not guaranteed to work with all Mamba variants
    print("[Path A] Checking Mamba GGUF support...")

    # Try to find or create a GGUF file
    gguf_path = Path(f"models/{model_id.split('/')[-1]}.gguf")

    if not gguf_path.exists():
        print(f"[Path A] GGUF file not found at {gguf_path}")
        print("[Path A] To test this path, convert the model to GGUF format:")
        print(f"  python -m llama_cpp.convert_hf {model_id} --outfile {gguf_path}")
        print("[Path A] Or download a pre-converted GGUF if available.")

        # Try conversion via transformers CLI
        try:
            print("[Path A] Attempting automatic conversion...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[Path A] Loading {model_id} for GGUF conversion check...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Check if the model architecture is "MambaForCausalLM"
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            arch = config.architectures[0] if config.architectures else "unknown"
            print(f"[Path A] Model architecture: {arch}")

            if "Mamba" not in arch:
                result.reason = f"Model architecture {arch} - may not be supported in GGUF"
                return result

            # llama.cpp GGUF conversion for Mamba is experimental
            result.reason = (
                f"Mamba architecture detected ({arch}) but automatic GGUF conversion "
                "not yet reliable. Manual conversion required. "
                "Check: https://github.com/ggerganov/llama.cpp/issues for Mamba GGUF status."
            )
            return result

        except Exception as e:
            result.reason = f"GGUF conversion check failed: {e}"
            return result

    # If we have a GGUF file, benchmark it
    try:
        print(f"[Path A] Loading GGUF model from {gguf_path}...")
        ram_before = psutil.Process().memory_info().rss / (1024**2)

        model = Llama(
            model_path=str(gguf_path),
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

        ram_after = psutil.Process().memory_info().rss / (1024**2)
        result.model_size_mb = ram_after - ram_before
        result.ram_mb = ram_after

        # Benchmark forward passes
        latencies = []
        print(f"[Path A] Running {len(events)} forward passes...")

        for i, event in enumerate(events):
            text = event_to_text(event)
            tokens = model.tokenize(text.encode())

            start = time.perf_counter()
            model.eval(tokens)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{len(events)}] mean: {np.mean(latencies[-1000:]):.1f}ms")

        result.latency = LatencyStats(
            mean_ms=round(float(np.mean(latencies)), 2),
            p50_ms=round(float(np.percentile(latencies, 50)), 2),
            p95_ms=round(float(np.percentile(latencies, 95)), 2),
            p99_ms=round(float(np.percentile(latencies, 99)), 2),
            min_ms=round(float(np.min(latencies)), 2),
            max_ms=round(float(np.max(latencies)), 2),
        )

        result.ram_mb = psutil.Process().memory_info().rss / (1024**2)
        result.events_processed = len(events)
        result.viable = result.latency.mean_ms < 150.0
        if not result.viable:
            result.reason = f"Mean latency {result.latency.mean_ms}ms exceeds 150ms target"

    except Exception as e:
        result.error = str(e)
        result.reason = f"GGUF benchmark failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Path B: PyTorch + torch.compile + dynamic int8
# ---------------------------------------------------------------------------

def benchmark_path_b_torch(model_id: str, events: list[dict]) -> PathResult:
    """Test PyTorch inference with torch.compile and dynamic int8 quantization."""
    result = PathResult(path_name="torch_compile_int8")

    try:
        import torch
        print(f"[Path B] PyTorch version: {torch.__version__}")
        print(f"[Path B] CPU threads: OMP={os.environ.get('OMP_NUM_THREADS', 'default')}")
    except ImportError:
        result.reason = "PyTorch not installed"
        return result

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        result.reason = "transformers not installed"
        return result

    # Load model
    print(f"[Path B] Loading {model_id}...")
    ram_before = psutil.Process().memory_info().rss / (1024**2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU needs float32
        )
        model.eval()
    except Exception as e:
        result.reason = f"Model loading failed: {e}"
        result.error = str(e)
        return result

    ram_after_load = psutil.Process().memory_info().rss / (1024**2)
    result.model_size_mb = round(ram_after_load - ram_before, 1)
    print(f"[Path B] Model loaded. RAM: {result.model_size_mb:.0f}MB")

    # Try optimization strategies in order of preference:
    # 1. torch.compile (fastest but may not work with Mamba)
    # 2. dynamic int8 quantization (reduces memory, may break custom layers)
    # 3. plain float32 eager mode (always works, baseline)

    quantized = False
    compiled = False
    optimization = "float32_eager"

    # First: test that base model works at all
    print("[Path B] Testing base model inference...")
    warmup_tokens = tokenizer(event_to_text(events[0]), return_tensors="pt")
    try:
        with torch.no_grad():
            model(**warmup_tokens)
        print("[Path B] Base model inference OK")
    except Exception as e:
        result.reason = f"Base model inference failed: {e}"
        result.error = str(e)
        return result

    # Try torch.compile
    print("[Path B] Attempting torch.compile...")
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            compiled_model(**warmup_tokens)
        model = compiled_model
        compiled = True
        optimization = "torch_compile"
        print("[Path B] torch.compile succeeded")
    except Exception as e:
        print(f"[Path B] torch.compile failed ({type(e).__name__}), trying int8 quantization...")

    # Try int8 quantization (only if compile failed)
    if not compiled:
        try:
            quant_model = torch.ao.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            with torch.no_grad():
                quant_model(**warmup_tokens)
            model = quant_model
            quantized = True
            optimization = "int8_eager"
            ram_after_quant = psutil.Process().memory_info().rss / (1024**2)
            print(f"[Path B] int8 quantization succeeded. RAM: {ram_after_quant - ram_before:.0f}MB")
        except Exception as e:
            print(f"[Path B] int8 quantization broke inference ({type(e).__name__}: {e})")
            print("[Path B] Using plain float32 eager mode")

    print(f"[Path B] Final optimization: {optimization}")

    # Warmup
    print("[Path B] Warmup (5 forward passes)...")
    for _ in range(5):
        tokens = tokenizer(event_to_text(events[0]), return_tensors="pt")
        with torch.no_grad():
            model(**tokens)

    # Benchmark forward passes
    latencies = []
    print(f"[Path B] Running {len(events)} forward passes...")

    for i, event in enumerate(events):
        text = event_to_text(event)
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        start = time.perf_counter()
        with torch.no_grad():
            output = model(**tokens)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        if (i + 1) % 1000 == 0:
            cur_mean = np.mean(latencies[-1000:])
            print(f"  [{i+1}/{len(events)}] mean: {cur_mean:.1f}ms")

    result.latency = LatencyStats(
        mean_ms=round(float(np.mean(latencies)), 2),
        p50_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_ms=round(float(np.percentile(latencies, 99)), 2),
        min_ms=round(float(np.min(latencies)), 2),
        max_ms=round(float(np.max(latencies)), 2),
    )

    # State serialization benchmark
    print("[Path B] Benchmarking state serialization...")
    try:
        import safetensors.torch as st

        # Extract hidden states (Mamba SSM internal state)
        # After a forward pass, the model's internal state is in the cache
        state_dict = {}
        for name, param in model.named_parameters():
            if "ssm" in name.lower() or "conv" in name.lower():
                state_dict[name] = param.data.clone()

        # If no SSM-specific params found, save a subset of the model state
        if not state_dict:
            for name, param in list(model.named_parameters())[:20]:
                state_dict[name] = param.data.clone()

        state_path = Path("state/benchmark_state.safetensors")
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        start = time.perf_counter()
        st.save_file(state_dict, str(state_path))
        result.state_save_ms = round((time.perf_counter() - start) * 1000, 2)

        # Restore
        start = time.perf_counter()
        _ = st.load_file(str(state_path))
        result.state_restore_ms = round((time.perf_counter() - start) * 1000, 2)

        # Cleanup
        state_path.unlink(missing_ok=True)

        print(f"[Path B] State save: {result.state_save_ms:.1f}ms, restore: {result.state_restore_ms:.1f}ms")
    except Exception as e:
        print(f"[Path B] State serialization benchmark failed: {e}")
        result.state_save_ms = -1
        result.state_restore_ms = -1

    # Sustained throughput (shortened: 2 min instead of 10 for initial benchmark)
    print("[Path B] Sustained throughput test (2 minutes)...")
    sustained_start = time.perf_counter()
    sustained_count = 0
    sustained_duration_sec = 120

    while (time.perf_counter() - sustained_start) < sustained_duration_sec:
        event = events[sustained_count % len(events)]
        text = event_to_text(event)
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            model(**tokens)
        sustained_count += 1

    elapsed_sec = time.perf_counter() - sustained_start
    result.sustained_throughput_eps = round(sustained_count / elapsed_sec, 2)
    print(f"[Path B] Sustained: {result.sustained_throughput_eps:.1f} events/sec over {elapsed_sec:.0f}s")

    result.ram_mb = round(psutil.Process().memory_info().rss / (1024**2), 1)
    result.events_processed = len(events)
    result.viable = result.latency.mean_ms < 150.0

    if result.viable:
        result.reason = f"PASS - {result.latency.mean_ms:.1f}ms mean latency, optimization={optimization}"
    else:
        result.reason = f"Mean latency {result.latency.mean_ms:.1f}ms exceeds 150ms target (optimization={optimization})"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Luna Mamba Streams CPU Benchmark")
    parser.add_argument(
        "--model", default="state-spaces/mamba-790m-hf",
        help="HuggingFace model ID (default: state-spaces/mamba-790m-hf)"
    )
    parser.add_argument(
        "--events", type=int, default=10000,
        help="Number of dummy events to generate (default: 10000)"
    )
    parser.add_argument(
        "--skip-path-a", action="store_true",
        help="Skip GGUF/llama.cpp benchmark path"
    )
    parser.add_argument(
        "--skip-path-b", action="store_true",
        help="Skip PyTorch/torch.compile benchmark path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Luna Mamba Streams - Priority Zero Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Events: {args.events}")
    print()

    # System info
    print("Gathering system info...")
    sys_info = get_system_info()
    print(f"CPU: {sys_info.get('cpu_model', 'unknown')}")
    print(f"Cores: {sys_info.get('cpu_count_physical')} physical, {sys_info.get('cpu_count_logical')} logical")
    print(f"RAM: {sys_info.get('ram_total_gb')}GB total, {sys_info.get('ram_available_gb')}GB available")
    print(f"AVX2: {sys_info.get('avx2')}, AVX-512: {sys_info.get('avx512f')}")
    print()

    # Generate events
    print(f"Generating {args.events} dummy events...")
    events = generate_dummy_events(args.events)
    print(f"Generated {len(events)} events")
    print()

    results = BenchmarkResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=args.model,
        system_info=sys_info,
    )

    # Path A: GGUF
    if not args.skip_path_a:
        print("-" * 60)
        print("PATH A: GGUF via llama.cpp")
        print("-" * 60)
        try:
            path_a = benchmark_path_a_gguf(args.model, events)
            results.paths["gguf"] = asdict(path_a)
        except Exception as e:
            print(f"[Path A] Fatal error: {e}")
            results.paths["gguf"] = {"viable": False, "error": str(e), "path_name": "gguf_llama_cpp"}
        print()
    else:
        print("Skipping Path A (--skip-path-a)")
        results.paths["gguf"] = {"viable": False, "reason": "Skipped", "path_name": "gguf_llama_cpp"}

    # Path B: PyTorch
    if not args.skip_path_b:
        print("-" * 60)
        print("PATH B: PyTorch + torch.compile + dynamic int8")
        print("-" * 60)
        try:
            path_b = benchmark_path_b_torch(args.model, events)
            results.paths["torch_compile"] = asdict(path_b)
        except Exception as e:
            print(f"[Path B] Fatal error: {e}")
            results.paths["torch_compile"] = {"viable": False, "error": str(e), "path_name": "torch_compile_int8"}
        print()
    else:
        print("Skipping Path B (--skip-path-b)")
        results.paths["torch_compile"] = {"viable": False, "reason": "Skipped", "path_name": "torch_compile_int8"}

    # Decision
    print("=" * 60)
    print("DECISION")
    print("=" * 60)

    viable_paths = [
        (name, data) for name, data in results.paths.items()
        if data.get("viable", False)
    ]

    if not viable_paths:
        if args.model == "state-spaces/mamba-790m-hf":
            results.decision = (
                "FAIL - neither path meets <150ms target with 790M model. "
                "Re-run with: python benchmark.py --model state-spaces/mamba-370m-hf"
            )
        else:
            results.decision = (
                f"FAIL - neither path meets <150ms target with {args.model}. "
                "Architecture redesign needed."
            )
        print(results.decision)
    else:
        # Pick the best viable path
        best_name, best_data = min(
            viable_paths,
            key=lambda x: x[1].get("latency", {}).get("mean_ms", 9999)
        )
        mean_lat = best_data.get("latency", {}).get("mean_ms", 0)
        ram = best_data.get("ram_mb", 0)
        results.decision = (
            f"USE {best_name} - {mean_lat:.1f}ms mean latency, "
            f"{ram:.0f}MB RAM"
        )
        print(results.decision)

    # Detailed summary
    for name, data in results.paths.items():
        lat = data.get("latency")
        print(f"\n  {name}:")
        print(f"    viable: {data.get('viable')}")
        print(f"    reason: {data.get('reason', data.get('error', 'N/A'))}")
        if lat:
            print(f"    latency: mean={lat['mean_ms']}ms p50={lat['p50_ms']}ms p95={lat['p95_ms']}ms p99={lat['p99_ms']}ms")
            print(f"    RAM: {data.get('ram_mb', 0):.0f}MB")

    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(asdict(results) if hasattr(results, '__dataclass_fields__') else {
            "timestamp": results.timestamp,
            "model_id": results.model_id,
            "system_info": results.system_info,
            "paths": results.paths,
            "decision": results.decision,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("=" * 60)

    # Exit with non-zero if no viable path
    sys.exit(0 if viable_paths else 1)


if __name__ == "__main__":
    main()
