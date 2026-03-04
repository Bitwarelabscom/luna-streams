"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "info"

    # State persistence
    state_dir: str = "./state"
    model_dir: str = "./models"
    auto_save_interval_sec: int = 300
    snapshot_retention: int = 3

    # GGUF inference (llama.cpp)
    gguf_model: str = "mamba-370m/mamba-370m-q8_0.gguf"
    gguf_n_ctx: int = 256
    gguf_n_threads: int = 8

    # EMA buffer
    ema_decay: float = 0.999
    delta_threshold: float = 0.01

    # Qwen bridge (Option B - text summaries)
    qwen_ollama_url: str = "http://10.0.0.30:11434"
    qwen_model: str = "qwen3.5:9b"

    # Stream control
    streams_enabled: bool = True
    stream_user_enabled: bool = True
    stream_knowledge_enabled: bool = False  # Phase 8
    stream_dynamics_enabled: bool = False   # Phase 8

    model_config = {"env_prefix": "STREAMS_"}


settings = Settings()
