"""Configuration definitions for the DAC-3D assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv_file(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding real env vars."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    parts = [part.strip() for part in raw_value.split(",")]
    values = tuple(part for part in parts if part)
    return values or default


@dataclass(slots=True)
class AppConfig:
    """Typed runtime configuration for the assistant."""

    provider: str = "mock"
    api_key: str = ""
    api_base_url: str = ""
    model_name: str = "claude-3-5-haiku-latest"
    max_generation_tokens: int = 700
    temperature: float = 0.1
    timeout_seconds: float = 30.0
    retry_count: int = 1
    streaming: bool = True
    vector_store_type: str = "chroma"
    vector_store_collection: str = "dac3d_iim_documents"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimensions: int = 384
    embedding_download_allowed: bool = False
    chunk_size: int = 320
    chunk_overlap: int = 80
    retrieval_top_k: int = 4
    retrieval_candidate_k: int = 8
    min_retrieval_score: float = 0.16
    history_window: int = 4
    language: str = "zh-CN"
    document_globs: tuple[str, ...] = ("**/*",)
    mock_mode: bool = True
    dac3d_endpoint: str = "mock://dac3d"
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860
    gradio_share: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    frontend_dev_url: str = ""
    agent_model_name: str = ""
    agent_api_key: str = ""
    agent_api_base_url: str = ""
    agent_api_type: str = "auto"
    agent_max_turns: int = 8
    agent_tracing_disabled: bool = True
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    knowledge_base_dir: Path = field(init=False)
    documents_dir: Path = field(init=False)
    vector_store_dir: Path = field(init=False)
    vector_store_path: Path = field(init=False)
    vector_store_manifest_path: Path = field(init=False)
    vector_store_history_path: Path = field(init=False)
    frontend_dir: Path = field(init=False)
    frontend_dist_dir: Path = field(init=False)
    legacy_frontend_dir: Path = field(init=False)
    legacy_frontend_dist_dir: Path = field(init=False)
    temp_root_dir: Path = field(init=False)
    upload_temp_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.knowledge_base_dir = self.base_dir / "knowledge_base"
        self.documents_dir = self.knowledge_base_dir / "documents"
        self.vector_store_dir = self.knowledge_base_dir / "vector_store"
        self.vector_store_path = self.vector_store_dir / "chroma"
        self.vector_store_manifest_path = self.vector_store_path / "index.json"
        self.vector_store_history_path = self.vector_store_path / "build_history.json"
        self.frontend_dir = self.base_dir / "frontend"
        self.frontend_dist_dir = self.frontend_dir / "dist"
        self.legacy_frontend_dir = self.base_dir / "ui2"
        self.legacy_frontend_dist_dir = self.legacy_frontend_dir / "dist"
        self.temp_root_dir = self.base_dir / ".tmp"
        self.upload_temp_dir = self.temp_root_dir / "uploads"
        local_status_file = self.temp_root_dir / "dac3d_runtime_status.json"
        if self.dac3d_endpoint == "mock://dac3d" and local_status_file.exists():
            self.dac3d_endpoint = local_status_file.resolve().as_uri()

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "AppConfig":
        """Load configuration from environment variables."""
        resolved_base_dir = base_dir or Path(__file__).resolve().parent
        _load_dotenv_file(resolved_base_dir / ".env")
        provider = os.getenv("DAC3D_LLM_PROVIDER", "mock")
        api_key = os.getenv("DAC3D_LLM_API_KEY", "")
        if not api_key and provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        api_base_url = (
            os.getenv("DAC3D_LLM_API_BASE_URL")
            or os.getenv("DAC3D_LLM_API_BASE")
            or os.getenv("ANTHROPIC_BASE_URL", "")
        )
        llm_model_name = os.getenv("DAC3D_LLM_MODEL_NAME", "claude-3-5-haiku-latest")
        openai_compatible_model_name = (
            llm_model_name if provider == "openai_compatible" else ""
        )
        openai_compatible_api_key = api_key if provider == "openai_compatible" else ""
        openai_compatible_base_url = api_base_url if provider == "openai_compatible" else ""
        return cls(
            provider=provider,
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=llm_model_name,
            max_generation_tokens=int(os.getenv("DAC3D_MAX_GENERATION_TOKENS", "700")),
            temperature=float(os.getenv("DAC3D_TEMPERATURE", "0.1")),
            timeout_seconds=float(os.getenv("DAC3D_TIMEOUT_SECONDS", "30")),
            retry_count=int(os.getenv("DAC3D_RETRY_COUNT", "1")),
            streaming=_read_bool("DAC3D_STREAMING", True),
            vector_store_type=os.getenv("DAC3D_VECTOR_STORE_TYPE", "chroma"),
            vector_store_collection=os.getenv(
                "DAC3D_VECTOR_STORE_COLLECTION",
                "dac3d_iim_documents",
            ),
            embedding_model_name=os.getenv(
                "DAC3D_EMBEDDING_MODEL_NAME",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            embedding_dimensions=int(os.getenv("DAC3D_EMBEDDING_DIMENSIONS", "384")),
            embedding_download_allowed=_read_bool("DAC3D_EMBEDDING_DOWNLOAD_ALLOWED", False),
            chunk_size=int(os.getenv("DAC3D_CHUNK_SIZE", "320")),
            chunk_overlap=int(os.getenv("DAC3D_CHUNK_OVERLAP", "80")),
            retrieval_top_k=int(os.getenv("DAC3D_RETRIEVAL_TOP_K", "4")),
            retrieval_candidate_k=int(os.getenv("DAC3D_RETRIEVAL_CANDIDATE_K", "8")),
            min_retrieval_score=float(os.getenv("DAC3D_MIN_RETRIEVAL_SCORE", "0.16")),
            history_window=int(os.getenv("DAC3D_HISTORY_WINDOW", "4")),
            language=os.getenv("DAC3D_LANGUAGE", "zh-CN"),
            document_globs=_read_csv(
                "DAC3D_DOCUMENT_GLOBS",
                ("**/*",),
            ),
            mock_mode=_read_bool("DAC3D_MOCK_MODE", True),
            dac3d_endpoint=os.getenv("DAC3D_ENDPOINT", "mock://dac3d"),
            gradio_host=os.getenv("DAC3D_GRADIO_HOST", "127.0.0.1"),
            gradio_port=int(os.getenv("DAC3D_GRADIO_PORT", "7860")),
            gradio_share=_read_bool("DAC3D_GRADIO_SHARE", False),
            web_host=os.getenv("DAC3D_WEB_HOST", "127.0.0.1"),
            web_port=int(os.getenv("DAC3D_WEB_PORT", "8000")),
            frontend_dev_url=os.getenv("DAC3D_FRONTEND_DEV_URL", "").strip(),
            agent_model_name=(
                os.getenv("DAC3D_AGENT_MODEL_NAME")
                or os.getenv("OPENAI_DEFAULT_MODEL", "")
                or openai_compatible_model_name
            ).strip(),
            agent_api_key=(
                os.getenv("DAC3D_AGENT_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or openai_compatible_api_key
            ).strip(),
            agent_api_base_url=(
                os.getenv("DAC3D_AGENT_API_BASE_URL")
                or os.getenv("DAC3D_AGENT_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or openai_compatible_base_url
            ).strip(),
            agent_api_type=(
                os.getenv("DAC3D_AGENT_API_TYPE")
                or ("chat_completions" if provider == "openai_compatible" else "auto")
            ).strip().lower(),
            agent_max_turns=int(os.getenv("DAC3D_AGENT_MAX_TURNS", "8")),
            agent_tracing_disabled=_read_bool("DAC3D_AGENT_TRACING_DISABLED", True),
            base_dir=resolved_base_dir,
        )

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.upload_temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def vector_store_ready(self) -> bool:
        """Return whether the persisted vector-store manifest exists."""
        return self.vector_store_manifest_path.exists()
