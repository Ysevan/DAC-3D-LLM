"""Configuration definitions for the assistant."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Application-level configuration placeholder."""

    api_key: str = "YOUR_API_KEY"
    model_name: str = "YOUR_MODEL_NAME"
    base_dir: Path = Path(__file__).resolve().parent
    knowledge_base_dir: Path = base_dir / "knowledge_base"
