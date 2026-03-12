"""
Central configuration loaded from environment variables / .env file.
All modules import from here — never read os.environ directly elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pydantic v2

# Resolve project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # OpenAI
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    # ------------------------------------------------------------------ #
    # NCBI / PubMed
    # ------------------------------------------------------------------ #
    ncbi_api_key: str | None = Field(None, env="NCBI_API_KEY")
    ncbi_email: str = Field("researcher@example.com", env="NCBI_EMAIL")

    # ------------------------------------------------------------------ #
    # Qdrant
    # ------------------------------------------------------------------ #
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        "research_limitations", env="QDRANT_COLLECTION"
    )
    # Set to True to use in-memory Qdrant (no server required)
    qdrant_in_memory: bool = Field(True, env="QDRANT_IN_MEMORY")

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    retrieval_top_k: int = Field(8, env="RETRIEVAL_TOP_K")

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    raw_data_dir: Path = Field(PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(PROJECT_ROOT / "data" / "processed")
    log_dir: Path = Field(PROJECT_ROOT / "logs")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        """Create data/log directories if they don't exist."""
        for d in (self.raw_data_dir, self.processed_data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()
settings.ensure_dirs()
