"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_tokens: str = ""
    admin_token: str = ""
    database_url: str = "sqlite:///./usage.sqlite"
    model_source: str = "file"
    model_path: Path = Path("models/release/eurosat_int8.onnx")
    reference_stats_path: Path = Path("models/release/reference_stats.json")
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    model_name: str = "eurosat-classifier"
    model_alias: str = "production"
    cors_origins: str = "http://127.0.0.1:8080"
    max_upload_bytes: int = Field(default=2_000_000, ge=1_000)
    drift_window: int = Field(default=200, ge=20)

    def token_map(self) -> dict[str, str]:
        """Return {token: user} parsed from API_TOKENS ("user:token,user:token")."""
        pairs: dict[str, str] = {}
        for item in filter(None, (p.strip() for p in self.api_tokens.split(","))):
            user, sep, token = item.partition(":")
            if sep and user and token:
                pairs[token] = user
        return pairs

    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
