"""
src/utils/config.py
-------------------
Settings management via Pydantic BaseSettings.
Values are read from  (in priority order):
  1. Environment variables
  2. .env file
  3. Hard-coded defaults below
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────
    app_name: str        = Field("BreastCancerML", alias="APP_NAME")
    app_env: str         = Field("development",    alias="APP_ENV")
    app_debug: bool      = Field(True,             alias="APP_DEBUG")
    app_version: str     = Field("1.0.0",          alias="APP_VERSION")

    # ── API ───────────────────────────────────────────────
    api_host: str        = Field("0.0.0.0",        alias="API_HOST")
    api_port: int        = Field(8000,             alias="API_PORT")
    api_workers: int     = Field(1,                alias="API_WORKERS")

    # ── Paths ─────────────────────────────────────────────
    model_dir: Path      = Field(Path("models/saved"), alias="MODEL_DIR")
    log_dir: Path        = Field(Path("logs"),         alias="LOG_DIR")

    # ── Model training ────────────────────────────────────
    default_model: str   = Field("ensemble_soft_voting", alias="DEFAULT_MODEL")
    random_state: int    = Field(42,   alias="RANDOM_STATE")
    test_size: float     = Field(0.30, alias="TEST_SIZE")
    cv_folds: int        = Field(10,   alias="CV_FOLDS")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ── Singleton ─────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings