"""
src/api/main.py
----------------
FastAPI application factory.

Start the server
----------------
uvicorn src.api.main:app --reload --port 8000

Interactive docs
----------------
http://localhost:8000/docs      (Swagger UI)
http://localhost:8000/redoc     (ReDoc)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.models.registry import list_models
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_settings()


# ─────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before serving requests."""
    log.info("=" * 55)
    log.info(f"  {cfg.app_name} v{cfg.app_version}  [{cfg.app_env}]")
    log.info("=" * 55)

    saved = list_models()
    if saved:
        log.info(f"  ✔ {len(saved)} trained models found in {cfg.model_dir}")
    else:
        log.warning("  ⚠ No trained models found. Run: python scripts/train.py")

    yield  # — application runs here —

    log.info("Shutting down …")


# ─────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title       = cfg.app_name,
        description = (
            "Production ML API for breast cancer prediction.\n\n"
            "Based on the CTEMS-2018 paper comparing Random Forest, "
            "kNN, and Naive Bayes on the Wisconsin Diagnosis Breast Cancer dataset."
        ),
        version     = cfg.app_version,
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins  = cfg.api_host and ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # ── Routes ────────────────────────────────────────────────────
    application.include_router(router)

    return application


app = create_app()


# ─────────────────────────────────────────────────────────────────
# Run directly (python src/api/main.py)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host    = cfg.api_host,
        port    = cfg.api_port,
        reload  = cfg.app_debug,
        workers = 1 if cfg.app_debug else cfg.api_workers,
    )