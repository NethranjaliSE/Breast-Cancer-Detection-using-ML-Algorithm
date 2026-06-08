"""
src/utils/logger.py
-------------------
Centralised structured logger for the entire project.
All modules import from here to ensure consistent formatting.
"""
import logging
import logging.handlers
import os
from pathlib import Path


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a project-wide logger with both console and rotating file handlers.

    Usage
    -----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
    """
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ──────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger