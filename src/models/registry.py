"""
src/models/registry.py
-----------------------
Model persistence layer: save, load, list, and version models.

Saved structure
---------------
models/saved/
├── random_forest.pkl
├── knn.pkl
├── naive_bayes.pkl
├── svm.pkl
├── logistic_regression.pkl
├── ensemble_soft_voting.pkl   ← default model for inference
└── manifest.json              ← metadata (versions, sizes, metrics)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Naming helpers
# ─────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """'Random Forest' → 'random_forest'"""
    return (name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", ""))


def _model_path(name: str, model_dir: Path) -> Path:
    return model_dir / f"{_slug(name)}.pkl"


# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────

def save_models(
    models: dict,
    metrics: Optional[dict] = None,
    model_dir: Optional[Path] = None,
    compress: int = 3,
) -> List[Path]:
    """
    Persist every model to disk and write a manifest.json.

    Parameters
    ----------
    models    : {name: fitted_pipeline}  from trainer.train_all_models()
    metrics   : optional {name: ModelMetrics}  to embed in manifest
    model_dir : override default from settings
    compress  : joblib compression level 0-9

    Returns
    -------
    saved_paths : list of Path objects
    """
    cfg = get_settings()
    model_dir = model_dir or cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(models)} models → {model_dir}")

    saved_paths: List[Path] = []
    manifest_entries = {}

    for name, model in models.items():
        path = _model_path(name, model_dir)
        joblib.dump(model, path, compress=compress)
        size_kb = path.stat().st_size / 1024

        entry = {
            "slug":       _slug(name),
            "path":       str(path),
            "size_kb":    round(size_kb, 1),
            "saved_at":   datetime.now(timezone.utc).isoformat(),
        }
        if metrics and name in metrics:
            m = metrics[name]
            entry["metrics"] = m.to_dict() if hasattr(m, "to_dict") else {}

        manifest_entries[_slug(name)] = entry
        saved_paths.append(path)
        log.info(f"  ✔ {_slug(name)+'.pkl':<45}  {size_kb:.1f} KB")

    # ── Write manifest ─────────────────────────────────────────
    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    log.info(f"  ✔ manifest.json written")

    return saved_paths


# ─────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────

def load_model(name: str, model_dir: Optional[Path] = None):
    """
    Load a single model by display name or slug.

    Examples
    --------
    load_model("Random Forest")
    load_model("ensemble_soft_voting")
    """
    cfg = get_settings()
    model_dir = model_dir or cfg.model_dir
    path = _model_path(name, model_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run  python scripts/train.py  to train and save models first."
        )

    model = joblib.load(path)
    log.info(f"Model loaded ← {path.name}")
    return model


def load_all_models(model_dir: Optional[Path] = None) -> Dict[str, object]:
    """Load all .pkl files from the model directory."""
    cfg = get_settings()
    model_dir = model_dir or cfg.model_dir

    models = {}
    for pkl in sorted(model_dir.glob("*.pkl")):
        model = joblib.load(pkl)
        name  = pkl.stem                  # filename without .pkl
        models[name] = model
        log.debug(f"  Loaded {pkl.name}")

    log.info(f"Loaded {len(models)} models from {model_dir}")
    return models


def load_default_model(model_dir: Optional[Path] = None):
    """Load whichever model is set as DEFAULT_MODEL in settings."""
    cfg = get_settings()
    return load_model(cfg.default_model, model_dir)


# ─────────────────────────────────────────────────────────────────
# Manifest / listing
# ─────────────────────────────────────────────────────────────────

def list_models(model_dir: Optional[Path] = None) -> List[dict]:
    """Return metadata for all saved models from manifest.json."""
    cfg = get_settings()
    model_dir = model_dir or cfg.model_dir
    manifest_path = model_dir / "manifest.json"

    if not manifest_path.exists():
        log.warning("manifest.json not found — run training first.")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    return list(manifest.values())


def model_exists(name: str, model_dir: Optional[Path] = None) -> bool:
    cfg = get_settings()
    model_dir = model_dir or cfg.model_dir
    return _model_path(name, model_dir).exists()