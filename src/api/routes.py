"""
src/api/routes.py
------------------
All route handlers.  The router is included by main.py.

Endpoints
---------
GET  /health           — liveness check
GET  /models           — list saved models + metadata
GET  /metrics          — model performance metrics (loaded from manifest)
POST /predict          — single-sample prediction
POST /predict/batch    — CSV upload → batch predictions
"""
from __future__ import annotations

import io
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from src.api.schemas import (
    PredictRequest, PredictResponse,
    MetricsResponse, HealthResponse, ModelInfo,
)
from src.models.registry import list_models, load_model, load_default_model
from src.utils.config import get_settings, Settings
from src.utils.logger import get_logger

log    = get_logger(__name__)
router = APIRouter()

# WDBC sklearn feature name order (used for DataFrame column alignment)
FEATURE_NAMES = [
    "mean radius","mean texture","mean perimeter","mean area",
    "mean smoothness","mean compactness","mean concavity",
    "mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error",
    "smoothness error","compactness error","concavity error",
    "concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area",
    "worst smoothness","worst compactness","worst concavity",
    "worst concave points","worst symmetry","worst fractal dimension",
]


# ─────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(cfg: Settings = Depends(get_settings)):
    saved = list_models()
    return HealthResponse(
        status       = "ok",
        app_name     = cfg.app_name,
        version      = cfg.app_version,
        environment  = cfg.app_env,
        models_ready = len(saved) > 0,
        model_count  = len(saved),
    )


# ─────────────────────────────────────────────────────────────────
# /models
# ─────────────────────────────────────────────────────────────────

@router.get("/models", response_model=List[ModelInfo], tags=["Models"])
def get_models():
    """List all saved models with metadata."""
    entries = list_models()
    if not entries:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run `python scripts/train.py` first.",
        )
    return [ModelInfo(**e) for e in entries]


# ─────────────────────────────────────────────────────────────────
# /metrics
# ─────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse, tags=["Models"])
def get_metrics():
    """Return performance metrics embedded in the manifest."""
    entries = list_models()
    result  = {}
    for e in entries:
        result[e["slug"]] = e.get("metrics", {})
    return MetricsResponse(models=result)


# ─────────────────────────────────────────────────────────────────
# /predict  (single)
# ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(body: PredictRequest, cfg: Settings = Depends(get_settings)):
    """
    Run inference on a single sample.
    Returns prediction label, confidence, and both class probabilities.
    """
    model_name = body.model_name or cfg.default_model

    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    features = np.array(body.to_feature_list()).reshape(1, -1)
    X        = pd.DataFrame(features, columns=FEATURE_NAMES)

    try:
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]        # [P(malignant), P(benign)]
    except Exception as exc:
        log.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    label      = "Benign" if pred == 1 else "Malignant"
    confidence = float(proba[pred])

    log.info(f"Predict → {label}  conf={confidence:.3f}  model={model_name}")

    return PredictResponse(
        prediction            = pred,
        label                 = label,
        confidence            = round(confidence, 4),
        malignant_probability = round(float(proba[0]), 4),
        benign_probability    = round(float(proba[1]), 4),
        model_used            = model_name,
    )


# ─────────────────────────────────────────────────────────────────
# /predict/batch  (CSV upload)
# ─────────────────────────────────────────────────────────────────

@router.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    file:       UploadFile = File(..., description="CSV with 30 WDBC feature columns"),
    model_name: str        = Query(None, description="Model slug (default: ensemble)"),
    cfg:        Settings   = Depends(get_settings),
):
    """
    Upload a CSV file with one row per sample (30 feature columns).
    Returns predictions for every row.
    """
    model_name = model_name or cfg.default_model

    # ── Validate file type ────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # ── Drop non-feature columns if present ───────────────────────
    for col in ["id", "diagnosis"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if df.shape[1] != 30:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 30 feature columns, got {df.shape[1]}.",
        )

    # ── Rename to sklearn names if needed ─────────────────────────
    df.columns = FEATURE_NAMES

    try:
        model    = load_model(model_name)
        preds    = model.predict(df).tolist()
        probas   = model.predict_proba(df).tolist()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")

    results = [
        {
            "row":                    i,
            "prediction":             int(p),
            "label":                  "Benign" if p == 1 else "Malignant",
            "malignant_probability":  round(pr[0], 4),
            "benign_probability":     round(pr[1], 4),
        }
        for i, (p, pr) in enumerate(zip(preds, probas))
    ]

    log.info(f"Batch predict → {len(results)} rows  model={model_name}")

    return {
        "model_used": model_name,
        "total_rows": len(results),
        "predictions": results,
    }