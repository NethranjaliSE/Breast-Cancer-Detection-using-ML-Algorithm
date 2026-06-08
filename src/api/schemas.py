"""
src/api/schemas.py
------------------
Pydantic v2 request / response models for the FastAPI layer.
All 30 WDBC features are declared with default=0.0 so the API
can be tested with partial payloads during development.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Single-sample prediction request.
    Accepts either a named-feature dict OR a raw 30-value list.
    """
    # Named features (preferred)
    mean_radius:             float = Field(0.0, ge=0)
    mean_texture:            float = Field(0.0, ge=0)
    mean_perimeter:          float = Field(0.0, ge=0)
    mean_area:               float = Field(0.0, ge=0)
    mean_smoothness:         float = Field(0.0, ge=0)
    mean_compactness:        float = Field(0.0, ge=0)
    mean_concavity:          float = Field(0.0, ge=0)
    mean_concave_points:     float = Field(0.0, ge=0)
    mean_symmetry:           float = Field(0.0, ge=0)
    mean_fractal_dimension:  float = Field(0.0, ge=0)
    se_radius:               float = Field(0.0, ge=0)
    se_texture:              float = Field(0.0, ge=0)
    se_perimeter:            float = Field(0.0, ge=0)
    se_area:                 float = Field(0.0, ge=0)
    se_smoothness:           float = Field(0.0, ge=0)
    se_compactness:          float = Field(0.0, ge=0)
    se_concavity:            float = Field(0.0, ge=0)
    se_concave_points:       float = Field(0.0, ge=0)
    se_symmetry:             float = Field(0.0, ge=0)
    se_fractal_dimension:    float = Field(0.0, ge=0)
    worst_radius:            float = Field(0.0, ge=0)
    worst_texture:           float = Field(0.0, ge=0)
    worst_perimeter:         float = Field(0.0, ge=0)
    worst_area:              float = Field(0.0, ge=0)
    worst_smoothness:        float = Field(0.0, ge=0)
    worst_compactness:       float = Field(0.0, ge=0)
    worst_concavity:         float = Field(0.0, ge=0)
    worst_concave_points:    float = Field(0.0, ge=0)
    worst_symmetry:          float = Field(0.0, ge=0)
    worst_fractal_dimension: float = Field(0.0, ge=0)

    # Optional: override which model to use
    model_name: Optional[str] = Field(None, description="Model slug to use (default: ensemble)")

    def to_feature_list(self) -> List[float]:
        """Return features in WDBC column order (excluding model_name)."""
        return [
            self.mean_radius, self.mean_texture, self.mean_perimeter,
            self.mean_area, self.mean_smoothness, self.mean_compactness,
            self.mean_concavity, self.mean_concave_points, self.mean_symmetry,
            self.mean_fractal_dimension,
            self.se_radius, self.se_texture, self.se_perimeter,
            self.se_area, self.se_smoothness, self.se_compactness,
            self.se_concavity, self.se_concave_points, self.se_symmetry,
            self.se_fractal_dimension,
            self.worst_radius, self.worst_texture, self.worst_perimeter,
            self.worst_area, self.worst_smoothness, self.worst_compactness,
            self.worst_concavity, self.worst_concave_points, self.worst_symmetry,
            self.worst_fractal_dimension,
        ]

    model_config = {"json_schema_extra": {
        "example": {
            "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
            "mean_area": 1001.0,  "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
            "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
            "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.07871,
            "se_radius": 1.095, "se_texture": 0.9053, "se_perimeter": 8.589,
            "se_area": 153.4, "se_smoothness": 0.006399, "se_compactness": 0.04904,
            "se_concavity": 0.05373, "se_concave_points": 0.01587,
            "se_symmetry": 0.03003, "se_fractal_dimension": 0.006193,
            "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6,
            "worst_area": 2019.0, "worst_smoothness": 0.1622, "worst_compactness": 0.6656,
            "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
            "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189,
        }
    }}


# ─────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    prediction:   int    = Field(..., description="0=Malignant  1=Benign")
    label:        str    = Field(..., description="'Malignant' or 'Benign'")
    confidence:   float  = Field(..., description="Probability of predicted class (0-1)")
    malignant_probability: float
    benign_probability:    float
    model_used:   str


class ModelInfo(BaseModel):
    slug:     str
    size_kb:  float
    saved_at: str
    metrics:  Optional[dict] = None


class MetricsResponse(BaseModel):
    models: Dict[str, dict]


class HealthResponse(BaseModel):
    status:       str
    app_name:     str
    version:      str
    environment:  str
    models_ready: bool
    model_count:  int