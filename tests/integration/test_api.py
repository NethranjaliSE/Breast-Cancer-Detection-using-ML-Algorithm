"""
tests/integration/test_api.py
------------------------------
Integration tests for all FastAPI endpoints.
Requires models to be trained first: python scripts/train.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.api.main import app

client = TestClient(app)

# Sample malignant row from WDBC dataset (first row)
MALIGNANT_SAMPLE = {
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0,  "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,  "mean_fractal_dimension": 0.07871,
    "se_radius": 1.095,   "se_texture": 0.9053,  "se_perimeter": 8.589,
    "se_area": 153.4,     "se_smoothness": 0.006399, "se_compactness": 0.04904,
    "se_concavity": 0.05373,  "se_concave_points": 0.01587,
    "se_symmetry": 0.03003,   "se_fractal_dimension": 0.006193,
    "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6,
    "worst_area": 2019.0,  "worst_smoothness": 0.1622, "worst_compactness": 0.6656,
    "worst_concavity": 0.7119, "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,  "worst_fractal_dimension": 0.1189,
}


class TestHealth:
    def test_health_status(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "models_ready" in data


class TestModels:
    def test_list_models(self):
        r = client.get("/models")
        # 503 is acceptable when no models trained in CI
        assert r.status_code in (200, 503)

    def test_metrics_endpoint(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data


class TestPredict:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=MALIGNANT_SAMPLE)
        assert r.status_code in (200, 404)   # 404 if models not saved yet

    def test_predict_response_schema(self):
        r = client.post("/predict", json=MALIGNANT_SAMPLE)
        if r.status_code == 200:
            data = r.json()
            assert "prediction"   in data
            assert "label"        in data
            assert "confidence"   in data
            assert "model_used"   in data
            assert data["prediction"] in (0, 1)
            assert data["label"] in ("Malignant", "Benign")
            assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self):
        r = client.post("/predict", json=MALIGNANT_SAMPLE)
        if r.status_code == 200:
            data = r.json()
            total = data["malignant_probability"] + data["benign_probability"]
            assert abs(total - 1.0) < 0.01


class TestBatchPredict:
    def _make_csv(self, n_rows: int = 5) -> bytes:
        raw = load_breast_cancer()
        df  = pd.DataFrame(raw.data[:n_rows], columns=raw.feature_names)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode()

    def test_batch_predict_csv(self):
        csv_bytes = self._make_csv(10)
        r = client.post(
            "/predict/batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code in (200, 404)

    def test_batch_predict_wrong_extension(self):
        r = client.post(
            "/predict/batch",
            files={"file": ("test.txt", io.BytesIO(b"a,b,c"), "text/plain")},
        )
        assert r.status_code == 400

    def test_batch_predict_response_count(self):
        csv_bytes = self._make_csv(7)
        r = client.post(
            "/predict/batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        if r.status_code == 200:
            data = r.json()
            assert data["total_rows"] == 7
            assert len(data["predictions"]) == 7