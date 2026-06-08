"""
scripts/predict.py
-------------------
CLI tool for running batch predictions on a CSV file.

Usage
-----
python scripts/predict.py --input data/samples.csv --output predictions.csv
python scripts/predict.py --input data/samples.csv --model svm
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.models.registry import load_model, load_default_model
from src.utils.config    import get_settings
from src.utils.logger    import get_logger

log = get_logger("predict")

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch prediction on a CSV")
    p.add_argument("--input",  required=True, help="Path to input CSV")
    p.add_argument("--output", default="predictions.csv", help="Output CSV path")
    p.add_argument("--model",  default=None, help="Model slug (default: ensemble)")
    return p.parse_args()


def run_predict(args: argparse.Namespace) -> None:
    cfg   = get_settings()
    model = load_model(args.model or cfg.default_model)

    log.info(f"Loading input  {args.input}")
    df = pd.read_csv(args.input)

    # Drop non-feature columns
    for col in ["id", "diagnosis"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.columns = FEATURE_NAMES[:df.shape[1]]

    preds  = model.predict(df)
    probas = model.predict_proba(df)

    out = pd.DataFrame({
        "prediction":             preds,
        "label":                  ["Benign" if p == 1 else "Malignant" for p in preds],
        "malignant_probability":  probas[:, 0].round(4),
        "benign_probability":     probas[:, 1].round(4),
    })

    out.to_csv(args.output, index=False)
    log.info(f"Predictions saved → {args.output}  ({len(out)} rows)")
    log.info(f"  Benign:    {(out.prediction == 1).sum()}")
    log.info(f"  Malignant: {(out.prediction == 0).sum()}")


if __name__ == "__main__":
    run_predict(parse_args())