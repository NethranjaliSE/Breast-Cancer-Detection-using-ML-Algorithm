"""
src/data/loader.py
------------------
Responsible for loading and validating the WDBC dataset.

Supports three sources (configured via settings.yaml):
  • sklearn  – load directly from scikit-learn (default / zero-setup)
  • csv      – load from a local CSV file
  • url      – download from a remote URL

Returns a validated DataBundle (X, y, feature_names).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Expected WDBC feature count
EXPECTED_FEATURES = 30


@dataclass
class DataBundle:
    """Immutable container for the loaded dataset."""
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]

    def __post_init__(self) -> None:
        assert len(self.X) == len(self.y), "X and y length mismatch"
        assert len(self.feature_names) == self.X.shape[1], "Feature name count mismatch"

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def class_distribution(self) -> dict:
        return self.y.value_counts().to_dict()

    def summary(self) -> str:
        dist = self.class_distribution
        return (
            f"Samples={self.n_samples}  Features={self.n_features}  "
            f"Benign={dist.get(1, 0)}  Malignant={dist.get(0, 0)}"
        )


# ─────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────

def _load_from_sklearn() -> DataBundle:
    """Load the canonical WDBC dataset bundled with scikit-learn."""
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="diagnosis")   # 1=benign, 0=malignant
    log.debug("Loaded dataset from sklearn  (569 samples, 30 features)")
    return DataBundle(X=X, y=y, feature_names=list(raw.feature_names))


def _load_from_csv(path: str | Path) -> DataBundle:
    """
    Load from a local CSV.
    Expected columns: id, diagnosis (M/B), then 30 feature columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # Drop id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "diagnosis" not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' column (M or B)")

    y = df["diagnosis"].map({"B": 1, "M": 0}).rename("diagnosis")
    X = df.drop(columns=["diagnosis"])

    log.debug(f"Loaded dataset from CSV: {path}  ({len(df)} samples)")
    return DataBundle(X=X, y=y, feature_names=list(X.columns))


def _load_from_url(url: str) -> DataBundle:
    """Download CSV from a URL (e.g. UCI repo direct link)."""
    import urllib.request
    log.info(f"Downloading dataset from {url} …")
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode("utf-8")
    return _load_from_csv(io.StringIO(content))   # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_data(source: str = "sklearn",
              csv_path: Optional[str] = None,
              url: Optional[str] = None) -> DataBundle:
    """
    Load the WDBC dataset from the specified source and validate it.

    Parameters
    ----------
    source   : "sklearn" | "csv" | "url"
    csv_path : local file path (required when source="csv")
    url      : remote URL      (required when source="url")

    Returns
    -------
    DataBundle
    """
    log.info(f"Loading data  source={source}")

    if source == "sklearn":
        bundle = _load_from_sklearn()
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path must be supplied when source='csv'")
        bundle = _load_from_csv(csv_path)
    elif source == "url":
        if not url:
            raise ValueError("url must be supplied when source='url'")
        bundle = _load_from_url(url)
    else:
        raise ValueError(f"Unknown source '{source}'. Use: sklearn | csv | url")

    _validate(bundle)
    log.info(f"Data loaded   {bundle.summary()}")
    return bundle


def _validate(bundle: DataBundle) -> None:
    """Run basic sanity checks on the loaded dataset."""
    if bundle.n_samples == 0:
        raise ValueError("Dataset is empty.")
    if bundle.n_features != EXPECTED_FEATURES:
        log.warning(
            f"Expected {EXPECTED_FEATURES} features, got {bundle.n_features}. "
            "Results may differ from the paper."
        )
    if bundle.y.isnull().any():
        raise ValueError("Target column contains NaN values.")
    if bundle.X.isnull().any().any():
        missing = bundle.X.isnull().sum().sum()
        log.warning(f"Dataset contains {missing} missing values — will be handled in preprocessing.")
    log.debug("Data validation passed ✔")