"""
src/features/preprocessor.py
------------------------------
Feature engineering & preprocessing utilities.

The WDBC dataset is already numeric with no missing values (per paper),
so the main step is StandardScaler.  This module wraps it in a reusable
class so the same fitted scaler can be saved and applied at inference time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import joblib

from src.utils.logger import get_logger

log = get_logger(__name__)


class BreastCancerPreprocessor:
    """
    Stateful preprocessor that fits on training data and transforms
    both train and test splits identically.

    Note: Each model Pipeline already wraps its own StandardScaler,
    so this class is provided for standalone preprocessing, batch
    inference scripts, and future feature-engineering extensions.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False
        self.feature_names_: list[str] = []

    # ── Fit ───────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame) -> "BreastCancerPreprocessor":
        """Fit the scaler on training features."""
        self._scaler.fit(X)
        self.feature_names_ = list(X.columns)
        self._fitted = True
        log.debug(f"Preprocessor fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self

    # ── Transform ─────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features using fitted parameters."""
        self._check_fitted()
        self._check_columns(X)
        return self._scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── Inverse ───────────────────────────────────────────────────
    def inverse_transform(self, X_scaled: np.ndarray) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame(
            self._scaler.inverse_transform(X_scaled),
            columns=self.feature_names_,
        )

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        log.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BreastCancerPreprocessor":
        obj = joblib.load(path)
        log.info(f"Preprocessor loaded ← {path}")
        return obj

    # ── Helpers ───────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call .fit() first.")

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = set(self.feature_names_) - set(X.columns)
        extra   = set(X.columns) - set(self.feature_names_)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        if extra:
            log.warning(f"Extra features in input (will be ignored): {extra}")

    # ── Feature statistics ─────────────────────────────────────────
    def feature_stats(self) -> pd.DataFrame:
        """Return a DataFrame with mean and std for each feature."""
        self._check_fitted()
        return pd.DataFrame({
            "feature": self.feature_names_,
            "mean":    self._scaler.mean_,
            "std":     np.sqrt(self._scaler.var_),
        })