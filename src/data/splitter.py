"""
src/data/splitter.py
--------------------
Reproducible, stratified train/test splitting.
Mirrors the paper's 398/171 split strategy.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataBundle
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SplitBundle:
    X_train: pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_test:  pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def split_data(
    bundle: DataBundle,
    test_size: float = 0.30,
    random_state: int = 42,
) -> SplitBundle:
    """
    Stratified train/test split.

    Parameters
    ----------
    bundle       : DataBundle from loader.load_data()
    test_size    : fraction held out for testing  (default 0.30 ≈ 171/569)
    random_state : RNG seed for reproducibility

    Returns
    -------
    SplitBundle
    """
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_state,
        stratify=bundle.y,          # preserve class ratio
    )

    log.info(
        f"Split  train={len(X_train)}  test={len(X_test)}  "
        f"test_ratio={test_size:.2f}  stratified=True"
    )
    return SplitBundle(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )