"""
src/models/trainer.py
----------------------
Train ALL classifiers — individual + Soft-Voting Ensemble — in ONE method.

Design principles
-----------------
• Every estimator is wrapped in a Pipeline(StandardScaler → clf) so the
  same interface is used at training and inference with no data leakage.
• Config is loaded from settings.yaml / env vars via get_settings().
• Returns a dict of {name: fitted_pipeline} consumed by evaluator & registry.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.ensemble  import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import GaussianNB
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm          import SVC

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)
Settings = get_settings()

# Public type alias
ModelDict = Dict[str, Pipeline]


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ModelDict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  Train every classifier + Soft-Voting Ensemble in one call. │
    └─────────────────────────────────────────────────────────────┘

    Models trained
    --------------
    1. Random Forest       (72 estimators — paper value)
    2. kNN                 (k=5, Euclidean)
    3. Naive Bayes         (GaussianNB)
    4. SVM                 (RBF kernel, probability=True)
    5. Logistic Regression (lbfgs, L2)
    6. Ensemble            (Soft Voting over all 5 above)

    Parameters
    ----------
    X_train : pd.DataFrame  (n_samples, 30)
    y_train : pd.Series     0=malignant  1=benign

    Returns
    -------
    models : dict  {model_name: fitted_sklearn_Pipeline}
    """
    log.info("=" * 60)
    log.info("  Starting model training …")
    log.info("=" * 60)

    # ── 1. Define estimators (settings-driven hyperparams) ─────
    estimator_map = _build_estimators()

    # ── 2. Wrap each in a Pipeline with StandardScaler ─────────
    pipelines: ModelDict = {}
    for name, clf in estimator_map.items():
        pipelines[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])

    # ── 3. Fit individual models ────────────────────────────────
    for name, pipe in pipelines.items():
        log.info(f"  Training  {name} …")
        pipe.fit(X_train, y_train)
        log.info(f"  ✔ {name} trained")

    # ── 4. Build Soft-Voting Ensemble over all trained pipelines ─
    # VotingClassifier needs (label, estimator) pairs.
    # Each sub-pipeline already contains its own scaler.
    ensemble = VotingClassifier(
        estimators=[
            (name.lower().replace(" ", "_"), pipe)
            for name, pipe in pipelines.items()
        ],
        voting="soft",      # average class probabilities
        weights=None,       # equal weight; tune as needed
    )
    log.info("  Training  Ensemble (Soft Voting) …")
    ensemble.fit(X_train, y_train)
    log.info("  ✔ Ensemble (Soft Voting) trained")

    all_models: ModelDict = {**pipelines, "Ensemble (Soft Voting)": ensemble}

    log.info(f"Training complete — {len(all_models)} models ready.")
    return all_models


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _build_estimators() -> dict:
    """Instantiate all sklearn estimators from config/settings."""
    cfg = get_settings()
    rs  = cfg.random_state

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=72,          # paper value
            max_features="sqrt",
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            weights="uniform",
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,         # required for soft-voting & AUC
            random_state=rs,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=10_000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=rs,
        ),
    }