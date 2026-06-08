"""
src/models/evaluator.py
------------------------
Compute, log, and return all performance metrics for trained models.

Metrics (from the paper)
------------------------
• Accuracy
• Precision
• Recall (sensitivity) — most critical for cancer detection
• F1 Score
• AUC-ROC
• 10-fold Cross-Validation accuracy (mean ± std)

Also generates:
• Per-model confusion matrix
• Classification report
• ROC curve data (for plotting / API)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Result data-classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    model_name:       str
    accuracy:         float
    precision:        float
    recall:           float
    f1_score:         float
    auc_roc:          float
    cv_mean:          float
    cv_std:           float
    confusion_matrix: List[List[int]]
    report:           str = field(repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("report")   # keep dict compact for API responses
        return d

    def summary_line(self) -> str:
        return (
            f"{self.model_name:<30} "
            f"Acc={self.accuracy:.4f}  Prec={self.precision:.4f}  "
            f"Rec={self.recall:.4f}  F1={self.f1_score:.4f}  "
            f"AUC={self.auc_roc:.4f}  CV={self.cv_mean:.4f}±{self.cv_std:.4f}"
        )


EvalResults = Dict[str, ModelMetrics]


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def evaluate_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: Optional[int] = None,
) -> EvalResults:
    """
    Evaluate every model and return a dict of ModelMetrics.

    Parameters
    ----------
    models   : dict returned by trainer.train_all_models()
    X_test   : held-out feature matrix
    y_test   : held-out labels
    X_train  : training features (for cross-validation)
    y_train  : training labels
    cv_folds : number of CV folds (default from settings)

    Returns
    -------
    results : {model_name: ModelMetrics}
    """
    cfg      = get_settings()
    cv_folds = cv_folds or cfg.cv_folds
    skf      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)

    log.info("=" * 60)
    log.info(f"  Evaluating {len(models)} models  (CV folds={cv_folds})")
    log.info("=" * 60)

    results: EvalResults = {}

    for name, model in models.items():
        log.info(f"  Evaluating  {name} …")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)
        auc  = roc_auc_score  (y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred).tolist()
        rep  = classification_report(
                   y_test, y_pred,
                   target_names=["Malignant", "Benign"])

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

        metrics = ModelMetrics(
            model_name       = name,
            accuracy         = round(acc,        4),
            precision        = round(prec,       4),
            recall           = round(rec,        4),
            f1_score         = round(f1,         4),
            auc_roc          = round(auc,        4),
            cv_mean          = round(cv_scores.mean(), 4),
            cv_std           = round(cv_scores.std(),  4),
            confusion_matrix = cm,
            report           = rep,
        )
        results[name] = metrics
        log.info(f"  ✔ {metrics.summary_line()}")

    _log_winner(results)
    return results


def results_to_dataframe(results: EvalResults) -> pd.DataFrame:
    """Convert EvalResults to a tidy DataFrame for reporting."""
    rows = [m.to_dict() for m in results.values()]
    df   = pd.DataFrame(rows).set_index("model_name")
    return df.sort_values("recall", ascending=False)   # recall first for cancer


def best_model_name(results: EvalResults, metric: str = "recall") -> str:
    """Return the name of the best-performing model by a given metric."""
    return max(results, key=lambda n: getattr(results[n], metric))


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _log_winner(results: EvalResults) -> None:
    best_acc = best_model_name(results, "accuracy")
    best_rec = best_model_name(results, "recall")
    log.info("─" * 60)
    log.info(f"  🏆 Best Accuracy : {best_acc}  ({results[best_acc].accuracy:.4f})")
    log.info(f"  🎯 Best Recall   : {best_rec}  ({results[best_rec].recall:.4f})")
    log.info("─" * 60)