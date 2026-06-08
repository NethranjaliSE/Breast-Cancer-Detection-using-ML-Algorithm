"""
scripts/train.py
-----------------
CLI entry point for training the full pipeline.

Usage
-----
python scripts/train.py                    # default settings
python scripts/train.py --test-size 0.2   # custom split
python scripts/train.py --no-plots        # skip matplotlib
"""
from __future__ import annotations

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore")

from src.data.loader   import load_data
from src.data.splitter import split_data
from src.models.trainer    import train_all_models
from src.models.evaluator  import evaluate_models, results_to_dataframe, best_model_name
from src.models.registry   import save_models
from src.utils.config  import get_settings
from src.utils.logger  import get_logger

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train breast cancer ML pipeline")
    p.add_argument("--test-size",    type=float, default=None,
                   help="Test split ratio (default from settings)")
    p.add_argument("--random-state", type=int,   default=None)
    p.add_argument("--cv-folds",     type=int,   default=None)
    p.add_argument("--no-plots",     action="store_true",
                   help="Skip generating matplotlib figures")
    p.add_argument("--source",       default="sklearn",
                   choices=["sklearn", "csv", "url"],
                   help="Dataset source (default: sklearn)")
    p.add_argument("--csv-path",     default=None,
                   help="Path to CSV file (when --source=csv)")
    return p.parse_args()


def run_training(args: argparse.Namespace) -> None:
    cfg = get_settings()

    test_size    = args.test_size    or cfg.test_size
    random_state = args.random_state or cfg.random_state
    cv_folds     = args.cv_folds     or cfg.cv_folds

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   Breast Cancer ML — Production Training Pipeline    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── 1. Load data ─────────────────────────────────────────────
    bundle = load_data(source=args.source, csv_path=args.csv_path)

    # ── 2. Split ─────────────────────────────────────────────────
    split  = split_data(bundle, test_size=test_size, random_state=random_state)

    # ── 3. Train ALL models in one call ──────────────────────────
    models = train_all_models(split.X_train, split.y_train)

    # ── 4. Evaluate ──────────────────────────────────────────────
    metrics = evaluate_models(
        models,
        split.X_test,  split.y_test,
        split.X_train, split.y_train,
        cv_folds=cv_folds,
    )

    # ── 5. Print summary table ────────────────────────────────────
    df = results_to_dataframe(metrics)
    log.info("\n" + df[["accuracy","precision","recall","f1_score","auc_roc"]].to_string())

    # ── 6. Save models + manifest ─────────────────────────────────
    save_models(models, metrics=metrics)

    # ── 7. Optional plots ────────────────────────────────────────
    if not args.no_plots:
        _generate_plots(models, metrics, split)

    best = best_model_name(metrics, "recall")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info(f"║  🏆 Best Recall  → {best:<34}║")
    log.info(f"║     Recall = {metrics[best].recall:.4f}"
             f"   Accuracy = {metrics[best].accuracy:.4f}{'':>22}║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("Training complete.  Start the API with:")
    log.info("  uvicorn src.api.main:app --reload --port 8000")


def _generate_plots(models, metrics, split) -> None:
    """Generate and save performance plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, roc_auc_score
        import pandas as pd

        cfg = get_settings()
        out = cfg.model_dir.parent / "plots"
        out.mkdir(exist_ok=True)

        names   = list(metrics.keys())
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        palette = sns.color_palette("Set2", len(names))

        # ── Bar chart ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(metric_keys), figsize=(22, 5))
        fig.suptitle("Breast Cancer Prediction — Performance Metrics", fontsize=13, fontweight="bold")
        for ax, key in zip(axes, metric_keys):
            vals = [getattr(metrics[n], key) for n in names]
            bars = ax.bar(range(len(names)), vals, color=palette, edgecolor="white")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7)
            ax.set_ylim(0.85, 1.01)
            ax.set_title(key.replace("_", " ").title(), fontweight="bold")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "performance_metrics.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── ROC curves ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, model), color in zip(models.items(), sns.color_palette("tab10", len(models))):
            y_prob = model.predict_proba(split.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(split.y_test, y_prob)
            auc = roc_auc_score(split.y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                    lw=2.5 if "Ensemble" in name else 1.5)
        ax.plot([0,1],[0,1],"k--", lw=0.8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

        log.info(f"Plots saved → {out}/")

    except ImportError:
        log.warning("matplotlib/seaborn not installed — skipping plots")
    except Exception as e:
        log.warning(f"Plot generation failed: {e}")


if __name__ == "__main__":
    run_training(parse_args())