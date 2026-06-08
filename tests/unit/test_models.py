"""tests/unit/test_models.py"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import numpy as np
from src.data.loader      import load_data
from src.data.splitter    import split_data
from src.models.trainer   import train_all_models
from src.models.evaluator import evaluate_models, best_model_name, results_to_dataframe


@pytest.fixture(scope="module")
def trained_models():
    bundle = load_data()
    split  = split_data(bundle, test_size=0.3, random_state=42)
    models = train_all_models(split.X_train, split.y_train)
    return models, split


class TestTrainer:
    def test_all_models_returned(self, trained_models):
        models, _ = trained_models
        expected = {"Random Forest", "kNN", "Naive Bayes",
                    "SVM", "Logistic Regression", "Ensemble (Soft Voting)"}
        assert set(models.keys()) == expected

    def test_models_can_predict(self, trained_models):
        models, split = trained_models
        for name, model in models.items():
            preds = model.predict(split.X_test)
            assert len(preds) == len(split.y_test), f"{name} wrong pred length"

    def test_models_have_predict_proba(self, trained_models):
        models, split = trained_models
        for name, model in models.items():
            proba = model.predict_proba(split.X_test)
            assert proba.shape == (len(split.y_test), 2), f"{name} wrong proba shape"
            assert np.allclose(proba.sum(axis=1), 1.0), f"{name} probas don't sum to 1"


class TestEvaluator:
    def test_metrics_range(self, trained_models):
        models, split = trained_models
        results = evaluate_models(
            models,
            split.X_test, split.y_test,
            split.X_train, split.y_train,
            cv_folds=3,   # fast for unit tests
        )
        for name, m in results.items():
            assert 0.0 <= m.accuracy  <= 1.0, f"{name} accuracy out of range"
            assert 0.0 <= m.recall    <= 1.0, f"{name} recall out of range"
            assert 0.0 <= m.precision <= 1.0, f"{name} precision out of range"
            assert 0.0 <= m.auc_roc   <= 1.0, f"{name} AUC out of range"

    def test_accuracy_above_threshold(self, trained_models):
        models, split = trained_models
        results = evaluate_models(
            models,
            split.X_test, split.y_test,
            split.X_train, split.y_train,
            cv_folds=3,
        )
        for name, m in results.items():
            assert m.accuracy >= 0.90, (
                f"{name} accuracy {m.accuracy:.4f} below 90% threshold"
            )

    def test_best_model_name(self, trained_models):
        models, split = trained_models
        results = evaluate_models(
            models,
            split.X_test, split.y_test,
            split.X_train, split.y_train,
            cv_folds=3,
        )
        best = best_model_name(results, "recall")
        assert best in results

    def test_results_to_dataframe(self, trained_models):
        models, split = trained_models
        results = evaluate_models(
            models,
            split.X_test, split.y_test,
            split.X_train, split.y_train,
            cv_folds=3,
        )
        df = results_to_dataframe(results)
        assert "accuracy" in df.columns
        assert "recall" in df.columns
        assert len(df) == len(models)