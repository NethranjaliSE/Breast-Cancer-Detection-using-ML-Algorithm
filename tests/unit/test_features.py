"""tests/unit/test_features.py"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import pytest
from src.data.loader          import load_data
from src.features.preprocessor import BreastCancerPreprocessor


@pytest.fixture(scope="module")
def bundle():
    return load_data()


class TestPreprocessor:
    def test_fit_transform(self, bundle):
        pre = BreastCancerPreprocessor()
        X_scaled = pre.fit_transform(bundle.X)
        assert X_scaled.shape == bundle.X.shape

    def test_zero_mean_after_scaling(self, bundle):
        pre = BreastCancerPreprocessor()
        X_scaled = pre.fit_transform(bundle.X)
        means = X_scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-8)

    def test_unit_variance_after_scaling(self, bundle):
        pre = BreastCancerPreprocessor()
        X_scaled = pre.fit_transform(bundle.X)
        stds = X_scaled.std(axis=0)
        assert np.allclose(stds, 1, atol=1e-6)

    def test_transform_without_fit_raises(self, bundle):
        pre = BreastCancerPreprocessor()
        with pytest.raises(RuntimeError):
            pre.transform(bundle.X)

    def test_feature_stats(self, bundle):
        pre = BreastCancerPreprocessor()
        pre.fit(bundle.X)
        stats = pre.feature_stats()
        assert isinstance(stats, pd.DataFrame)
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert len(stats) == 30