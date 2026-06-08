"""tests/unit/test_data.py"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
import pandas as pd
from src.data.loader   import load_data, DataBundle
from src.data.splitter import split_data


class TestLoader:
    def test_sklearn_load(self):
        bundle = load_data(source="sklearn")
        assert isinstance(bundle.X, pd.DataFrame)
        assert isinstance(bundle.y, pd.Series)
        assert bundle.n_samples == 569
        assert bundle.n_features == 30

    def test_class_distribution(self):
        bundle = load_data()
        dist = bundle.class_distribution
        assert dist[1] == 357   # benign
        assert dist[0] == 212   # malignant

    def test_no_missing_values(self):
        bundle = load_data()
        assert bundle.X.isnull().sum().sum() == 0

    def test_invalid_source(self):
        with pytest.raises(ValueError):
            load_data(source="unknown")

    def test_csv_source_requires_path(self):
        with pytest.raises(ValueError):
            load_data(source="csv")


class TestSplitter:
    def setup_method(self):
        self.bundle = load_data()

    def test_split_sizes(self):
        split = split_data(self.bundle, test_size=0.3)
        assert split.train_size + split.test_size == 569

    def test_stratification(self):
        split = split_data(self.bundle, test_size=0.3)
        train_ratio = split.y_train.mean()
        test_ratio  = split.y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05   # within 5%

    def test_reproducible(self):
        s1 = split_data(self.bundle, random_state=42)
        s2 = split_data(self.bundle, random_state=42)
        assert (s1.X_train.values == s2.X_train.values).all()