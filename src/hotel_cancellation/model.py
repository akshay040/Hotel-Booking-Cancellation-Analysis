"""Leakage-safe preprocessing and modelling pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build an interpretable baseline with all transformations fitted in-fold."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    classifier = LogisticRegression(
        max_iter=1_000, class_weight="balanced", random_state=random_state, solver="liblinear"
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
