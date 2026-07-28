import pandas as pd
import pytest

from hotel_cancellation.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from hotel_cancellation.train import temporal_split, train


def test_temporal_split_keeps_future_in_holdout():
    frame = pd.DataFrame({"arrival_date": pd.date_range("2020-01-01", periods=10)})
    train, test = temporal_split(frame, test_fraction=0.2)
    assert len(train) == 8
    assert train.arrival_date.max() < test.arrival_date.min()


def test_training_smoke_writes_loadable_artifacts(tmp_path):
    rows = []
    for index, arrival_date in enumerate(pd.date_range("2020-01-01", periods=30)):
        row = {feature: index % 5 for feature in NUMERIC_FEATURES}
        row.update({feature: f"category-{index % 2}" for feature in CATEGORICAL_FEATURES})
        row.update({"arrival_date": arrival_date, "is_canceled": index % 2})
        rows.append(row)

    data_path = tmp_path / "bookings.csv"
    output_path = tmp_path / "artifacts"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    metrics = train(data_path, output_path, importance_repeats=1)

    assert 0 <= metrics["roc_auc"] <= 1
    assert (output_path / "model.joblib").is_file()
    assert (output_path / "metrics.json").is_file()
    assert (output_path / "permutation_importance.csv").is_file()


def test_training_rejects_invalid_importance_repeats(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        train(tmp_path / "missing.csv", tmp_path, importance_repeats=0)
