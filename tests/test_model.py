import pandas as pd

from hotel_cancellation.evaluate import classification_metrics
from hotel_cancellation.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from hotel_cancellation.model import build_pipeline


def test_pipeline_handles_missing_and_unseen_categories():
    rows = []
    for index in range(20):
        row = {name: float(index) for name in NUMERIC_FEATURES}
        row.update({name: "known" for name in CATEGORICAL_FEATURES})
        rows.append(row)
    frame = pd.DataFrame(rows)[FEATURES]
    model = build_pipeline().fit(frame, [0, 1] * 10)
    unseen = frame.iloc[[0]].copy()
    unseen.loc[:, CATEGORICAL_FEATURES[0]] = "never-seen"
    unseen.loc[:, NUMERIC_FEATURES[0]] = None
    probability = model.predict_proba(unseen)[0, 1]
    assert 0 <= probability <= 1


def test_metrics_include_confusion_counts():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.4])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["accuracy"] == 0.5
