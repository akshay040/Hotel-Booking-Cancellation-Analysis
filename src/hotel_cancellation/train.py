"""Command-line, reproducible temporal training workflow."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import classification_metrics
from .features import DATE_COLUMN, FEATURES, TARGET, validate_training_frame
from .model import build_pipeline

LOGGER = logging.getLogger(__name__)


def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Hold out the newest bookings, mirroring prospective deployment."""
    ordered = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    unique_dates = ordered[DATE_COLUMN].drop_duplicates().sort_values()
    boundary = max(1, int(len(unique_dates) * (1 - test_fraction)))
    if boundary >= len(unique_dates):
        raise ValueError("At least two distinct arrival dates are required for a temporal split")
    first_test_date = unique_dates.iloc[boundary]
    return ordered[ordered[DATE_COLUMN] < first_test_date], ordered[
        ordered[DATE_COLUMN] >= first_test_date
    ]


def train(data_path: Path, output_dir: Path, random_state: int = 42) -> dict:
    frame = validate_training_frame(pd.read_csv(data_path, low_memory=False))
    train_frame, test_frame = temporal_split(frame)
    LOGGER.info("Training on %d rows; evaluating on %d rows", len(train_frame), len(test_frame))

    pipeline = build_pipeline(random_state)
    pipeline.fit(train_frame[FEATURES], train_frame[TARGET])
    probability = pipeline.predict_proba(test_frame[FEATURES])[:, 1]
    metrics = classification_metrics(test_frame[TARGET], probability)
    metrics.update(
        {
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "train_end": train_frame[DATE_COLUMN].max().date().isoformat(),
            "test_start": test_frame[DATE_COLUMN].min().date().isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "LogisticRegression(class_weight='balanced')",
        }
    )

    importance = permutation_importance(
        pipeline,
        test_frame[FEATURES],
        test_frame[TARGET],
        scoring="roc_auc",
        n_repeats=3,
        random_state=random_state,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)
    LOGGER.info("Artifacts saved to %s; holdout ROC-AUC %.4f", output_dir, metrics["roc_auc"])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("hotel_bookings_cleaned_enhanced.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    train(args.data, args.output, args.random_state)


if __name__ == "__main__":
    main()
