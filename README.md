# Hotel Booking Cancellation Risk

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

An end-to-end, leakage-aware machine-learning portfolio project for estimating whether a hotel booking will be cancelled. It turns the original exploratory notebook into a tested Python package, reproducible training command, explainable evaluation artifact, and deployable Streamlit application.

> **Responsible-use note:** this is a decision-support demonstration. A risk score should prompt proportionate, routine follow-up—not automatic cancellation, pricing, or adverse treatment.

## Business problem

Cancellations make occupancy and staffing forecasts harder. The model estimates cancellation probability from information plausibly available around booking time. A hotel could use the estimate to prioritise confirmation messages, provided it validates the model on its own recent data and monitors guest impact.

## Architecture

```text
CSV → schema validation → chronological split
    → in-pipeline imputation + scaling/one-hot encoding
    → class-balanced logistic regression
    → holdout metrics + permutation importance + model artifact
    → Streamlit inference (same fitted pipeline)
```

The production workflow lives in `src/hotel_cancellation`; the notebook remains historical analysis rather than the deployment source of truth.

## Leakage and evaluation design

- The newest 20% of arrivals are held out. This more closely represents predicting future bookings than a random split.
- Every learned preprocessing step is inside the scikit-learn pipeline and is fitted only on training data.
- Outcome-derived `reservation_status` and `reservation_status_date` are excluded. `assigned_room_type` is also excluded because assignment timing may make it unavailable at scoring time.
- Metrics include ROC-AUC, average precision, accuracy, precision, recall, F1, Brier score, and the confusion matrix at a documented 0.50 demonstration threshold.
- Class balancing changes the probability distribution; calibration and threshold selection should be revisited using local intervention costs before operational use.
- Model-agnostic permutation importance is calculated against the untouched chronological holdout. Importance describes model reliance, not causation.

The old README reported notebook metrics from a different experimental setup. They are not presented as results of this pipeline. Run training to generate truthful, machine-readable metrics for the current code and environment in `artifacts/metrics.json`; generated artifacts are intentionally not committed.

## Quick start

Python 3.10+ is supported (CI uses 3.11).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# Tests and static analysis
ruff check .
pytest

# Train and evaluate
python -m hotel_cancellation.train \
  --data hotel_bookings_cleaned_enhanced.csv \
  --output artifacts

# Explore generated evidence
cat artifacts/metrics.json
head artifacts/permutation_importance.csv

# Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. If an artifact is absent, it gives an explicit training instruction rather than crashing with an opaque file error.

## Docker

```bash
docker build -t hotel-cancellation .
docker run --rm -p 8501:8501 hotel-cancellation
```

The image installs pinned runtime dependencies, runs as an unprivileged user, trains from the packaged dataset at startup, and exposes a Streamlit health check. For a real release, train in a controlled pipeline, version the dataset/model together, and deploy an immutable reviewed artifact instead of training at container start.

## Repository map

| Path | Purpose |
|---|---|
| `src/hotel_cancellation/features.py` | Feature contract and input validation |
| `src/hotel_cancellation/model.py` | Reusable preprocessing/model pipeline |
| `src/hotel_cancellation/train.py` | Temporal training, evaluation, explainability, serialization, logging |
| `src/hotel_cancellation/evaluate.py` | Central metric computation |
| `app.py` | Streamlit scoring interface and model limitations |
| `tests/` | Unit tests for schema, unseen categories, metrics, and splitting |
| `.github/workflows/ci.yml` | Lint, tests, full training smoke test, artifact upload, Docker build |
| `Hotel_Booking_Cancellation_Analysis.ipynb` | Original exploratory work (not production inference code) |

## Dataset and limitations

The included cleaned CSV derives from the [Hotel Booking Demand dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191), describing two Portuguese hotels and arrivals from 2015–2017. Important limitations:

- performance may not transfer across hotels, countries, booking systems, or time;
- historic policies and channel behaviour can become stale (concept drift);
- country can proxy geography or socioeconomic factors, so impact must be reviewed and its inclusion reconsidered for each use case;
- the dataset does not establish why a booking was cancelled, so feature importance is not causal;
- the default threshold is not a business recommendation.

Before production: define intervention costs, assess calibration and subgroup errors, use a time-based backtest over several periods, add drift/performance monitoring, obtain privacy and legal review, and document a retraining/rollback process.

## Interview walkthrough

1. **Why a pipeline?** It prevents preprocessing leakage and guarantees training/inference parity, including safe handling of unseen categories.
2. **Why chronological holdout?** Deployment predicts later bookings; a random split can overstate performance when patterns change over time.
3. **Why logistic regression?** It is a strong, explainable baseline. Complexity should be earned through repeated time-based validation, not selected from one test set.
4. **Why more than accuracy?** Cancellation classes and intervention costs differ. Recall measures missed cancellations, precision measures wasted follow-ups, ROC-AUC measures ranking, average precision focuses on the positive class, and Brier score assesses probability error.
5. **Why permutation importance?** It measures the drop in holdout ROC-AUC after disrupting a feature and works on the whole pipeline. Correlated predictors can share or mask importance, and no importance is causal.
6. **What would you improve next?** Rolling-window validation, threshold/cost analysis, calibration curves, subgroup review, experiment tracking, artifact/data versioning, and production drift monitoring.

## Reproducibility

- Runtime and development dependencies are separated and pinned in `pyproject.toml`.
- Randomised model and importance operations use an explicit seed.
- CI recreates installation, linting, unit tests, complete training, and a Docker build on every pull request.
- Logs record sample sizes, output location, and holdout ROC-AUC without logging guest-level data.
