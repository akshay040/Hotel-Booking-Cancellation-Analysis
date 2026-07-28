"""Feature contract and validation used by training and inference."""

from __future__ import annotations

import pandas as pd

TARGET = "is_canceled"
DATE_COLUMN = "arrival_date"

# Only information plausibly available when a reservation is made is included.
# Outcome-derived reservation status/date and assigned room are deliberately excluded.
NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy with parseable dates and a binary target."""
    required = set(FEATURES + [TARGET, DATE_COLUMN])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    clean = frame.copy()
    clean[DATE_COLUMN] = pd.to_datetime(clean[DATE_COLUMN], errors="coerce")
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    clean = clean.dropna(subset=[DATE_COLUMN, TARGET])
    if not set(clean[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_canceled must contain only 0 and 1")
    if clean.empty:
        raise ValueError("No valid rows remain after validation")
    return clean
