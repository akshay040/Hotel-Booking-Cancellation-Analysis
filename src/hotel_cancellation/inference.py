"""Inference validation and presentation helpers shared by the app and tests."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .features import FEATURES


def booking_frame(values: Mapping[str, object]) -> pd.DataFrame:
    """Validate one UI booking and return columns in the model's feature order."""
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing prediction inputs: {', '.join(missing)}")
    if sum(float(values[name]) for name in ("adults", "children", "babies")) <= 0:
        raise ValueError("A booking must include at least one guest")
    if (
        sum(float(values[name]) for name in ("stays_in_weekend_nights", "stays_in_week_nights"))
        <= 0
    ):
        raise ValueError("A booking must include at least one night")
    if float(values["adr"]) < 0:
        raise ValueError("Average daily rate cannot be negative")
    return pd.DataFrame([{feature: values[feature] for feature in FEATURES}])


def threshold_message(probability: float, threshold: float = 0.5) -> tuple[str, str]:
    """Return a neutral label and action based on the documented demo threshold."""
    if not 0 <= probability <= 1:
        raise ValueError("Predicted probability must be between 0 and 1")
    if probability >= threshold:
        return "Above demo threshold", "Consider a routine confirmation message."
    return "Below demo threshold", "No additional follow-up is suggested by the demo threshold."
