import pytest

from hotel_cancellation.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from hotel_cancellation.inference import booking_frame, threshold_message


@pytest.fixture
def valid_booking():
    values = {feature: 0 for feature in NUMERIC_FEATURES}
    values.update({feature: "known" for feature in CATEGORICAL_FEATURES})
    values.update({"adults": 2, "stays_in_week_nights": 1, "adr": 100})
    return values


def test_booking_frame_preserves_complete_feature_contract(valid_booking):
    frame = booking_frame(valid_booking)
    assert frame.columns.tolist() == FEATURES
    assert frame.iloc[0].to_dict() == valid_booking


@pytest.mark.parametrize("feature", FEATURES)
def test_every_prediction_input_is_forwarded(feature, valid_booking):
    valid_booking[feature] = "changed" if feature in CATEGORICAL_FEATURES else 3
    if feature in {"adults", "children", "babies"}:
        valid_booking["adults"] = 2
    if feature in {"stays_in_weekend_nights", "stays_in_week_nights"}:
        valid_booking["stays_in_week_nights"] = 1
    assert booking_frame(valid_booking).iloc[0][feature] == valid_booking[feature]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"adults": 0, "children": 0, "babies": 0}, "at least one guest"),
        ({"stays_in_week_nights": 0, "stays_in_weekend_nights": 0}, "at least one night"),
        ({"adr": -1}, "cannot be negative"),
    ],
)
def test_booking_frame_rejects_unrealistic_inputs(valid_booking, updates, message):
    valid_booking.update(updates)
    with pytest.raises(ValueError, match=message):
        booking_frame(valid_booking)


def test_booking_frame_reports_missing_inputs(valid_booking):
    del valid_booking[FEATURES[0]]
    with pytest.raises(ValueError, match="Missing prediction inputs"):
        booking_frame(valid_booking)


@pytest.mark.parametrize(
    ("probability", "label"),
    [(0.0, "Below demo threshold"), (0.5, "Above demo threshold"), (1.0, "Above demo threshold")],
)
def test_threshold_boundaries(probability, label):
    assert threshold_message(probability)[0] == label


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_threshold_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_message(probability)
