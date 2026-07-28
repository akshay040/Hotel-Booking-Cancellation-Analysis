import pandas as pd
import pytest

from hotel_cancellation.features import DATE_COLUMN, FEATURES, TARGET, validate_training_frame


def valid_frame():
    row = {feature: 0 for feature in FEATURES}
    row.update({DATE_COLUMN: "2017-01-01", TARGET: 1})
    return pd.DataFrame([row])


def test_validation_parses_date_and_target():
    result = validate_training_frame(valid_frame())
    assert pd.api.types.is_datetime64_any_dtype(result[DATE_COLUMN])
    assert result[TARGET].iloc[0] == 1


def test_validation_lists_missing_columns():
    with pytest.raises(ValueError, match="required columns"):
        validate_training_frame(pd.DataFrame({TARGET: [0]}))


def test_validation_rejects_non_binary_target():
    frame = valid_frame()
    frame[TARGET] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_training_frame(frame)
