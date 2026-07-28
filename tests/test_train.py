import pandas as pd

from hotel_cancellation.train import temporal_split


def test_temporal_split_keeps_future_in_holdout():
    frame = pd.DataFrame({"arrival_date": pd.date_range("2020-01-01", periods=10)})
    train, test = temporal_split(frame, test_fraction=0.2)
    assert len(train) == 8
    assert train.arrival_date.max() < test.arrival_date.min()
