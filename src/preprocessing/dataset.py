"""Timeseries dataset."""

import logging

import pandas as pd
import numpy as np


def get_is_day(
    size: int,
    shift: int = 0,
    start_day: int = 16,
    end_day: int = 65,
    time_day: int = 96,
):
    """Get if the sun is up or not."""
    return np.array(
        [
            1.0 if start_day < ((step - shift) % time_day) < end_day else 0.0
            for step in range(size)
        ]
    )


def generate_dataset(data: np.ndarray, shift=20, make_diff: bool = True, add_is_day: bool = False):
    """Generate a dataset for time series models."""
    df_data = pd.Series(data) / 255
    # Compute the difference
    if make_diff:
        logging.info("Computing the difference...")
        diff_data = df_data.diff()
    else:
        logging.info("Not computing the difference...")
        diff_data = df_data

    DATA = pd.DataFrame(
        {
            **{f"{i}": -diff_data.shift(i) for i in range(shift)},
            "Y": -diff_data.shift(shift),
        }
    )
    if add_is_day:
        DATA["is_day"] = get_is_day(len(diff_data), shift=shift)

    DATA = DATA.dropna()

    features = DATA.drop(columns=["Y"])
    labels = DATA["Y"]
    return features.values, labels.values
