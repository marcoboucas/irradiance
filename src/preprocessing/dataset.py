"""Timeseries dataset."""

import logging
from typing import Optional

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
    return pd.Series([
            1.0 if start_day < ((step - shift) % time_day) < end_day else 0.0
            for step in range(size)
        ]).shift(shift).dropna().to_numpy()



def generate_dataset(
    data: np.ndarray,
    shift=20,
    make_diff: bool = True,
    remove_some_0: bool = True
):
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
            **{f"{i}": diff_data.shift(i) for i in range(shift)},
            "Y": diff_data.shift(shift),
        }
    )

    DATA = DATA.dropna()

    features = DATA.drop(columns=["Y"])
    labels = DATA["Y"]

    dataset_X, dataset_Y = features.values, labels.values

    if remove_some_0:
        indexes = ~((dataset_Y == 0) * (np.random.random(dataset_Y.shape) > 0.1))
        dataset_X = dataset_X[indexes]
        dataset_Y = dataset_Y[indexes]
    else:
        indexes = None

    return dataset_X, dataset_Y, indexes
