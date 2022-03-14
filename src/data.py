"""Data functions."""

import os
import h5py

from src.config import DATA_FOLDER


def load_data(data_path: str, year: int):
    """Load the data for one year."""
    f = h5py.File(os.path.join(data_path, f"ghi_{year}.h5"), "r")
    return f["/set"]


def get_africa_mask():
    """Get africa mask (no oceans)."""
    data = load_data(DATA_FOLDER, 2020)[8] == 0
    return data
