"""Data functions."""

import os
import h5py

def load_data(data_path: str, year: int):
    """Load the data for one year."""
    f = h5py.File(os.path.join(data_path, f"ghi_{year}.h5"), 'r')
    return f['/set']