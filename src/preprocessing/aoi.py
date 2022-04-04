"""AOI research."""

import pandas as pd
from src.utils import convert_pixel_to_coordinates
from pvlib import pvsystem, location, modelchain, irradiance

def get_aoi_year_location(loc_x: int, loc_y: int, year: int, shift:int = 0):
    """Get the AOI for one year at a specific location."""
    # Convert the pixel locations to coordinates
    lat, long = convert_pixel_to_coordinates(loc_x, loc_y)
    loc = location.Location(lat, long)

    times = pd.date_range(
        f"{year}-01-01 00:00", f"{year}-12-31 23:44", freq="15min", tz="Etc/GMT+1"
    )
    weather = loc.get_solarposition(times)
    aoi = irradiance.aoi(0, 0, weather.apparent_zenith, weather.azimuth)
    return pd.Series(aoi).shift(shift).dropna().to_numpy()