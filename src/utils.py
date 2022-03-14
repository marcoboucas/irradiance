"""Utils."""

from src import config


def convert_pixel_to_coordinates(
    loc_x: int,
    loc_y: int,
    min_lat: int = -37.1,
    max_lat: int = 40.1,
    min_lon: int = -20.1,
    max_lon: int = 54.1,
):
    """Convert pixels to coordinates.

    Latitude: between the poles
    Longitude: Equator
    """
    lat = min_lat + (max_lat - min_lat) * loc_y / config.RAW_DATA_HEIGHT
    lon = min_lon + (max_lon - min_lon) * loc_x / config.RAW_DATA_WIDTH
    return lat, lon
