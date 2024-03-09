import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely import Point

from mitools.google import (
    DummyResponse,
    create_dummy_response,
    meters_to_degree,
    nearby_search_request,
)


class TestDummyResponse(unittest.TestCase):

    latitude = 35.5
    longitude = 120.7
    radius = 200
    circle = gpd.GeoSeries([Point(latitude, longitude).buffer(meters_to_degree(radius, latitude))])

    query = nearby_search_request(circle, radius)

    print(query)

    
if __name__ == '__main__':
    unittest.main()
