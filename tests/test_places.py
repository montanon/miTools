import math
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase

import geopandas as gpd
from pandas import Series
from shapely import Point
from shapely.geometry import MultiPolygon, Polygon

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
from mitools.google.places import (
    GOOGLE_PLACES_API_KEY,
    QUERY_HEADERS,
    AccessibilityOptions,
    AddressComponent,
    CircleCenterInsidePolygon,
    CircleInsidePolygon,
    CircleIntersectsPolygon,
    CityGeojson,
    ConditionProtocol,
    DummyResponse,
    NearbySearchRequest,
    NewNearbySearchRequest,
    NewPlace,
    Place,
    Viewport,
    ViewportCoordinate,
    calculate_degree_steps,
    create_dummy_place,
    create_dummy_response,
    create_subsampled_circles,
    generate_unique_place_id,
    get_circles_search,
    get_response_places,
    get_saturated_area,
    intersection_condition_factory,
    meters_to_degree,
    nearby_search_request,
    places_search_step,
    polygon_plot_with_circles_and_points,
    polygon_plot_with_points,
    polygon_plot_with_sampling_circles,
    polygons_folium_map,
    polygons_folium_map_with_pois,
    process_circles,
    read_or_initialize_places,
    sample_polygon_with_circles,
    sample_polygons_with_circles,
    search_and_update_places,
    search_places_in_polygon,
    update_progress_and_save,
)


class TestMetersToDegree(TestCase):
    def test_zero_distance(self):
        result = meters_to_degree(0, 0)
        self.assertEqual(result, 0.0)

    def test_equator_reference_latitude(self):
        distance = 1000.0  # 1 km
        result = meters_to_degree(distance, 0)
        expected = distance / 111_132.95  # Same for lat/lon at equator
        self.assertAlmostEqual(result, expected, places=6)

    def test_polar_reference_latitude(self):
        distance = 1000.0  # 1 km
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(distance, 90)

    def test_mid_latitude(self):
        distance = 1000.0  # 1 km
        reference_latitude = 45.0
        meters_per_degree_latitude = 111_132.95
        meters_per_degree_longitude = 111_132.95 * math.cos(
            math.radians(reference_latitude)
        )
        expected = max(
            distance / meters_per_degree_latitude,
            distance / meters_per_degree_longitude,
        )
        result = meters_to_degree(distance, reference_latitude)
        self.assertAlmostEqual(result, expected, places=6)

    def test_negative_distance(self):
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(-1000.0, 45)

    def test_invalid_latitude(self):
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(1000.0, 100)  # Latitude must be between -90 and 90

    def test_large_distance(self):
        distance = 10_000_000.0  # 10,000 km
        reference_latitude = 0.0  # Equator
        expected = distance / 111_132.95  # Large distance at equator
        result = meters_to_degree(distance, reference_latitude)
        self.assertAlmostEqual(result, expected, places=6)

    def test_extreme_latitude(self):
        distance = 1000.0  # 1 km
        for latitude in [-89.9, 89.9]:  # Close to poles
            result = meters_to_degree(distance, latitude)
            expected = distance / (
                111_132.95 * math.cos(math.radians(latitude))
            )  # Very close to 0 for longitude
            self.assertAlmostEqual(result, expected, places=6)


if __name__ == "__main__":
    unittest.main()
