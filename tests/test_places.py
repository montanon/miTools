import unittest
from dataclasses import asdict
from unittest import TestCase

import geopandas as gpd
from pandas import Series
from shapely import Point

from mitools.exceptions import ArgumentKeyError, ArgumentValueError
from mitools.google.places import (
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


class TestPlace(TestCase):
    def setUp(self):
        self.valid_json = {
            "place_id": "12345",
            "name": "Test Place",
            "geometry": {"location": {"lat": 40.7128, "lng": -74.0060}},
            "types": ["restaurant", "food"],
            "price_level": 2,
            "rating": 4.5,
            "user_ratings_total": 100,
            "vicinity": "New York, NY",
            "permanently_closed": False,
        }
        self.missing_keys_json = {
            "name": "Incomplete Place",
            "geometry": {"location": {"lat": 40.7128}},
        }  # Missing 'place_id' and 'longitude'
        self.empty_json = {}

    def test_place_from_valid_json(self):
        place = Place.from_json(self.valid_json)
        self.assertEqual(place.id, "12345")
        self.assertEqual(place.name, "Test Place")
        self.assertAlmostEqual(place.latitude, 40.7128)
        self.assertAlmostEqual(place.longitude, -74.0060)
        self.assertEqual(place.types, "restaurant,food")
        self.assertEqual(place.price_level, 2)
        self.assertAlmostEqual(place.rating, 4.5)
        self.assertEqual(place.total_ratings, 100)
        self.assertEqual(place.vicinity, "New York, NY")
        self.assertFalse(place.permanently_closed)

    def test_place_from_json_missing_keys(self):
        with self.assertRaises(ArgumentValueError) as context:
            Place.from_json(self.missing_keys_json)
        self.assertIn("Invalid place data", str(context.exception))

    def test_place_from_empty_json(self):
        with self.assertRaises(ArgumentValueError):
            Place.from_json(self.empty_json)

    def test_optional_fields_with_none(self):
        json_data = {
            "place_id": "67890",
            "name": "No Extras Place",
            "geometry": {"location": {"lat": 34.0522, "lng": -118.2437}},
            "types": [],
        }
        place = Place.from_json(json_data)
        self.assertEqual(place.id, "67890")
        self.assertEqual(place.name, "No Extras Place")
        self.assertEqual(place.latitude, 34.0522)
        self.assertEqual(place.longitude, -118.2437)
        self.assertEqual(place.types, "")
        self.assertIsNone(place.price_level)
        self.assertIsNone(place.rating)
        self.assertIsNone(place.total_ratings)
        self.assertIsNone(place.vicinity)
        self.assertIsNone(place.permanently_closed)

    def test_to_series(self):
        place = Place.from_json(self.valid_json)
        series = place.to_series()
        expected_series = Series(asdict(place))
        self.assertTrue(series.equals(expected_series))

    def test_to_series_with_none_fields(self):
        json_data = {
            "place_id": "12345",
            "name": "Test Place",
            "geometry": {"location": {"lat": 40.7128, "lng": -74.0060}},
            "types": [],
        }
        place = Place.from_json(json_data)
        series = place.to_series()
        self.assertIsNone(series["price_level"])
        self.assertIsNone(series["rating"])


if __name__ == "__main__":
    unittest.main()
