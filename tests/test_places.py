import math
import unittest
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import requests
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from shapely import Point
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from tqdm import tqdm

from mitools.context import ContextVar
from mitools.exceptions import (
    ArgumentKeyError,
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.google.places import (
    GOOGLE_PLACES_API_KEY,
    NEW_NEARBY_SEARCH_URL,
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
    filter_saturated_circles,
    generate_unique_place_id,
    get_circles_search,
    get_response_places,
    get_saturated_area,
    global_requests_counter,
    global_requests_counter_limit,
    intersection_condition_factory,
    meters_to_degree,
    nearby_search_request,
    places_search_step,
    process_circles,
    process_single_circle,
    sample_polygon_with_circles,
    sample_polygons_with_circles,
    search_and_update_places,
    search_places_in_polygon,
    should_save_state,
    update_progress_bar,
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


class TestSamplePolygonWithCircles(TestCase):
    def setUp(self):
        self.valid_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.complex_polygon = orient(
            Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1), (-1, -1)]), sign=1.0
        )
        self.invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        self.radius_in_meters = 1000.0  # 1 km
        self.step_in_degrees = 0.01
        self.condition_rule = "center"  # Default rule

    def test_valid_polygon_with_center_rule(self):
        circles = sample_polygon_with_circles(
            self.valid_polygon,
            self.radius_in_meters,
            self.step_in_degrees,
            condition_rule=self.condition_rule,
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)
        for circle in circles:
            self.assertTrue(self.valid_polygon.contains(circle.centroid))

    def test_valid_polygon_with_intersection_rule(self):
        circles = sample_polygon_with_circles(
            self.valid_polygon,
            self.radius_in_meters,
            self.step_in_degrees,
            condition_rule="intersection",
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)
        for circle in circles:
            self.assertTrue(self.valid_polygon.intersects(circle))

    def test_complex_polygon_sampling(self):
        circles = sample_polygon_with_circles(
            self.complex_polygon,
            self.radius_in_meters,
            self.step_in_degrees,
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

    def test_invalid_polygon_raises_error(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(
                self.invalid_polygon,
                self.radius_in_meters,
                self.step_in_degrees,
            )

    def test_negative_radius_raises_error(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(
                self.valid_polygon,
                -1000.0,  # Negative radius
                self.step_in_degrees,
            )

    def test_invalid_condition_rule(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(
                self.valid_polygon,
                self.radius_in_meters,
                self.step_in_degrees,
                condition_rule="invalid_rule",
            )

    def test_empty_polygon_bounds(self):
        degenerate_polygon = Polygon([(0, 0), (0, 0), (0, 0)])
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(
                degenerate_polygon,
                self.radius_in_meters,
                self.step_in_degrees,
            )

    def test_large_step_in_degrees(self):
        circles = sample_polygon_with_circles(
            self.valid_polygon,
            self.radius_in_meters,
            step_in_degrees=0.5,  # Large step size
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

    def test_zero_step_in_degrees_raises_error(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(
                self.valid_polygon,
                self.radius_in_meters,
                step_in_degrees=0.0,
            )

    def test_json_query_output(self):
        condition = intersection_condition_factory("center")
        circle = Point(0.5, 0.5).buffer(
            meters_to_degree(self.radius_in_meters, reference_latitude=0.5)
        )
        self.assertTrue(condition.check(self.valid_polygon, circle))


class TestSamplePolygonsWithCircles(TestCase):
    def setUp(self):
        self.square_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.rectangular_polygon = Polygon([(0, 0), (0, 2), (1, 2), (1, 0), (0, 0)])
        self.multipolygon = MultiPolygon(
            [self.square_polygon, self.rectangular_polygon]
        )
        self.radius_in_meters = 1000.0  # 1 km radius
        self.step_in_degrees = 0.01  # Small step size for dense sampling
        self.condition_rule = "center"  # Default rule

    def test_single_polygon(self):
        circles = sample_polygons_with_circles(
            polygons=self.square_polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule=self.condition_rule,
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)
        for circle in circles:
            self.assertTrue(self.square_polygon.contains(circle.centroid))

    def test_multipolygon(self):
        circles = sample_polygons_with_circles(
            polygons=self.multipolygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule="intersection",
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

    def test_empty_polygon_list(self):
        circles = sample_polygons_with_circles(
            polygons=[],
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
        )
        self.assertIsInstance(circles, list)
        self.assertEqual(len(circles), 0)

    def test_invalid_polygon_type(self):
        with self.assertRaises(ArgumentTypeError):
            sample_polygons_with_circles(
                polygons=Point(0, 0),  # Invalid input: Not a polygon
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
            )

    def test_large_step_in_degrees(self):
        circles = sample_polygons_with_circles(
            polygons=self.square_polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=0.5,  # Large step size for sparse sampling
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

    def test_zero_radius(self):
        circles = sample_polygons_with_circles(
            polygons=self.square_polygon,
            radius_in_meters=0.0,
            step_in_degrees=self.step_in_degrees,
        )
        self.assertEqual(len(circles), 0)

    def test_invalid_condition_rule(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygons_with_circles(
                polygons=self.square_polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule="invalid_rule",  # Invalid rule
            )

    def test_multiple_polygons(self):
        polygons = [self.square_polygon, self.rectangular_polygon]
        circles = sample_polygons_with_circles(
            polygons=polygons,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)

    def test_intersecting_circles(self):
        circles = sample_polygons_with_circles(
            polygons=self.square_polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule="intersection",
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)
        for circle in circles:
            self.assertTrue(self.square_polygon.intersects(circle))

    def test_multiple_polygons_and_rules(self):
        polygons = [self.square_polygon, self.rectangular_polygon]
        for rule in ["center", "intersection", "circle"]:
            circles = sample_polygons_with_circles(
                polygons=polygons,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule=rule,
            )
            self.assertIsInstance(circles, list)


class TestGetCirclesSearch(TestCase):
    def setUp(self):
        self.test_path = Path("./test_circles.geojson")
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.radius_in_meters = 1000.0
        self.step_in_degrees = 0.01
        self.condition_rule = "center"
        self.recalculate = False

    def tearDown(self):
        if self.test_path.exists():
            self.test_path.unlink()

    def test_generate_circles_and_save(self):
        circles = get_circles_search(
            circles_path=self.test_path,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule=self.condition_rule,
        )
        self.assertTrue(self.test_path.exists())
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertIn("searched", circles.columns)
        self.assertFalse(circles["searched"].any())

    def test_load_existing_file(self):
        circles1 = get_circles_search(
            circles_path=self.test_path,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule=self.condition_rule,
        )
        circles2 = get_circles_search(
            circles_path=self.test_path,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule=self.condition_rule,
            recalculate=False,
        )
        self.assertTrue(self.test_path.exists())
        self.assertIsInstance(circles2, GeoDataFrame)
        self.assertListEqual(circles1.index.tolist(), circles2.index.tolist())

    def test_recalculate_for_existing_file(self):
        get_circles_search(
            circles_path=self.test_path,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
        )
        circles = get_circles_search(
            circles_path=self.test_path,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            recalculate=True,
        )
        self.assertTrue(self.test_path.exists())
        self.assertIsInstance(circles, GeoDataFrame)

    def test_empty_polygon(self):
        empty_polygon = Polygon()
        with self.assertRaises(ArgumentValueError):
            get_circles_search(
                circles_path=self.test_path,
                polygon=empty_polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
            )

    def test_invalid_path(self):
        invalid_path = Path("/invalid_directory/test_circles.geojson")
        with self.assertRaises(Exception):
            get_circles_search(
                circles_path=invalid_path,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
            )

    def test_invalid_radius(self):
        with self.assertRaises(ArgumentValueError):
            get_circles_search(
                circles_path=self.test_path,
                polygon=self.polygon,
                radius_in_meters=-100.0,  # Invalid radius
                step_in_degrees=self.step_in_degrees,
            )

    def test_invalid_step_in_degrees(self):
        with self.assertRaises(ArgumentValueError):
            get_circles_search(
                circles_path=self.test_path,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=0,  # Invalid step size
            )

    def test_invalid_condition_rule(self):
        with self.assertRaises(ArgumentValueError):
            get_circles_search(
                circles_path=self.test_path,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule="invalid_rule",  # Invalid rule
            )


class TestCreateSubsampledCircles(TestCase):
    def setUp(self):
        self.center = Point(0, 0)  # Origin as the center
        self.large_radius = 5000  # Meters
        self.small_radius = 1000  # Meters
        self.radial_samples = 8
        self.factor = 1.0

    def test_successful_creation(self):
        circles = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=self.factor,
        )
        self.assertIsInstance(circles, list)
        self.assertGreater(len(circles), 0)
        for circle in circles:
            self.assertIsInstance(circle, Polygon)

    def test_zero_large_radius(self):
        with self.assertRaises(ArgumentValueError) as cm:
            create_subsampled_circles(
                large_circle_center=self.center,
                large_radius=0,
                small_radius=self.small_radius,
                radial_samples=self.radial_samples,
            )
        self.assertEqual(str(cm.exception), "Radius values must be positive.")

    def test_negative_small_radius(self):
        with self.assertRaises(ArgumentValueError) as cm:
            create_subsampled_circles(
                large_circle_center=self.center,
                large_radius=self.large_radius,
                small_radius=-1000,
                radial_samples=self.radial_samples,
            )
        self.assertEqual(str(cm.exception), "Radius values must be positive.")

    def test_zero_radial_samples(self):
        with self.assertRaises(ArgumentValueError) as cm:
            create_subsampled_circles(
                large_circle_center=self.center,
                large_radius=self.large_radius,
                small_radius=self.small_radius,
                radial_samples=0,
            )
        self.assertEqual(
            str(cm.exception), "radial_samples must be a positive integer."
        )

    def test_large_radius_contains_all_circles(self):
        circles = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=self.factor,
        )
        large_circle_deg = self.center.buffer(
            meters_to_degree(self.large_radius, self.center.y)
        )
        for circle in circles:
            self.assertTrue(large_circle_deg.contains(circle))

    def test_factor_influence_on_circle_placement(self):
        circles_default = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=1.0,
        )
        circles_increased = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=10.0,
        )
        self.assertGreater(len(circles_default), len(circles_increased))

    def test_edge_case_single_radial_sample(self):
        circles = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=1,
        )
        self.assertEqual(len(circles), 2)  # Center circle + 1 radial circle

    def test_non_intersecting_large_circle(self):
        large_circle_center = Point(0, 0)
        circles = create_subsampled_circles(
            large_circle_center=large_circle_center,
            large_radius=1000,  # Small large circle
            small_radius=500,
            radial_samples=4,
            factor=5.0,  # Large factor causing non-intersecting circles
        )
        self.assertEqual(len(circles), 1)  # Only the center circle is generated

    def test_result_with_different_latitudes(self):
        circles = create_subsampled_circles(
            large_circle_center=Point(0, 45),  # Higher latitude
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=self.factor,
        )
        self.assertGreater(len(circles), 0)

    def test_invalid_large_circle_center_type(self):
        with self.assertRaises(ArgumentTypeError):
            create_subsampled_circles(
                large_circle_center=(0, 0),  # Invalid type
                large_radius=self.large_radius,
                small_radius=self.small_radius,
                radial_samples=self.radial_samples,
            )

    def test_empty_result_with_large_factor(self):
        circles = create_subsampled_circles(
            large_circle_center=self.center,
            large_radius=self.large_radius,
            small_radius=self.small_radius,
            radial_samples=self.radial_samples,
            factor=10.0,  # Large factor making circles non-overlapping
        )
        self.assertEqual(len(circles), 1)  # Only the center circle


class TestCreateDummyPlace(TestCase):
    def setUp(self):
        self.query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": 35.6895, "longitude": 139.6917},  # Tokyo
                    "radius": 1000,
                }
            }
        }

    def test_create_place(self):
        place_data = create_dummy_place(self.query, place_class=Place)
        self.assertIn("place_id", place_data)
        self.assertIn("name", place_data)
        self.assertIn("geometry", place_data)
        self.assertIn("types", place_data)
        self.assertIn("vicinity", place_data)
        self.assertIsInstance(place_data["place_id"], str)
        self.assertIsInstance(place_data["name"], str)
        self.assertIsInstance(place_data["geometry"]["location"]["latitude"], float)
        self.assertIsInstance(place_data["geometry"]["location"]["longitude"], float)

    def test_create_new_place(self):
        new_place_data = create_dummy_place(self.query, place_class=NewPlace)
        self.assertIn("id", new_place_data)
        self.assertIn("displayName", new_place_data)
        self.assertIn("location", new_place_data)
        self.assertIn("primaryType", new_place_data)
        self.assertIsInstance(new_place_data["id"], str)
        self.assertIsInstance(new_place_data["displayName"]["text"], str)
        self.assertIsInstance(new_place_data["location"]["latitude"], float)
        self.assertIsInstance(new_place_data["location"]["longitude"], float)

    def test_random_types(self):
        place_data = create_dummy_place(self.query, place_class=Place)
        types = place_data["types"]
        self.assertIsInstance(types, list)
        self.assertGreaterEqual(len(types), 1)
        self.assertLessEqual(len(types), 5)

    def test_random_coordinates_within_bounds(self):
        latitude = self.query["locationRestriction"]["circle"]["center"]["latitude"]
        longitude = self.query["locationRestriction"]["circle"]["center"]["longitude"]
        radius = self.query["locationRestriction"]["circle"]["radius"]
        place_data = create_dummy_place(self.query, place_class=Place)
        place_lat = place_data["geometry"]["location"]["latitude"]
        place_lon = place_data["geometry"]["location"]["longitude"]
        distance_in_deg = meters_to_degree(radius, latitude)
        self.assertTrue(
            latitude - distance_in_deg <= place_lat <= latitude + distance_in_deg
        )
        self.assertTrue(
            longitude - distance_in_deg <= place_lon <= longitude + distance_in_deg
        )

    def test_randomness_in_place_ids(self):
        place1 = create_dummy_place(self.query, place_class=Place)
        place2 = create_dummy_place(self.query, place_class=Place)
        self.assertNotEqual(place1["place_id"], place2["place_id"])

    def test_missing_location_restriction(self):
        invalid_query = {"locationRestriction": {"circle": {}}}
        with self.assertRaises(KeyError):
            create_dummy_place(invalid_query, place_class=Place)

    def test_new_place_has_display_name(self):
        new_place_data = create_dummy_place(self.query, place_class=NewPlace)
        display_name = new_place_data["displayName"]["text"]
        self.assertTrue(display_name.startswith("Name"))

    def test_large_radius(self):
        large_query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": 40.7128, "longitude": -74.0060},  # NYC
                    "radius": 50_000,  # Large radius
                }
            }
        }
        place_data = create_dummy_place(large_query, place_class=Place)
        self.assertIsInstance(place_data, dict)

    def test_generate_new_place_with_no_types(self):
        query_no_types = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": 34.0522, "longitude": -118.2437},  # LA
                    "radius": 500,
                }
            }
        }
        new_place_data = create_dummy_place(query_no_types, place_class=NewPlace)
        self.assertIn("types", new_place_data)

    def test_query_with_invalid_coordinates(self):
        invalid_query = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": 200.0,
                        "longitude": -118.2437,
                    },  # Invalid lat
                    "radius": 500,
                }
            }
        }
        with self.assertRaises(ArgumentValueError):
            create_dummy_place(invalid_query, place_class=Place)


class TestCreateDummyResponse(TestCase):
    def setUp(self):
        self.query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": 40.748817, "longitude": -73.985428},
                    "radius": 1000,
                }
            }
        }

    def test_response_type(self):
        response = create_dummy_response(self.query)
        self.assertIsInstance(response, DummyResponse)

    def test_empty_places(self):
        for _ in range(10):
            response = create_dummy_response(self.query)
            data = response.json()
            if "places" not in data:
                self.assertEqual(
                    data, {}
                )  # Should return an empty dictionary if no places

    def test_non_empty_places(self):
        found_non_empty = False
        for _ in range(100):
            response = create_dummy_response(self.query)
            data = response.json()
            if "places" in data:
                self.assertIsInstance(data["places"], list)
                self.assertTrue(1 <= len(data["places"]) <= 21)
                found_non_empty = True
                break
        self.assertTrue(
            found_non_empty, "No non-empty response found after multiple runs."
        )

    def test_places_structure(self):
        for _ in range(50):  # Repeat to ensure we hit non-empty cases
            response = create_dummy_response(self.query)
            data = response.json()
            if "places" in data:
                for place in data["places"]:
                    self.assertIn("id", place)
                    self.assertIn("types", place)
                    self.assertIn("location", place)
                    self.assertIn("latitude", place["location"])
                    self.assertIn("longitude", place["location"])

    def test_randomness(self):
        empty_count = 0
        non_empty_count = 0
        for _ in range(100):
            response = create_dummy_response(self.query)
            data = response.json()
            if "places" in data:
                non_empty_count += 1
            else:
                empty_count += 1
        self.assertGreater(non_empty_count, 0, "No non-empty responses found.")
        self.assertGreater(empty_count, 0, "No empty responses found.")


class TestNearbySearchRequest(TestCase):
    def setUp(self):
        self.circle = Point(151.2099, -33.865143)  # Sydney
        self.radius_in_meters = 1000.0  # 1 km radius
        self.valid_headers = {"X-Goog-Api-Key": "valid_api_key"}
        self.restaurants = ["restaurant", "cafe", "bar"]

    @patch("requests.post")
    def test_valid_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK", "results": []}
        mock_post.return_value = mock_response
        response = nearby_search_request(
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            query_headers=self.valid_headers,
            included_types=self.restaurants,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK", "results": []})
        mock_post.assert_called_once_with(
            NEW_NEARBY_SEARCH_URL,
            headers=self.valid_headers,
            json=NewNearbySearchRequest(
                location=self.circle.centroid,
                distance_in_meters=self.radius_in_meters,
                included_types=self.restaurants,
            ).json_query(),
            timeout=10,
        )

    def test_dummy_response_without_api_key(self):
        response = nearby_search_request(
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            query_headers={"X-Goog-Api-Key": ""},
            has_places=True,
        )
        self.assertIsInstance(response, DummyResponse)
        self.assertIn("places", response.json())

    @patch("requests.post", side_effect=requests.exceptions.RequestException)
    def test_request_failure(self, mock_post):
        with self.assertRaises(RuntimeError):
            nearby_search_request(
                circle=self.circle,
                radius_in_meters=self.radius_in_meters,
                query_headers=self.valid_headers,
            )

    def test_headers_fallback_to_global(self):
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            response = nearby_search_request(
                circle=self.circle,
                radius_in_meters=self.radius_in_meters,
                included_types=[],
            )
            self.assertEqual(response.status_code, 200)
            mock_post.assert_called_once_with(
                NEW_NEARBY_SEARCH_URL,
                headers=QUERY_HEADERS,
                json=NewNearbySearchRequest(
                    location=self.circle.centroid,
                    distance_in_meters=self.radius_in_meters,
                    included_types=[],
                ).json_query(),
                timeout=10,
            )

    def test_invalid_circle_type(self):
        with self.assertRaises(ArgumentTypeError):
            nearby_search_request(
                circle={"invalid": "object"},
                radius_in_meters=self.radius_in_meters,
                query_headers=self.valid_headers,
            )

    def test_timeout_handling(self):
        with patch("requests.post", side_effect=requests.exceptions.Timeout):
            with self.assertRaises(RuntimeError) as context:
                nearby_search_request(
                    circle=self.circle,
                    radius_in_meters=self.radius_in_meters,
                    query_headers=self.valid_headers,
                )
            self.assertIn("Request to", str(context.exception))


class TestGetResponsePlaces(TestCase):
    def setUp(self):
        self.valid_place_data = {
            "id": "place_1",
            "types": ["restaurant"],
            "location": {"latitude": 40.748817, "longitude": -73.985428},
            "displayName": {"text": "Sample Place"},
            "primaryType": "restaurant",
        }
        self.valid_response = DummyResponse(data={"places": [self.valid_place_data]})
        self.multi_place_response = DummyResponse(
            data={
                "places": [
                    {**self.valid_place_data, "id": f"place_{i}"} for i in range(3)
                ]
            }
        )
        self.empty_response = DummyResponse(data={"places": []})
        self.invalid_response = DummyResponse(data={"invalid_key": []})

    def test_valid_response_single_place(self):
        df = get_response_places("circle_1", self.valid_response)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], "place_1")
        self.assertEqual(df.iloc[0]["circle"], "circle_1")

    def test_valid_response_multiple_places(self):
        df = get_response_places("circle_1", self.multi_place_response)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(df["id"].tolist(), ["place_0", "place_1", "place_2"])

    def test_empty_response(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_response_places("circle_1", self.empty_response)
        self.assertEqual(str(context.exception), "No places found in the response.")

    def test_invalid_response_structure(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_response_places("circle_1", self.invalid_response)
        self.assertIn("No places found in the response.", str(context.exception))

    def test_requests_response_integration(self):
        import requests

        response = requests.Response()
        response._content = b'{"places": [{"id": "place_1", "location": {"latitude": 40.748817, "longitude": -73.985428}, "displayName": {"text": "Sample Place"}, "primaryType": "restaurant", "types": ["restaurant"]}]}'
        response.status_code = 200
        df = get_response_places("circle_2", response)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["id"], "place_1")
        self.assertEqual(df.iloc[0]["circle"], "circle_2")

    def test_invalid_json_in_response(self):
        invalid_response = DummyResponse(data=None)
        with self.assertRaises(ArgumentValueError) as context:
            get_response_places("circle_1", invalid_response)
        self.assertIn("No places found in the response.", str(context.exception))


class TestSearchAndUpdatePlaces(TestCase):
    def setUp(self):
        self.circle = Point(151.2099, -33.865143)  # Small area near Sydney
        self.radius_in_meters = 1000.0
        self.response_id = "test_circle"
        self.query_headers = {"X-Goog-Api-Key": ""}
        self.valid_place_1 = {
            "id": "place_1",
            "types": ["restaurant"],
            "location": {"latitude": -33.865, "longitude": 151.209},
            "displayName": {"text": "Test Place 1"},
            "primaryType": "restaurant",
        }
        self.valid_place_2 = {
            "id": "place_2",
            "types": ["cafe"],
            "location": {"latitude": -33.866, "longitude": 151.208},
            "displayName": {"text": "Test Place 2"},
            "primaryType": "cafe",
        }

    def create_dummy_response(self, has_places: bool) -> DummyResponse:
        data = (
            {"places": [self.valid_place_1, self.valid_place_2]} if has_places else {}
        )
        return DummyResponse(data=data)

    def test_successful_search_with_places(self):
        success, places_df = search_and_update_places(
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            response_id=self.response_id,
            query_headers=self.query_headers,
        )
        self.assertTrue(success)
        self.assertIsInstance(places_df, DataFrame)
        self.assertIn("id", places_df.columns)
        self.assertIn("circle", places_df.columns)

    def test_unsucessful_search_without_places(self):
        searched, places_df = search_and_update_places(
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            response_id=self.response_id,
            query_headers=self.query_headers,
            has_places=False,
        )
        self.assertTrue(searched)
        self.assertIsNone(places_df)

    def test_failed_request_with_invalid_key(self):
        with self.assertRaises(RuntimeError):
            search_and_update_places(
                circle=self.circle,
                radius_in_meters=self.radius_in_meters,
                response_id=self.response_id,
                query_headers={"X-Goog-Api-Key": "invalid_key"},  # Invalid key
            )

    def test_large_radius_handling(self):
        success, places_df = search_and_update_places(
            circle=self.circle,
            radius_in_meters=100000.0,  # Large search radius
            response_id=self.response_id,
            query_headers=self.query_headers,
        )
        self.assertTrue(success)  # Ensure the search succeeds
        self.assertIsInstance(places_df, DataFrame)

    def test_request_with_included_types(self):
        success, places_df = search_and_update_places(
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            response_id=self.response_id,
            query_headers=self.query_headers,
            included_types=["restaurant"],
        )
        self.assertTrue(success)
        self.assertIsInstance(places_df, DataFrame)


class TestUpdateProgressBar(TestCase):
    def setUp(self):
        self.circles = GeoDataFrame(
            {"id": [1, 2, 3, 4, 5], "searched": [False, True, False, True, False]}
        )
        self.found_places = DataFrame({"id": [1, 2, 3, 2]})  # Duplicate ID (2)
        self.tqdm_out = StringIO()
        self.pbar = tqdm(total=len(self.circles), file=self.tqdm_out)

    def tearDown(self):
        self.pbar.close()

    def test_update_progress_bar_values(self):
        update_progress_bar(self.pbar, self.circles, self.found_places)
        postfix = self.pbar.format_dict["postfix"]
        self.assertEqual(
            postfix, "Remaining Circles=3, Found Places=3, Searched Circles=2"
        )

    def test_update_progress_bar_complete(self):
        self.circles["searched"] = True  # All circles searched
        update_progress_bar(self.pbar, self.circles, self.found_places)
        postfix = self.pbar.format_dict["postfix"]
        self.assertEqual(
            postfix, "Remaining Circles=0, Found Places=3, Searched Circles=5"
        )

    def test_update_progress_bar_empty_data(self):
        empty_circles = GeoDataFrame(columns=["id", "searched"])
        empty_places = DataFrame(columns=["id"])
        update_progress_bar(self.pbar, empty_circles, empty_places)
        postfix = self.pbar.format_dict["postfix"]
        self.assertEqual(
            postfix, "Remaining Circles=0, Found Places=0, Searched Circles=0"
        )

    def test_progress_bar_increment(self):
        initial_progress = self.pbar.n  # Initial progress value
        update_progress_bar(self.pbar, self.circles, self.found_places)
        self.assertEqual(self.pbar.n, initial_progress + 1)

    def test_progress_bar_output(self):
        update_progress_bar(self.pbar, self.circles, self.found_places)
        output = self.tqdm_out.getvalue()
        self.assertIn("Remaining Circles", output)
        self.assertIn("Found Places", output)
        self.assertIn("Searched Circles", output)


class TestShouldSaveState(TestCase):
    def setUp(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 1000

    def test_save_on_exact_n_amount(self):
        self.assertTrue(should_save_state(response_id=200, total_circles=500))

    def test_save_on_last_circle(self):
        self.assertTrue(should_save_state(response_id=499, total_circles=500))

    def test_save_on_global_counter_limit(self):
        global_requests_counter.value = 999
        self.assertTrue(should_save_state(response_id=150, total_circles=500))

    def test_no_save_mid_execution(self):
        self.assertFalse(should_save_state(response_id=150, total_circles=500))

    def test_save_on_counter_limit_edge_case(self):
        global_requests_counter.value = 998
        global_requests_counter_limit.value = 999
        self.assertTrue(should_save_state(response_id=150, total_circles=500))

    def test_not_save_when_global_counter_not_reached(self):
        global_requests_counter.value = 500
        self.assertFalse(should_save_state(response_id=150, total_circles=500))

    def test_save_when_total_circles_is_one(self):
        self.assertTrue(should_save_state(response_id=0, total_circles=1))


class TestProcessSingleCircle(TestCase):
    def setUp(self):
        self.response_id = 0
        self.radius_in_meters = 1000.0
        self.circle = Point(0, 1)
        self.found_places = DataFrame(columns=["id", "name", "circle"])
        self.circles = GeoDataFrame(
            {"geometry": [None], "searched": [False]}, index=[self.response_id]
        )
        self.file_path = Path("./test_found_places.parquet")
        self.circles_path = Path("./test_circles.geojson")
        self.pbar = tqdm(total=1)
        self.included_types = ["restaurant", "cafe"]
        self.query_headers = {"X-Goog-Api-Key": ""}

    def tearDown(self):
        if self.file_path.exists():
            self.file_path.unlink()
        if self.circles_path.exists():
            self.circles_path.unlink()

    def test_successful_processing(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        found_places = process_single_circle(
            response_id=self.response_id,
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            found_places=self.found_places,
            circles=self.circles,
            file_path=self.file_path,
            circles_path=self.circles_path,
            pbar=self.pbar,
            included_types=self.included_types,
            query_headers=self.query_headers,
        )
        self.assertTrue(self.circles.loc[self.response_id, "searched"])
        self.assertGreater(len(found_places), 0)
        self.assertTrue(self.file_path.exists())
        self.assertTrue(self.circles_path.exists())

    def test_no_places_found(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        found_places = process_single_circle(
            response_id=self.response_id,
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            found_places=self.found_places,
            circles=self.circles,
            file_path=self.file_path,
            circles_path=self.circles_path,
            pbar=self.pbar,
            included_types=self.included_types,
            query_headers=self.query_headers,
            has_places=False,
        )
        self.assertTrue(self.circles.loc[self.response_id, "searched"])
        self.assertTrue(found_places.empty)

    def test_should_save_state(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        process_single_circle(
            response_id=self.response_id,
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            found_places=self.found_places,
            circles=self.circles,
            file_path=self.file_path,
            circles_path=self.circles_path,
            pbar=self.pbar,
            included_types=self.included_types,
            query_headers=self.query_headers,
        )
        self.assertTrue(self.file_path.exists())
        self.assertTrue(self.circles_path.exists())

    def test_invalid_query_headers(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        with self.assertRaises(RuntimeError):
            process_single_circle(
                response_id=self.response_id,
                circle=self.circle,
                radius_in_meters=self.radius_in_meters,
                found_places=self.found_places,
                circles=self.circles,
                file_path=self.file_path,
                circles_path=self.circles_path,
                pbar=self.pbar,
                included_types=self.included_types,
                query_headers={"X-Goog-Api-Key": "invalid_key"},
            )

    def test_no_search_due_to_limit(self):
        global_requests_counter.value = 1000
        global_requests_counter_limit.value = 1000
        found_places = process_single_circle(
            response_id=self.response_id,
            circle=self.circle,
            radius_in_meters=self.radius_in_meters,
            found_places=self.found_places,
            circles=self.circles,
            file_path=self.file_path,
            circles_path=self.circles_path,
            pbar=self.pbar,
            included_types=self.included_types,
            query_headers=self.query_headers,
        )
        self.assertFalse(self.circles.loc[self.response_id, "searched"])
        self.assertTrue(found_places.empty)
        self.assertFalse(self.file_path.exists())
        self.assertFalse(self.circles_path.exists())


class TestProcessCircles(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "found_places.parquet"
        self.circles_path = Path(self.temp_dir.name) / "circles.geojson"
        geometry = [Point(0, 0), Point(1, 1)]
        self.circles = GeoDataFrame({"searched": [False, False], "geometry": geometry})
        self.found_places = DataFrame(
            columns=["circle", *list(NewPlace.__annotations__.keys())]
        )
        self.query_headers = {"X-Goog-Api-Key": ""}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_circles_no_recalculation(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        result = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=False,
        )
        self.assertGreater(len(result), len(self.found_places))

    def test_process_circles_with_recalculation(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        result1 = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=False,
        )
        self.assertTrue(self.circles["searched"].all())
        result2 = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=True,
        )
        self.assertFalse(result1.equals(result2))  # Random generation of places

    def test_process_circles_empty_circles(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        empty_circles = GeoDataFrame(columns=["searched", "geometry"])
        result = process_circles(
            circles=empty_circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
        )
        self.assertTrue(result.empty)

    def test_process_circles_partial_processing(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        self.circles.loc[0, "searched"] = True
        result = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=False,
        )
        self.assertTrue(self.circles.loc[1, "searched"])
        self.assertGreater(len(result), len(self.found_places))

    def test_process_circles_save_to_file(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        result = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=True,
        )
        self.assertTrue(self.file_path.exists())
        self.assertTrue(self.circles_path.exists())
        saved_places = pd.read_parquet(self.file_path)
        saved_circles = gpd.read_file(self.circles_path)
        assert_frame_equal(saved_places, result, check_dtype=False)
        assert_frame_equal(saved_circles, self.circles, check_dtype=False)

    def test_process_circles_with_included_types(self):
        global_requests_counter.value = 0
        global_requests_counter_limit.value = 100
        result = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            included_types=["restaurant", "cafe"],
            query_headers=self.query_headers,
            recalculate=True,
        )
        self.assertFalse(result.empty)

    def test_process_circles_timeout_behavior(self):
        global_requests_counter.value = global_requests_counter_limit.value - 1
        result = process_circles(
            circles=self.circles,
            radius_in_meters=1000,
            file_path=self.file_path,
            circles_path=self.circles_path,
            query_headers=self.query_headers,
            recalculate=False,
        )
        self.assertTrue(self.circles["searched"].values[0])
        self.assertFalse(self.circles["searched"].values[1])
        self.assertFalse(result.empty)


class TestFilterSaturatedCircles(TestCase):
    def setUp(self):
        self.found_places = pd.DataFrame(
            {"circle": [1, 1, 1, 2, 2, 3, 4], "id": ["A", "B", "C", "D", "E", "F", "G"]}
        )
        self.circles = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(1, 1).buffer(1),
                    Point(2, 2).buffer(1),
                    Point(3, 3).buffer(1),
                    Point(4, 4).buffer(1),
                ]
            },
            index=[1, 2, 3, 4],
        )

    def test_filter_with_valid_threshold(self):
        threshold = 2
        result = filter_saturated_circles(self.found_places, self.circles, threshold)
        expected_indices = [1, 2]  # Circles 1 and 2 meet the threshold
        self.assertListEqual(result.index.tolist(), expected_indices)

    def test_filter_with_no_saturated_circles(self):
        threshold = 4  # No circle has 4 or more places
        result = filter_saturated_circles(self.found_places, self.circles, threshold)
        self.assertTrue(result.empty)

    def test_filter_with_all_circles_saturated(self):
        threshold = 1  # All circles have at least 1 place
        result = filter_saturated_circles(self.found_places, self.circles, threshold)
        expected_indices = [1, 2, 3, 4]
        self.assertListEqual(result.index.tolist(), expected_indices)

    def test_filter_with_empty_found_places(self):
        found_places = DataFrame(columns=["circle", "id"])  # Empty DataFrame
        threshold = 1
        result = filter_saturated_circles(found_places, self.circles, threshold)
        self.assertTrue(result.empty)

    def test_filter_with_empty_circles(self):
        empty_circles = GeoDataFrame(columns=["geometry"])  # Empty GeoDataFrame
        empty_circles.index.name = "id"
        threshold = 1
        with self.assertRaises(ArgumentValueError):
            filter_saturated_circles(self.found_places, empty_circles, threshold)

    def test_filter_with_invalid_threshold(self):
        with self.assertRaises(ArgumentValueError):
            filter_saturated_circles(
                self.found_places, self.circles, -1
            )  # Negative threshold

    def test_filter_with_missing_circle_in_circles(self):
        self.found_places = pd.concat(
            [self.found_places, DataFrame({"circle": [5], "id": ["H"]})]
        )
        threshold = 1
        with self.assertRaises(ArgumentValueError):
            filter_saturated_circles(self.found_places, self.circles, threshold)


if __name__ == "__main__":
    unittest.main()
