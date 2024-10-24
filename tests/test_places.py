import math
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import geopandas as gpd
import requests
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from shapely import Point
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
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


if __name__ == "__main__":
    unittest.main()
