import math
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series
from shapely import Point
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

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
        with self.assertRaises(ZeroDivisionError):
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
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
