import unittest
from dataclasses import asdict
from unittest import TestCase

import geopandas as gpd
from pandas import Series
from shapely import Point

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
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
        self.assertIn("Invalid place data schema", str(context.exception))

    def test_place_from_empty_json(self):
        with self.assertRaises(ArgumentValueError) as context:
            Place.from_json(self.empty_json)
        self.assertIn("Invalid place data schema", str(context.exception))

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


class TestNewPlace(TestCase):
    def setUp(self):
        self.valid_data = {
            "id": "123",
            "types": ["restaurant", "food"],
            "formattedAddress": "123 Main St, Springfield",
            "addressComponents": [
                {
                    "longText": "Main St",
                    "shortText": "Main",
                    "types": ["route"],
                    "languageCode": "en",
                },
            ],
            "plusCode": {"globalCode": "849VCWC8+R9", "compoundCode": "CWC8+R9"},
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "viewport": {
                "low": {"latitude": 40.7, "longitude": -74.01},
                "high": {"latitude": 40.73, "longitude": -73.98},
            },
            "googleMapsUri": "https://maps.google.com/",
            "utcOffsetMinutes": -240,
            "adrFormatAddress": "<span>123 Main St</span>",
            "businessStatus": "OPERATIONAL",
            "iconMaskBaseUri": "https://example.com/icon.png",
            "iconBackgroundColor": "#FFFFFF",
            "displayName": {"text": "Sample Place"},
            "primaryTypeDisplayName": {"text": "Restaurant"},
            "primaryType": "restaurant",
            "shortFormattedAddress": "Main St",
            "accessibilityOptions": {"wheelchairAccessibleEntrance": True},
            "internationalPhoneNumber": "+1 234-567-890",
            "nationalPhoneNumber": "(234) 567-890",
            "priceLevel": "3",
            "rating": 4.5,
            "userRatingCount": 200,
            "websiteUri": "https://example.com",
            "currentOpeningHours": "Open now",
            "currentSecondaryOpeningHours": "Closed at 8 PM",
            "regularOpeningHours": "Mon-Fri: 9 AM - 5 PM",
            "regularSecondaryOpeningHours": "Sat: 10 AM - 4 PM",
        }

    def test_from_json_valid_data(self):
        place = NewPlace.from_json(self.valid_data)
        self.assertEqual(place.id, "123")
        self.assertEqual(place.types, "restaurant,food")
        self.assertEqual(place.latitude, 40.7128)
        self.assertEqual(place.longitude, -74.0060)
        self.assertEqual(place.globalCode, "849VCWC8+R9")
        self.assertEqual(place.businessStatus, "OPERATIONAL")
        self.assertEqual(place.rating, 4.5)
        self.assertEqual(place.userRatingCount, 200)

    def test_from_json_missing_fields(self):
        minimal_data = {
            "id": "123",
            "types": ["restaurant"],
            "location": {"latitude": 40.7128, "longitude": -74.0060},
        }
        place = NewPlace.from_json(minimal_data)
        self.assertEqual(place.id, "123")
        self.assertEqual(place.types, "restaurant")
        self.assertEqual(place.businessStatus, "")  # Optional field

    def test_from_json_invalid_data(self):
        with self.assertRaises(ArgumentValueError):
            NewPlace.from_json({"invalid_key": "value"})

    def test_parse_address_components(self):
        components = NewPlace._parse_address_components(
            self.valid_data["addressComponents"]
        )
        self.assertIsInstance(components, list)
        self.assertEqual(components[0].longText, "Main St")

    def test_parse_viewport(self):
        viewport = NewPlace._parse_viewport(self.valid_data["viewport"])
        self.assertIsInstance(viewport, Viewport)
        self.assertEqual(viewport.low.latitude, 40.7)
        self.assertEqual(viewport.high.longitude, -73.98)

    def test_parse_plus_code(self):
        global_code, compound_code = NewPlace._parse_plus_code(
            self.valid_data["plusCode"]
        )
        self.assertEqual(global_code, "849VCWC8+R9")
        self.assertEqual(compound_code, "CWC8+R9")

    def test_to_series(self):
        place = NewPlace.from_json(self.valid_data)
        series = place.to_series()
        self.assertIsInstance(series, Series)
        self.assertEqual(series["id"], "123")
        self.assertNotIn("addressComponents", series)  # Non-serialized field

    def test_invalid_viewport_data(self):
        invalid_data = {"location": {"latitude": 40.7128, "longitude": -74.0060}}
        viewport = NewPlace._parse_viewport(invalid_data.get("viewport", {}))
        self.assertEqual(viewport.low.latitude, 0.0)
        self.assertEqual(viewport.high.longitude, 0.0)

    def test_invalid_plus_code(self):
        plus_code = NewPlace._parse_plus_code({})
        self.assertEqual(plus_code, ("", ""))

    def test_missing_file_path(self):
        with self.assertRaises(ArgumentTypeError):
            NewPlace.from_json(None)

    def test_invalid_accessibility_options(self):
        invalid_data = self.valid_data.copy()
        invalid_data["accessibilityOptions"] = {"unknown_option": True}
        with self.assertRaises(ArgumentTypeError):
            NewPlace._parse_accessibility_options(invalid_data["accessibilityOptions"])


if __name__ == "__main__":
    unittest.main()
