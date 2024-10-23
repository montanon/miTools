import unittest
from dataclasses import asdict
from unittest import TestCase

import geopandas as gpd
from pandas import Series
from shapely import Point

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


class TestPlace(TestCase):
    def setUp(self):
        self.valid_json = {
            "place_id": "12345",
            "name": "Test Place",
            "geometry": {"location": {"lat": 40.7128, "lng": -74.0060}},
            "displayName": {"text": "Sample Place"},
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
            "displayName": {"text": "Sample Place"},
        }
        place = NewPlace.from_json(minimal_data)
        self.assertEqual(place.id, "123")
        self.assertEqual(place.types, "restaurant")
        self.assertEqual(place.businessStatus, "")  # Optional field

    def test_from_json_invalid_data(self):
        with self.assertRaises(ArgumentTypeError):
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


class TestDummyResponse(TestCase):
    def test_default_response(self):
        response = DummyResponse()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.reason, "OK")
        self.assertEqual(response.json(), {})

    def test_custom_data_response(self):
        data = {"key": "value", "status": "success"}
        response = DummyResponse(data=data)
        self.assertEqual(response.json(), data)

    def test_custom_status_code(self):
        response = DummyResponse(status_code=404)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.reason, "Error")
        self.assertEqual(response.json(), {})

    def test_custom_data_and_status_code(self):
        data = {"error": "Not found"}
        response = DummyResponse(data=data, status_code=404)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.reason, "Error")
        self.assertEqual(response.json(), data)

    def test_empty_data(self):
        response = DummyResponse(data={})
        self.assertEqual(response.json(), {})

    def test_invalid_status_code(self):
        response = DummyResponse(status_code=500)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.reason, "Error")
        self.assertEqual(response.json(), {})

    def test_large_data_payload(self):
        data = {"key_" + str(i): i for i in range(1000)}
        response = DummyResponse(data=data)
        self.assertEqual(response.json(), data)


class TestNearbySearchRequest(TestCase):
    def setUp(self):
        self.valid_location = Point(151.2099, -33.865143)  # Sydney coordinates
        self.valid_distance = 1000.0
        self.valid_type = "restaurant"
        self.valid_language = "en"

    def test_successful_creation(self):
        request = NearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            type=self.valid_type,
        )
        self.assertEqual(request.formatted_location, "-33.865143, 151.2099")
        self.assertEqual(request.distance_in_meters, self.valid_distance)
        self.assertEqual(request.type, self.valid_type)
        self.assertEqual(request.language_code, self.valid_language)
        self.assertEqual(request.key, GOOGLE_PLACES_API_KEY)

    def test_custom_language_code(self):
        request = NearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            type=self.valid_type,
            language_code="es",
        )
        self.assertEqual(request.language_code, "es")

    def test_invalid_location_type(self):
        with self.assertRaises(TypeError):
            NearbySearchRequest(
                location=(-33.865143, 151.2099),  # Not a Point object
                distance_in_meters=self.valid_distance,
                type=self.valid_type,
            )

    def test_negative_distance(self):
        with self.assertRaises(ArgumentValueError):
            NearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=-500.0,
                type=self.valid_type,
            )

    def test_empty_type(self):
        with self.assertRaises(ArgumentValueError):
            NearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                type="",
            )

    def test_invalid_language_code(self):
        with self.assertRaises(ArgumentValueError):
            NearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                type=self.valid_type,
                language_code=None,  # Invalid language code type
            )

    def test_json_query_output(self):
        request = NearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            type=self.valid_type,
        )
        expected_query = {
            "location": "-33.865143, 151.2099",
            "radius": "1000.0",
            "type": "restaurant",
            "key": GOOGLE_PLACES_API_KEY,
            "language": "en",
        }
        self.assertDictEqual(request.json_query(), expected_query)

    def test_large_distance(self):
        request = NearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=100000.0,  # Large distance
            type=self.valid_type,
        )
        self.assertEqual(request.distance_in_meters, 100000.0)

    def test_mutability_violation(self):
        request = NearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            type=self.valid_type,
        )
        with self.assertRaises(AttributeError):
            request.distance_in_meters = 500.0

    def test_missing_key_config(self):
        self.assertEqual(
            NearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                type=self.valid_type,
            ).key,
            GOOGLE_PLACES_API_KEY,
        )


class TestNewNearbySearchRequest(TestCase):
    def setUp(self):
        self.valid_location = Point(151.2099, -33.865143)  # Sydney coordinates
        self.valid_distance = 1000.0
        self.valid_included_types = ["restaurant", "cafe"]
        self.valid_language = "en"

    def test_successful_creation(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            included_types=self.valid_included_types,
        )
        self.assertEqual(request.location, self.valid_location)
        self.assertEqual(request.distance_in_meters, self.valid_distance)
        self.assertEqual(request.included_types, self.valid_included_types)
        self.assertEqual(request.language_code, self.valid_language)

    def test_custom_language_code(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            included_types=self.valid_included_types,
            language_code="es",
        )
        self.assertEqual(request.language_code, "es")

    def test_invalid_location_type(self):
        with self.assertRaises(ArgumentTypeError):
            NewNearbySearchRequest(
                location=(-33.865143, 151.2099),  # Not a Point object
                distance_in_meters=self.valid_distance,
            )

    def test_negative_distance(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=-500.0,
            )

    def test_invalid_language_code(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                language_code="eng",  # Invalid length
            )

    def test_empty_included_types(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            included_types=[],
        )
        self.assertEqual(request.included_types, [])

    def test_invalid_max_result_count(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                max_result_count=0,
            )

    def test_json_query_output(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
            included_types=self.valid_included_types,
        )
        expected_query = {
            "includedTypes": self.valid_included_types,
            "maxResultCount": 20,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": self.valid_location.centroid.y,
                        "longitude": self.valid_location.centroid.x,
                    },
                    "radius": self.valid_distance,
                }
            },
            "languageCode": self.valid_language,
        }
        self.assertDictEqual(request.json_query(), expected_query)

    def test_large_distance(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=100000.0,  # Large distance
        )
        self.assertEqual(request.distance_in_meters, 100000.0)

    def test_mutability_violation(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
        )
        with self.assertRaises(AttributeError):
            request.distance_in_meters = 500.0

    def test_missing_included_types(self):
        request = NewNearbySearchRequest(
            location=self.valid_location,
            distance_in_meters=self.valid_distance,
        )
        self.assertEqual(request.included_types, [])

    def test_invalid_language_code_type(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.valid_location,
                distance_in_meters=self.valid_distance,
                language_code=None,
            )


if __name__ == "__main__":
    unittest.main()
