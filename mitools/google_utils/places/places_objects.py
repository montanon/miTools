import os
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Protocol, Tuple

import geopandas as gpd
import seaborn as sns
from geopandas import GeoDataFrame, GeoSeries
from jsonschema import ValidationError, validate
from matplotlib.pyplot import Axes
from pandas import Series
from shapely import Point, Polygon
from shapely.ops import unary_union

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
from mitools.google_utils.places.json_schemas import NEWPLACE_SCHEMA, PLACE_SCHEMA

CircleType = NewType("CircleType", Polygon)

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")


@dataclass
class AddressComponent:
    longText: str
    shortText: str
    types: List[str]
    languageCode: str


@dataclass
class ViewportCoordinate:
    latitude: float
    longitude: float


@dataclass
class Viewport:
    low: ViewportCoordinate
    high: ViewportCoordinate


@dataclass
class AccessibilityOptions:
    wheelchairAccessibleSeating: Optional[bool] = None
    wheelchairAccessibleParking: Optional[bool] = None
    wheelchairAccessibleEntrance: Optional[bool] = None
    wheelchairAccessibleRestroom: Optional[bool] = None


@dataclass
class Place:
    id: str
    name: str
    latitude: float
    longitude: float
    types: str
    price_level: int = None
    rating: float = None
    total_ratings: int = None
    vicinity: str = None
    permanently_closed: bool = None

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Place":
        try:
            validate(instance=data, schema=PLACE_SCHEMA)
            return Place(
                id=data["place_id"],
                name=data["name"],
                latitude=data["geometry"]["location"].get("lat"),
                longitude=data["geometry"]["location"].get("lng"),
                types=",".join(data.get("types")),
                price_level=data.get("price_level"),
                rating=data.get("rating"),
                total_ratings=data.get("user_ratings_total"),
                vicinity=data.get("vicinity"),
                permanently_closed=data.get("permanently_closed"),
            )
        except ValidationError as e:
            raise ArgumentValueError(f"Invalid place data schema: {data}. {e}")
        except KeyError as e:
            raise ArgumentValueError(f"Invalid place data: {data}. {e}")

    def to_series(self) -> Series:
        place_dict = asdict(self)
        return Series(place_dict)


@dataclass
class NewPlace:
    _NON_SERIALIZED_DATA = ["addressComponents", "viewport", "accessibilityOptions"]

    id: str
    types: str
    formattedAddress: str
    addressComponents: List[AddressComponent]
    globalCode: str
    compoundCode: str
    latitude: float
    longitude: float
    viewport: Viewport
    googleMapsUri: str
    utcOffsetMinutes: int
    adrFormatAddress: str
    businessStatus: str
    iconMaskBaseUri: str
    iconBackgroundColor: str
    displayName: str
    primaryType: str
    shortFormattedAddress: str
    accessibilityOptions: AccessibilityOptions
    primaryTypeDisplayName: str
    internationalPhoneNumber: str
    nationalPhoneNumber: str
    priceLevel: str
    rating: float
    userRatingCount: int
    websiteUri: str
    currentOpeningHours: str
    currentSecondaryOpeningHours: str
    regularSecondaryOpeningHours: str
    regularOpeningHours: str

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "NewPlace":
        try:
            validate(instance=data, schema=NEWPLACE_SCHEMA)
            global_code, compound_code = NewPlace._parse_plus_code(
                data.get("plusCode", {})
            )
            return NewPlace(
                id=data["id"],
                types=",".join(data["types"]),
                formattedAddress=data.get("formattedAddress", ""),
                addressComponents=NewPlace._parse_address_components(
                    data.get("addressComponents")
                ),
                globalCode=global_code,
                compoundCode=compound_code,
                latitude=data["location"]["latitude"],
                longitude=data["location"]["longitude"],
                viewport=NewPlace._parse_viewport(data.get("viewport")),
                googleMapsUri=data.get("googleMapsUri", ""),
                utcOffsetMinutes=data.get("utcOffsetMinutes", 0),
                adrFormatAddress=data.get("adrFormatAddress", ""),
                businessStatus=data.get("businessStatus", ""),
                iconMaskBaseUri=data.get("iconMaskBaseUri", ""),
                iconBackgroundColor=data.get("iconBackgroundColor", ""),
                displayName=data["displayName"]["text"],
                primaryTypeDisplayName=data.get("primaryTypeDisplayName", {}).get(
                    "text", ""
                ),
                primaryType=data.get("primaryType", ""),
                shortFormattedAddress=data.get("shortFormattedAddress", ""),
                accessibilityOptions=NewPlace._parse_accessibility_options(
                    data.get("accessibilityOptions", {})
                ),
                internationalPhoneNumber=data.get("internationalPhoneNumber", ""),
                nationalPhoneNumber=data.get("nationalPhoneNumber", ""),
                priceLevel=data.get("priceLevel", ""),
                rating=data.get("rating", -1.0),
                userRatingCount=data.get("userRatingCount", 0),
                websiteUri=data.get("websiteUri", ""),
                regularOpeningHours=str(data.get("regularOpeningHours", "")),
                regularSecondaryOpeningHours=str(
                    data.get("regularSecondaryOpeningHours", "")
                ),
                currentOpeningHours=str(data.get("currentOpeningHours", "")),
                currentSecondaryOpeningHours=str(
                    data.get("currentSecondaryOpeningHours", "")
                ),
            )
        except KeyError as e:
            raise ArgumentValueError(f"Missing expected field: {e}")
        except (AttributeError, ArgumentTypeError, TypeError, ValidationError) as e:
            raise ArgumentTypeError(f"Invalid 'data={data}' structure or type: {e}")

    @staticmethod
    def _parse_address_components(
        components: List[Dict[str, Any]],
    ) -> List[AddressComponent]:
        return [AddressComponent(**comp) for comp in components] if components else []

    @staticmethod
    def _parse_viewport(viewport_data: Dict[str, Any]) -> Viewport:
        if not viewport_data:
            return Viewport(
                low=ViewportCoordinate(0.0, 0.0), high=ViewportCoordinate(0.0, 0.0)
            )
        return Viewport(
            low=ViewportCoordinate(**viewport_data.get("low", {})),
            high=ViewportCoordinate(**viewport_data.get("high", {})),
        )

    @staticmethod
    def _parse_plus_code(plus_code_data: Dict[str, Any]) -> Tuple[str, str]:
        return (
            plus_code_data.get("globalCode", ""),
            plus_code_data.get("compoundCode", ""),
        )

    @staticmethod
    def _parse_accessibility_options(data: Dict[str, Any]) -> AccessibilityOptions:
        try:
            return AccessibilityOptions(**data) if data else AccessibilityOptions()
        except TypeError as e:
            raise ArgumentTypeError(f"Invalid accessibility options data: {data}. {e}")

    def to_series(self) -> Series:
        place_dict = asdict(self)
        place_dict = {
            key: value
            for key, value in place_dict.items()
            if key not in self._NON_SERIALIZED_DATA
        }
        return Series(place_dict)


class DummyResponse:
    def __init__(self, data: Dict[str, Any] = None, status_code: int = 200):
        self.data = data or {}
        self.status_code = status_code
        self.reason = "OK" if status_code == 200 else "Error"

    def json(self) -> Dict[str, Any]:
        return self.data


@dataclass(frozen=True)
class NearbySearchRequest:
    location: CircleType
    distance_in_meters: float
    type: str
    language_code: str = "en"
    key: str = field(default=GOOGLE_PLACES_API_KEY, init=False)

    def __post_init__(self):
        if not isinstance(self.location, Polygon):
            raise ArgumentTypeError(
                f"'location' must be a Polygon, got {type(self.location)}"
            )
        if self.distance_in_meters <= 0:
            raise ArgumentValueError("Distance in meters must be a positive number.")
        if not isinstance(self.type, str) or not self.type:
            raise ArgumentValueError("'type' must be a non-empty string.")
        if not isinstance(self.language_code, str):
            raise ArgumentValueError("'language_code' must be a string.")

    @property
    def formatted_location(self) -> str:
        return f"{self.location.y}, {self.location.x}"

    def json_query(self) -> Dict[str, str]:
        return {
            "location": self.formatted_location,
            "radius": str(self.distance_in_meters),
            "type": self.type,
            "key": self.key,
            "language": self.language_code,
        }


@dataclass(frozen=True)
class NewNearbySearchRequest:
    location: CircleType
    distance_in_meters: float
    max_result_count: int = 20
    included_types: List[str] = field(default_factory=list)
    language_code: str = "en"

    def __post_init__(self):
        if not isinstance(self.location, Polygon):
            raise ArgumentTypeError(
                f"Invalid location type: {type(self.location)}. Expected a 'Polygon'."
            )
        if self.distance_in_meters <= 0:
            raise ArgumentValueError(
                f"Distance must be positive. Received: {self.distance_in_meters}"
            )
        if not isinstance(self.language_code, str) or len(self.language_code) != 2:
            raise ArgumentValueError(
                f"Invalid language code: {self.language_code}. Expected a two-letter code."
            )
        if self.max_result_count <= 0:
            raise ArgumentValueError(
                f"Max result count must be positive. Received: {self.max_result_count}"
            )

    @property
    def location_restriction(self) -> Dict[str, Dict]:
        return {
            "circle": {
                "center": {
                    "latitude": self.location.centroid.y,
                    "longitude": self.location.centroid.x,
                },
                "radius": self.distance_in_meters,
            }
        }

    def json_query(self) -> Dict[str, Any]:
        return {
            "includedTypes": self.included_types,
            "maxResultCount": self.max_result_count,
            "locationRestriction": self.location_restriction,
            "languageCode": self.language_code,
        }


class CityGeojson:
    TOKYO_WARDS_NAMES = [
        "Chiyoda Ward",
        "Koto Ward",
        "Nakano",
        "Meguro",
        "Shinagawa Ward",
        "Ota-ku",
        "Setagaya",
        "Suginami",
        "Nerima Ward",
        "Itabashi Ward",
        "Adachi Ward",
        "Katsushika",
        "Edogawa Ward",
        "Sumida Ward",
        "Chuo-ku",
        "Minato-ku",
        "North Ward",
        "Toshima ward",
        "Shibuya Ward",
        "Arakawa",
        "Bunkyo Ward",
        "Shinjuku ward",
        "Taito",
    ]

    def __init__(self, geojson_path: PathLike, name: str):
        self.geojson_path = self._validate_path(geojson_path)
        self.data = self._load_geojson(self.geojson_path)
        self.name = name
        self.plots_width = 14
        self.plots_aspect_ratio = 16.0 / 9.0
        self.plots_height = self.plots_width / self.plots_aspect_ratio
        self.polygons = self._process_polygons()
        self.merged_polygon = self.polygons.unary_union
        self.bounds = self.polygons.bounds.iloc[0].values

    @staticmethod
    def _validate_path(geojson_path: PathLike) -> Path:
        try:
            path = Path(geojson_path).resolve(strict=True)
        except Exception as e:
            raise ArgumentValueError(f"Invalid GeoJSON path: {geojson_path}. {e}")
        return path

    @staticmethod
    def _load_geojson(geojson_path: Path) -> GeoDataFrame:
        try:
            return gpd.read_file(geojson_path)
        except Exception as e:
            raise ArgumentValueError(
                f"Failed to load GeoJSON file: {geojson_path}. Error: {e}"
            )

    def _process_polygons(self) -> GeoSeries:
        if self.geojson_path.name == "translated_tokyo_wards.geojson":
            return self._merge_tokyo_wards()
        return self.data["geometry"]

    def _merge_tokyo_wards(self) -> GeoSeries:
        polygons = [
            unary_union(self.data.loc[self.data["Wards"] == ward, "geometry"])
            for ward in self.TOKYO_WARDS_NAMES
        ]
        return GeoSeries(polygons).explode(index_parts=True).reset_index(drop=True)

    def plot_unary_polygon(self) -> Axes:
        return self._plot_geoseries(
            GeoSeries(self.merged_polygon), f"{self.name.title()} Polygon"
        )

    def plot_polygons(self) -> Axes:
        return self._plot_geoseries(
            GeoSeries(self.polygons), f"{self.name.title()} Wards Polygons"
        )

    def _plot_geoseries(self, geoseries: GeoSeries, title: str) -> Axes:
        ax = geoseries.plot(
            facecolor="none",
            edgecolor=sns.color_palette("Paired")[0],
            figsize=(self.plots_width, self.plots_height),
        )
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        ax.set_title(title)
        return ax


class ConditionProtocol(Protocol):
    def check(self, polygon: Polygon, circle: CircleType) -> bool: ...


class CircleInsidePolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return circle.within(polygon)


class CircleCenterInsidePolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return polygon.contains(circle.centroid)


class CircleIntersectsPolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return polygon.intersects(circle)


def intersection_condition_factory(condition_type: str) -> ConditionProtocol:
    if condition_type == "circle":
        return CircleInsidePolygon()
    elif condition_type == "center":
        return CircleCenterInsidePolygon()
    elif condition_type == "intersection":
        return CircleIntersectsPolygon()
    else:
        raise ArgumentValueError(f"Unknown condition type: {condition_type}")
