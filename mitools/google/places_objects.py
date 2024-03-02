import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import seaborn as sns
from pandas import Series
from shapely import Point
from shapely.ops import unary_union

GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY', '')

class CityGeojson:
    def __init__(self, geojson_path: PathLike, name: str):
        self.geojson_path = Path(geojson_path)
        self.data = gpd.read_file(geojson_path)
        self.name = name
        self.plots_width = 14
        self.plots_aspect_ratio = 16/9
        self.plots_height = self.plots_width / self.plots_aspect_ratio

        if self.geojson_path.name == 'translated_tokyo_wards.geojson':
            wards = ['Chiyoda Ward', "Koto Ward", "Nakano", "Meguro", "Shinagawa Ward", "Ota-ku", "Setagaya",
                "Suginami", "Nerima Ward", "Itabashi Ward", "Adachi Ward", "Katsushika",
                "Edogawa Ward", "Sumida Ward", "Chuo-ku", "Minato-ku", "North Ward",
                "Toshima ward", 'Shibuya Ward', 'Arakawa', 'Bunkyo Ward',
                'Shinjuku ward', 'Taito'
                     ]
            polygons = [unary_union(self.data.loc[self.data['Wards'] == ward, 'geometry']) for ward in wards]
            self.polygons = gpd.GeoSeries(polygons).explode(index_parts=True).reset_index(drop=True)
        else:
            self.polygons = self.data['geometry']
        self.merged_polygon = self.polygons.unary_union
        self.bounds = self.polygons.bounds.iloc[0].values
    
    def plot_unary_polygon(self):
        ax = gpd.GeoSeries(self.merged_polygon).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0],
                                figsize=(self.plots_width, self.plots_height))
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        ax.set_title(f"{self.name.title()} Polygon")
        return ax

    def plot_polygons(self):
        ax = gpd.GeoSeries(self.polygons).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0],
                                figsize=(self.plots_width, self.plots_height))
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        ax.set_title(f"{self.name.title()} Wards Polygons")
        return ax
    
class NewNearbySearchRequest:
    def __init__(self, 
                 location: Point,
                 distance_in_meters: float,
                 max_result_count: Optional[int]=20, 
                 included_types: Optional[List[str]]=None,
                 language_code: Optional[str]='en'):
        self.location = location
        self.distance_in_meters = distance_in_meters
        self.language_code = language_code
        self.location_restriction = {
            'circle': {
                'center': {
                    'latitude': self.location.centroid.y,
                    'longitude': self.location.centroid.x
                },
                'radius': self.distance_in_meters
            }
        }
        self.included_types = included_types if included_types else []
        self.max_result_count = max_result_count

    def json_query(self) -> Dict:
        query = {
            'includedTypes': self.included_types,
            'maxResultCount': self.max_result_count,
            'locationRestriction': self.location_restriction,
            'languageCode': self.language_code,
            }
        return query
    
class NearbySearchRequest:
    def __init__(self, 
                 location: Point,
                 distance_in_meters: float,
                 type: str,
                 language_code: Optional[str]='en'):
        self.location = f"{location.centroid.y}, {location.centroid.x}"
        self.distance_in_meters = distance_in_meters
        self.type = type
        self.key = GOOGLE_PLACES_API_KEY
        self.language_code = language_code

    def json_query(self) -> Dict:
        query = {
            'location': self.location,
            'radius': self.distance_in_meters,
            'type': self.type,
            'key': self.key,
            'language': self.language_code,
            }
        return query
    

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
    wheelchairAccessibleSeating: Optional[bool]=None
    wheelchairAccessibleParking: Optional[bool]=None
    wheelchairAccessibleEntrance: Optional[bool]=None
    wheelchairAccessibleRestroom: Optional[bool]=None

@dataclass
class NewPlace:
    _NON_SERIALIZED_DATA = ['addressComponents', 'viewport', 'accessibilityOptions']
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

    @staticmethod
    def from_json(data: dict) -> 'NewPlace':
        global_code, compound_code = NewPlace.parse_plus_code(data.get('plusCode', {}))
        return NewPlace(
            id=data.get('id', ''),
            types=','.join(data.get('types', [])),
            formattedAddress=data.get('formattedAddress', ''),
            addressComponents=NewPlace.parse_address_components(data.get('addressComponents')),
            globalCode=global_code,
            compoundCode=compound_code,
            latitude=data.get('location', {}).get('latitude', 0.0),
            longitude=data.get('location', {}).get('longitude', 0.0),
            viewport=NewPlace.parse_viewport(data.get('viewport')),
            googleMapsUri=data.get('googleMapsUri', ''),
            utcOffsetMinutes=data.get('utcOffsetMinutes', 0),
            adrFormatAddress=data.get('adrFormatAddress', ''),
            businessStatus=data.get('businessStatus', ''),
            iconMaskBaseUri=data.get('iconMaskBaseUri', ''),
            iconBackgroundColor=data.get('iconBackgroundColor', ''),
            displayName=data.get('displayName', {}).get('text', ''),
            primaryTypeDisplayName=data.get('primaryTypeDisplayName', {}).get('text', ''),
            primaryType=data.get('primaryType', ''),
            shortFormattedAddress=data.get('shortFormattedAddress', ''),
            accessibilityOptions=AccessibilityOptions(**data.get('accessibilityOptions', {}))
        )
    
    @staticmethod
    def parse_address_components(components: List[dict]) -> List[AddressComponent]:
        return [AddressComponent(**comp) for comp in components] if components else []

    @staticmethod
    def parse_viewport(viewport_data: dict) -> Viewport:
        if not viewport_data:
            return Viewport(low=ViewportCoordinate(0.0, 0.0), high=ViewportCoordinate(0.0, 0.0))
        return Viewport(
            low=ViewportCoordinate(**viewport_data.get('low', {})),
            high=ViewportCoordinate(**viewport_data.get('high', {}))
        )

    @staticmethod
    def parse_plus_code(plus_code_data: dict) -> tuple:
        return (
            plus_code_data.get('globalCode', ''),
            plus_code_data.get('compoundCode', '')
        )
    
    def to_series(self) -> Series:
        place_dict = asdict(self)
        place_dict = {key: value for key, value in place_dict.items() if key not in self._NON_SERIALIZED_DATA}
        return Series(place_dict)
    
@dataclass
class Place:
    id: str
    name: str
    latitude: float
    longitude: float
    types: str
    price_level: Optional[int]=None
    rating: Optional[float]=None
    total_ratings: Optional[int]=None
    vicinity: Optional[str]=None
    permanently_closed: Optional[bool]=None

    @staticmethod
    def from_json(data: dict) -> 'Place':
        return Place(
            id=data['place_id'],
            name=data['name'],
            latitude=data['geometry']['location'].get('lat'),
            longitude=data['geometry']['location'].get('lng'),
            types=','.join(data.get('types')),
            price_level=data.get('price_level'),
            rating=data.get('rating'),
            total_ratings=data.get('user_ratings_total'),
            vicinity=data.get('vicinity'),
            permanently_closed=data.get('permanently_closed')
        )
    def to_series(self) -> Series:
        place_dict = asdict(self)
        return Series(place_dict)
    
class DummyResponse(dict):
    def __init__(self):
        super().__init__()
        self.reason = 'OK'
        self.status_code = 200
    def json(self):
        return self