import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, NewType, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from IPython.display import display
from matplotlib.axes import Axes
from pandas import DataFrame, Series
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform, unary_union
from tqdm import tqdm

CircleType = NewType('CircleType', Polygon)

# https://mapsplatform.google.com/pricing/#pricing-grid
# https://developers.google.com/maps/documentation/places/web-service/search-nearby
# https://developers.google.com/maps/documentation/places/web-service/usage-and-billing#nearby-search
# https://developers.google.com/maps/documentation/places/web-service/nearby-search#fieldmask
# https://developers.google.com/maps/documentation/places/web-service/search-nearby#PlaceSearchPaging
# https://developers.google.com/maps/documentation/places/web-service/place-types
NEW_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
NEARBY_SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?parameters'
GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY')
RESTAURANT_TYPES = [
    'american_restaurant',
    'bakery',
    'bar',
    'barbecue_restaurant',
    'brazilian_restaurant',
    'breakfast_restaurant',
    'brunch_restaurant',
    'cafe',
    'chinese_restaurant',
    'coffee_shop',
    'fast_food_restaurant',
    'french_restaurant',
    'greek_restaurant',
    'hamburger_restaurant',
    'ice_cream_shop',
    'indian_restaurant',
    'indonesian_restaurant',
    'italian_restaurant',
    'japanese_restaurant',
    'korean_restaurant',
    'lebanese_restaurant',
    'meal_delivery',
    'meal_takeaway',
    'mediterranean_restaurant',
    'mexican_restaurant', 
    'middle_eastern_restaurant',
    'pizza_restaurant',
    'ramen_restaurant',
    'restaurant',
    'sandwich_shop',
    'seafood_restaurant',
    'spanish_restaurant',
    'steak_house',
    'sushi_restaurant',
    'thai_restaurant',
    'turkish_restaurant',
    'vegan_restaurant',
    'vegetarian_restaurant',
    'vietnamese_restaurant',
]
FIELD_MASK = 'places.accessibilityOptions,places.addressComponents,places.adrFormatAddress,places.businessStatus,' \
        + 'places.displayName,places.formattedAddress,places.googleMapsUri,places.iconBackgroundColor,' \
            + 'places.iconMaskBaseUri,places.id,places.location,places.name,places.primaryType,places.primaryTypeDisplayName,places.plusCode,' \
                + 'places.shortFormattedAddress,places.subDestinations,places.types,places.utcOffsetMinutes,places.viewport'
QUERY_HEADERS = {
    'Content-Type': 'aplication/json',
    'X-Goog-Api-Key': GOOGLE_PLACES_API_KEY,
    'X-Goog-FieldMask': FIELD_MASK
}

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


def meters_to_degree(distance_in_meters: float, 
                     reference_latitude:float) -> float:
    meters_per_degree_latitude = 111_132.95
    meters_per_degree_longitude = 111_132.95 * math.cos(math.radians(reference_latitude))
    lat_degrees = distance_in_meters / meters_per_degree_latitude
    lon_degrees = distance_in_meters / meters_per_degree_longitude
    return max(lat_degrees, lon_degrees)

def circle_inside_polygon(polygon: Polygon, circle: CircleType) -> bool:
    return circle.within(polygon)

def circle_center_inside_polygon(polygon: Polygon, circle: CircleType) -> bool:
    return polygon.contains(circle.centroid)

def circle_intersects_polygon(polygon: Polygon, circle: CircleType) -> bool:
    return polygon.intersects(circle)

def sample_polygon_with_circles(polygon: Polygon, 
                               radius_in_meters: float, 
                               step_in_degrees: float,
                               condition_rule: Optional[str]='center') -> List[CircleType]:
    assert condition_rule in ['center', 'circle', 'intersection'], 'condition_rule must be one of "center", "intersection" or "circle"'
    conditions = {
        'circle': circle_inside_polygon,
        'center': circle_center_inside_polygon,
        'intersection': circle_intersects_polygon
    }
    if not polygon.is_valid:
        raise ValueError('Invalid Polygon')
    minx, miny, maxx, maxy = polygon.bounds
    latitudes, longitudes = np.arange(miny, maxy, step_in_degrees), np.arange(minx, maxx, step_in_degrees)
    circles = []
    for lat, lon in itertools.product(latitudes, longitudes):
        deg_radius = meters_to_degree(distance_in_meters=radius_in_meters, reference_latitude=lat)
        circle = Point(lon, lat).buffer(deg_radius)
        if conditions[condition_rule](polygon=polygon, circle=circle):
            circles.append(circle)
    return circles

def sample_polygons_with_circles(polygons: Union[Iterable[Polygon], Polygon], 
                                 radius_in_meters: float, 
                                 step_in_degrees: float,
                                 condition_rule: Optional[int]='center') -> List[CircleType]:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons, MultiPolygon):
        polygons = list(polygons.geoms)
    circles = []
    for polygon in polygons:
        circles.extend(sample_polygon_with_circles(polygon=polygon, 
                                                  radius_in_meters=radius_in_meters, 
                                                  step_in_degrees=step_in_degrees,
                                                  condition_rule=condition_rule))
    return circles

def polygons_folium_map(polygons: Union[Iterable[Polygon], Polygon], 
                        output_file_path: Optional[PathLike]=None) -> folium.Map:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    reversed_polygons = [transform(lambda lat, lon: (
        lon, lat), polygon) for polygon in polygons]
    centroids = [polygon.centroid.coords[0] for polygon in reversed_polygons]
    lons, lats = zip(*centroids)
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)
    centroid = (centroid_lon, centroid_lat)
    folium_map = folium.Map(location=centroid, zoom_start=15)
    for polygon in polygons:
        folium.GeoJson(
            polygon,
            style_function=lambda feature: {
                'fillColor': 'blue',  # Color the polygon
                'color': 'black',     # Color for the outline
                'weight': 2,          # Width of the outline
                'fillOpacity': 0.3,   # Opacity of the fill color
            }
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map

def polygons_folium_map_with_pois(polygons: Union[Iterable[Polygon], Polygon], 
                                  pois: Iterable[Tuple[str, Point]], 
                                  output_file_path: Optional[PathLike]=None) -> folium.Map:
    folium_map = polygons_folium_map(polygons=polygons, output_file_path=None)
    for poi_name, poi_point in pois:
        folium.Marker(
            location=[poi_point.x, poi_point.y],
            popup=folium.Popup(poi_name, max_width=250),
            icon=folium.Icon(color='orange', icon='cutlery',
                             prefix='fa')
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map

def polygon_plot_with_sampling_circles(polygon: Polygon, 
                                       circles: List[CircleType],
                                       point_of_interest: Optional[Point]=None,
                                       zoom_level: Optional[float]=1.0,
                                       output_file_path: Optional[PathLike]=None) -> Axes:
    width = 14
    aspect_ratio = 16/9
    height = width / aspect_ratio
    minx, miny, maxx, maxy = polygon.bounds
    out_circles, in_circles = [], []
    for circle in tqdm(circles):
        if polygon.contains(circle) or polygon.intersects(circle):
            in_circles.append(circle)
        else:
            out_circles.append(circle)
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
            facecolor=sns.color_palette('Paired')[0], edgecolor='none', alpha=0.5,
            figsize=(width, height))
    ax = polygon.plot(
        facecolor='none', edgecolor=sns.color_palette('Paired')[0], linewidth=3, ax=ax)
    point1 = (minx, miny)
    point2 = (maxx, maxy)
    rectangle = patches.Rectangle((point1[0], point1[1]),  # (x,y)
                             point2[0] - point1[0],  # width
                             point2[1] - point1[1],  # height
                             edgecolor='k',
                             facecolor='none',
                             linestyle='--',)
    ax.add_patch(rectangle)
    if out_circles:
        ax = gpd.GeoSeries(out_circles).plot(
            facecolor='none', edgecolor='r', ax=ax, alpha=0.5, label='Out Circles')
        out_circle_proxy = mlines.Line2D(
            [], [], color='r', marker='o', markersize=10, label='Out Circles', linestyle='None')
    ax = gpd.GeoSeries(in_circles).plot(
        facecolor='none', edgecolor='g', ax=ax, alpha=0.5, label='In Circles')
    in_circle_proxy = mlines.Line2D(
        [], [], color='g', marker='o', markersize=10, label='In Circles', linestyle='None')
    ax.legend(handles=[out_circle_proxy, in_circle_proxy] if out_circles else [in_circle_proxy],
              loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title('Circles to Sample')
    ax.set_xticks([])
    ax.set_yticks([])
    if point_of_interest:
        ax.set_xlim([point_of_interest.centroid.x - zoom_level, point_of_interest.centroid.x + zoom_level])
        ax.set_ylim([point_of_interest.centroid.y - zoom_level, point_of_interest.centroid.y + zoom_level])
    if output_file_path:
        plt.savefig(output_file_path, dpi=500)
    return ax

def polygon_plot_with_circles_and_points(polygon, 
                                         circles, 
                                         points, 
                                         point_of_interest=None, 
                                         zoom_level=None, 
                                         output_file_path=None):
    ax = polygon_plot_with_sampling_circles(polygon=polygon, 
                                            circles=circles,
                                            point_of_interest=point_of_interest, 
                                            zoom_level=zoom_level,
                                            output_file_path=output_file_path)
    for point in points:
        ax.plot(point[0], point[1], 'ro', markersize=0.25)
    if output_file_path:
        plt.savefig(output_file_path, dpi=500)
    return ax

def polygon_plot_with_points(polygon, 
                             points, 
                             point_of_interest: Optional[Point]=None,
                             zoom_level: Optional[float]=1.0,
                             output_file_path=None):
    width = 14
    aspect_ratio = 16/9
    height = width / aspect_ratio
    minx, miny, maxx, maxy = polygon.bounds
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
            facecolor=sns.color_palette('Paired')[0], edgecolor='none', alpha=0.5, figsize=(width, height))
    ax = polygon.plot(
        facecolor='none', edgecolor=sns.color_palette('Paired')[0], linewidth=3, ax=ax)
    point1 = (minx, miny)
    point2 = (maxx, maxy)
    rectangle = patches.Rectangle((point1[0], point1[1]),  # (x,y)
                             point2[0] - point1[0],  # width
                             point2[1] - point1[1],  # height
                             edgecolor='k',
                             facecolor='none',
                             linestyle='--',)
    ax.add_patch(rectangle)
    for point in points:
        ax.plot(point[0], point[1], 'ro', markersize=0.25)
    places_proxy = mlines.Line2D(
        [], [], color='g', marker='o', markersize=10, label='In Circles', linestyle='None')
    ax.legend(handles=[places_proxy],
              loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title('Sampled Points')
    ax.set_xticks([])
    ax.set_yticks([])
    if point_of_interest:
        ax.set_xlim([point_of_interest.centroid.x - zoom_level, point_of_interest.centroid.x + zoom_level])
        ax.set_ylim([point_of_interest.centroid.y - zoom_level, point_of_interest.centroid.y + zoom_level])
    if output_file_path:
        plt.savefig(output_file_path, dpi=500)
    return ax

def run_new_nearby_search():
    queries = [NewNearbySearchRequest(circle, 
                                      distance_in_meters=RADIUS_IN_METERS, 
                                      included_types=RESTAURANT_TYPES
                                      ).json_query() for circle in circles]
    for query in queries:
        response = requests.post(NEARBY_SEARCH_URL, headers=QUERY_HEADERS, json=query)
        response_data = response.json()['places']
        for place in response_data:
            place = Place.from_json(place).to_series()
            break
        break

def run_nearby_search():
    random.shuffle(circles)
    all_places = []
    for circle in circles[:10]:
        query = NearbySearchRequest(location=circle, distance_in_meters=RADIUS_IN_METERS, type='restaurant').json_query()
        places = []
        while True:
            response = requests.get(NEARBY_SEARCH_URL, params=query)
            if response.status_code == 200:
                response = response.json()
                for place in response['results']:
                    place = Place.from_json(place).to_series()
                    places.append(place)
                next_page_token = response.get('next_page_token')
                if not next_page_token:
                    break
                query['pagetoken'] = next_page_token
                time.sleep(0.001)
            else:
                print('Response failed: ', response)
                break
        print('Found Places: ', len(places))
        if places:
            places = pd.concat(places, axis=1).T.reset_index(drop=True)
            all_places.append(places)
    all_places = pd.concat(all_places, axis=0).reset_index(drop=True)
    all_places.to_excel('dehli_places.xlsx', index=False)
    display(all_places)

def get_circles_search(circles_path, 
                       polygon, 
                       radius_in_meters, 
                       step_in_degrees, 
                       condition_rule='center', 
                       recalculate=False):
    if not circles_path.exists() or recalculate:
        circles = sample_polygons_with_circles(polygons=polygon, 
                                            radius_in_meters=radius_in_meters, 
                                            step_in_degrees=step_in_degrees,
                                            condition_rule=condition_rule)
        circles = gpd.GeoDataFrame(geometry=circles).reset_index(drop=True)
        circles['searched'] = False
        circles.to_file(circles_path, driver='GeoJSON')
    else:
        circles = gpd.read_file(circles_path)
    return circles

def create_subsampled_circles(large_circle_center, large_radius, small_radius, radial_samples, factor):
    large_radius_in_deg = meters_to_degree(large_radius, large_circle_center.y)
    small_radius_in_deg = meters_to_degree(small_radius, large_circle_center.y)
    large_circle = Point(large_circle_center).buffer(large_radius_in_deg)
    subsampled_points = [Point(large_circle_center).buffer(small_radius_in_deg)]
    angle_step = 2 * np.pi / radial_samples
    for i in range(radial_samples):
        angle = i * angle_step
        dx = factor * small_radius_in_deg * np.cos(angle)
        dy = factor * small_radius_in_deg * np.sin(angle)
        point = Point(large_circle_center.x + dx, large_circle_center.y + dy)
        if large_circle.contains(point):  # Check if small circle is within the large circle
            subsampled_points.append(point.buffer(small_radius_in_deg))
    return subsampled_points

def read_or_initialize_places(file_path, recalculate=False):
    if file_path.exists() and not recalculate:
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=['circle', *list(NewPlace.__annotations__.keys())])

def generate_unique_place_id():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

def create_dummy_place(query):
    latitude = query['locationRestriction']['circle']['center']['latitude']
    longitude = query['locationRestriction']['circle']['center']['longitude']
    radius = query['locationRestriction']['circle']['radius']
    distance_in_deg = meters_to_degree(radius, latitude)
    random_types = random.sample(RESTAURANT_TYPES, random.randint(1, min(len(RESTAURANT_TYPES), random.randint(1,5))))
    unique_id = generate_unique_place_id()
    random_latitude = random.uniform(latitude - distance_in_deg, latitude + distance_in_deg)
    random_longitude = random.uniform(longitude - distance_in_deg, longitude + distance_in_deg)
    place_json = {
        'id': unique_id,
        'types': random_types,
        'location': {
            'latitude': random_latitude,
            'longitude': random_longitude,
        },
        'displayName': {
            'text': f'Name {unique_id}'
            },
        'primaryType': random.choice(random_types)
    }
    return place_json

class DummyResponse(dict):
    def __init__(self):
        super().__init__()
        self.status_code = 'OK'
    def json(self):
        return self
    
def create_dummy_response(query):
    dummy_response = DummyResponse()
    has_places = random.choice([True, False, False])
    if has_places:
        places_n = random.randint(1, 21)
        dummy_response['places'] = [create_dummy_place(query) for _ in range(places_n)]
    return dummy_response

def nearby_search_request(circle, radius_in_meters):
    query = NewNearbySearchRequest(circle.geometry, 
                                distance_in_meters=radius_in_meters, 
                                included_types=RESTAURANT_TYPES
                                   ).json_query()
    if QUERY_HEADERS['X-Goog-Api-Key'] != '':
        return requests.post(NEW_NEARBY_SEARCH_URL, headers=QUERY_HEADERS, json=query)
    else:
        return create_dummy_response(query)

def get_response_places(response_id, response):
    for n, place in enumerate(response.json()['places']):
        place_series = NewPlace.from_json(place).to_series()
        place_series['circle'] = response_id
        if n == 0:
            places_df = pd.DataFrame(place_series).T
        else:
            places_df = pd.concat([places_df, pd.DataFrame(place_series).T], axis=0, ignore_index=True)
    return places_df

def search_and_update_places(circle, radius_in_meters, response_id):
    response = nearby_search_request(circle, radius_in_meters)
    places_df = None
    if response.status_code == 'OK':
        if 'places' in response.json():
            places_df = get_response_places(response_id, response)
        searched = True
    else:
        print(response.status_code, response.reason, response.text)
        searched = False
    return searched, places_df
    
def process_circles(circles, radius_in_meters, file_path, circles_path, recalculate=False):
    if (~circles['searched']).any() or recalculate:
        circles_search = circles[~circles['searched']]
        found_places = read_or_initialize_places(file_path, recalculate)
        pbar = tqdm(total=len(circles_search), desc="Processing circles")
        for response_id, circle in circles_search.iterrows():
            searched, places_df = search_and_update_places(circle, radius_in_meters, response_id)
            if places_df is not None:
                found_places = pd.concat([found_places, places_df], axis=0, ignore_index=True)
            update_progress_and_save(searched, circles, response_id, found_places, file_path, circles_path, pbar)
    else:
        found_places = pd.read_parquet(file_path)
    return found_places

def update_progress_and_save(searched, circles, index, found_places, file_path, circles_path, pbar):
    circles.loc[index, 'searched'] = searched
    if index % 5_000 == 0 or index == circles.shape[0] - 1:
        found_places.to_parquet(file_path)
        circles.to_file(circles_path, driver='GeoJSON')
    pbar.update()
    pbar.set_postfix({'Remaining Circles': circles['searched'].value_counts()[False] if False in circles['searched'].value_counts() else 0, 
                      'Found Places': found_places['id'].nunique(),
                      'Searched Circles': circles['searched'].sum()})

def run_level_circles_search(root_folder, city, radius_in_meters, step_in_degrees, process=False, recalculate=True, show=False):
    tokyo_circles_path = root_folder / Path(f"{city.name}_{radius_in_meters}_radius_{step_in_degrees}_step_circles.geojson")
    tokyo_found_places = root_folder / Path(f"{city.name}_{radius_in_meters}_radius_{step_in_degrees}_step_places.parquet")
    circles = get_circles_search(tokyo_circles_path, city, radius_in_meters, step_in_degrees, recalculate=recalculate)
    if show:
        _ = polygon_plot_with_sampling_circles(polygon=city.merged_polygon, circles=circles.geometry.tolist())
        plt.show()
    if process:
        found_places = process_circles(circles, radius_in_meters, tokyo_found_places, tokyo_circles_path)
    else:
        found_places = None
    return circles, found_places

def plot_subsampled_circles_schema(original_circle, original_radius, circle_subsamples, plot_path, show=False):
    if not plot_path.exists() or show:
        fig, ax = plt.subplots()
        x, y = Point(original_circle.centroid).buffer(meters_to_degree(original_radius, original_circle.centroid.y)).exterior.xy
        ax.fill(x, y, alpha=0.25, fc='r', ec='none')
        for circle_subsample in circle_subsamples:
            x, y = circle_subsample.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='b', ec='none')
        fig.savefig(plot_path, dpi=500)

def get_subsampled_circles(city, 
                           circles, 
                           found_places, 
                           original_radius, 
                           subsampling_radius, 
                           subsampling_count, 
                           subsamplig_factor, 
                           project_folder,
                           show=False):
    subsampled_circles_path = project_folder / Path(f"{city.name}_{subsampling_radius}_radius_{subsampling_count}-{subsamplig_factor}_subsampled_circles.geojson")
    subsampling_circles_schema_plot = project_folder / Path(f"{city.name}_{original_radius}-{subsampling_radius}_radius_{subsampling_count}-{subsamplig_factor}_subsampled_circles.png")

    places_by_circle = found_places.groupby('circle')['id'].nunique().sort_values(ascending=False)
    saturated_circles = places_by_circle[places_by_circle == 20].index

    if not subsampled_circles_path.exists():
        subsampled_circles = DataFrame(columns=['circle', 'subsampled_circle'])
        for saturated_circle in saturated_circles:
            circle = circles.loc[saturated_circle, 'geometry']
            circle_subsamples = create_subsampled_circles(circle.centroid, original_radius, subsampling_radius, subsampling_count, subsamplig_factor)
            for subsampled_circle in circle_subsamples:
                    subsampled_circles = pd.concat([subsampled_circles, DataFrame({'circle': saturated_circle, 'subsampled_circle': subsampled_circle}, index=[0])], axis=0, ignore_index=True)
        plot_subsampled_circles_schema(circle, original_radius, circle_subsamples, subsampling_circles_schema_plot, show=show)
        subsampled_circles = subsampled_circles.reset_index(drop=True)
        subsampled_circles['searched'] = False
        subsampled_circles = gpd.GeoDataFrame(subsampled_circles, geometry='subsampled_circle')
        subsampled_circles.to_file(subsampled_circles_path, driver='GeoJSON')
    else:
        subsampled_circles = gpd.read_file(subsampled_circles_path)
    if show:
        _ = polygon_plot_with_sampling_circles(polygon=city.merged_polygon, circles=subsampled_circles.geometry.tolist())
        plt.show()
    return subsampled_circles

def resample_subsampled_area(city, subsampled_circles, project_folder, subsample_radius_in_meters, subsample_step_in_degrees, show=False):
    subsampled_area = subsampled_circles.geometry.unary_union
    if show:
        gpd.GeoSeries(subsampled_area).plot(facecolor='none', edgecolor=sns.color_palette('Paired')[0])
        plt.show()
    resampled_subsampled_circles_path = project_folder / Path(f"{city.name}_{subsample_radius_in_meters}_radius_{subsample_step_in_degrees}_step_resubsampled_circles.geojson") 
    if not resampled_subsampled_circles_path.exists():
        resampled_circles = sample_polygons_with_circles(subsampled_area, subsample_radius_in_meters, subsample_step_in_degrees)
        resampled_circles = gpd.GeoDataFrame(geometry=resampled_circles).reset_index(drop=True)
        resampled_circles['searched'] = False
        resampled_circles.to_file(resampled_subsampled_circles_path, driver='GeoJSON')
    else:
        resampled_circles = gpd.read_file(resampled_subsampled_circles_path)
    if show:
        _ = polygon_plot_with_sampling_circles(polygon=subsampled_area, circles=resampled_circles.geometry.tolist())
        plt.show()
    return subsampled_area, resampled_circles

def search_places_in_polygon(root_folder,
                             plot_folder, 
                                 tag, 
                                 polygon,
                                 radius_in_meters, 
                                 step_in_degrees, 
                                 condition_rule, 
                                 recalculate=False, 
                                 show=False):
        circles_path = root_folder / Path(f"{tag}_{radius_in_meters}_radius_{step_in_degrees}_step_circles.geojson")
        places_path = root_folder / Path(f"{tag}_{radius_in_meters}_radius_{step_in_degrees}_step_places.parquet")
        polygon_with_circles_plot_path = plot_folder / Path(f"{tag}_polygon_with_circles_plot.png")
        polygon_with_circles_zoom_plot_path = plot_folder / Path(f"{tag}_polygon_with_circles_zoom_plot.png")
        polygon_with_circles_and_points_plot_path = plot_folder / Path(f"{tag}_polygon_with_circles_and_places_plot.png")
        polygon_with_circles_and_points_zoom_plot_path = plot_folder / Path(f"{tag}_polygon_with_circles_and_places_zoom_plot.png")
        circles = get_circles_search(circles_path, 
                                    polygon, 
                                    radius_in_meters, 
                                    step_in_degrees, 
                                    condition_rule=condition_rule,
                                    recalculate=recalculate)
        if show or recalculate:
            _ = polygon_plot_with_sampling_circles(polygon=polygon, 
                                                   circles=circles.geometry.tolist(),
                                                   output_file_path=polygon_with_circles_plot_path)
            if show:
                plt.show()
            random_circle = random.choice(circles.geometry.tolist())
            _ = polygon_plot_with_sampling_circles(polygon=polygon, 
                                                   circles=circles.geometry.tolist(),
                                                   point_of_interest=random_circle,
                                                   zoom_level=5*meters_to_degree(radius_in_meters, random_circle.centroid.y),
                                                   output_file_path=polygon_with_circles_zoom_plot_path)
            if show:
                plt.show()
        found_places = process_circles(circles,
                                    radius_in_meters, 
                                    places_path, 
                                    circles_path, 
                                    recalculate=recalculate)
        if show or recalculate:
            _ = polygon_plot_with_circles_and_points(polygon=polygon, 
                                                    circles=circles.geometry.tolist(), 
                                                    points=found_places[['longitude', 'latitude']].values.tolist(), 
                                                    output_file_path=polygon_with_circles_and_points_plot_path)
            if show:
                plt.show()
            random_circle = random.choice(circles.geometry.tolist())
            _ = polygon_plot_with_circles_and_points(polygon=polygon, 
                                                    circles=circles.geometry.tolist(), 
                                                    point_of_interest=random_circle,
                                                    zoom_level=5*meters_to_degree(radius_in_meters, random_circle.centroid.y),
                                                    points=found_places[['longitude', 'latitude']].values.tolist(), 
                                                    output_file_path=polygon_with_circles_and_points_zoom_plot_path)
            if show:
                plt.show()
        return circles, found_places

def get_saturated_circles(polygon, found_places, circles, threshold, show=False, output_file_path=None):
    
    places_by_circle = found_places.groupby('circle')["id"].nunique().sort_values(ascending=False)
    saturated_circles = places_by_circle[places_by_circle >= threshold].index
    saturated_circles = circles.loc[saturated_circles, :]
    _ = polygon_plot_with_circles_and_points(polygon=polygon, 
                                            circles=saturated_circles.geometry.tolist(), 
                                            points=found_places.loc[found_places['circle'].isin(saturated_circles.index), ['longitude', 'latitude']].values.tolist(), 
                                            output_file_path=output_file_path)
    if show:
        plt.show()
    return saturated_circles

def get_saturated_area(polygon, saturated_circles, show=False, output_path=None):
    saturated_area = saturated_circles.geometry.unary_union
    width = 14
    aspect_ratio = 16/9
    height = width / aspect_ratio
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
            facecolor=sns.color_palette('Paired')[0], edgecolor='none', alpha=0.5, figsize=(width, height))
    gpd.GeoSeries(saturated_area).plot(ax=ax, facecolor='none', edgecolor=sns.color_palette('Paired')[0])
    ax.set_title('Saturated Sampled Areas')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    if output_path:
        plt.savefig(output_path, dpi=500)
    if show:
            plt.show()
    return saturated_area

if __name__ == '__main__':
    
    cities_geojsons = {
        'delhi': '/Users/sebastian/Desktop/MontagnaInc/Projects/India_shapefiles/city/delhi/district/delhi_1997-2012_district.json',
        'tokyo': '/Users/sebastian/Desktop/MontagnaInc/Research/Cities_Restaurants/translated_tokyo_wards.geojson'
    }

    PROJECT_FOLDER = Path('/Users/sebastian/Desktop/MontagnaInc/Research/Cities_Restaurants/test')
    PROJECT_FOLDER.mkdir(exist_ok=True)
    PLOTS_FOLDER = PROJECT_FOLDER / 'plots'
    PLOTS_FOLDER.mkdir(exist_ok=True)
    CITY = 'tokyo'
    SHOW = True
    RECALCULATE = True

    city = CityGeojson(cities_geojsons[CITY], CITY)
    city_wards_plot_path = PLOTS_FOLDER / f"{city.name}_wards_polygons_plot.png"
    city_plot_path = PLOTS_FOLDER / f"{city.name}_polygon_plot.png"
    if SHOW:
        ax = city.plot_polygons()
        if not city_wards_plot_path.exists() or RECALCULATE:
            ax.get_figure().savefig(city_wards_plot_path, dpi=500)
        plt.show()
        ax = city.plot_unary_polygon()
        if not city_plot_path.exists() or RECALCULATE:
            ax.get_figure().savefig(city_plot_path, dpi=500)
        plt.show()

    print('STEP 1')
    RADIUS_IN_METERS = 500 #50
    STEP_IN_DEGREES = 0.0039*2 #0.00075
    TAG = f"Step-1_{city.name}"
    circles, found_places = search_places_in_polygon(PROJECT_FOLDER,
                                                     PLOTS_FOLDER,
                                                     TAG,
                                                     city.merged_polygon,
                                                     RADIUS_IN_METERS,
                                                     STEP_IN_DEGREES,
                                                     condition_rule='center',
                                                     recalculate=RECALCULATE,
                                                     show=SHOW)
    print('SATURED STEP 1')
    saturated_circles_plot_path = PLOTS_FOLDER / f"{TAG}_saturated_circles_plot.png"
    saturated_area_plot_path = PLOTS_FOLDER / f"{TAG}_saturated_area_plot.png"
    saturated_circles = get_saturated_circles(city.merged_polygon, 
                                              found_places, 
                                              circles, 
                                              threshold=12, 
                                              show=SHOW, 
                                              output_file_path=saturated_circles_plot_path)
    saturated_area = get_saturated_area(city.merged_polygon, saturated_circles, show=SHOW, output_path=saturated_area_plot_path)
    plt.close('all')
    print('STEP 2')
    RADIUS_IN_METERS2 = 100
    STEP_IN_DEGREES2 = STEP_IN_DEGREES * (RADIUS_IN_METERS2 / RADIUS_IN_METERS)
    TAG = f"Step-2_{city.name}"
    circles, found_places = search_places_in_polygon(PROJECT_FOLDER,
                                                     PLOTS_FOLDER,
                                                     TAG,
                                                     saturated_area,
                                                     RADIUS_IN_METERS2,
                                                     STEP_IN_DEGREES2,
                                                     condition_rule='center',
                                                     recalculate=RECALCULATE,
                                                     show=SHOW)
    print('SATURED STEP 2')
    saturated_circles_plot_path = PLOTS_FOLDER / f"{TAG}_saturated_circles_plot.png"
    saturated_area_plot_path = PLOTS_FOLDER / f"{TAG}_saturated_area_plot.png"
    saturated_circles = get_saturated_circles(saturated_area, 
                                              found_places, 
                                              circles, 
                                              threshold=15, 
                                              show=SHOW, 
                                              output_file_path=saturated_circles_plot_path)
    saturated_area = get_saturated_area(city.merged_polygon, saturated_circles, show=SHOW, output_path=saturated_area_plot_path)
    plt.close('all')
    print('STEP 3')
    TAG = f"Step-3_{city.name}"
    RADIUS_IN_METERS3 = 50
    STEP_IN_DEGREES3 = STEP_IN_DEGREES2 * (RADIUS_IN_METERS3 / RADIUS_IN_METERS2)
    circles, found_places = search_places_in_polygon(PROJECT_FOLDER,
                                                     PLOTS_FOLDER,
                                                     TAG,
                                                     saturated_area,
                                                     RADIUS_IN_METERS3,
                                                     STEP_IN_DEGREES3,
                                                     condition_rule='center',
                                                     recalculate=RECALCULATE,
                                                     show=SHOW)
