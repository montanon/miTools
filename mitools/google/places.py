import math
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, List, NewType, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import shapely
from matplotlib.axes import Axes
from pandas import DataFrame, Series
from shapely.geometry import Point, Polygon
from shapely.ops import transform

CircleType = NewType('CircleType', Polygon)

NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
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
            + 'places.iconMaskBaseUri,places.location,places.primaryType,places.primaryTypeDisplayName,places.plusCode,' \
                + 'places.shortFormattedAddress,places.subDestinations,places.types,places.utcOffsetMinutes,places.viewport'
QUERY_HEADERS = {
    'Content-Type': 'aplication/json',
    'X-Goog-Api-Key': GOOGLE_PLACES_API_KEY,
    'X-Goog-FieldMask': FIELD_MASK
}

class CityGeojson:
    def __init__(self, geojson_path: PathLike):
        self.geojson_path = Path(geojson_path)
        self.data = gpd.read_file(geojson_path)
        self.polygons = self.data['geometry']
        self.merged_polygon = self.polygons.unary_union
        self.bounds = self.polygons.bounds.iloc[0].values
    
    def plot_unary_polygon(self):
        ax = gpd.GeoSeries(self.merged_polygon).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0])
        return ax

    def plot_polygons(self):
        ax = gpd.GeoSeries(self.polygons).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0])
        return ax

class NearbySearchRequest:
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

    def json_query(self):
        query = {
            'includedTypes': self.included_types,
            'maxResultCount': self.max_result_count,
            'locationRestriction': self.location_restriction,
            'languageCode': self.language_code,
            }
        return query
    
@dataclass
class AddressComponent:
    longText: str
    shortText: str
    types: List[str]
    languageCode: str

@dataclass
class PlusCode:
    globalCode: str
    compoundCode: str

@dataclass
class Location:
    latitude: float
    longitude: float

@dataclass
class ViewportCoordinate:
    latitude: float
    longitude: float

@dataclass
class Viewport:
    low: ViewportCoordinate
    high: ViewportCoordinate

@dataclass
class DisplayName:
    text: str
    languageCode: str

@dataclass
class AccessibilityOptions:
    wheelchairAccessibleSeating: bool

@dataclass
class Place:
    types: List[str]
    formattedAddress: str
    addressComponents: List[AddressComponent]
    plusCode: PlusCode
    location: Location
    viewport: Viewport
    googleMapsUri: str
    utcOffsetMinutes: int
    adrFormatAddress: str
    businessStatus: str
    iconMaskBaseUri: str
    iconBackgroundColor: str
    displayName: DisplayName
    primaryTypeDisplayName: DisplayName
    primaryType: str
    shortFormattedAddress: str
    accessibilityOptions: AccessibilityOptions

    @staticmethod
    def from_json(data: str) -> 'Place':
        return Place(
            types=data['types'],
            formattedAddress=data['formattedAddress'],
            addressComponents=[AddressComponent(**comp) for comp in data['addressComponents']],
            plusCode=PlusCode(**data['plusCode']),
            location=Location(**data['location']),
            viewport=Viewport(
                low=ViewportCoordinate(**data['viewport']['low']),
                high=ViewportCoordinate(**data['viewport']['high'])
            ),
            googleMapsUri=data['googleMapsUri'],
            utcOffsetMinutes=data['utcOffsetMinutes'],
            adrFormatAddress=data['adrFormatAddress'],
            businessStatus=data['businessStatus'],
            iconMaskBaseUri=data['iconMaskBaseUri'],
            iconBackgroundColor=data['iconBackgroundColor'],
            displayName=DisplayName(**data['displayName']),
            primaryTypeDisplayName=DisplayName(**data['primaryTypeDisplayName']),
            primaryType=data['primaryType'],
            shortFormattedAddress=data['shortFormattedAddress'],
            accessibilityOptions=AccessibilityOptions(**data['accessibilityOptions'])
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

def sample_polygon_with_circle(polygon: Polygon, 
                               radius_in_meters: float, 
                               step_in_degrees: float) -> List[CircleType]:
    if not polygon.is_valid:
        raise ValueError('Invalid Polygon')
    minx, miny, maxx, maxy = polygon.bounds
    circles = []
    for lat in np.arange(miny, maxy, step_in_degrees):
        for lon in np.arange(minx, maxx, step_in_degrees):
            point = Point(lon, lat)
            deg_radius = meters_to_degree(distance_in_meters=radius_in_meters, reference_latitude=lat)
            circle = point.buffer(deg_radius)
            if polygon.contains(point) or polygon.intersects(circle):
                circles.append(circle)
    return circles

def sample_polygons_with_circles(polygons: Union[Iterable[Polygon], Polygon], 
                                 radius_in_meters: float, 
                                 step_in_degrees: float) -> List[CircleType]:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    circles = []
    for polygon in polygons:
        circles.extend(sample_polygon_with_circle(polygon=polygon, 
                                                  radius_in_meters=radius_in_meters, 
                                                  step_in_degrees=step_in_degrees))
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
                                       output_file_path: Optional[PathLike]=None) -> Axes:
    minx, miny, maxx, maxy = polygon.bounds
    out_circles, in_circles = [], []
    for circle in circles:
        if polygon.contains(circle) or polygon.intersects(circle):
            in_circles.append(circle)
        else:
            out_circles.append(circle)
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
            facecolor=sns.color_palette('Paired')[0], edgecolor='none', alpha=0.5)
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
    ax = gpd.GeoSeries(out_circles).plot(
        facecolor='none', edgecolor='r', ax=ax, alpha=0.5, label='Out Circles')
    ax = gpd.GeoSeries(in_circles).plot(
        facecolor='none', edgecolor='g', ax=ax, alpha=0.5, label='In Circles')
    out_circle_proxy = mlines.Line2D(
        [], [], color='r', marker='o', markersize=10, label='Out Circles', linestyle='None')
    in_circle_proxy = mlines.Line2D(
        [], [], color='g', marker='o', markersize=10, label='In Circles', linestyle='None')
    ax.legend(handles=[out_circle_proxy, in_circle_proxy],
              loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title('Sampling of POIs')
    ax.set_xticks([])
    ax.set_yticks([])
    if output_file_path:
        plt.savefig(output_file_path)
    return ax


if __name__ == '__main__':
    
    dehli_geojson = 'delhi_1997-2012_district.json'
    dehli = CityGeojson(dehli_geojson)

    ax1 = dehli.plot_unary_polygon()
    ax2 = dehli.plot_polygons()

    RADIUS_IN_METERS = 1_750
    STEP_IN_DEGREES = 0.025
    circles = sample_polygons_with_circles(polygons=dehli.merged_polygon, 
                                           radius_in_meters=RADIUS_IN_METERS, 
                                           step_in_degrees=STEP_IN_DEGREES)
    ax3 = polygon_plot_with_sampling_circles(polygon=dehli.merged_polygon, 
                                             circles=circles)

    queries = [NearbySearchRequest(circle, distance_in_meters=RADIUS_IN_METERS, included_types=RESTAURANT_TYPES).json_query() for circle in circles]

    for query in queries:
        response = requests.post(NEARBY_SEARCH_URL, headers=QUERY_HEADERS, json=query)
        import json
        response_data = response.json()['places']
        json.dump(response_data, open('response.json', 'w'))
        for place in response_data:
            place = Place.from_json(place)
            print(place.to_series())
            break
        break
